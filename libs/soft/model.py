import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import network

from evaluate import mse
from layers import DenseForSparse
from layers import TransposedSharedDense
from layers import AddBetaLoss, ReparameterizeBeta
from layers import AddGaussianLoss, ReparameterizeGaussian

class MLP(network.Network):
    '''
        Multilayer Perceptron (MLP). Suitable for both dense and sparse 
        input. If the input is sparsethe input_size must be specified.
    '''
    def __init__(self, 
                 hidden_sizes,
                 activations,
                 sparse_input=False,
                 input_size=None,
                 l2_normalize=True,
                 dropout_rate=0.5,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)

        self.dense_list = []
        self.sparse_input = sparse_input
        self.dropout_rate = dropout_rate
        self.l2_normalize = l2_normalize

        if self.sparse_input:
            assert input_size is not None, "if the inputs is sparse,\
                you must specify the input shape."
            self.dense_list.append(
                DenseForSparse(input_size, hidden_sizes[0], 
                    activation=activations[0], name="mlp_dense_0"
            ))
            hidden_sizes = hidden_sizes[1:]
            activations = activations[1:]

        for i, (size, activation) in enumerate(zip(hidden_sizes, activations)):
            self.dense_list.append(
                layers.Dense(size, activation=activation,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="mlp_dense_{}".format(i+1 if self.sparse_input else i)
            ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:], sparse=self.sparse_input)
        h_mid = x_in
        if self.dropout_rate and not self.sparse_input:
            if self.l2_normalize:
                h_mid = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})(h_mid)
            h_mid = layers.Dropout(self.dropout_rate)(h_mid)
        h_out = self.dense_list[0](h_mid)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, self.m_name)
        super(MLP, self).build(input_shapes)


class SymetricMLP(network.Network):
    '''
        The symetric version of an MLP. if reuse_weights is true, the transposed 
        weights, and bias of the corresponding layers of source MLP are reused.
    '''
    def __init__(self,
                 source_mlp,
                 activations,
                 reuse_weights=False,
                 **kwargs):
        super(SymetricMLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_list = []

        if reuse_weights:
            '''
                If resuse_weights is true, for the ith (i<L) layer of 
                the decoder, use the kernel and bias from the L-i+1th 
                and L-ith layer of the encoder.
            '''
            for i, (dense_W, dense_b) in enumerate(
                zip(source_mlp.dense_list[-1:0:-1], 
                    source_mlp.dense_list[-2::-1])):
                ### Fetch the weights from source layers.
                weights = [dense_W.weights[0], dense_b.weights[1]]
                self.dense_list.append(
                    TransposedSharedDense(weights=weights,
                    activation=activations[i],
                    name="sym_mlp_dense_{}".format(i)
                ))
            '''
                And for the Lth layer of the decoder, it only uses the 
                tranposed weights from the first layer of the encoder, 
                the bias is reiniatilized from the scratch.
            '''
            weights = [source_mlp.dense_list[0].weights[0]]
            self.dense_list.append(
                TransposedSharedDense(weights=weights, activation=activations[-1]
            ))
        else:
            '''
                Else, only get the hidden sizes from the source encoder
                and  all the weights are reiniatialized from the scratch.
            '''
            for i, dense in enumerate(source_mlp.dense_list[-1::-1]):
                hidden_size = dense.weights[0].shape.as_list()[0]
                self.dense_list.append(
                    layers.Dense(hidden_size, activation=activations[i],
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="sym_mlp_dense_{}".format(i)
                ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_out = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, name=self.m_name)
        super(SymetricMLP, self).build(input_shapes)


class SamplerGaussian(network.Network):
    '''
        Sample from the variational Gaussian, and add its KL 
        with the prior to loss
    '''
    def __init__(self, **kwargs):
        super(SamplerGaussian, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.rep_gauss = ReparameterizeGaussian()

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        self._init_graph_network([mean, std], [sample], name=self.m_name)
        super(SamplerGaussian, self).build(input_shapes)


class SamplerBeta(network.Network):
    '''
        Sample from the variational Beta, and add its KL 
        with the prior to loss
    '''
    def __init__(self, fixed_std, **kwargs):
        super(SamplerBeta, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.rep_beta = ReparameterizeBeta(fixed_std)

    def build(self, input_shapes):
        logits = layers.Input(input_shapes[1:])
        sample = self.rep_beta(logits)
        self._init_graph_network([logits], [sample], name=self.m_name)
        super(SamplerBeta, self).build(input_shapes)


class ContentLatentCore(network.Network):
    '''
        The latent core for the content network, which takes the ouput 
        from the encoder, gets the mean and logstd of the hidden Gaussian
        draws a sample, and applies the first dense layer of the decoder.
    '''
    def __init__(self,  
                 latent_size, 
                 out_size, 
                 activation, 
                 **kwargs):
        super(ContentLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
        self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")
        self.z_sampler = SamplerGaussian(name="z_sampler")
        self.dense_out = layers.Dense(out_size, activation=activation, name="latent_out")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean = self.dense_mean(x_in)
        std  = self.exp(self.clip(self.dense_std(x_in)))
        h_mid = self.z_sampler([mean, std]) 
        y_out = self.dense_out(h_mid)
        self._init_graph_network(x_in, [y_out, mean, std, h_mid], name=self.m_name)
        super(ContentLatentCore, self).build(input_shapes)


class CollaborativeLatentCore(network.Network):
    '''
        The latent core for the collaborative network, which takes 
        the ouput from the encoder, gets the mean and logstd of the
        hidden Gaussian, and the prob of the hidden Beta, draws
        a sample, and applies the first dense layer of the decoder.
    '''
    def __init__(self,  
                 latent_size, 
                 out_size, 
                 activation,
                 fixed_std,
                 kernel_init=1.0,
                 bias_init=0.0,
                 **kwargs):
        super(CollaborativeLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        kernel_init = initializers.Constant(value=kernel_init)
        bias_init = initializers.Constant(value=bias_init)
        self.dense_channel = layers.Dense(1, kernel_initializer=kernel_init, 
                                          bias_initializer=bias_init, name="channel")
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
        self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")
        self.l2_normalize = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})
        self.get_l2_norm  = layers.Lambda(lambda x:-tf.norm(x, axis=1, keepdims=True))
        self.batch_norm = layers.BatchNormalization()
        self.stop_grad = layers.Lambda(lambda x:tf.stop_gradient(x))
        self.z_sampler = SamplerGaussian(name="z_sampler")
        self.d_sampler = SamplerBeta(fixed_std, name="d_sampler")
        self.dense_out = layers.Dense(out_size, activation=activation, name="latent_out")

    def build(self, input_shapes):
        h_mid = layers.Input(input_shapes[0][1:])
        z_t = layers.Input(input_shapes[1][1:])
        h_mid_dirc = self.l2_normalize(h_mid)        
        h_mid_norm = self.get_l2_norm(h_mid)
        mean = self.dense_mean(h_mid_dirc)
        std = self.exp(self.clip(self.dense_std(h_mid_dirc)))
        z_b = self.z_sampler([mean, std])
        logits = self.dense_channel(self.batch_norm(self.stop_grad(h_mid_norm)))
        d = self.d_sampler(logits)
        v_mid = layers.Lambda(lambda x: (x[0] + x[1]*x[2]))([z_b, d, z_t])
        y_out = self.dense_out(v_mid)
        self._init_graph_network([h_mid, z_t], [y_out, mean, std, logits, z_b, d], name=self.m_name)
        super(CollaborativeLatentCore, self).build(input_shapes)


class LayerwisePretrainableContentVAE():
    '''
        The Layerwise Pretrainable Content Variational Auto-encoder
    '''
    def __init__(self, 
                 input_shapes,
                 hidden_sizes,
                 encoder_activs,
                 decoder_activs,
                 latent_size,
                 latent_activ):
        self.input_shapes = input_shapes
        self.latent_size = latent_size
        self.encoder = MLP(hidden_sizes, encoder_activs, dropout_rate=0, l2_normalize=False, name="encoder")
        self.encoder.build(input_shapes=input_shapes)
        self.latent_core = ContentLatentCore(latent_size, hidden_sizes[-1], 
                                             latent_activ, name="latent_core")
        self.decoder = SymetricMLP(self.encoder, decoder_activs, reuse_weights=True, name="decoder")
        
        ### Initialize the weights for latent core and decoder (Optitional)
        self.latent_core.build(input_shapes=[None, hidden_sizes[-1]])
        self.decoder.build(input_shapes=[None, hidden_sizes[-1]])

    def build_peri_pretrain(self, index):
        '''
            Pair the ith layer of encoder and the L-i+1th 
            layer of decoder as an auto-encoder for pretraining
        '''
        depth = len(self.encoder.dense_list)
        assert index < depth, "index out {} of range {}!".format(index, depth)
        if not hasattr(self, "peri_pretrains"):
            self.peri_pretrains = {}

        if not index in self.peri_pretrains.keys():
            src_dense = self.encoder.dense_list[index]
            sym_dense = self.decoder.dense_list[depth-index-1]
            x_in = layers.Input(shape=(src_dense.input.shape.as_list()[-1],),
                                name="peri_pretrain_{}_input".format(index))
            x_rec = sym_dense(src_dense(x_in))
            self.peri_pretrains[index] = models.Model(inputs=x_in, outputs=x_rec)
        return self.peri_pretrains[index]

    def build_core_pretrain(self):
        '''
            Get the latent core for pretraining
        '''
        if not hasattr(self, "core_pretrain"):
            x_in = layers.Input(shape=(self.latent_core.input.shape.as_list()[-1],),
                                name="core_pretrain_input")
            x_rec, mean, std, _ = self.latent_core(x_in)
            self.core_pretrain = models.Model(inputs=x_in, outputs=x_rec)
            self.core_pretrain.add_loss(AddGaussianLoss()([mean, std]))
        return self.core_pretrain

    def build_vae_pretrain(self):
        '''
            Get the whole vae model
        '''
        if not hasattr(self, "vae_pretrain"):
            x_in = layers.Input(shape=self.input_shapes[1:], name="contents")
            h_mid = self.encoder(x_in)
            h_mid, mean, std, _ = self.latent_core(h_mid)
            x_rec = self.decoder(h_mid)
            self.vae_pretrain = models.Model(inputs=x_in, outputs=x_rec)
            kl_loss = AddGaussianLoss()([mean, std])
            self.vae_pretrain.add_loss(kl_loss)
        return self.vae_pretrain

    def build_vbae_tstep(self, collab_decoder, lambda_W):
        '''
            Get the content module for the vbae model
        '''
        if not hasattr(self, "vbae_tstep"):
            x_in = layers.Input(shape=self.input_shapes[1:], name="contents")
            z_b  = layers.Input(shape=[self.latent_size,], name="collab_embed")
            d = layers.Input(shape=[1,], name="channel")
            h_mid = self.encoder(x_in)
            h_mid, mean, std, z_t = self.latent_core(h_mid)
            x_rec = self.decoder(h_mid)
            v_mid = layers.Lambda(lambda x: (x[0] + x[1]*x[2]))([z_b, d, z_t])
            r_rec = collab_decoder(v_mid)
            self.vbae_tstep = models.Model(inputs=[x_in, z_b, d], outputs=[x_rec, r_rec])
            kl_loss = AddGaussianLoss()([mean, std])
            reg_loss = tf.nn.l2_loss(self.vbae_tstep.layers[1].dense_list[0].weights[0]) + \
                       tf.nn.l2_loss(self.vbae_tstep.layers[1].dense_list[1].weights[0]) + \
                       tf.nn.l2_loss(tf.transpose(self.vbae_tstep.layers[1].dense_list[0].weights[0])) + \
                       tf.nn.l2_loss(tf.transpose(self.vbae_tstep.layers[1].dense_list[1].weights[0]))
            self.vbae_tstep.add_loss(kl_loss + lambda_W*reg_loss)
        return self.vbae_tstep

    def build_vbae_infer_tstep(self):
        '''
            Get the inference part of the vbae model
        '''
        if not hasattr(self, "vbae_infer_tstep"):
            x_in = layers.Input(shape=self.input_shapes[1:], name="contents")
            h_mid = self.encoder(x_in)
            _, mean, _, _ = self.latent_core(h_mid)
            self.vbae_infer_tstep = models.Model(inputs=x_in, outputs=mean)
        return self.vbae_infer_tstep

    def load_weights(self, weight_path):
        '''
            Load weights from pretrained vae
        '''
        vae = self.build_vae_pretrain()
        vae.load_weights(weight_path)


class CollarboativeBandwidthVAE():
    '''
       	The collaborative VAE with bandwidth bottleneck structure
    '''
    def __init__(self, 
                 input_shapes,
                 hidden_sizes,
                 latent_size,
                 encoder_activs,
                 decoder_activs,
                 latent_activ,
                 prior_a=20,
                 dropout_rate=0.5,
                 kernel_init=1.0,
                 bias_init=0.0):

        self.input_shape = input_shapes[-1]
        self.latent_size = latent_size
        self.prior_a = prior_a

        self.encoder = MLP(hidden_sizes, activations=encoder_activs, l2_normalize=False, 
                           input_size=input_shapes[-1], dropout_rate=dropout_rate, name="Encoder")
        self.encoder.build(input_shapes=input_shapes)
        self.latent_core = CollaborativeLatentCore(latent_size, hidden_sizes[-1], fixed_std=0.1, 
                                                   activation=latent_activ, kernel_init=kernel_init,
                                                   bias_init=bias_init, name="LatentCore")
        self.decoder = SymetricMLP(self.encoder, activations=decoder_activs, name="Decoder")

    def build_vbae_bstep(self, lambda_W):
        '''
            Get the collaborative module for the vbae model
        '''
        self.lambda_W = lambda_W
        if not hasattr(self, "vbae_bstep") or self.lambda_W != lambda_W:
            r_in = layers.Input(shape=[self.input_shape,], name="ratings")
            z_t  = layers.Input(shape=[self.latent_size,], name="content_embed")
            h_mid = self.encoder(r_in)
            h_mid, mu, std, logits, z_b, d = self.latent_core([h_mid, z_t])
            r_rec = self.decoder(h_mid)
            inputs = [r_in, z_t]

            self.vbae_bstep = models.Model(inputs=inputs, outputs=r_rec)
            reg_loss = tf.nn.l2_loss(self.vbae_bstep.layers[1].dense_list[0].weights[0]) + \
                       tf.nn.l2_loss(self.vbae_bstep.layers[-1].dense_list[0].weights[0])

            self.vbae_bstep.Wc0 = self.vbae_bstep.add_weight(name="wc0", shape=(), trainable=False)
            self.vbae_bstep.Wc1 = self.vbae_bstep.add_weight(name="wc1", shape=(), trainable=False)
            chn_loss = tf.nn.l2_loss(self.vbae_bstep.layers[3].dense_channel.weights[0]) * self.vbae_bstep.Wc0 + \
                       tf.nn.l2_loss(self.vbae_bstep.layers[3].dense_channel.weights[1]) * self.vbae_bstep.Wc1
                       
            self.add_gauss_loss = AddGaussianLoss()
            self.add_beta_loss = AddBetaLoss(prior_a=self.prior_a)
            kl_loss = self.add_gauss_loss([mu, std]) \
                    + self.add_beta_loss(logits)
            self.vbae_bstep.add_loss(lambda_W*reg_loss + chn_loss + kl_loss)
        return self.vbae_bstep

    def build_vbae_eval(self):
        '''
            For evaluation, use the mean deterministically
        '''
        if not hasattr(self, "vbae_eval"):
            r_in = layers.Input(shape=[self.input_shape,], name="ratings")
            mu_t  = layers.Input(shape=[self.latent_size,], name="content_embed")
            h_mid = self.encoder(r_in)
            _, mu_b, _, _, _, d = self.latent_core([h_mid, mu_t])
            v_mid = layers.Lambda(lambda x: (x[0] + x[1]*x[2]))([mu_b, d, mu_t])
            h_mid = self.latent_core.dense_out(v_mid)
            r_rec = self.decoder(h_mid)
            self.vbae_eval = models.Model(inputs=[r_in, mu_t], outputs=r_rec)
        return self.vbae_eval        

    def build_vbae_infer_bstep(self, return_logits=True):
        '''
            Get the inference part of the vbae model
        '''
        if not hasattr(self, "vbae_infer_bstep") or self.return_logits != return_logits:
            self.return_logits = return_logits
            r_in = layers.Input(shape=[self.input_shape,], name="ratings")
            h_mid = self.encoder(r_in)
            mu = self.latent_core.dense_mean(self.latent_core.l2_normalize(h_mid))
            logits = self.latent_core.dense_channel(
            	self.latent_core.batch_norm(self.latent_core.get_l2_norm(h_mid)
            ))
            d  = self.latent_core.d_sampler(logits)
            outputs = [mu, logits, d] if return_logits else [mu, d]
            self.vbae_infer_bstep = models.Model(inputs=r_in, outputs=outputs)
        return self.vbae_infer_bstep

    def build_vbae_recon_bstep(self):
        '''
            Get the reconstruct part of the model
        '''
        if not hasattr(self, "vbae_recon_bstep"):
            r_in = layers.Input(shape=[self.latent_size], name="ratings")
            h_mid = self.latent_core.dense_out(r_in)
            r_rec = self.decoder(h_mid)
            self.vbae_recon_bstep = models.Model(inputs=r_in, outputs=r_rec, name="collab_decoder")
        return self.vbae_recon_bstep

    def load_weights(self, weight_path):
        '''
            Load weights from pretrained vbae
        '''
        if not hasattr(self, "lambda_W"):
            self.lambda_W = 0
        vae_model = self.build_vbae_bstep(self.lambda_W)
        vae_model.load_weights(weight_path)


if __name__ == "__main__":
    pass