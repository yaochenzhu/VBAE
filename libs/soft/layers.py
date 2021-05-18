import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from tensorflow_probability import distributions as tfp


class DenseForSparse(layers.Layer):
    '''
        Dense layer where the input is a sparse matrix.
    '''
    def __init__(self, in_dim, out_dim, activation=None, **kwargs):
        super(DenseForSparse, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activations.get(activation)

        self.kernel = self.add_weight(
            "kernel", shape=(self.in_dim, self.out_dim),
            trainable=True, initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            "bias", shape=(self.out_dim),
            trainable=True, initializer="zeros",
        )

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            outputs = tf.add(tf.sparse.matmul(inputs, self.kernel), self.bias)
        else:
            outputs = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        return self.activation(outputs)


class TransposedSharedDense(layers.Layer):
    '''
        Dense layer that shares weights (transposed) and bias 
        with another dense layer.
    '''

    def __init__(self, weights, activation=None, **kwargs):
        super(TransposedSharedDense, self).__init__(**kwargs)
        assert(len(weights) in [1, 2]), \
            "Specify the [kernel] or the [kernel] and [bias]."
        self.W = weights[0]

        if len(weights) == 1:
            b_shape = self.W.shape.as_list()[0]
            self.b = self.add_weight(shape=(b_shape),
                name="bias",
                trainable=True,
                initializer="zeros")
        else:
            self.b = weights[1]
        self.activate = activations.get(activation)

    def call(self, inputs):
        return self.activate(K.dot(inputs, K.transpose(self.W))+self.b)


class AddGaussianLoss(layers.Layer):
    '''
        Add the KL divergence between the variational 
        Gaussian distribution and the prior to loss.
    '''
    def __init__(self,  **kwargs):
        super(AddGaussianLoss, self).__init__(**kwargs)             
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)

    def call(self, inputs):
        mu, std  = inputs
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                              scale_diag=K.ones_like(std))    
        kl_loss  = self.lamb_kl*K.mean(tfp.kl_divergence(var_dist, pri_dist))
        return kl_loss


class AddBetaLoss(layers.Layer):
    '''
        Add the KL divergence between the variational 
        Beta distribution and the prior to loss
    '''
    def __init__(self, prior_a=1, offset=1, **kwargs):
        super(AddBetaLoss, self).__init__(**kwargs)
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)
        self.a = np.ones((1, 2)).astype(np.float32)*prior_a
        self.a[0, 0] = self.a[0, 0]*offset
        self.pri_mu = tf.constant((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.pri_var = tf.constant((((1.0/self.a) * (1 - (2.0/2.0))).T +
                                     (1.0/(2.0*2.0))*np.sum(1.0/self.a,1)).T)

    def call(self, inputs):
        if len(inputs) == 2:
            mu_2, std = inputs
            mu = mu_2 / 2
            kl_loss = self.lamb_kl*0.5*tf.reduce_mean((tf.reduce_sum(tf.div(std**2,self.pri_var),1)+\
            tf.reduce_sum(tf.multiply(tf.div((self.pri_mu - mu), self.pri_var),
                                             (self.pri_mu - mu)), 1) - 2 +\
                          tf.reduce_sum(tf.log(self.pri_var), 1) - 2*tf.reduce_sum(tf.log(std), 1)))
        else:
            mu = inputs / 2
            kl_loss = self.lamb_kl*tf.reduce_mean(tf.reduce_sum(tf.multiply(
                tf.div((self.pri_mu[None,:,0] - mu), self.pri_var[None,:,0]),
                (self.pri_mu[None,:,0] - mu)), 1))
        return kl_loss

class ReparameterizeGaussian(layers.Layer):
    '''
        Rearameterization trick for Gaussian
    '''
    def __init__(self, **kwargs):
        super(ReparameterizeGaussian, self).__init__(**kwargs)

    def call(self, stats):
        mu, std = stats
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return dist.sample()


class ReparameterizeBeta(layers.Layer):
    '''
        Rearameterization trick for Beta distribution
    '''
    def __init__(self, fixed_std=None, **kwargs):
        super(ReparameterizeBeta, self).__init__(**kwargs)
        self.std = fixed_std

    def call(self, stats):
        if len(stats) == 2:
            mu_2, std = stats
            gauss_add = tfp.MultivariateNormalDiag(loc=mu_2/2, scale_diag=std).sample()
            gauss_sub = tfp.MultivariateNormalDiag(loc=-mu_2/2, scale_diag=std).sample()
            beta_samples = K.sigmoid(gauss_add - gauss_sub)
        else:
            mu_2 = stats
            assert self.std is not None
            gauss_dist = tfp.Normal(0, 1)
            gauss_add = gauss_dist.sample(K.shape(mu_2))
            gauss_sub = gauss_dist.sample(K.shape(mu_2))
            beta_samples = K.sigmoid(mu_2 + self.std * (gauss_add - gauss_sub))
        return beta_samples