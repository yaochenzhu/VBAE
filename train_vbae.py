import os
import time
import json
import logging
import argparse

import sys
sys.path.append(os.path.join("libs", "soft"))

from utils import Init_logging
from utils import PiecewiseSchedule

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from data import ContentVaeDataGenerator
from data import CollaborativeVAEDataGenerator
from pretrain_vae import get_content_vae
from model import CollarboativeBandwidthVAE

from evaluate import binary_crossentropy
from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k
from evaluate import multinomial_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

class Params():
    def __init__(self, W):
        self.lambda_W = W

citeulike_a_args = {
    "hidden_sizes":[300], 
    "latent_size":50,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["softmax"],
    "latent_activ" : "tanh",
    "dropout_rate" : 0.5,
    "bias_init" : 0.4
}

citeulike_t_args = {
    "hidden_sizes": [200], 
    "latent_size": 50,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["softmax"],
    "latent_activ" : "tanh",
    "dropout_rate" : 0.5,
    "bias_init" : 0.4
}

toys_args = {
    "hidden_sizes":[150], 
    "latent_size":100,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["softmax"],
    "latent_activ" : "tanh",
    "dropout_rate" : 0.5,
    "bias_init" : 0.4    
}

name_args_dict = {
    "citeulike-a"  : citeulike_a_args,
    "citeulike-t"  : citeulike_t_args,
    "toys" : toys_args,
}

name_loss_dict = {
    "citeulike-a"  : binary_crossentropy,
    "citeulike-t"  : binary_crossentropy,
    "toys" : binary_crossentropy,
}


def get_collabo_vae(params, input_shapes):
    get_collabo_vae = CollarboativeBandwidthVAE(
         input_shapes = input_shapes,        
         **params,
    )
    return get_collabo_vae


def infer_bstep(vbae_infer_bstep, X, batch_size=4000):
    num_users = X.shape[0]
    z_size, logits_size, d_size = [out.shape.as_list()[-1] for out in vbae_infer_bstep.outputs]
    z_b = np.zeros((num_users, z_size), dtype=np.float32)
    logits = np.zeros((num_users, logits_size), dtype=np.float32)
    d = np.zeros((num_users, d_size), dtype=np.float32)
    for i in range(num_users//batch_size+1):
        z_b[i*batch_size:(i+1)*batch_size], logits[i*batch_size:(i+1)*batch_size], \
        d[i*batch_size:(i+1)*batch_size] = vbae_infer_bstep.predict_on_batch(X[i*batch_size:(i+1)*batch_size].toarray())
    return z_b, logits, d


def infer_tstep(vbae_infer_tstep, features, batch_size=4000):
    num_users = len(features)
    z_size = vbae_infer_tstep.output.shape.as_list()[-1]
    z_t = np.zeros((num_users, z_size), dtype=np.float32)
    for i in range(num_users//batch_size+1):
        z_t[i*batch_size:(i+1)*batch_size] \
             = vbae_infer_tstep.predict_on_batch(features[i*batch_size:(i+1)*batch_size])
    return z_t


def summary(save_root, logs, epoch):
    save_train = os.path.join(save_root, "train")
    save_val = os.path.join(save_root, "val")
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    if not os.path.exists(save_val):
        os.makedirs(save_val)
    writer_train = tf.summary.FileWriter(save_train)
    writer_val = tf.summary.FileWriter(save_val)
    for metric, value in logs.items():
        if isinstance(value, list):
            value = value[0]
        summary = tf_summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        if "val" in metric:
            summary_value.tag = metric[4:]
            writer_val.add_summary(summary, epoch)
        else:
            summary_value.tag = metric
            writer_train.add_summary(summary, epoch)
    writer_val.flush(); writer_val.flush()


def train_vbae_model():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, \
        help="specify the dataset for experiment")
    parser.add_argument("--split", type=int, default=1,
        help="specify the split of dataset for experiment")
    parser.add_argument("--batch_size", type=int, default=500,
        help="specify the batch size for updating vbae")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    parser.add_argument("--pretrain_root", type=str, default=None,
        help="specify the root for pretrained model (optional)")
    parser.add_argument("--param_path", type=str, default=None,
        help="specify the path of hyper parameter (if any)")
    parser.add_argument("--save_root", type=str, default=None,
        help="specify the prefix for save root (if any)")
    parser.add_argument("--summary", default=False, action="store_true",
        help="whether or not write summaries to the results")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the train, val data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split))

    tstep_train_gen = ContentVaeDataGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size, joint=True,
    )
    tstep_valid_gen =  ContentVaeDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8, joint=True,
    )

    ### Get the train, val data generator for vbae
    bstep_train_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size, joint=True,
    )
    bstep_valid_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8, joint=True,
    )

    if args.dataset == "citeulike-a" or args.dataset == "toys":
        blr_schedule = PiecewiseSchedule([[0, 1e-3], [100, 1e-3], [101, 1e-4]], outside_value=1e-4)
        tlr_schedule = PiecewiseSchedule([[0, 1e-3], [50,  1e-3], [51 , 1e-4]], outside_value=1e-4)
        w0_schedule = PiecewiseSchedule([[0, 0.25], [21, 0.25]], outside_value=0.50)
        w1_schedule = PiecewiseSchedule([[0, 2.50], [20, 2.50]], outside_value=2.50)
        params = Params(2e-4); epochs = 125
    elif args.dataset == "citeulike-t":
        blr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [101, 1e-4]], outside_value=1e-4)
        tlr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 1e-4]], outside_value=1e-4)
        w0_schedule = PiecewiseSchedule([[0, 0.25], [21, 0.25]], outside_value=0.50)
        w1_schedule = PiecewiseSchedule([[0, 2.50], [20, 2.50]], outside_value=2.50)
        params = Params(8e-4); epochs = 125
        
    ### Make sure the data order is aligned between two data generator
    assert np.all(tstep_train_gen.user_ids==bstep_train_gen.user_ids)
    assert np.all(tstep_valid_gen.user_ids==bstep_valid_gen.user_ids)
    tstep_train_gen.set_ratings(bstep_train_gen.Y)
    tstep_valid_gen.set_ratings(bstep_valid_gen.Y)

    ### Build the t and b step vbae model
    if not args.pretrain_root:
        pretrain_root = os.path.join("models", args.dataset, str(args.split), "pretrained")
    else:
        pretrain_root = args.pretrain_root

    pretrain_weight_path = os.path.join(pretrain_root, "weights.model")
    pretrain_params_path = os.path.join(pretrain_root, "hyperparams.json")
    with open(pretrain_params_path, "r") as param_file:
        pretrain_params = json.load(param_file)
    content_vae = get_content_vae(pretrain_params, tstep_train_gen.feature_dim)
    content_vae.load_weights(pretrain_weight_path)

    if args.param_path is not None:
        try:
            with open(args.param_path, "r") as param_file:
                train_params = json.load(param_file)
        except:
            print("Fail to load hyperparams from file, use default instead!")
            train_params = name_args_dict[args.dataset]
    else:
        train_params = name_args_dict[args.dataset]

    collabo_vae = get_collabo_vae(train_params, input_shapes=[None, bstep_train_gen.num_items])
    collab_decoder = collabo_vae.build_vbae_recon_bstep()    
    
    vbae_bstep = collabo_vae.build_vbae_bstep(lambda_W=params.lambda_W)
    vbae_tstep = content_vae.build_vbae_tstep(collab_decoder=collab_decoder, lambda_W=params.lambda_W)
    loss_tstep = [name_loss_dict[args.dataset], multinomial_crossentropy]
    
    vbae_infer_tstep = content_vae.build_vbae_infer_tstep()
    vbae_infer_bstep = collabo_vae.build_vbae_infer_bstep()
    vbae_eval = collabo_vae.build_vbae_eval()

    ### Some configurations for training
    best_Recall_20, best_Recall_40, best_NDCG_100, best_sum = -np.inf, -np.inf, -np.inf, -np.inf

    if args.save_root:
        save_root = args.save_root
    else:
        save_root = os.path.join("models", args.dataset, str(args.split), "vbae-soft")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(os.path.join(save_root, "hyperparams.json"), "w") as f:
        json.dump(train_params, f)

    with open(os.path.join(save_root, "pretrain_hyperparams.json"), "w") as f:
        json.dump(pretrain_params, f)

    training_dynamics = os.path.join(save_root, "training_dynamics.csv")
    with open(training_dynamics, "w") as f:
        f.write("Recall@20,Recall@40,NDCG@100,chn_mean,chn_std\n")

    best_bstep_path = os.path.join(save_root, "best_bstep.model")
    best_tstep_path = os.path.join(save_root, "best_tstep.model")

    lamb_schedule_gauss = PiecewiseSchedule([[0, 0.0], [80, 0.2]], outside_value=0.2)
    lamb_schedule_beta = PiecewiseSchedule([[0, 2.0], [80, 2.0]], outside_value=2.0)

    vbae_bstep.compile(loss=multinomial_crossentropy, optimizer=optimizers.Adam(), 
                       metrics=[multinomial_crossentropy])    
    collab_decoder.trainable = False
    vbae_tstep.compile(optimizer=optimizers.Adam(), loss=loss_tstep)

    ### Train the content and collaborative part of vbae in an EM-like style
    for epoch in range(epochs):
        ### Set the value of annealing parameters
        K.set_value(vbae_bstep.optimizer.lr, blr_schedule.value(epoch))
        K.set_value(vbae_bstep.Wc0, w0_schedule.value(epoch))
        K.set_value(vbae_bstep.Wc1, w1_schedule.value(epoch))
        K.set_value(collabo_vae.add_gauss_loss.lamb_kl, lamb_schedule_gauss.value(epoch))
        K.set_value(collabo_vae.add_beta_loss.lamb_kl, lamb_schedule_beta.value(epoch))

        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)
        print("Begin bstep:")
        bstep_train_gen.update_previous_tstep(infer_tstep(vbae_infer_tstep, tstep_train_gen.features.A))
        bstep_valid_gen.update_previous_tstep(infer_tstep(vbae_infer_tstep, tstep_valid_gen.features.A))

        vbae_bstep.fit_generator(bstep_train_gen, workers=4, epochs=1, validation_data=bstep_valid_gen)
    
        Recall_20 = EvaluateModel(vbae_eval, bstep_valid_gen, Recall_at_k, k=20)
        Recall_40 = EvaluateModel(vbae_eval, bstep_valid_gen, Recall_at_k, k=40)
        NDCG_100 = EvaluateModel(vbae_eval, bstep_valid_gen, NDCG_at_k, k=100)

        tstep_valid_gen.update_previous_bstep(*infer_bstep(vbae_infer_bstep, bstep_valid_gen.X))
        chn_mean = tstep_valid_gen.alpha.mean()
        chn_std  = tstep_valid_gen.alpha.std()

        if Recall_20 > best_Recall_20:
            best_Recall_20 = Recall_20

        if Recall_40 > best_Recall_40:
            best_Recall_40 = Recall_40

        if NDCG_100 > best_NDCG_100:
            best_NDCG_100 = NDCG_100

        cur_sum =  Recall_20 + Recall_40 + NDCG_100
        if cur_sum > best_sum:
            best_sum = cur_sum
            vbae_bstep.save_weights(best_bstep_path, save_format="tf")
            vbae_tstep.save_weights(best_tstep_path, save_format="tf")

        with open(training_dynamics, "a") as f:
            f.write("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".\
                format(Recall_20, Recall_40, NDCG_100, chn_mean, chn_std))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@20: {:5f}, best recall@20: {:5f}".format(Recall_20, best_Recall_20))
        print("cur recall@40: {:5f}, best recall@40: {:5f}".format(Recall_40, best_Recall_40))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(NDCG_100, best_NDCG_100))
        print("cur channel mean: {:5f}, std: {:5f}".format(chn_mean, chn_std))

        print("Begin tstep:")
        K.set_value(vbae_tstep.optimizer.lr, tlr_schedule.value(epoch))
        tstep_train_gen.update_previous_bstep(*infer_bstep(vbae_infer_bstep, bstep_train_gen.X))
        vbae_tstep.fit_generator(tstep_train_gen, workers=4, epochs=1, validation_data=tstep_valid_gen)

    print("Done training!")

if __name__ == '__main__':
    train_vbae_model()