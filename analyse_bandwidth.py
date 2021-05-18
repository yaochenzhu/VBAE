import os
import time
import json
import logging
import argparse

import sys
sys.path.append(os.path.join("libs", "soft"))

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import stats
from tensorflow.keras import backend as K

from data import CollaborativeVAEDataGenerator
from train_vbae_soft import get_collabo_vae
from train_vbae_soft import infer_bstep

from utils import sigmoid
from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k


def predict_and_evaluate():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--split", type=int,
        help="specify the split of the dataset")
    parser.add_argument("--batch_size", type=int, default=128,
        help="specify the batch size prediction")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Fix the random seeds.
    np.random.seed(98765)
    tf.set_random_seed(98765)

    ### Get the test data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split))
    model_root = os.path.join("models", args.dataset, str(args.split), "vbae-soft")

    params_path = os.path.join(model_root, "hyperparams.json")
    with open(params_path, "r") as params_file:
        params = json.load(params_file)

    bstep_test_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, joint=True,
        shuffle=False
    )

    ### Build test model and load trained weights
    collabo_vae = get_collabo_vae(params, [None, bstep_test_gen.num_items])
    collabo_vae.load_weights(os.path.join(model_root, "best_bstep.model"))
    vbae_infer_bstep = collabo_vae.build_vbae_infer_bstep()

    _, logits, _ = infer_bstep(vbae_infer_bstep, bstep_test_gen.X)
    alpha = sigmoid(logits.squeeze())
    density = bstep_test_gen.X.toarray().sum(axis=-1)

    chn_mean, chn_std = alpha.mean(), alpha.std()
    chn_corr, _ = stats.pearsonr(alpha, density)

    save_path = os.path.join(model_root, "bandwidth.csv")
    with open(save_path, "w") as f:
        f.write("chn_mean,chn_std,chn_corr\n")
        f.write("{:.4f},{:.4f},{:.4f}".format(chn_mean, chn_std, chn_corr))

    print("Done analysing bandwidth! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()