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
from tensorflow.keras import backend as K

from data import ContentVaeDataGenerator
from data import CollaborativeVAEDataGenerator
from pretrain_vae import get_content_vae
from train_vbae_soft import get_collabo_vae, infer_tstep

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
    parser.add_argument("--model_root", type=str, default=None,
        help="specify the trained model root (optional)")
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
    if args.model_root:
        model_root = args.model_root
    else:
        model_root = os.path.join("models", args.dataset, str(args.split), "vbae-soft")

    params_path = os.path.join(model_root, "hyperparams.json")
    with open(params_path, "r") as params_file:
        params = json.load(params_file)

    pretrain_params_path = os.path.join(model_root, "pretrain_hyperparams.json")
    with open(pretrain_params_path, "r") as params_file:
        pretrain_params = json.load(params_file)

    tstep_test_gen =  ContentVaeDataGenerator(
        data_root = data_root, phase="test",
        batch_size = 1000, joint=True,
        shuffle=False
    )

    bstep_test_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, joint=True,
        shuffle=False
    )
    ### Make sure the data order is aligned between two data generator
    assert np.all(tstep_test_gen.user_ids==bstep_test_gen.user_ids)

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(params, [None, bstep_test_gen.num_items])
    collab_vae.load_weights(os.path.join(model_root, "best_bstep.model"))
    collab_decoder = collab_vae.build_vbae_recon_bstep()
    
    content_vae = get_content_vae(pretrain_params, tstep_test_gen.feature_dim)
    content_vae.build_vbae_tstep(collab_decoder, 0).load_weights(os.path.join(model_root, "best_tstep.model"))
    vbae_infer_tstep = content_vae.build_vbae_infer_tstep()

    vbae_eval = collab_vae.build_vbae_eval()
    bstep_test_gen.update_previous_tstep(infer_tstep(vbae_infer_tstep, tstep_test_gen.features.A))

    ### Evaluate and save the results
    k4recalls = [20, 40]
    k4ndcgs = [100]
    recalls, NDCGs = [], []
    for k in k4recalls:
        recalls.append("{:.4f}".format(EvaluateModel(vbae_eval, bstep_test_gen, Recall_at_k, k=k)))
    for k in k4ndcgs:
        NDCGs.append("{:.4f}".format(EvaluateModel(vbae_eval, bstep_test_gen, NDCG_at_k, k=k)))

    recall_table = pd.DataFrame({"k":k4recalls, "recalls":recalls}, columns=["k", "recalls"])
    recall_table.to_csv(os.path.join(model_root, "recalls.csv"), index=False)

    ndcg_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs}, columns=["k", "NDCGs"])
    ndcg_table.to_csv(os.path.join(model_root, "NDCGs.csv"), index=False)

    print("Done evaluation! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()