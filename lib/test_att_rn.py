import sys
sys.path.append('../..')
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data.h5_data_loader import DataLoader
from lib.eval_h5 import get_descs_and_labels, average_precisions
import random
import tensorflow as tf
import numpy as np
from models.cmplaces_rgcn.yul_rn_att import Model
from main.cmplaces_rgcn.utils import unpack_onr_npz

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--sess", help="session name, such as cmplaces1_1536482758")
parser.add_argument("--ckpts", help="comma seperated checkpoint names, such as 1,5,10,20")
parser.add_argument("--samples", help="number of test samples")
args = parser.parse_args()

sess_name = args.sess if args.sess else 'yul_rn1543201298'
ckpt_names = args.ckpts.split(',') if args.ckpts else ['15','16','17','18','19','20','21','22','23','24','25']
n_samples = int(args.samples) if args.samples else 1000
label_start_with_zero = True

text_h5_path = "/data1/yjgroup/yzq/data/cmplaces/text_bow_unified.h5"
image_h5_path = "/data1/yjgroup/yzq/data/cmplaces/natural50k_onr.h5"

n_classes = 205
batch_size = 256
desc_dims = 1024
out_dims = 1
max_rgcn_nodes=25

text_loader = DataLoader(text_h5_path, 0,batch_size, whole_batches=True)
image_loader = DataLoader(image_h5_path,0, batch_size, whole_batches=True)

text_val_list = text_loader.get_split_indices("val")
text_test_list = text_loader.get_split_indices("test")
text_train_list = text_loader.get_split_indices("train")
image_val_list = image_loader.get_split_indices("val")
image_test_list = image_loader.get_split_indices("test")
image_train_list = image_loader.get_split_indices("train")

text_val_list = random.sample(text_val_list, n_samples)
text_test_list = random.sample(text_test_list, n_samples)
image_val_list = random.sample(image_val_list, n_samples)
image_test_list = random.sample(image_test_list, n_samples)

out_dir = os.path.join('..', '..', 'out', sess_name)
ckpt_dir = os.path.join(out_dir, 'checkpoints')

def test_ckpt(ckpt_path, m1_q_indices, m1_r_indices, m2_q_indices, m2_r_indices):

    # get descriptors

    print("computing descriptors", end="\r")

    net = Model(max_rgcn_nodes, batch_size, desc_dims, out_dims)
    net.build()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print('building desc')
        net.restore(ckpt_path, sess)
        text_loader.set_indices(m1_q_indices)
        m1_q_descs, m1_q_labels = get_descs_and_labels(net, sess, 1, text_loader)
        image_loader.set_indices(m2_q_indices)
        m2_q_descs, m2_q_labels = get_descs_and_labels(net, sess, 2, image_loader)
        text_loader.set_indices(m1_r_indices)
        m1_r_descs, m1_r_labels = get_descs_and_labels(net, sess, 1, text_loader)
        image_loader.set_indices(m2_r_indices)
        m2_r_descs, m2_r_labels = get_descs_and_labels(net, sess, 2, image_loader)

    m1_rs_indices = random.sample(m1_r_indices, n_samples)
    m1_rs_descs = m1_r_descs[m1_rs_indices]
    m1_rs_labels = m1_r_labels[m1_rs_indices]

    m2_rs_indices = random.sample(m2_r_indices, n_samples)
    m2_rs_descs = m2_r_descs[m2_rs_indices]
    m2_rs_labels = m2_r_labels[m2_rs_indices]

    print("m1_rs_descs shape",m1_rs_descs.shape )
    print("m2_rs_descs shape",m2_rs_descs.shape )

    del net
    tf.reset_default_graph()

    # retrieval

    net = Model(max_rgcn_nodes, batch_size, desc_dims, out_dims, is_retrieving=True)
    net.build()
    with tf.Session(config=config) as sess:
        net.restore(ckpt_path, sess)

        APs_2 = average_precisions(net, sess,
                                   m2_q_descs, m2_q_labels,
                                   m1_r_descs, m1_r_labels,1,
                                   100, batch_size)
        mAP2 = sum(APs_2) / len(APs_2)
        print('img query',mAP2)

        APs_1 = average_precisions(net, sess,
                                   m1_q_descs, m1_q_labels,
                                   m2_r_descs, m2_r_labels,0,
                                   100, batch_size)
        mAP1 = sum(APs_1) / len(APs_1)

        print('text query',mAP1)


        # APs_3 = average_precisions(net, sess,
        #                            m1_rs_descs, m1_rs_labels,
        #                            m2_r_descs, m2_r_labels,
        #                            100, batch_size)
        # mAP3 = sum(APs_3) / len(APs_3)

        # APs_4 = average_precisions(net, sess,
        #                            m2_rs_descs, m2_rs_labels,
        #                            m1_r_descs, m1_r_labels,
        #                            100, batch_size)
        # mAP4 = sum(APs_4) / len(APs_4)

    del net
    tf.reset_default_graph()

    # return mAP1, mAP2, mAP3, mAP4
    return mAP1, mAP2

if __name__ == '__main__':
    print("running validation")
    results = []
    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        mAP1, mAP2 = test_ckpt(ckpt_path, text_val_list, text_train_list, image_val_list, image_train_list)
        results.append((ckpt_name, mAP1, mAP2, (mAP1 + mAP2) / 2))
        print("    %10s | %6.4f, %6.4f, %6.4f |" % (ckpt_name, mAP1, mAP2, (mAP1 + mAP2) / 2))

    results = sorted(results, key=lambda x: x[3], reverse=True)
    best = results[0]
    ckpt_path = os.path.join(ckpt_dir, best[0])
    mAP1, mAP2 = test_ckpt(ckpt_path, text_test_list, text_train_list, image_test_list, image_train_list)
    print("best model validation results:")
    print("    %6s | %6.4f, %6.4f, %6.4f " %
          (best[0], best[1], best[2], best[3]))
    print("best model test results:")
    print("    %6s | %6.4f, %6.4f, %6.4f " %
          (best[0], mAP1, mAP2, (mAP1 + mAP2) / 2))

