import sys
sys.path.append('../..')
from models.cmplaces_rgcn.yul_rn import Model
from data.h5_data_pair_loader import PosNegLoader
from main.utils import parse_device_str
import time
import os
import tensorflow as tf
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--sess", help="session name, such as cmplaces1_1536482758")
parser.add_argument("--ckpt", help="checkpoint to restore")
parser.add_argument("--epochs", help="number of epochs")
parser.add_argument("--bsize", help="batch size")
parser.add_argument("--lthreads", help="number of loader threads")
parser.add_argument("--lswone", help="if labels start with one instead of zero, add this flag",
                    action="store_true")
parser.add_argument("--m1device", help="specify modal 1 device. e.g.: cpu0 or gpu1 or ...")
parser.add_argument("--m2device", help="specify modal 2 device. e.g.: cpu0 or gpu1 or ...")
parser.add_argument("--mtrdevice", help="specify metrics device. e.g.: cpu0 or gpu1 or ...")
args = parser.parse_args()

sess_name = args.sess if args.sess else 'yul_rn%d' % int(time.time())
ckpt_name = args.ckpt if args.sess else None
n_epochs = args.epochs if args.epochs else 25
batch_size = int(args.bsize) if args.bsize else 256
label_start_with_zero = False if args.lswone else True

modal_1_device = parse_device_str(args.m1device) if args.m1device else None
modal_2_device = parse_device_str(args.m1device) if args.m2device else None
metrics_device = parse_device_str(args.mtrdevice) if args.mtrdevice else None

# initialize data loader

text_h5_path = "/home1/yul/yzq/data/cmplaces/text_bow_unified.h5"
image_h5_path = "/home1/yul/yzq/data/cmplaces/natural50k_onr.h5"
adjmat_path = '/home1/yul/yzq/data/cmplaces/txt_graph_knn_unified.txt'

n_classes = 205
label_start_with_zero = True
n_train = 819200
n_val = 8192

train_loader = PosNegLoader(text_h5_path, image_h5_path, "train", "train", n_train, n_train,
                            batch_size=batch_size, n_classes=n_classes, shuffle=True, whole_batches=True)
val_loader = PosNegLoader(text_h5_path, image_h5_path, "val", "val", n_val, n_val,
                            batch_size=batch_size, n_classes=n_classes, shuffle=True, whole_batches=True)

# build network, loss, train

desc_dims = 1024
out_dims = 1
max_rgcn_nodes = 25
net = Model(max_rgcn_nodes, batch_size, desc_dims, out_dims, is_training=True)
net.build(modal_1_device, modal_2_device, metrics_device)

lamda = 0.35
mu = 0.8
regularization = 5e-3
net.build_loss(lamda, mu, regularization)

learning_rate = 1e-4
decay_rate = 0.95
decay_steps = train_loader.n_batches
net.build_train(learning_rate, decay_rate, decay_steps)

# start training

eval_freq = 400
dropout = 0.6
out_dir = os.path.join('..', '..', 'out', sess_name)

log_dir = os.path.join(out_dir, 'log')
ckpt_dir = os.path.join(out_dir, 'checkpoints')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

with tf.Session(config=config) as sess:
    net.build_summary(log_dir, sess)
    net.initialize(sess)
    if ckpt_name is not None:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        net.restore(ckpt_path, sess)
    net.train(n_epochs, dropout, eval_freq, ckpt_dir, sess, train_loader, val_loader)
