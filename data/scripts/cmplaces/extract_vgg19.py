import sys
sys.path.append('../../..')
import os
import cv2
import tensorflow as tf
from modules.vgg19 import Vgg19
import numpy as np
import re
import math
import time

dataset_dir = '/mnt/db/data/cmplaces'
in_series = 'natural50k'
out_series = 'natural50k_vgg19_relu7'
os.makedirs(os.path.join(dataset_dir, out_series), exist_ok=True)

batch_size = 128
vgg19_path = '/mnt/db/data/vgg19.npy'
ph_image = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg19 = Vgg19(vgg19_path)
vgg19.build(ph_image, format='bgr')

def resize_bgr(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def parse(str):
    pattern = re.compile(r"(\S+)_(\d+)_(\d+).(\S+)")
    match = pattern.match(str)
    start, end = match.regs[1]
    series = str[start:end]
    start, end = match.regs[2]
    number = int(str[start:end])
    start, end = match.regs[3]
    label = int(str[start:end])
    return series, number, label

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for role in ['val', 'test', 'train']:

    print(role)
    list_name = '%s_%s.txt' % (in_series, role)
    list_path = os.path.join(dataset_dir, list_name)
    with open(list_path, 'r') as f:
        data_list = [line.strip() for line in f.readlines()]
    new_list_name = '%s_%s.txt' % (out_series, role)
    new_list = []

    n_batches = int(math.ceil(1.0*len(data_list)/batch_size))

    for b in range(n_batches):
        print("%d/%d " % (b+1, n_batches), end='')
        time1 = time.time()
        batch_list = data_list[b*batch_size:(b+1)*batch_size]
        batch_img = np.zeros([len(batch_list), 224, 224, 3])
        for i in range(len(batch_list)):
            file_name = batch_list[i]
            file_path = os.path.join(dataset_dir, in_series, file_name)
            img = resize_bgr(file_path)
            batch_img[i] = img
        batch_features = vgg19.relu7.eval(feed_dict={ph_image: batch_img}, session=sess)
        for i in range(len(batch_list)):
            file_name = batch_list[i]
            file_path = os.path.join(dataset_dir, in_series, file_name)
            features = batch_features[i]
            series, number, label = parse(file_name)
            new_name = '%s_%08d_%d.npy' % (out_series, number, label)
            new_list.append(new_name)
            new_path = os.path.join(dataset_dir, out_series, new_name)
            np.save(new_path, features)
        time2 = time.time()
        ellapsed = time2-time1
        eta = ellapsed * (n_batches-b-1) / 60
        print("eta %.3f mins" % eta)
    with open(os.path.join(dataset_dir, new_list_name), 'w') as f:
        f.write('\n'.join(new_list))

sess.close()