import sys
sys.path.append('../..')
import os
import random
import time
import data.utils as utils

root_dir = '/mnt/db/data/eng-wiki'
series = 'text_bow_train'
list_name = series + '.txt'
n = 1000
new_list_name = series + '_%d_sample.txt' % n

with open(os.path.join(root_dir, list_name), 'r') as f:
    lines = f.readlines()

l = [line for line in lines]
random.seed(int(1e6 * (time.time() % 1)))
random.shuffle(l)

new_l = l[:n]
new_str = ''
for line in new_l: new_str += line

with open(os.path.join(root_dir, new_list_name), 'w') as f:
    f.write(new_str)

