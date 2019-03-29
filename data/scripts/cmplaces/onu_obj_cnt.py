from data.utils import parse_data_file_list
import numpy as np
import random
import matplotlib.pyplot as plt

image_dir = '/mnt/db/data/mscoco/processed/image_onu'
image_train_path = '/mnt/db/data/mscoco/processed/image_onu_val.txt'
image_train_list = parse_data_file_list(image_dir, image_train_path, True)
image_train_list = random.sample(image_train_list, 3000)

n_files = len(image_train_list)
n_objs = []

for i, tup in enumerate(image_train_list):
    if (i+1) % 1000 == 0: print("loading %d/%d" % (i+1, n_files))
    path, label = tup
    npzd = np.load(path)
    objs = npzd['objs']
    n_obj, _ = objs.shape
    n_objs.append(n_obj)

bins = [i for i in range(1, 40)]
plt.hist(n_objs, bins)
plt.show()