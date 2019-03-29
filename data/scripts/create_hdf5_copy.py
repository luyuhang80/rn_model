import sys
sys.path.append('../..')
import os
import h5py
import numpy as np
import time
from data.utils import *

# configurations

dataset_dir = "/mnt/db/data/mscoco/processed"
series = "image_onu"
series_dir = os.path.join(dataset_dir, series)
splits = ["train", "val", "test"]
col_keys = ["v_objs", "v_unis"]
col_item_shapes = [[36, 2048], [36, 36, 2048]]
col_dtypes = ["float32", "float32"]
label_start_with_zero = True

compression = "gzip"

origin_series = "image"
origin_ext = "jpg"

# def process_fn(path):
#     return [np.load(path)]

def process_fn(npz_path, N=36):

    npzd = np.load(npz_path)
    v_objs = npzd['v_objs'][:N, :]
    v_rels = npzd['v_unis'][:N, :N, :]

    n_obj, F_o = v_objs.shape
    _, _, F_r = v_rels.shape
    x = np.zeros([N, F_o])
    R = np.zeros([N, N, F_r])

    x[:n_obj, :] = v_objs
    R[:n_obj, :n_obj, :] = v_rels

    return [x, R]

# get total number of data items

n_data = 0
for split in splits:
    list_name = "%s_%s.txt" % (series, split)
    list_path = os.path.join(dataset_dir, list_name)
    data_list = parse_data_file_list(series_dir, list_path, label_start_with_zero)
    n_data += len(data_list)

# create placeholders

h5_name = "%s.h5" % series
h5_path = os.path.join(dataset_dir, h5_name)
hdf = h5py.File(h5_path, 'w')

g_data = hdf.create_group("data")
for col_key, col_item_shape, col_dtype in zip(col_keys, col_item_shapes, col_dtypes):
    col_shape = [n_data] + col_item_shape
    g_data.create_dataset(col_key, shape=col_shape, dtype=col_dtype, chunks=True,
                          compression=compression)
g_splits = hdf.create_group("splits")
ds_labels = hdf.create_dataset("labels", shape=[n_data], dtype="int32")
ds_origins = hdf.create_dataset("origins", shape=[n_data], dtype="S50")
ds_col_keys = hdf.create_dataset("col_keys", shape=[len(col_keys)], dtype="S50")
for i, col_key in enumerate(col_keys):
    ds_col_keys[i] = np.string_(col_key)

# fill data

idx = 0

for split in splits:

    print("processing split: %s" % split)

    list_name = "%s_%s.txt" % (series, split)
    list_path = os.path.join(dataset_dir, list_name)
    data_list = parse_data_file_list(series_dir, list_path, label_start_with_zero)

    indices = []

    t_int = time.time()
    n_samples = len(data_list)
    avg_int = 0.0

    for num, tup in enumerate(data_list):

        print("    %d/%d" % (num, len(data_list)), end=" ")

        path, label = tup
        _, fname = os.path.split(path)
        _, id_str, label_str = parse_data_file_name_raw(fname)
        ds_labels[idx] = label
        origin_name = "%s_%s_%s.%s" % (origin_series, id_str, label_str, origin_ext)
        ds_origins[idx] = np.string_(origin_name)
        data = process_fn(path)
        for i, datum in enumerate(data):
            col_key = col_keys[i]
            ds_data = g_data.get(col_key)
            ds_data[idx] = datum
        indices.append(idx)
        idx += 1

        t_tmp = time.time()
        interval, t_int = t_tmp - t_int, t_tmp
        avg_int = 0.98 * avg_int + 0.02 * interval
        eta = (n_samples - num - 1) * avg_int
        eta_str = humanize_duration(eta)
        print('%.2fms, eta %s' % (1000 * avg_int, eta_str), end="\n")

    indices = np.array(indices, dtype="int32")
    g_splits.create_dataset(split, data=indices)

hdf.close()
