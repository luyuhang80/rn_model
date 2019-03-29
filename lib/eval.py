from lib.mlnet import MLNet
from data.data_pair_loader import MAPLoader
from data.utils import parse_data_file_list
import tensorflow as tf
import numpy as np
import time

def average_precisions(net: MLNet, sess: tf.Session, query_modal, at,
                       paths_with_labels_1, paths_with_labels_2,
                       process_fn_1, process_fn_2,
                       batch_size, n_classes, process_id=None):

    """
    :param net: an MLNet model
    :param sess: a tensorflow session
    :param query_modal: should be 1 or 2. indicates which modal is query and which modal is retrieved
    e.g.: query_idx=4, query_modal=2, then the traversed indices pairs would be:
    [ (0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4), (7, 4), ...]
    :param at: if mAP@100 is desired, assign at with 100, if mAP@ALL is desired, assign at with 0
    :param paths_with_labels_1: data list with labels parsed with data.utils.parse_data_list
    :param paths_with_labels_2: data list with labels parsed with data.utils.parse_data_list
    :param process_fn_1: preprocess function for modal 1
    :param process_fn_2: preprocess function for modal 2
    :param batch_size: batch size
    :param n_classes: number of classes
    :param process_id: designate a process id
    :return: mAP
    """

    if net.is_training: raise Exception("cannot run this in training mode")

    n_samples, n_entries = 0, 0
    if query_modal == 1:
        n_samples, n_entries = len(paths_with_labels_1), len(paths_with_labels_2)
    elif query_modal == 2:
        n_samples, n_entries = len(paths_with_labels_2), len(paths_with_labels_1)

    if process_id is not None: print("process %d: " % process_id, end='')
    print("retrieving %d samples from %d entries" % (n_samples, n_entries))

    average_precisions = []

    for query_idx in range(n_samples):

        time1 = time.time()

        loader = MAPLoader(paths_with_labels_1, paths_with_labels_2,
                           query_idx, query_modal, batch_size, n_classes, shuffle=False,
                           process_fn_1=process_fn_1, process_fn_2=process_fn_2)
        loader.async_load_batch(0)

        preds = []
        labels = []

        for batch in range(loader.n_batches):
            batch_data_1, batch_data_2, batch_labels = loader.get_async_loaded()
            if batch + 1 < loader.n_batches: loader.async_load_batch(batch + 1)
            if loader.n_remain > 0 and batch + 1 == loader.n_batches: loader.async_load_batch(batch+1)

            feed_dict = {}
            for ph, data in zip(net.ph1, batch_data_1):
                feed_dict[ph] = data
            for ph, data in zip(net.ph2, batch_data_2):
                feed_dict[ph] = data

            batch_pred = net.logits.eval(session=sess, feed_dict=feed_dict)
            preds.append(batch_pred)
            labels.append(batch_labels)

        if loader.n_remain > 0:
            batch_data_1, batch_data_2, batch_labels = loader.get_async_loaded()

            feed_dict = {}
            for ph, data in zip(net.ph1, batch_data_1):
                feed_dict[ph] = data
            for ph, data in zip(net.ph2, batch_data_2):
                feed_dict[ph] = data

            batch_pred = net.logits.eval(session=sess, feed_dict=feed_dict)
            preds.append(batch_pred[:loader.n_remain])
            labels.append(batch_labels[:loader.n_remain])

        time2 = time.time()

        preds = np.concatenate(preds, axis=0).tolist()
        labels = np.concatenate(labels, axis=0).tolist()
        zipped = list(zip(preds, labels))
        zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
        preds, labels = zip(*zipped)

        n_relavant = 0
        precisions = []
        piv = len(labels) if at <= 0 or at > len(labels) else at
        for j in range(piv):
            if labels[j] == 1:
                n_relavant += 1
                precisions.append(1.0 * n_relavant / (j+1))

        if n_relavant == 0: precisions = [0]

        average_precision = sum(precisions)/len(precisions)
        average_precisions.append(average_precision)

        time3 = time.time()
        ellapsed = time3 - time1
        if process_id is not None: print("process %d: " % process_id, end='')
        print("modal %d, sample %d/%d, AP: %.3f, gpu: %.2fs, cpu: %.2fs, total: %.2fs eta: %.1f mins" %
              (query_modal, query_idx + 1, n_samples, average_precision, time2-time1, time3-time2, ellapsed,
               ellapsed * (n_samples - query_idx - 1) / 60))

    return average_precisions


def mAP(net: MLNet, sess: tf.Session, query_modal, at,
        paths_with_labels_1, paths_with_labels_2,
        process_fn_1, process_fn_2,
        batch_size, n_classes):

    APs = average_precisions(net, sess, query_modal, at,
                             paths_with_labels_1, paths_with_labels_2,
                             process_fn_1, process_fn_2,
                             batch_size, n_classes)

    return sum(APs) / len(APs)

