import sys
sys.path.append('../../..')
from gensim import corpora
from gensim.models import Word2Vec
import numpy as np
import os
import pathlib
import random
import math
from data.scripts.cmplaces.text_extract_feature import preprocess

if __name__ == '__main__':
    dataset_path = '/mnt/db/data/cmplaces'
    old_train_list_path = '/mnt/db/data/cmplaces/labels/text_train.txt'
    old_test_list_path = '/mnt/db/data/cmplaces/labels/text_test.txt'
    out_path = '/mnt/db/data/cmplaces/wv_8nn_graph.txt'
    K = 8

    print("preprocessing")
    with open(old_train_list_path, 'r') as f:
        train_list = [line.strip() for line in f.readlines()]
    train_list = [(l[:l.find(' ')], int(l[l.find(' ')+1:])) for l in train_list]

    with open(old_test_list_path, 'r') as f:
        test_list = [line.strip() for line in f.readlines()]
    test_list = [(l[:l.find(' ')], int(l[l.find(' ')+1:])) for l in test_list]

    combined_list = train_list + test_list

    processed_essays, labels = preprocess(dataset_path, combined_list)

    print("generating word vector")
    dictionary = corpora.Dictionary(processed_essays)
    model = Word2Vec(processed_essays, size=100, window=5, min_count=1, workers=4)
    wv = model.wv

    lines = []
    for id, word in dictionary.items():
        KNN = wv.most_similar(word, topn=K)
        line = [str(id)]
        readable_line = [word, ' : ']
        for neighbor, similarity in KNN:
            readable_line.append(neighbor)
            line.append(str(dictionary.token2id[neighbor]))
        print(' '.join(readable_line))
        lines.append(' '.join(line))

    with open(out_path, 'w') as f: f.write('\n'.join(lines))












