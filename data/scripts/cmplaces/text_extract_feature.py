import sys
sys.path.append('../../..')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import numpy as np
import os
import pathlib
import math
import re

lemmatizer = WordNetLemmatizer()
word_list = np.load('/mnt/db/data/cmplaces/word_list.npy')

def sentence_to_words(sen):
    sen = sen.lstrip()
    sen = sen.rstrip()
    words = word_tokenize(sen)
    word_tag = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(word[0]) for word in word_tag
             if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP']
    words = [word.lower() for word in words]
    words = [word for word in words if word in word_list]
    return words

def preprocess(series_path, data_file_list):
    """
    loads and extracts useful words from text files
    :param dataset_path: the path to the CMPlaces dataset
    :param data_file_list_path: the path to a list of text files to be preprocessed
    :return: a list of processed essays and a list of labels
    """

    data_files = data_file_list
    processed_essays = []

    # extract useful words from text files
    for file in data_files:
        file_path = os.path.join(series_path, file)
        with open(file_path, 'r') as f: sen = f.read()
        words = sentence_to_words(sen)
        processed_essays.append(words)

    return processed_essays

def corpus_to_np_array(corpus, n_essays, n_words, dtype='float32'):
    """
    converts a gensim corpus to a numpy array
    """
    corpus = list(corpus)
    corpus_arr = np.zeros([n_essays, n_words], dtype=dtype)

    for i in range(n_essays):
        essay = corpus[i]
        for word, value in essay:
            corpus_arr[i][word] = value

    return corpus_arr

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

if __name__ == '__main__':
    dataset_path = '/mnt/db/data/cmplaces'
    old_series = 'text'
    series = 'text_bow'
    output_path = os.path.join(dataset_path, series)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) # recursively makes directories if they don't already exist
    n_classes = 205
    val_portion = 0.1
    test_portion = 0.1

    print("preprocessing")

    with open(os.path.join(dataset_path, old_series+'_train.txt'), 'r') as f:
        train_list = [line.strip() for line in f.readlines()]
    with open(os.path.join(dataset_path, old_series+'_test.txt'), 'r') as f:
        test_list = [line.strip() for line in f.readlines()]
    with open(os.path.join(dataset_path, old_series+'_val.txt'), 'r') as f:
        val_list = [line.strip() for line in f.readlines()]

    n_train = len(train_list)
    n_test = len(test_list)
    n_val = len(val_list)
    n = n_train + n_test + n_val

    combined_list = test_list + val_list + train_list

    processed_essays = preprocess(os.path.join(dataset_path, old_series), combined_list)

    print("extracting features")
    dictionary = corpora.Dictionary(processed_essays)
    dictionary.save(os.path.join(dataset_path, 'dictionary.dict'))
    corpus = [dictionary.doc2bow(essay) for essay in processed_essays]
    corpora.MmCorpus.serialize(os.path.join(dataset_path, 'corpus.mm'), corpus)
    corpus = corpora.MmCorpus(os.path.join(dataset_path, 'corpus.mm'))

    # tf_idf_model = models.TfidfModel(corpus)
    # corpus_tf_idf = tf_idf_model[corpus]
    # corpora.MmCorpus.serialize(os.path.join(dataset_path, 'corpus_tf_idf.mm'), corpus_tf_idf)
    # corpus = corpora.MmCorpus(os.path.join(dataset_path, 'corpus_tf_idf.mm'))

    array = corpus_to_np_array(corpus, corpus.num_docs, corpus.num_terms)
    print("%d samples, %d words" % (corpus.num_docs, corpus.num_terms))

    new_train_list = []
    new_val_list = []
    new_test_list = []

    for i in range(n):
        old_name = combined_list[i]
        _, number, label = parse(old_name)
        new_name = "%s_%08d_%d.npy" % (series, number, label)
        new_path = os.path.join(dataset_path, series, new_name)
        np.save(new_path, array[i])
        if i < n_test: new_test_list.append(new_name)
        elif n_test <= i < n_test + n_val: new_val_list.append(new_name)
        else: new_train_list.append(new_name)

    train_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'train'))
    val_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'val'))
    test_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'test'))

    with open(train_list_path, 'w') as f:
        f.write('\n'.join(new_train_list))
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(new_val_list))
    with open(test_list_path, 'w') as f:
        f.write('\n'.join(new_test_list))

    print('total %d samples, %d train, %d val, %d test' % (n, n_train, n_val, n_test))





