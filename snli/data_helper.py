# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import json
import collections

import numpy as np
from tensorflow.contrib import learn


def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Building dataset ...')
        start = time.time()
        incsv = csv.reader(f)  # 读取文件
        header = next(incsv)  # Header
        label_idx = header.index('label')  # 第一列标题
        content_idx = header.index('content')  # 第二列标题

        labels = []
        sentences = []
        print(label_idx)
        print(content_idx)
        print(labels)
        print(sentences)
        if sw_path is not None:
            sw = _stop_words(sw_path)   # _stop_words这个命令是干啥的？
        else:
            sw = None

        for line in incsv:
            sent = line[content_idx].strip()   # 一行一行得读，读取content列的内容

            if language == 'ch':
                sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
            elif language == 'en':
                sent = sent.lower()
            else:
                raise ValueError('language should be one of [ch, en].')

            sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

            if len(sent) < 1:
                continue

            if language == 'ch':
                sent = _word_segmentation(sent)
            sentences.append(sent)

            if int(line[label_idx]) < 0:
                labels.append(2)
            else:
                labels.append(int(line[label_idx]))     #这一段啥意思，干啥的 .append就是在后面加一列

    labels = np.array(labels)
    print(labels)

    # Real lengths   list(seq) 将元组转换为列表
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in sentences])))  # map() 会根据提供的函数对指定序列做映射，如map(square, [1,2,3,4,5])   # 计算列表各个元素的平方[1, 4, 9, 16, 25]
    print(lengths)
    # 这是干啥的
    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    # learn.preprocessing.VocabularyProcessor 实现的功能就是，根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
    # max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。
    # min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中。
    
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(sentences)))
    # fit (raw_documents, unused_y=None) 作用：从原始文档raw_documents中学习到一个词汇表。参数：raw_documents：一个可产生str或uncode的迭代器。
    # fit_transform (raw_documents, unused_y=None)
    # 与上面一个函数一样，但是返回它返回原始文档的id矩阵[n_samples, max_document_length]
    else:
        data = np.array(list(vocab_processor.transform(sentences)))
    # transform (raw_documents)
    # 将raw_documents中的词转化为id
    data_size = len(data)

    print(data_size)
    print(data)
    # shuffle 是啥意思，随机排序一下
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))   #  即随机排列序列，或返回随机范围。我的理解就是返回一个乱序的序列
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]
        lengths = lengths[shuffle_indices]

    end = time.time()
    print(data)
    print(labels)
    print(lengths)
    print(vocab_processor)
    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}\n'.format(vocab_processor.max_document_length))
    
    return data, labels, lengths, vocab_processor


def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length

# --------------- Private Methods ---------------

def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    # Please download langconv.py and zh_wiki.py first
    # langconv.py and zh_wiki.py are used for converting between languages
    try:
        import langconv
    except ImportError as e:
        error = "Please download langconv.py and zh_wiki.py at "
        error += "https://github.com/skydark/nstools/tree/master/zhtools."
        print(str(e) + ': ' + error)
        sys.exit()

    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    """ Tokenizer for Chinese """
    import jieba
    sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent
