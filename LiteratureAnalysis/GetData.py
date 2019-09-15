# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 21:35
# @Author  : Kurt

import numpy as np
import pandas as pd

def create_dataset(filename, targetname,trainpath, testpath):
    """
    生成测试集和训练集
    """
    rawdata = pd.read_excel(filename)
    # 去除无效数据
    rawdata2 = rawdata[(rawdata[targetname] != 0) & (rawdata[targetname] != '0') & (rawdata[targetname] != ' ')]
    trainset = rawdata2.sample(frac=0.9, axis=0)
    testset = rawdata2[~rawdata2.index.isin(trainset.index)]
    trainset.to_excel(trainpath, index=None)
    testset.to_excel(testpath, index=None)


def load_dataset(trainsetpath, testsetpath, separator):
    """
    加载训练集和测试集
    """
    trainset = pd.read_csv(trainsetpath, sep=separator)
    testset = pd.read_csv(testsetpath, sep=separator)
    return trainset, testset


def get_content(dataset, col):
    """
    获取要分析的内容
    """
    if isinstance(col, str):
        return dataset[col]
    if isinstance(col, list):
        col0, col1 = col
        data = []
        for i in range(len(dataset)):
            title = dataset.iloc[i][col0]
            keywords = dataset.iloc[i][col1]
            titkw = "%s%s%s" % (title, '. ', keywords)
            data.append(titkw)
        return data


def get_stopwords(filename):
    """
    读取停用词
    """
    stop_words = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            stop_words.append(line)

    return stop_words

