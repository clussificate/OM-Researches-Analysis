# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:26:40 2019

@author: Kurt
"""
import re
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler

def word_tokenize(string):
    """
    根据标点符号分词
    """
    return re.split(r'[;,()\s\?\:-]\s*', string)


def del_tail_dot(word):
    """
    删除单词某端的句号
    """
    if word[-1]=='.':
        return word[0:-1]
    else:
        return word

def get_single_label(label):
    """
    产生单标签
    """
    label = str(label)
    if 'EX' in label:
        return 'EX'
    elif 'EM' in label:
         return 'EM'
    elif 'T' in label:
        return 'T'
    else:
        return label


def get_multiple_label(label):
    """
    产生多标签
    """
    labels = ['EM', 'EX', 'T']
    output = []
    split_label = sorted(label.strip().split("+"))
    for label in labels:
        if label in split_label:
            output.append(1)
        else:
            output.append(0)
    return output


def MergeWord(wordlist):
    """
    连接字符串
    """
    return " ".join(wordlist)

def data_scaler(train, test):
    """
    数据标准化
    """
    ss = StandardScaler()
    train_scale = ss.fit_transform(train)
    test_scale = ss.transform(test)
    return train_scale,test_scale

def cross_validate(X_train, y_train, clf, paras):
    """
    网格搜索 + 交叉验证
    """
    grid_search = GridSearchCV(clf, param_grid=paras, cv=5, verbose=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("最优参数为：")
    print(grid_search.best_params_)
    print("最优配置的得分：")
    print(grid_search.best_score_)
    return grid_search.best_estimator_


def save_model(model, path):
    """
    存储模型
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """
    加载模型
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def merge_features(*features):
    """
    特征融合函数
    """
    return np.hstack(features)

def single_smote(X, y):
    """
    为multi-class样本过采样
    """
    sm = SMOTE(random_state=1994)
    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm, y_sm

def multiple_smote(X, y):
    """
    为multi-label样本过采样
    """

    # Import a dataset with X and multi-label y
    y = np.array(y)
    lp = LabelPowerset()
    oversampler = ADASYN(random_state=1994, n_neighbors=1)
    # oversampler = SMOTE(random_state=1994, k_neighbors=1)

    # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
    yt = lp.transform(y)

    X_resampled, y_resampled = oversampler.fit_resample(X, yt)

    # Inverts the ML-MC transformation to recreate the ML set
    y_resampled = lp.inverse_transform(y_resampled) # return a sparse matrix

    return X_resampled, y_resampled.toarray()

def over_sampling(X, y, sign):
    """
    对训练集过采样
    """
    if sign=="single_label":
        X_sm, y_sm = single_smote(X, y)
    elif sign == "multiple_label":
        X_sm, y_sm = multiple_smote(X,y)
    else:
        raise Exception("Correct label format：single_label or multiple_label")
    return X_sm, y_sm