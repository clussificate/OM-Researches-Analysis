# -*- coding: utf-8 -*-
"""
Created on 2019/7/28 14:09

@Author: Kurt

"""
import os
import pandas as pd
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import ClassifierChain
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from tools import *
import GetData


def load_dataset(trainsetpath, testsetpath, separator):
    """
    加载训练集和测试集
    """
    trainset = pd.read_csv(trainsetpath, sep=separator)
    testset = pd.read_csv(testsetpath, sep=separator)
    return trainset, testset

def get_label(trainset, testset, target):
    y_train = list(map(get_multiple_label, trainset[target]))
    y_test = list(map(get_multiple_label, testset[target]))
    return np.array(y_train), np.array(y_test)


def get_TfIdf(trainset, testset):
    """
    获取TfIdf特征
    """
    TfidfVec = TfidfVectorizer(max_df=0.3, min_df=0.1, ngram_range=(1,2),stop_words='english')
    trainTfIdf = TfidfVec.fit_transform(trainset)
    testTfIdf = TfidfVec.transform(testset)
    print("the shape of tfidf features: ",trainTfIdf.shape[1])
    return trainTfIdf.toarray(), testTfIdf.toarray()

def get_LDA(trainset, testset):
    """
    获取LDA特征
    """
    CntVec = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    trainCntLs = CntVec.fit_transform(trainset)
    testCntLs = CntVec.transform(testset)
    lda = LatentDirichletAllocation(n_components=100,
                                    learning_offset=50.,
                                    random_state=0)
    X_trainLDA = lda.fit_transform(trainCntLs)
    X_testLDA = lda.transform(testCntLs)
    print("the shape of LDA features: ", X_trainLDA.shape[1])
    return X_trainLDA, X_testLDA

def train_model(X, y, strategy):
    # clf = SVC(C=1,kernel='rbf',probability=True, gamma='scale') # svc without class_weight
    # clf = SVC(C=10,kernel='rbf',class_weight='balanced',probability=True, gamma='scale')  # svc with class_weight
    clf = XGBClassifier(subsample=0.8, colsample_bytree=0.8)
    # clf = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5,
    #                     min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
    #                     objective='binary:logistic', nthread=4, scale_pos_weight=1)
    print(clf)
    if strategy=='ovr':  # OneVsRest strategy also known as BinaryRelevance strategy
        ovr = OneVsRestClassifier(clf)
        ovr.fit(X, y)
        return ovr
    elif strategy=='classifier_chains':
        cc = ClassifierChain(clf)
        cc.fit(X, y)
        return cc
    else:
        raise Exception("Correct strategies：ovr or classifier_chains")

def evaluation(y_test, preds):
    print(classification_report(y_test, preds))

if __name__ =="__main__":
    # os.system("python init.py") # 重新加载原数据
    trainset, testset = GetData.load_dataset("data/trainset.csv", "data/testset.csv", separator='|')

    # 加载label
    y_train, y_test = get_label(trainset, testset, 'TYPE')

    # 加载title和keywords数据
    X_train_titkw = load_model("data/TitleAndKeywordsTrainset")
    X_test_titkw = load_model("data/TitleAndKeywordsTestset")
    # print(X_train_titkw[0])

    # 加载abstract数据
    X_train_abs = load_model("data/AbstractTrainset")
    X_test_abs = load_model("data/AbstractTestset")
    # print(X_train_abs[0])

    # LDA for title and keywords
    X_train_titkw_lda,X_test_titkw_lda = get_LDA(X_train_titkw, X_test_titkw)

    # TfIdf for abstrct
    X_train_abs_TfIdf,X_test_abs_TfIdf = get_TfIdf(X_train_abs, X_test_abs)
    # print(X_train_abs_TfIdf.shape)
    # print(X_train_titkw_lda.shape)

    # 融合特征
    X_train = merge_features(X_train_abs_TfIdf , X_train_titkw_lda)
    X_test = merge_features(X_test_abs_TfIdf, X_test_titkw_lda)

    # 数据标准化
    X_train, X_test = data_scaler(X_train, X_test)
    # y_train2 = [''.join(list(map(str, e))) for e in y_train]

    # smote过采样
    # print(sorted(Counter(y_train2).items()))
    oversampling = True
    if oversampling:
        print("oversampling: True")
        X_train, y_train = over_sampling(X_train,y_train,"multiple_label")
    else:
        print("oversampling: False")
    # # print(y_train2)
    # print("过采样后的类别分布：")
    # y_train3 = [''.join(list(map(str, e))) for e in y_train]
    # print(sorted(Counter(y_train3).items()))

    # 训练模型
    print("the shape of trainset: ",X_train.shape)
    strategy = 'classifier_chains' # supported strategies : ovr, classifier_chains
    print("multi-label classification strategy: %s" % strategy)
    model = train_model(X_train, y_train,strategy)
    train_preds = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)
    if strategy == 'classifier_chains':
        train_preds = train_preds.toarray()
        train_proba = train_proba.toarray()
        test_preds = test_preds.toarray()
        test_proba = test_proba.toarray()

    # 效果评估
    print("results of trainset: ")
    print("-"*20)
    for x,y,z in zip(train_preds, train_proba, y_train):
        print("pred: %s, proba: %s, label: %s" %(x, y, z))

    print("the classification report of trainset:")
    evaluation(y_train, train_preds)
    print("the classification report of testset:")
    evaluation(y_test, test_preds)

    print("results of testset")
    print("-" * 20)
    for x,y,z in zip(test_preds, test_proba, y_test):
        print("pred: %s, proba: %s, label: %s" %(x, y, z))