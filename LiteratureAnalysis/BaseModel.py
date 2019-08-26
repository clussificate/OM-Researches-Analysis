# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:12:56 2019

@author: Kurt

本模块根据单一的摘要或关键词题目信息建立分类模型

"""
import nltk
import os
import GetData
from tools import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
lancaster_stemmer = nltk.LancasterStemmer()
stop_words = GetData.get_stopwords("data/stopwords.txt")


def process_data(trainset, testset, content, target):
    """
    原数据预处理函数，对文本进行分词、stemming、清洗、去停用词等操作
    使用get_single_label产生单标签，使用get_multi_label产生多标签
    """
    X_train_wt = [word_tokenize(x) for x in GetData.get_content(trainset, content)]
    X_test_wt = [word_tokenize(x) for x in GetData.get_content(testset, content)]
    y_train = list(map(get_single_label, trainset[target]))
    y_test = list(map(get_single_label, testset[target]))

    # 去停用词及文末句号
    X_train_st = [[del_tail_dot(word.lower()) for word in document if not (word.lower() in stop_words or word == "")]
                  for document in X_train_wt]
    X_test_st = [[del_tail_dot(word.lower()) for word in document if not (word.lower() in stop_words or word == "")] for
                 document in X_test_wt]
    # 词干提取
    X_train_ls = [[lancaster_stemmer.stem(word) for word in document] for document in X_train_st]
    X_test_ls = [[lancaster_stemmer.stem(word) for word in document] for document in X_test_st]

    # 字符串列表连接
    X_train_ls_merge = [MergeWord(document) for document in X_train_ls]
    X_test_ls_merge = [MergeWord(document) for document in X_test_ls]
    return X_train_ls_merge, y_train, X_test_ls_merge, y_test


def count_vectorizer(trainset, testset):
    CntVec = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    trainCntLs = CntVec.fit_transform(trainset)
    testCntLs = CntVec.transform(testset)
    return trainCntLs, testCntLs


def get_TfIdf(trainset, testset):
    """
    获取TfIdf特征
    """
    TfidfVec = TfidfVectorizer(max_df=0.15, min_df=0.1, ngram_range=(1,1),stop_words='english')
    trainTfIdf = TfidfVec.fit_transform(trainset)
    testTfIdf = TfidfVec.transform(testset)
    return trainTfIdf.toarray(), testTfIdf.toarray()


def get_LDA(trainset, testset):
    """
    获取LDA特征
    """
    lda = LatentDirichletAllocation(n_components=100,
                                    learning_offset=50.,
                                    random_state=0)
    X_trainLDA = lda.fit_transform(trainset)
    X_testLDA = lda.transform(testset)
    return X_trainLDA, X_testLDA

def train_model(X, y):
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=10, random_state=0)
    clf.fit(X, y)
    return clf

def evaluation(y_test, preds):
    print(classification_report(y_test, preds))


if __name__ == '__main__':
    os.system("python init.py")
    trainset, testset = GetData.load_dataset("data/trainset.csv", "data/testset.csv", separator='|')
    # X_train, y_train, X_test, y_test = process_data(trainset, testset, ['TITLE', 'KEY_WORDS'], 'TYPE')
    X_train, y_train, X_test, y_test = process_data(trainset, testset, 'N_ABS', 'TYPE')

    # 存储处理后的数据
    save_model(X_train, "data/AbstractTrainset")
    save_model(X_test, "data/AbstractTestset")

    # save_model(X_train, "data/TitleAndKeywordsTrainset")
    # save_model(X_test, "data/TitleAndKeywordsTestset")


    # 存储labels
    save_model(y_train, "TrainsetLabel")
    save_model(y_test, "TestsetLabel")

    # 生成LDA特征
    # X_train, X_test = count_vectorizer(X_train, X_test)
    # X_train, X_test = get_LDA(X_train, X_test)

    # 生成TfIdf特征
    X_train, X_test = get_TfIdf(X_train, X_test)

    print("样本维度为：", X_train.shape)
    model = train_model(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    print("训练集预测情况")
    evaluation(y_train, train_preds)
    print("测试集预测情况")
    evaluation(y_test, test_preds)







