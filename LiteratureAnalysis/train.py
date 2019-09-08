# -*- coding: utf-8 -*-
"""
Created on 2019/8/28 16:12

@Author: Kurt

"""
import pandas as pd
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import ClassifierChain
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import nltk
from tools import *
import GetData
from collections import Counter
from sklearn.utils import shuffle

lancaster_stemmer = nltk.LancasterStemmer()
stop_words = GetData.get_stopwords("data/stopwords.txt")
def read_data(filename):
    rawdata = pd.read_excel(filename)
    shdata=shuffle(rawdata,random_state=1994)  # shuffle data
    return shdata

def preprocess(data, content):

    # word tokenize
    X_wt = [word_tokenize(x) for x in GetData.get_content(data, content)]
    # 去停用词及文末句号
    X_st = [[del_tail_dot(word.lower()) for word in document if not (word.lower() in stop_words or word == "")]
                  for document in X_wt]
    X_ls = [[lancaster_stemmer.stem(word) for word in document] for document in X_st]
    # 字符串列表连接
    X_ls_merge = [MergeWord(document) for document in X_ls]
    return X_ls_merge

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

def data_scaler(data):
    """
    数据标准化
    """
    ss = StandardScaler()
    train_scale = ss.fit_transform(data)
    save_model(ss, "model/ss")
    return train_scale

def get_TfIdf(data):
    """
    获取TfIdf特征
    """
    TfidfVec = TfidfVectorizer(max_df=0.3, min_df=0.01, ngram_range=(1,2),stop_words='english')
    dataTfIdf = TfidfVec.fit_transform(data)
    print("the shape of tfidf features: ", dataTfIdf.shape[1])
    save_model(TfidfVec, "model/tfidf")
    return dataTfIdf.toarray()

def get_LDA(data):
    """
    获取LDA特征
    """
    CntVec = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    dataCntLs = CntVec.fit_transform(data)
    lda = LatentDirichletAllocation(n_components=100,
                                    learning_offset=50.,
                                    random_state=0)
    dataLDA = lda.fit_transform(dataCntLs)
    print("the shape of LDA features: ", dataLDA.shape[1])
    save_model(CntVec, "model/CntVec")
    save_model(lda, "model/LDA")
    return dataLDA

def train_model(X, y, strategy):
    X = np.array(X)
    y = np.array(y)
    # clf = SVC(C=1,kernel='rbf',probability=True, gamma='scale') # svc without class_weight
    # clf = SVC(C=10,kernel='rbf',class_weight='balanced',probability=True, gamma='scale')  # svc with class_weight
    clf = XGBClassifier(subsample=0.8, colsample_bytree=0.8, max_depth=5,n_estimators=200)
    # clf = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5,
    #                     min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
    #                     objective='binary:logistic', nthread=4, scale_pos_weight=1)
    print(clf)
    if strategy=='ovr':  # OneVsRest strategy also known as BinaryRelevance strategy
        ovr = OneVsRestClassifier(clf)
        ovr.fit(X, y)
        save_model(ovr, "model/ovr")
        return ovr
    elif strategy=='classifier_chains':
        cc = ClassifierChain(clf)
        cc.fit(X, y)
        save_model(cc, "model/cc")
        return cc
    else:
        raise Exception("Correct strategies：ovr or classifier_chains")

def evaluation(y_test, preds):
    print(classification_report(y_test, preds))



if __name__ =="__main__":
    # os.system("python init.py") # 重新加载原数据
    print("load data......")
    filename = "data/Data_0730/all_trainset.xlsx"
    data= read_data(filename)
    print("preprocess data......")
    X_abs = preprocess(data,'N_ABS')
    X_titkw = preprocess(data,['TITLE', 'KEY_WORDS'])
    # for i in range(len(X_titkw)):
    #     print(X_titkw[i])

    # label
    target = np.array(list(map(get_multiple_label, data['TYPE'])))
    # save DOIs of trainset
    save_model(data['DOI'],"data/Data_0730/trian_DOIs")

    # 标签计数
    print("statistics of labels:")
    target2 = [''.join(list(map(str, e))) for e in target]
    print(sorted(Counter(target2).items()))

    # LDA for title and keywords
    X_titkw_lda = get_LDA(X_titkw)

    # TfIdf for abstrct
    X_abs_TfIdf = get_TfIdf(X_abs)


    # 融合特征
    X_train = merge_features(X_abs_TfIdf , X_titkw_lda)

    # 数据标准化
    X_train = data_scaler(X_train)
    # y_train2 = [''.join(list(map(str, e))) for e in y_train]

    # smote过采样
    # print(sorted(Counter(y_train2).items()))
    oversampling = True
    if oversampling:
        print("oversampling: True")
        X_train, target = over_sampling(X_train,target,"multiple_label")
    else:
        print("oversampling: False")
    # # print(y_train2)
    # print("过采样后的类别分布：")
    # y_train3 = [''.join(list(map(str, e))) for e in y_train]
    # print(sorted(Counter(y_train3).items()))

    # 训练模型
    print("the shape of trainset: ", X_train.shape)
    strategy = 'classifier_chains' # supported strategies : ovr, classifier_chains
    print("multi-label classification strategy: %s" % strategy)
    model = train_model(X_train, target, strategy)
    train_preds = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    if strategy == 'classifier_chains':
        train_preds = train_preds.toarray()
        train_proba = train_proba.toarray()

    # 效果评估
    print("results of trainset: ")
    print("-"*20)
    for x,y,z in zip(train_preds, train_proba, target):
        print("pred: %s, proba: %s, label: %s" %(x, y, z))

    print("the classification report of trainset:")
    evaluation(target, train_preds)
    print("Done~!")