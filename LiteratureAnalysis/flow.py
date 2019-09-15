# -*- coding: utf-8 -*-
"""
Created on 2019/9/15 10:04

@Author: Kurt

"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import ClassifierChain
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import lightgbm
from xgboost import XGBClassifier
import nltk
from tools import *
import GetData
from collections import Counter
import sklearn.utils as us
import plot_flow



lancaster_stemmer = nltk.LancasterStemmer()
stop_words = GetData.get_stopwords("data/stopwords.txt")

def read_data(filename, shuffle=True):
    rawdata = pd.read_excel(filename)
    if not shuffle:
        print('Shuffle: False')
        return rawdata
    print('Shuffle: True')
    rawdata = us.shuffle(rawdata, random_state=1994)  # shuffle data
    return rawdata

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

def data_scaler(data, train=True):
    """
    数据标准化
    """
    if train:
        ss = StandardScaler()
        train_scale = ss.fit_transform(data)
        save_model(ss, "model/flow/ss")
        return train_scale
    elif not train:
        ss = load_model("model/flow/ss")
        data = ss.transform(data)
        return data
    else:
        raise Exception("Input correct values of train: True or false")

def get_TfIdf(data,train=True):
    """
    获取TfIdf特征
    """
    if train:
        TfidfVec = TfidfVectorizer(max_df=0.1, min_df=0.01, ngram_range=(1,1),stop_words='english')
        dataTfIdf = TfidfVec.fit_transform(data)
        save_model(TfidfVec, "model/flow/tfidf")
        return dataTfIdf.toarray()
    elif not train:
        model = load_model("model/flow/tfidf")
        return model.transform(data).toarray()
    else:
        raise Exception("Input correct values of train: True or false")


def get_LDA(data, train=True):
    """
    获取LDA特征
    """
    if train:
        CntVec = CountVectorizer(min_df=0.01, ngram_range=(1,1))
        lda = LatentDirichletAllocation(n_components=150,learning_method='batch',
                                    random_state=0)
        data = CntVec.fit_transform(data)
        data = lda.fit_transform(data)
        save_model(CntVec, "model/flow/CntVec")
        save_model(lda, "model/flow/LDA")
        return data
    elif not train:
        cntmodel = load_model("model/flow/CntVec")
        ldampdel = load_model("model/flow/LDA")
        data = cntmodel.transform(data)
        data = ldampdel.transform(data)
        return data
    else:
        raise Exception("Input correct values of train: True or false")

def train_model(X, y, strategy):
    X = np.array(X)
    y = np.array(y)
    clf = lightgbm.sklearn.LGBMClassifier(max_depth=9, num_leaves=600,
                                          n_estimators=500,subsample=0.8,n_jobs=-1) # 0.8
    print(clf)
    if strategy=='ovr':  # OneVsRest strategy also known as BinaryRelevance strategy
        ovr = OneVsRestClassifier(clf)
        ovr.fit(X, y)
        save_model(ovr, "model/flow/ovr")
        return ovr
    elif strategy=='classifier_chains':
        cc = ClassifierChain(clf)
        cc.fit(X, y)
        save_model(cc, "model/flow/cc")
        return cc
    else:
        raise Exception("Correct strategies：ovr or classifier_chains")

def save_predictions(data,preds, proba, filename,journal):
    preds = pd.DataFrame(preds,columns=["Sign_F","Sign_I","Sign_P"])
    proba = pd.DataFrame(proba,columns=["Proba_F","Proba_I","Proba_P"])
    merge_data = pd.concat([data,preds,proba],axis =1)
    merge_data['Journal']= journal
    merge_data.to_excel(filename,index = False)

def make_predictions(journals, clf):
    print("load models:")
    CntVec = load_model("model/flow/CntVec")
    LDA = load_model("model/flow/LDA")
    Tfidf = load_model("model/flow/tfidf")
    ss = load_model("model/flow/ss")
    train_DOIs = load_model("data/Data_flow/trian_DOIs").values
    for journal in journals:
        filename = "data/Data_flow/"+journal+".xlsx"
        X_data= pd.read_excel(filename)
        print("preprocess %s data......" % filename.split("/")[-1].split('.')[0])
        X_abs = preprocess(X_data,'N_ABS')
        X_titkw = preprocess(X_data,['TITLE', 'KEY_WORDS'])
        # print(X_abs[7])  # test case no.7 data
        # predict
        X_abs_TfIdf = Tfidf.transform(X_abs).toarray()
        X_titkw_cnt = CntVec.transform(X_titkw)
        X_titkw_lda = LDA.transform(X_titkw_cnt)

        X_merge = merge_features(X_abs_TfIdf, X_titkw_lda)
        X_std = ss.transform(X_merge)

        print("make and save predictions....")
        preds = clf.predict(X_std).toarray()
        proba = clf.predict_proba(X_std).toarray()

        save_predictions(X_data[['DOI','TITLE','KEY_WORDS','N_ABS','YEAR']],
                         preds,proba,filename.split(".")[0]+"_prediction.xlsx",
                         journal)

        print("%s done!" %journal)

    all_data_name = "data/Data_flow/"+journals[0]+"_prediction.xlsx"
    all_data = pd.read_excel(all_data_name)
    for journal in journals[1: ]:
        filename = "data/Data_flow/"+journal+"_prediction.xlsx"
        data = pd.read_excel(filename)
        all_data = pd.concat([all_data, data],axis =0)

    all_data['In_trainset'] = 'No'
    for i, value in enumerate(train_DOIs):
        all_data.loc[all_data.DOI==value, 'In_trainset'] = 'Yes'
    all_data.to_excel('data/Data_flow/all_prediction.xlsx', index = False)


def plot(filename, sigs):
    data = pd.read_excel(filename)
    data = plot_flow.process(data, sigs)
    # plot_flow.style_tri(data)
    plot_flow.style_one(data,sigs)


if __name__=="__main__":
    # Train process
    # data = read_data("data/Data_flow/rawdata.xlsx")
    # data = data[(data['FLOW'] != 0) & (data['FLOW'] != '0') & (data['FLOW'] != ' ')]
    # # print(data.FLOW.value_counts())
    # X_abs = preprocess(data, 'N_ABS')
    # X_titkw = preprocess(data, ['TITLE', 'KEY_WORDS'])
    # # label
    # y_train = [get_multiple_label(x, ['F', 'I', 'P']) for x in data['FLOW']]
    # # save DOIs of trainset
    # save_model(data['DOI'], "data/Data_0730/trian_DOIs")
    # # 标签计数
    # print("statistics of labels:")
    # target = [''.join(list(map(str, e))) for e in y_train]
    # print(sorted(Counter(target).items()))
    #
    # # LDA for title and keywords
    # X_titkw_lda = get_LDA(X_titkw)
    #
    # # TfIdf for abstrct
    # X_abs_TfIdf = get_TfIdf(X_abs)
    #
    # print("generating features......")
    # X_train_abs = get_TfIdf(X_abs, train=True)
    # print("the shape of tfidf features: ", X_train_abs.shape[1])
    #
    # X_train_titkw = get_LDA(X_titkw, train=True)
    # print("the shape of LDA features: ", X_train_titkw.shape[1])
    #
    # # merge data
    # X_train_merge = merge_features(X_train_abs, X_train_titkw)
    #
    # # scale data
    # X_train = data_scaler(X_train_merge, train=True)
    #
    # oversampling = True
    # if oversampling:
    #     print("oversampling: True")
    #     X_train, y_train = over_sampling(X_train, y_train, "multiple_label")
    # else:
    #     print("oversampling: False")

    # Predict process
    # strategy = 'classifier_chains'
    # # model = train_model(X_train, y_train, strategy)
    # # train_model(X_train, y_train, strategy)
    # if strategy=='classifier_chains':
    #     model = load_model("model/flow/cc")
    # elif strategy=='ovr':
    #     model = load_model("model/flow/ovr")
    # else:
    #     raise Exception("please input correct model path")
    #
    # print("make predictions on all data")
    # journals = ["OR","MSOM","MS","POM","JOM"]
    # make_predictions(journals, model)


    # plot process
    print("Draw trend diagrams.....")
    plot("data/Data_flow/all_prediction.xlsx", ['Sign_F', 'Sign_I', 'Sign_P'])





