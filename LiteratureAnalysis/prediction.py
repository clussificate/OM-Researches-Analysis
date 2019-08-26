# -*- coding: utf-8 -*-
"""
Created on 2019/7/31 12:00

@Author: Kurt

"""
from tools import *
import nltk
import  GetData
import pandas as pd


lancaster_stemmer = nltk.LancasterStemmer()
stop_words = GetData.get_stopwords("data/stopwords.txt")
def read_data(filename):
    rawdata = pd.read_excel(filename)
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

def save_predictions(data,preds, proba, filename):
    preds = pd.DataFrame(preds,columns=["Sign_Emp","Sign_Exp","Sign_Ana"])
    proba = pd.DataFrame(proba,columns=["Proba_Emp","Proba_Exp","Proba_Ana"])
    merge_data = pd.concat([data,preds,proba],axis =1)
    merge_data.to_excel(filename,index = False)


if __name__ =="__main__":

    # load classifier model
    strategy = 'classifier_chains'
    if strategy == 'classifier_chains':
        clf = load_model("model/cc")
    elif strategy == 'ovr':
        clf = load_model("model/ovr")
    else:
        raise Exception("Correct strategies：ovr or classifier_chains")

    # load feature models:
    print("load trained models......")
    CntVec = load_model("model/CntVec")
    LDA = load_model("model/LDA")
    Tfidf = load_model("model/tfidf")
    ss = load_model("model/ss")

    # load data
    print("load data......")
    filename = "data/Data_0730/OR.xlsx"
    X_data= read_data(filename)
    print("preprocess data......")
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
    preds = clf.predict(X_std)
    proba = clf.predict_proba(X_std)
    if strategy == 'classifier_chains':
        preds = preds.toarray()
        proba = proba.toarray()

    save_predictions(X_data[['DOI','TITLE','KEY_WORDS','N_ABS']],
                     preds,proba,filename.split(".")[0]+"_prediction.xlsx")
    print("draw histograms...")

    print("Done!")


