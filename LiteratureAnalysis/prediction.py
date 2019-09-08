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

def save_predictions(data,preds, proba, filename, train_DOIs,journal):
    train_DOIs = train_DOIs.values
    preds = pd.DataFrame(preds,columns=["Sign_Emp","Sign_Exp","Sign_Ana"])
    proba = pd.DataFrame(proba,columns=["Proba_Emp","Proba_Exp","Proba_Ana"])
    merge_data = pd.concat([data,preds,proba],axis =1)
    merge_data['Journal']= journal
    merge_data['In_trainset'] = 'No'

    for i, value in enumerate(train_DOIs):
        print(value)
        merge_data.loc[merge_data.DOI==value, 'In_trainset'] = 'Yes'

    merge_data.to_excel(filename,index = False)


def save_predictions2(data, preds, proba, filename, train_DOI):
    # load classifier model

    train_DOIs = train_DOI.values
    preds = pd.DataFrame(preds,columns=["Sign_Emp","Sign_Exp","Sign_Ana"])
    proba = pd.DataFrame(proba,columns=["Proba_Emp","Proba_Exp","Proba_Ana"])
    merge_data = pd.concat([data,preds,proba],axis =1)
    merge_data['In_trainset'] = 'No'
    for i, value in enumerate(train_DOIs):
        print(value)
        merge_data.loc[merge_data.DOI==value, 'In_trainset'] = 'Yes'

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
    train_DOI = load_model("data/Data_0730/trian_DOIs")

    # load data
    print("load data......")
    #  training sets splits by journals

    # journals = ["OR","MSOM","MS","POM"]
    # con_data = []
    # for journal in journals:
    #     filename = "data/Data_0730/"+journal+".xlsx"
    #     X_data= read_data(filename)
    #     print("preprocess %s data......" % filename.split("/")[-1].split('.')[0])
    #     X_abs = preprocess(X_data,'N_ABS')
    #     X_titkw = preprocess(X_data,['TITLE', 'KEY_WORDS'])
    #     # print(X_abs[7])  # test case no.7 data
    #     # predict
    #     X_abs_TfIdf = Tfidf.transform(X_abs).toarray()
    #     X_titkw_cnt = CntVec.transform(X_titkw)
    #     X_titkw_lda = LDA.transform(X_titkw_cnt)
    #
    #     X_merge = merge_features(X_abs_TfIdf, X_titkw_lda)
    #     X_std = ss.transform(X_merge)
    #     print("make and save predictions....")
    #     preds = clf.predict(X_std)
    #     proba = clf.predict_proba(X_std)
    #     if strategy == 'classifier_chains':
    #         preds = preds.toarray()
    #         proba = proba.toarray()
    #
    #     save_predictions(X_data[['DOI','TITLE','KEY_WORDS','N_ABS','YEAR']],
    #                      preds,proba,filename.split(".")[0]+"_prediction.xlsx",
    #                      train_DOI,journal)
    #
    #     print("%s done!" %journal)

    # full training set
    filename = "data/Data_0730/all_prediction.xlsx"
    X_data= read_data(filename)
    X_abs = preprocess(X_data,'N_ABS')
    X_titkw = preprocess(X_data,['TITLE', 'KEY_WORDS'])

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

    save_predictions2(X_data[['DOI','TITLE','KEY_WORDS','N_ABS','YEAR']],
                      preds, proba,"data/Data_0730/all_prediction.xlsx",train_DOI)

    print("succeed!")



