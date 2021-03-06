# -*- coding: utf-8 -*-
"""
Created on 2019/9/10 22:04
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
from sklearn import metrics
from sklearn.utils import  shuffle

lancaster_stemmer = nltk.LancasterStemmer()
stop_words = GetData.get_stopwords("data/stopwords.txt")


def read_data(filename, shuffle=True):
    rawdata = pd.read_excel(filename)
    if shuffle:
        rawdata = us.shuffle(rawdata,random_state=1994)  # shuffle data
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


def get_TfIdf(data, train=True):
    """
    获取TfIdf特征
    """
    if train:
        TfidfVec = TfidfVectorizer(max_df=0.3, min_df=0.01, ngram_range=(1, 1))
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
        CntVec = CountVectorizer(min_df=0.01, ngram_range=(1, 1))
        lda = LatentDirichletAllocation(n_components=200, learning_method='batch', random_state=0)
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
    y = np.array(y, dtype=int)
    # clf = SVC(C=1,kernel='rbf',probability=True, gamma='scale')  # svc with class_weight
    # clf = XGBClassifier(max_depth=9, n_estimators=50, n_jobs=-1)
    # clf = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5,
    #                     min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
    #                     objective='binary:logistic', nthread=4, scale_pos_weight=1)
    # clf = RandomForestClassifier(max_depth=20, n_estimators=2000,n_jobs=-1)
    clf = lightgbm.sklearn.LGBMClassifier(max_depth=9, num_leaves=500,
                                          n_estimators=50, n_jobs=-1)
    print(clf)
    if strategy == 'ovr':  # OneVsRest strategy also known as BinaryRelevance strategy
        ovr = OneVsRestClassifier(clf)
        ovr.fit(X, y)
        save_model(ovr, "model/flow/ovr")
        return ovr
    elif strategy == 'classifier_chains':
        cc = ClassifierChain(clf)
        cc.fit(X, y)
        save_model(cc, "model/flow/cc")
        return cc
    else:
        raise Exception("Correct strategies：ovr or classifier_chains")


def evaluation(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def downsampling(data, y, prop=0.5):
    y = pd.Series([''.join(list(map(str, e))) for e in y], name='y')
    data = pd.concat([data, y], axis=1)
    data1 = data[data.y=='00']
    data2 = data[~data.index.isin(data1.index)]
    data1 = data1.sample(frac=prop, axis=0)
    return_data = pd.concat([data1, data2],axis=0).reset_index()
    return return_data


def save_predictions(data,preds, proba, filename,journal):
    preds = pd.DataFrame(preds, columns=["Sign_F", "Sign_I", "Sign_P"])
    proba = pd.DataFrame(proba, columns=["Proba_F", "Proba_I", "Proba_P"])
    merge_data = pd.concat([data, preds, proba], axis=1)
    merge_data['Journal'] = journal
    merge_data.to_excel(filename, index=False)


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
    all_data.to_excel('data/Data_flow/all_prediction.xlsx', index=False)


if __name__=="__main__":

    # GetData.create_dataset_flow("data/Data_flow/rawdata.xlsx", "data/Data_flow/trainset.xlsx",
    #                        "data/Data_flow/testset.xlsx")

    train_name = "data/Data_flow/trainset.xlsx"
    trainset= read_data(train_name)[['DOI', 'TITLE', 'KEY_WORDS', 'N_ABS', 'FLOW']]

    # # add new samples
    trainset_others_name = 'data/Data_flow/dataset_others_p.xlsx'
    trainset_others = read_data(trainset_others_name)[['DOI', 'TITLE', 'KEY_WORDS', 'N_ABS', 'FLOW']]
    # # add new samples 2
    trainset_others_name_2 = 'data/Data_flow/dataset_others 2.xlsx'
    trainset_others_2 = read_data(trainset_others_name)[['DOI', 'TITLE', 'KEY_WORDS', 'N_ABS', 'FLOW']]

    trainset = shuffle(pd.concat([trainset, trainset_others, trainset_others_2], axis=0).reset_index())
    y_train = np.array([get_multiple_label(x, ['F', 'I', 'P'])[:-1] for x in trainset['FLOW']])

    # downsampling
    # trainset = downsampling(trainset, y_train, prop=0.3)
    # y_train = np.array([get_multiple_label(x, ['F', 'I', 'P'])[:-1] for x in trainset['FLOW']])

    # # save dois in transet
    save_model(trainset['DOI'], "data/Data_flow/trian_DOIs")

    print("preprocess data......")
    X_train_abs = preprocess(trainset, 'N_ABS')
    X_train_titkw = preprocess(trainset, ['TITLE', 'KEY_WORDS'])

    # ## 标签计数
    print("statistics of labels:")
    target = [''.join(list(map(str, e))) for e in y_train]
    print(sorted(Counter(target).items()))

    test_name = "data/Data_flow/testset.xlsx"
    testset= read_data(test_name, shuffle=False)
    X_test_abs = preprocess(testset, 'N_ABS')
    X_test_titkw = preprocess(testset, ['TITLE','KEY_WORDS'])
    y_test = np.array([get_multiple_label(x, ['F', 'I', 'P'])[:-1] for x in testset['FLOW']])
    # print(testset[0:10])
    # print(X_test_titkw[0:10])
    # # print(y_test)

    print("generating features......")
    X_train_abs = get_TfIdf(X_train_abs, train=True)
    print("the shape of tfidf features: ", X_train_abs.shape[1])
    X_test_abs = get_TfIdf(X_test_abs, train=False)
    # print(X_test_abs[0])
    # print(X_test_abs.shape)

    X_train_titkw = get_LDA(X_train_titkw, train=True)
    print("the shape of LDA features: ", X_train_titkw.shape[1])
    X_test_titkw = get_LDA(X_test_titkw, train=False)
    # print(X_test_titkw[0])
    # print(X_test_titkw.shape)

    # # merge data
    # print(X_train_abs.shape, X_train_titkw.shape)
    # print(X_test_abs.shape, X_test_titkw.shape)
    X_train_merge =  merge_features(X_train_abs , X_train_titkw)
    X_test_merge = merge_features(X_test_abs, X_test_titkw)

    # # scale data
    X_train = data_scaler(X_train_merge, train=True)
    X_test = data_scaler(X_test_merge,train=False)

    oversampling = True
    if oversampling:
        print("oversampling: True")
        X_train, y_train = over_sampling(X_train,y_train,"multiple_label")
        print("statistics of labels after oversampling:")
        target = [''.join(list(map(str, e))) for e in y_train]
        print(sorted(Counter(target).items()))
    else:
        print("oversampling: False")

    # # cv only
    # # choose the random forests model for its fast speed.
    # model = RandomForestClassifier(max_depth=3,n_estimators=1000,n_jobs=-1)
    # model = XGBClassifier(subsample=0.8,colsample_bytree=0.8,max_depth=9,n_estimators=50)
    # model = lightgbm.sklearn.LGBMClassifier(max_depth=9, num_leaves=600,
    #                                       n_estimators=50,n_jobs=-1)
    # print(model)
    # model = ClassifierChain(model)
    # cv_results = cross_val_score(model, X=X_train, y=y_train, cv=5,scoring='f1_micro', n_jobs=1)
    # print("5-kold cv results: ", np.mean(cv_results))

    # grid search + cv
    # params = {'classifier__max_depth': range(5, 10, 2),
    #           'classifier__n_estimators': [50,100,150],
    #           'classifier__subsample':[0.6, 0.8, 1],
    #           'classifier__colsample_bytree':[0.6,0.8,0.1]}
    # model = XGBClassifier(n_jobs=-1)
    #
    # params = {'classifier__max_depth': range(3, 10, 2),
    #           'classifier__n_estimators': [100, 200, 300, 400],
    #           'classifier__max_features': ['auto', 'log2', 'sqrt']
    #           }
    # model = RandomForestClassifier(max_depth=3, n_estimators=1000,n_jobs=-1)
    #
    # params = {'classifier__max_depth': range(5, 10, 2),
    #           'classifier__n_estimators': [100,200,300,400,500],
    #           'classifier__subsample':[0.6, 0.8, 1]}
    # model = lightgbm.sklearn.LGBMClassifier(num_leaves=32000)
    # print(model)
    # model = ClassifierChain(model)
    # cv_results = params_seach(X_train, y_train, model, params)
    # print("the best score:", cv_results.best_score_)
    # print("the best parameters:", cv_results.best_params_)

    # # hold-out evaluation
    strategy = 'classifier_chains'
    print("strategy: ",strategy)
    model = train_model(X_train, y_train, strategy)
    train_preds = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)
    if strategy == 'classifier_chains':
        train_preds = train_preds.toarray()
        # train_preds = np.array([list(map(int, x)) for x in train_preds])
        train_proba = train_proba.toarray()
        test_preds = test_preds.toarray()
        test_proba = test_proba.toarray()

    #     ## model evaluations
    # print("results of trainset: ")
    # print("-"*20)
    # for x,y,z in zip(train_preds, train_proba, y_train):
    #     print("pred: %s, proba: %s, label: %s" %(x, y, z))

    # #pay attention to the performance of F and I samples
    print("the classification report of trainset:")
    evaluation(y_train, train_preds)
    print("the classification report of testset:")
    evaluation(y_test, test_preds)

    # # save predictions
    save_model([y_test, test_proba], "data/Data_flow/tradeoff")

    # print("results of testset")
    # print("-" * 20)
    # for x,y,z in zip(test_preds, test_proba, y_test):
    #     print("pred: %s, proba: %s, label: %s" %(x, y, z))

    # print("make predictions on all data")
    # journals = ["OR","MSOM","MS","POM","JOM"]
    # make_predictions(journals, model)
