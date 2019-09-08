# -*- coding: utf-8 -*-
"""
Created on 2019/7/27 21:39

@Author: Kurt

"""
import tools
import pickle
import nltk
import matplotlib.pyplot as plt
import xgboost

model = tools.load_model("model/tfidf")
params = model.get_feature_names()

lancaster_stemmer = nltk.LancasterStemmer()
print(lancaster_stemmer.stem("experimental"))
print(lancaster_stemmer.stem("experiments"))
print(lancaster_stemmer.stem("experiment"))

for i in range(len(params)):
    print(params[i], end=', ')
    if i%10==0:
        print('\n')

# cc = tools.load_model("model/cc")
# clf0 = cc.classifiers_[0]
# clf1 = cc.classifiers_[1]
# clf2 = cc.classifiers_[2]
# plt.subplot(311)
# plt.bar(range(len(clf0.feature_importances_)), clf0.feature_importances_)
# plt.subplot(312)
# plt.bar(range(len(clf1.feature_importances_)), clf1.feature_importances_)
# plt.subplot(313)
# plt.bar(range(len(clf2.feature_importances_)), clf2.feature_importances_)
#
# plt.show()
# print(len(clf2.feature_importances_))
