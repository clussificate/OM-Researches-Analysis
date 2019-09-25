# -*- coding: utf-8 -*-
"""
Created on 2019/9/1 13:53

@Author: Kurt

"""
import pandas as pd
import tools
from tools import save_model

print("random sampling")
# size = 100
# journals = ["OR", "MS", "POM"]
# # alldata = []
# # for journal in journals:
# #     filename = "data/Data_0730/"+journal+"_prediction.xlsx"
# #     X_data= pd.read_excel(filename)
# #     alldata.append(X_data)
# #     print("shape of journal %s:" %journal,X_data.shape )
# # data = pd.concat(alldata, axis=0)
# # data.to_excel("data/Data_0730/all_prediction.xlsx", index=False)
#
# data = pd.read_excel("data/Data_0730/all_prediction.xlsx", index=False)
# print(data.shape)
# print("choosing data......")
# out_of_samples = data[data.In_trainset == 'No']
# chosen_samples = out_of_samples.sample(size)
#
# chosen_samples.to_excel("data/Data_0730/chosen_samples 9-5.xlsx", index=False)
# print("Done!")

# filename = 'data/Data_flow/all_prediction.xlsx'
# data = pd.read_excel(filename)
# data = tools.random_sampling(data, size=200)
# data.to_excel("data/Data_flow/chosen_samples 9-16.xlsx", index=False)
#
# print("Done!")


# filename = 'data/Data_flow/all_prediction.xlsx'
# data = pd.read_excel(filename)
# data = data[(data.Proba_F>0.1) | (data.Proba_I>0.3)]
# print(data.shape)
# print(data[['Proba_F','Proba_I','Proba_P']].head(20))
# data = tools.random_sampling(data, size=200)
# data.to_excel("data/Data_flow/chosen_samples 9-21.xlsx", index=False)

# 验证是否重复
# doi1 = pd.read_excel("data/Data_flow/chosen_samples 9-21.xlsx")['DOI'].values
# doi2 = pd.read_excel("data/Data_flow/chosen_samples 9-16.xlsx")['DOI'].values
# Flag = False
# for x in doi1:
#     if x in doi2:
#         Flag=True
#         break
# print(Flag)


print("Done!")
