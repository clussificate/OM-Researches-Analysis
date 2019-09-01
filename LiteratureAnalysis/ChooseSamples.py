# -*- coding: utf-8 -*-
"""
Created on 2019/9/1 13:53

@Author: Kurt

"""
import pandas as pd


if __name__ =="__main__":
    size = 50
    journals = ["OR", "MS", "POM"]
    alldata = []
    for journal in journals:
        filename = "data/Data_0730/"+journal+"_prediction.xlsx"
        X_data= pd.read_excel(filename)
        alldata.append(X_data)
        print("shape of journal %s:" %journal,X_data.shape )
    data = pd.concat(alldata, axis=0)
    print(data.shape)
    out_of_samples = data[data.In_trainset=='No']
    chosen_samples = out_of_samples.sample(size)
    print("choose data......")
    chosen_samples.to_excel("data/Data_0730/chosen_samples.xlsx", index=False)