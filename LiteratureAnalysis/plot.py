# -*- coding: utf-8 -*-
"""
Created on 2019/8/26 23:04

@Author: Kurt

"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_hist(data, title):
    em = data["Proba_Emp"]
    ex = data["Proba_Exp"]
    ana = data["Proba_Ana"]
    plt.figure(figsize=(5, 10), dpi=100)
    plt.subplot(311)
    plt.hist(em, bins=500,color="green",label='Empirical')
    plt.title("The Histogram of Empirical Papers")
    plt.subplot(312)
    plt.hist(ex, bins=500,color="red",label="Experimental")
    plt.title("The Histogram of Experimental Papers")
    plt.subplot(313)
    plt.hist(ana, bins = 500,color="blue",label="Analytical")
    plt.title("The Histogram of Analytical Papers")
    plt.suptitle("Histograms of Predictive Probability for %s" % title)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__=="__main__":
    filename = "data/Data_0730/OR_prediction.xlsx"
    title = filename.split("/")[-1].split(("_"))[0]
    print(title)
    data = pd.read_excel(filename)[["Proba_Emp","Proba_Exp","Proba_Ana"]]
    plot_hist(data,title)