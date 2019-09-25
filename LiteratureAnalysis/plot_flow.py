# -*- coding: utf-8 -*-
"""
Created on 2019/9/15 15:04

@Author: Kurt

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.25)

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def analysis(label):
    if label in ['Sign_F', 'Proba_F']:
        newlabel = "Financial flow"

    elif label in ["Sign_I", 'Proba_I']:
        newlabel = "Information flow"

    elif label in ["Sign_P", 'Proba_P']:
        newlabel = "Physical flow"

    else:
        raise Exception("wrong label")

    return newlabel

def style_tri(data):
    years = data.index
    names = data.columns[:-1]  # 去除掉year
    fig, ax = plt.subplots(len(names), 1)
    for i, label in enumerate(names):
        cnt = data[label]
        sns.barplot(x=years, y=cnt, color="salmon", saturation=.5, ax=ax[i])
        # delete spines
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

        # add values on bar plot
        for p in ax[i].patches:
            ax[i].annotate("%i" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                           textcoords='offset points')
        ax[i].set_title("%s" % label,fontsize=22)
        ax[i].set_xlabel("")
        ax[i].set_ylabel("")
        ax[i].set_xticklabels(years,rotation=45)
    # plt.suptitle("The Chancing Trends", fontsize = 15)
    fig.text(0.5, 0.04, 'Year', ha='center',fontsize=24)
    fig.text(0.06, 0.5, 'Count', va='center', fontsize=24, rotation='vertical')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return None

def style_one(rawdata,sigs):
    data = rawdata.melt('YEAR', var_name='Type', value_name='Count')
    g = sns.barplot(x='YEAR',y='Count',hue='Type',data=data)

    # modify legends
    plt.title('The Chancing Trends',  fontsize = 15)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    return None

def process(rawdata, sigs):
    min_year = rawdata.YEAR.min()
    max_year = rawdata.YEAR.max()
    newdata = pd.DataFrame(np.array(range(min_year, max_year + 1)), columns=['YEAR']).set_index('YEAR')
    for i, label in enumerate(sigs):
        group = rawdata[rawdata[label] == 1].groupby(['YEAR']).count().DOI.to_frame()
        group.rename(columns={'DOI':analysis(label)},inplace= True)
        newdata = newdata.join(group)

    newdata = newdata.fillna(0)
    newdata['YEAR'] = newdata.index
    return newdata


def process_proba(rawdata, probas, thetas):
    min_year = rawdata.YEAR.min()
    max_year = rawdata.YEAR.max()
    newdata = pd.DataFrame(np.array(range(min_year, max_year + 1)), columns=['YEAR']).set_index('YEAR')

    for label, theta in zip(probas, thetas):
        rawdata[label] = pd.Series(np.where(rawdata[label].values >= theta, 1, 0))

    for i, label in enumerate(probas):
        group = rawdata[rawdata[label] == 1].groupby(['YEAR']).count().DOI.to_frame()
        group.rename(columns={'DOI': analysis(label)}, inplace=True)
        newdata = newdata.join(group)

    newdata = newdata.fillna(0)
    newdata['YEAR'] = newdata.index
    return newdata

if __name__=="__main__":
    data = pd.read_excel("data/Data_flow/all_prediction.xlsx")
    # print(data.info())
    thetas = [0.5, 0.3]
    probs=['Proba_F', 'Proba_I']
    data = process_proba(data, probs, thetas)
    style_tri(data)
    # style_one(data,sigs)