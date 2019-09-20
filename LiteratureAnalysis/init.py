# -*- coding: utf-8 -*-
"""
Created on 2019/7/28 0:27

@Author: Kurt

"""
import GetData

# 生成初始的训练集和测试集
_, _ = GetData.create_dataset("data/alldata.xlsx", 'TYPE',"data/trainset.csv", "data/testset.csv","|")