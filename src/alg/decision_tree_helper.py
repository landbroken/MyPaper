#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/27
# @Author  : LinYulong

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split

data_table = [
    ["spring", "no", "breeze", "yes"],
    ["winter", "no", "no wind", "yes"],
    ["autumn", "yes", "breeze", "yes"],
    ["winter", "no", "no wind", "yes"],
    ["summer", "no", "breeze", "yes"],
    ["winter", "yes", "breeze", "yes"],
    ["winter", "no", "gale", "yes"],
    ["winter", "no", "no wind", "yes"],
    ["spring", "yes", "no wind", "no"],
    ["summer", "yes", "gale", "no"],
    ["summer", "no", "gale", "no"],
    ["autumn", "yes", "breeze", "no"],
]

# 指定列
data_title = ['season', 'after 8', 'wind', 'lay bed']
data = pd.DataFrame(data_table, columns=data_title)

# sparse=False意思是不产生稀疏矩阵
vec = DictVectorizer(sparse=False)
# 先用 pandas 对每行生成字典，然后进行向量化
feature = data[['season', 'after 8', 'wind']]
X_train = vec.fit_transform(feature.to_dict(orient='record'))
result = data[['lay bed']]
Y_train = vec.fit_transform(result.to_dict(orient='record'))
# 打印各个变量
print('show feature\n', feature)
print('show vector\n', X_train)
print('show vector name\n', vec.get_feature_names())

# 决策树训练
# 划分成训练集，交叉集，验证集，不过这里我们数据量不够大，没必要
# train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size = 0.3)
# 训练决策树
clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, Y_train)
