#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 LinYulong. All Rights Reserved 
#
# @Time    : 2022/2/5
# @Author  : LinYulong
# @Description: 参考
# https://blog.csdn.net/qq_34720818/article/details/105865215
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

# import numpy as np
# from numpy import genfromtxt
# from sklearn import linear_model
#
# # 读取数据
# data = genfromtxt(r'longley.csv', delimiter=',')
#
# # 切分数据
# x_data = data[1:, 2:]
# y_data = data[1:, 1, np.newaxis]
#
# # 训练模型
# model = linear_model.LassoCV()
# model.fit(x_data, y_data)
#
# # 训练后选择的lasso系数
# print(model.alpha_)
# # 训练后线性模型参数
# print(model.coef_)
#
# # 预测值
# print(model.predict(x_data[-2, np.newaxis]))
# print(y_data[-2])  # 真实值

from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)
clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(clf.coef_)

print(clf.intercept_)
