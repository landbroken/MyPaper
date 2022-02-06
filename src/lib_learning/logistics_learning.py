#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 LinYulong. All Rights Reserved 
#
# @Time    : 2022/2/7
# @Author  : LinYulong
# @Description:
# https://blog.csdn.net/u013421629/article/details/78470020

# from __future__ import division

# 逻辑斯蒂回归模型Logostics regression
# 导入pandas与numpy工具包。
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# 从sklearn.preprocessing里导入StandardScaler。
from sklearn.preprocessing import StandardScaler
# 从sklearn.linear_model里导入LogisticRegression与SGDClassifier。
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
# 从sklearn.metrics里导入classification_report模块。
from sklearn.metrics import classification_report

# 创建特征列表。
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据。
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)

# print data
# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

# 输出data的数据量和维度。
# print  data.shape

# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25,
                                                    random_state=33)

# 查验训练样本的数量和类别分布。
# print  y_train.value_counts()

# 查验测试样本的数量和类别分布。
# print y_test.value_counts()

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 初始化LogisticRegression与SGDClassifier。
lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(x_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(x_test)

# print lr_y_predict
# 调用SGDClassifier中的fit函数/模块用来训练模型参数。
sgdc.fit(x_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中。
sgdc_y_predict = sgdc.predict(x_test)

# print sgdc_y_predict


# 打印混淆矩阵
labels1 = list(set(lr_y_predict))
conf_mat1 = confusion_matrix(y_test, lr_y_predict, labels=labels1)
print("Logistics regression")
print(conf_mat1)

labels2 = list(set(sgdc_y_predict))
conf_mat2 = confusion_matrix(y_test, sgdc_y_predict, labels=labels2)
print("sgdc_y_predict")
print(conf_mat2)

# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuracy of LR Classifier:', lr.score(x_test, y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果。
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuarcy of SGD Classifier:', sgdc.score(x_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果。
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
