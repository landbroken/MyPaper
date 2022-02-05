#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 LinYulong. All Rights Reserved 
#
# @Time    : 2022/2/5
# @Author  : LinYulong

# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


def main():
    iris = datasets.load_iris()  # 典型分类数据模型
    # 这里我们数据统一用pandas处理
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['class'] = iris.target

    # 这里只取两类
    #     data = data[data['class']!=2]
    # 为了可视化方便，这里取两个属性为例
    x = data[data.columns.drop('class')]
    y = data['class']

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x_train, y_train)

    # 显示训练结果
    print(lda.means_)  # 中心点
    print(lda.score(x_test, y_test))  # score是指分类的正确率
    print(lda.scalings_)  # score是指分类的正确率

    x_2d = lda.transform(x)  # 现在已经降到二维x_2d=np.dot(X-lda.xbar_,lda.scalings_)
    # 对于二维数据，我们做个可视化
    # 区域划分
    lda.fit(x_2d, y)
    h = 0.02
    x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
    y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    tmp = np.c_[xx.ravel(), yy.ravel()]
    z = lda.predict(tmp)
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired)

    # 做出原来的散点图
    class1_x = x_2d[y == 0, 0]
    class1_y = x_2d[y == 0, 1]
    l1 = plt.scatter(class1_x, class1_y, color='b', label=iris.target_names[0])
    class1_x = x_2d[y == 1, 0]
    class1_y = x_2d[y == 1, 1]
    l2 = plt.scatter(class1_x, class1_y, color='y', label=iris.target_names[1])
    class1_x = x_2d[y == 2, 0]
    class1_y = x_2d[y == 2, 1]
    l3 = plt.scatter(class1_x, class1_y, color='r', label=iris.target_names[2])

    plt.legend(handles=[l1, l2, l3], loc='best')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
