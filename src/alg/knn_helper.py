#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import operator

import numpy
from numpy import array, tile
from sklearn import neighbors, datasets


def classify(in_x: list, data_set: array, labels: list, k: int):
    """
    分类
    :param in_x: 待分类点坐标
    :param data_set: 已分类数据集
    :param labels: data_set 每一项对应的分类结果
    :param k: knn的k值
    :return: 分类结果
    """
    # 距离计算，使用欧式距离
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indices = distances.argsort()
    # 选择距离最小的 k 个点
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def ski_classify(wait_predict: numpy.ndarray, data_set: numpy.ndarray, labels: numpy.ndarray, k: int):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(data_set, labels)
    result = clf.predict(wait_predict)
    return result
