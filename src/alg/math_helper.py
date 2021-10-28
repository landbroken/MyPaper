#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/14
# @Author  : LinYulong

import numpy
from sklearn import metrics

def min_id(arr: numpy.ndarray):
    min_idx = 0
    min_val = arr[0]
    for i in range(arr.size):
        if min_val > arr[i]:
            min_idx = i
            min_val = arr[i]

    return min_idx


def mean_absolute_error(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    平均绝对值误差(Mean Absolute Error,MAE)
    https://blog.csdn.net/wydbyxr/article/details/82894256
    https://zhuanlan.zhihu.com/p/353125247
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    """
    return metrics.mean_absolute_error(y_true, y_pred)