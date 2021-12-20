#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/14
# @Author  : LinYulong
import math

import numpy
from sklearn import metrics
from sklearn.metrics import mean_squared_error


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


def my_mean_squared_error(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    平均方差误差
    :param y_true: 真实值
    :param y_pred：预测值
    :return:
    """
    return metrics.mean_squared_error(y_true, y_pred)


def root_mean_square_error(_predict: numpy.ndarray, real: numpy.ndarray):
    ret = numpy.sqrt(metrics.mean_squared_error(real, _predict))
    return ret


def my_average_error(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    average error
    :param y_true:
    :param y_pred:
    :return:
    """
    gap = y_pred - y_true
    ret = sum(gap) * 1.0 / len(y_true)
    return ret


def my_pearson_correlation_coefficient(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    皮尔逊相关系数
    https://blog.csdn.net/qq_40260867/article/details/90667462
    :param y_true:
    :param y_pred:
    :return:
    """
    pccs = numpy.corrcoef(y_true, y_pred)
    return pccs


def my_mean_of_residuals(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    residuals = y_pred - y_true
    ret = residuals * 1.0 / len(y_true)
    return ret


def my_standard_deviation_of_residuals(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    残差的标准差
    :param y_true:
    :param y_pred:
    :return:
    """
    residuals = y_pred - y_true
    ret = numpy.std(residuals)
    return ret


def my_mean_predicted(y_pred: numpy.ndarray):
    ret = numpy.average(y_pred)
    return ret
