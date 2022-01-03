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
    ret = pccs[0][1]
    return ret


def my_pearson_correlation_coefficient_square(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    皮尔逊相关系数
    https://blog.csdn.net/qq_40260867/article/details/90667462
    :param y_true:
    :param y_pred:
    :return:
    """
    pccs = numpy.corrcoef(y_true, y_pred)
    tmp = pccs[0][1]
    ret = tmp * tmp
    return ret


def my_coefficient_of_correlation(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """
    相关系数
    https://baike.baidu.com/item/%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0/3109424?fr=aladdin
    https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html
    https://zhuanlan.zhihu.com/p/37609917
    https://blog.csdn.net/ybdesire/article/details/6270328?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.pc_relevant_paycolumn_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.pc_relevant_paycolumn_v2&utm_relevant_index=2
    https://zhuanlan.zhihu.com/p/34380674
    :param y_true:
    :param y_pred:
    :return:
    """
    cov_x = numpy.stack((y_true, y_pred), axis=0)
    '''
    -- 0       , 1
    0: cov(a,a), cov(a,b)
    1: cov(b,a), cov(b,b)
    '''
    np_cov = numpy.cov(cov_x)
    # var（）函数默认计算总体方差。要计算样本的方差，必须将 ddof 参数设置为值1。
    var_true = numpy.var(y_true, ddof=1)
    var_predict = numpy.var(y_pred, ddof=1)
    ret_matrix = np_cov / ((var_true * var_predict) ** 0.5)
    ret = ret_matrix[0][1]
    return ret


def my_mean_of_residuals(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    residuals = y_pred - y_true
    tmp = residuals * 1.0 / len(y_true)
    ret = numpy.mean(tmp)
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
