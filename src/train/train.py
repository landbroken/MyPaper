#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/10
# @Author  : LinYulong
import numpy
import pandas

from src.alg import knn_helper


def data_and_label_split(df: pandas.DataFrame, columns_idx: int):
    """
    切分指定的列
    :param df:
    :param columns_idx:
    :return:
    """
    columns_size = df.columns.size
    columns_begin = 0
    columns_end = columns_idx
    df_part1: pandas.DataFrame = df.iloc[:, columns_begin:columns_end]
    columns_begin = columns_idx + 1
    columns_end = columns_size + 1
    df_part2 = df.iloc[:, columns_begin:columns_end]
    data_set = df_part1.join(df_part2)  # 所有行，除选定列外其它列
    labels = df.iloc[:, columns_idx]  # 所有行，选定列
    return data_set, labels


def root_mean_square_error(predict: numpy.ndarray, real: numpy.ndarray):
    tmp: numpy.ndarray = predict - real
    tmp = tmp ** 2
    ret: float = tmp.sum()
    ret = ret * 1.0 / tmp.size
    ret = ret ** 0.5
    return ret


def train(test_df: pandas.DataFrame, train_df: pandas.DataFrame):
    columns_size = train_df.columns.size
    rmse_columns = []
    for columns_idx in range(columns_size):
        test_data_set, test_labels = data_and_label_split(test_df, columns_idx)
        train_data_set, train_labels = data_and_label_split(train_df, columns_idx)
        np_test_data_set = numpy.array(test_data_set)
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels)
        result = knn_helper.ski_classify(np_test_data_set, np_train_data_set, np_train_labels, 10)
        np_test_labels = numpy.array(test_labels)
        # 求偏差，离散程度
        rmse = root_mean_square_error(result, np_test_labels)
        rmse_columns.append(rmse)
        # 求绝对误差
        err_arr = result - np_test_labels
    return numpy.array(rmse_columns)
