#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/10
# @Author  : LinYulong
import typing

import numpy
import pandas

from src.alg import knn_helper
from src.train import train_cfg


def column_split(df: pandas.DataFrame, columns_idx: int):
    """
    切分指定的列
    :param df:待切分的 DataFrame
    :param columns_idx: 指定列
    :return: 其余列，指定列
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


def root_mean_square_error(_predict: numpy.ndarray, real: numpy.ndarray):
    tmp: numpy.ndarray = _predict - real
    times = train_cfg.get_times()
    tmp = tmp * 1.0 / times  # 还原实际数量级
    tmp = tmp ** 2
    ret: float = tmp.sum()
    ret = ret * 1.0 / tmp.size
    ret = ret ** 0.5
    return ret


def caculate_err_percent(err_arr: numpy.ndarray):
    abs_arr = numpy.abs(err_arr)
    columns_size = err_arr.shape[1]
    line_size = err_arr.shape[0]
    err_ret = []
    for i in range(line_size):
        cur_column = abs_arr[i:i + 1]
        cur_sum = cur_column.sum()
        err_val = cur_sum * 1.0 / columns_size  # 平均误差绝对值
        times = train_cfg.get_times()
        table_range_max = train_cfg.get_table_range_max()
        err_percent = err_val / (times * table_range_max)  # 平均误差百分比
        err_ret.append(err_percent)
    return numpy.array(err_ret)


def train(test_df: pandas.DataFrame, train_df: pandas.DataFrame):
    columns_size = train_df.columns.size
    rmse_columns = []
    err_columns = []
    for columns_idx in range(columns_size):
        test_data_set, test_labels = column_split(test_df, columns_idx)
        train_data_set, train_labels = column_split(train_df, columns_idx)
        np_test_data_set = numpy.array(test_data_set)
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels)
        knn_k = train_cfg.get_knn_k()
        result = knn_helper.ski_classify(np_test_data_set, np_train_data_set, np_train_labels, knn_k)
        np_test_labels = numpy.array(test_labels)
        # 求偏差，离散程度
        rmse = root_mean_square_error(result, np_test_labels)
        rmse_columns.append(rmse)
        # 求绝对误差
        err_single = result - np_test_labels
        err_columns.append(err_single)
    err_arr = numpy.array(err_columns)
    err_percent = caculate_err_percent(err_arr)
    return numpy.array(rmse_columns), err_percent


def predict(df_predict: pandas.DataFrame, real_data: pandas.DataFrame, delete_idx: numpy.ndarray):
    pass


def is_negative(score: int) -> int:
    if score > 2:
        return 1  # 阴性
    else:
        return 0  # 阳性


def to_negative_and_positive_table(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    原分數表格转为阴阳性表格
    :param df: 原始表格
    :return: 阴阳性表格
    """
    ret = df.applymap(is_negative)
    return ret


def sub_table(df: pandas.DataFrame) -> pandas.DataFrame:
    pass


def table_sort(df: pandas.DataFrame, sort_fun) -> pandas.DataFrame:
    """
    对表格排序
    :param df: 待排序表格
    :param sort_fun: 排序算法
    :return: 排序后的表格
    """
    return sort_fun(df)


def get_question_group_mark(np: numpy.ndarray):
    """
    求题组得分
    :param np:
    :return:
    """
    pass


def get_question_mark(np: numpy.ndarray):
    """
    求题目得分
    :param np:
    :return:
    """


def cal_sensitivity(tp: int, fp: int) -> float:
    """
    灵敏度计算。
    灵敏度是指测试正确检测出患有这种疾病的患者的能力
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    :param tp: 真阳性人数
    :param fp: 假阳性人数
    :return:
    """
    tpr = tp / (tp + fp * 1.0)  # tp rate
    return tpr


def cal_specificity(tn: int, fp: int) -> float:
    """
    特异性计算
    :param tn: 真阴性人数
    :param fp: 假阳性人数
    :return: tnr: 特异性/选择性/真阴性率
    """
    tnr = tn / (tn + fp * 1.0)
    return tnr
