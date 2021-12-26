#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/10
# @Author  : LinYulong

import numpy
import pandas

from src.alg import knn_helper, math_helper
from src.alg.math_helper import root_mean_square_error
from src.alg.medicine_type import DiseaseCheckType
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
    r_columns = []
    r2_columns = []
    mae_columns = []
    rmsd_columns = []
    m_res_columns = []
    sd_res_columns = []
    for columns_idx in range(columns_size):
        test_data_set, test_labels = column_split(test_df, columns_idx)
        train_data_set, train_labels = column_split(train_df, columns_idx)

        test_data_set = test_data_set * train_cfg.get_times()
        test_data_set = pandas.DataFrame(test_data_set, dtype=int)
        test_labels = test_labels * train_cfg.get_times()
        test_labels = pandas.DataFrame(test_labels, dtype=int)
        train_data_set = train_data_set * train_cfg.get_times()
        train_data_set = pandas.DataFrame(train_data_set, dtype=int)
        train_labels = train_labels * train_cfg.get_times()
        train_labels = pandas.DataFrame(train_labels, dtype=int)

        np_test_data_set = numpy.array(test_data_set)
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels).ravel()

        knn_k = train_cfg.get_knn_k()
        result = knn_helper.ski_classify(np_test_data_set, np_train_data_set, np_train_labels, knn_k)

        result = result / train_cfg.get_times()
        np_test_labels: numpy.ndarray = numpy.array(test_labels) / train_cfg.get_times()

        # 求偏差，离散程度
        rmse = root_mean_square_error(result, np_test_labels)
        rmse_columns.append(rmse)
        # 求绝对误差
        err_single = result - np_test_labels
        err_columns.append(err_single)
        #
        r = math_helper.my_pearson_correlation_coefficient(np_test_labels.ravel(), result)
        r_columns.append(r)
        #
        r2 = math_helper.my_pearson_correlation_coefficient(np_test_labels.ravel(), result)
        r2_columns.append(r2)
        #
        mae = math_helper.mean_absolute_error(np_test_labels.ravel(), result)
        mae_columns.append(mae)
        #
        rmsd = math_helper.root_mean_square_error(np_test_labels.ravel(), result)
        rmsd_columns.append(rmsd)
        #
        m_res = math_helper.my_mean_of_residuals(np_test_labels.ravel(), result)
        m_res_columns.append(m_res)
        #
        sd_res = math_helper.my_standard_deviation_of_residuals(np_test_labels.ravel(), result)
        sd_res_columns.append(sd_res)
    err_arr = numpy.array(err_columns)
    err_percent = caculate_err_percent(err_arr)
    r_avg = numpy.average(r_columns)
    print("r avg = " + str(r_avg))
    r2_arr = numpy.array(r2_columns)
    print("r2 avg = " + str(numpy.average(r2_arr)))
    mae_avg = numpy.average(mae_columns)
    print("mae_avg = " + str(mae_avg))

    rmsd_avg = numpy.average(rmsd_columns)
    print("rmsd_columns = " + str(rmsd_avg))

    m_res_avg = numpy.average(m_res_columns)
    print("m_res_columns = " + str(m_res_avg))

    sd_res_avg = numpy.average(sd_res_columns)
    print("sd_res_columns = " + str(sd_res_avg))

    return numpy.array(rmse_columns), err_percent


def predict(df_predict: pandas.DataFrame, real_data: pandas.DataFrame, delete_idx: numpy.ndarray):
    pass


def is_negative(score: float) -> int:
    if score > 2.0:  # 人为指定的
        return DiseaseCheckType.negative.value  # 阴性
    else:
        return DiseaseCheckType.positive.value  # 阳性


def to_negative_and_positive_table(df: pandas.DataFrame, merged_columns_size: int = 1) -> pandas.DataFrame:
    """
    原分數表格转为阴阳性表格
    :param df: 原始表格
    :return: 阴阳性表格
    """
    df_float = df * 1.0 / merged_columns_size
    ret = df_float.applymap(is_negative)
    return ret


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
