#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import math

import numpy
import pandas

from src.alg import knn_helper
from src.train import train_cfg


def single_verify(test_df: pandas.DataFrame, train_df: pandas.DataFrame, fun):
    return fun(test_df, train_df)


def cross_verify(verify_cnt: int, df: pandas.DataFrame, fun):
    """
    n 折交叉验证
    :param fun: 交叉验证时使用的算法执行函数
    :param df: 待验证数据
    :param verify_cnt: 交叉验证的折数
    :return:
    """
    test_size = math.ceil(df.index.size / verify_cnt)  # 测试集大小
    sum_ret = numpy.array([])
    sum_err = numpy.array([])
    for i in range(verify_cnt):
        begin_line = 0 + test_size * i
        end_line = begin_line + test_size
        # 测试集
        test_df = df[begin_line:end_line]
        # 训练集
        train_df_part1 = df[:begin_line]
        train_df_part2 = df[end_line:]
        train_df = train_df_part1.append(train_df_part2)
        # 执行计算
        single_ret, single_err = single_verify(test_df, train_df, fun)
        if i == 0:
            sum_ret = single_ret
            sum_err = single_err
        else:
            sum_ret += single_ret
            sum_err += single_err

    ret = sum_ret / verify_cnt
    ret2 = sum_err / verify_cnt
    return ret, ret2


def cross_verify_2(verify_cnt: int, df_feature: pandas.DataFrame, df_result: pandas.DataFrame, fun):
    """
    n 折交叉验证
    :param verify_cnt: 交叉验证的折数
    :param df_result: 训练用结果
    :param df_feature: 预测用特征
    :param fun: 交叉验证时使用的算法执行函数
    :return:
    """
    test_size = math.ceil(df_feature.index.size / verify_cnt)  # 测试集大小
    for i in range(verify_cnt):
        begin_line = 0 + test_size * i
        end_line = begin_line + test_size
        # 选择 1 组作为测试集
        test_data_set = df_feature[begin_line:end_line]
        test_labels = df_result[begin_line:end_line]

        # 选择 verify_cnt -1 组作为训练集
        train_df_part1 = df_feature[:begin_line]
        train_df_part2 = df_feature[end_line:]
        train_data_set = train_df_part1.append(train_df_part2)

        train_labels_part1 = df_result[:begin_line]
        train_labels_part2 = df_result[end_line:]
        train_labels = train_labels_part1.append(train_labels_part2)

        # 转为算法识别类型
        np_test_data_set = numpy.array(test_data_set)
        np_test_labels = numpy.array(test_labels)
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels)

        if numpy.size(np_test_data_set) <= 0:
            # n 折计算时， (ceil（total_cnt / n）) * n 可能超过 total_cnt
            print("zero")
            break

        # 使用 XX 算法拟合，得到模型
        knn_k = train_cfg.get_knn_k()
        clf = knn_helper.ski_fit(np_train_data_set, np_train_labels, knn_k)
        # 计算得到预测分值
        clf.predict(np_test_data_set)

        # 预测分值划分成相应的阴阳性
        
        # 计算混淆矩阵
