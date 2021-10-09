#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import math

import pandas


def single_verify(test_df: pandas.DataFrame, train_df: pandas.DataFrame, fun):
    fun(test_df, train_df)


def cross_verify(verify_cnt: int, df: pandas.DataFrame, fun):
    """
    n 折交叉验证
    :param fun: 交叉验证时使用的算法执行函数
    :param df: 待验证数据
    :param verify_cnt: 交叉验证的折数
    :return:
    """
    test_size = math.ceil(df.index.size / verify_cnt)  # 测试集大小
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
        single_verify(test_df, train_df, fun)
