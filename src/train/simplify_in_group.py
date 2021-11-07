#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
# @Description 组内简化
import numpy
import pandas

from src.alg import cross_verify, math_helper
from src.excel import excel_helper
from src.train import train_cfg, train


def simplify_in_group(filepath: str):
    df_origin: pandas.DataFrame = excel_helper.read_resource(filepath)
    ret_np = train.negative_and_positive_split(df_origin)
    print("阴阳性计算")
    print(ret_np)

    merge_fun = train_cfg.get_merge_func()
    df = merge_fun(df_origin)
    # 简化计算表格
    question_size = df.columns.size
    df_train = df
    while question_size > 1:
        # 当前轮简化
        cross_verify_times = train_cfg.get_cross_verify_times()
        rmse_arr, err_percent = cross_verify.cross_verify(cross_verify_times, df_train, train.train)
        min_idx = math_helper.min_id(rmse_arr)
        min_err_percent = err_percent[min_idx]
        print("cur rmse = " + str(rmse_arr[min_idx]) +
              ", cur err percent = " + str(min_err_percent) +
              ", idx = " + str(min_idx))
        # 下一轮数据
        next_df, min_df = train.column_split(df_train, min_idx)
        df_train = next_df
        question_size -= 1


def chd_group_get() -> numpy.ndarray:
    chd_gp = [
        [1, 2, 9],
        [3, 4, 5, 6, 7],
        [11],
        [8, 10],
        [12, 13, 14],
    ]
    return numpy.array(chd_gp)


simplify_in_group("/冠心病.xlsx")
