#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import numpy
import pandas

from src.alg import cross_verify, math_helper
from src.excel import excel_helper
from src.train import train_cfg
from train import train


def merge_chd(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    将表格按组合并。
    注意：sklearn 算法要求不是 float，所以统一乘精度要求然后取整
    :param df:
    :return:
    """
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 5), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['G1', 'G2', 'G3', 'G4', 'G5'], dtype=int)
    ret["G1"] = (df["CHD1"] + df["CHD2"] + df["CHD9"]) / 3
    ret["G2"] = (df["CHD3"] + df["CHD4"] + df["CHD5"] + df["CHD6"] + df["CHD7"]) / 5
    ret["G3"] = df["CHD11"]
    ret["G4"] = (df["CHD8"] + df["CHD10"]) / 2
    ret["G5"] = (df["CHD12"] + df["CHD13"] + df["CHD14"]) / 3
    ret = ret * train_cfg.get_times()
    ret = pandas.DataFrame(ret, dtype=int)  # 截断（不是四舍五入）为整数
    return ret


def merge_chd2(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    将表格按组，且区分重要性合并。
    注意：sklearn 算法要求不是 float，所以统一乘精度要求然后取整
    :param df:
    :return:
    """
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 2), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['G1', 'G2'], dtype=int)
    ret["G1"] = df["CHD8"]
    ret["G2"] = df["CHD10"]
    ret = ret * train_cfg.get_times()
    ret = pandas.DataFrame(ret, dtype=int)  # 截断（不是四舍五入）为整数
    return ret


def merge_chd3(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    将表格按组，且区分重要性合并。
    注意：sklearn 算法要求不是 float，所以统一乘精度要求然后取整
    :param df:
    :return:
    """
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 3), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['G1', 'G2', 'G3'], dtype=int)
    ret["G1"] = df["CHD12"]
    ret["G2"] = df["CHD13"]
    ret["G3"] = df["CHD14"]
    ret = ret * train_cfg.get_times()
    ret = pandas.DataFrame(ret, dtype=int)  # 截断（不是四舍五入）为整数
    return ret


def merge_chd4(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    将表格按组，且区分重要性合并。
    注意：sklearn 算法要求不是 float，所以统一乘精度要求然后取整
    :param df:
    :return:
    """
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 5), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['G1', 'G2', 'G3', 'G4', 'G5'], dtype=int)
    ret["G1"] = df["CHD3"]
    ret["G2"] = df["CHD4"]
    ret["G3"] = df["CHD5"]
    ret["G4"] = df["CHD6"]
    ret["G5"] = df["CHD7"]
    ret = ret * train_cfg.get_times()
    ret = pandas.DataFrame(ret, dtype=int)  # 截断（不是四舍五入）为整数
    return ret


def merge_chd5(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    将表格按组，且区分重要性合并。
    注意：sklearn 算法要求不是 float，所以统一乘精度要求然后取整
    :param df:
    :return:
    """
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 3), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['G1', 'G2', 'G3'], dtype=int)
    ret["G1"] = df["CHD1"]
    ret["G2"] = df["CHD2"]
    ret["G3"] = df["CHD9"]
    ret = ret * train_cfg.get_times()
    ret = pandas.DataFrame(ret, dtype=int)  # 截断（不是四舍五入）为整数
    return ret


def merge_hy(df: pandas.DataFrame) -> pandas.DataFrame:
    line_size = df.index.size
    np_zero = numpy.zeros(shape=(line_size, 4))
    ret = pandas.DataFrame(np_zero, columns=['1', '2', '3', '4'])
    ret["1"] = (df["HY1"] + df["HY2"] + df["HY3"] + df["HY7"] + df["HY10"]) / 5
    ret["2"] = (df["HY4"] + df["HY5"] + df["HY6"] + df["HY8"]) / 4
    ret["3"] = (df["HY9"] + df["HY12"]) / 2
    ret["4"] = (df["HY11"] + df["HY13"]) / 2
    return ret


def merge_none(df: pandas.DataFrame) -> pandas.DataFrame:
    return df


def simplify_table(filepath: str):
    min_err_percent = 0
    df_origin: pandas.DataFrame = excel_helper.read_resource(filepath)
    merge_fun = train_cfg.get_merge_func()
    df = merge_fun(df_origin)
    # 简化计算表格
    question_size = df.columns.size
    df_train = df
    while question_size > 1:
        if min_err_percent > 0.4:
            # 错误率过高提前跳出
            break
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

    # 反向连续预测
    # 1, 0, 2, 1
    """
    0, 1, 2, 3, 4
    0,    1, 2, 3
          0, 1, 2
          0, 1
          0
    实际删除序号 1, 0, 4, 3 （依次是胸痛，心衰症状，生活，心理）
    重要性排序 2, 3, 4, 0, 1
    """
    df_predict = df
    delete_idx = numpy.array([1, 0, 4, 3])
    train.predict(df_predict, df, delete_idx)


train_cfg.set_merge_func(merge_chd)
train_cfg.set_times(10000)
train_cfg.set_knn_k(10)
train_cfg.set_cross_verify_times(10)
train_cfg.set_table_range_max(5)
simplify_table("/冠心病.xlsx")

train_cfg.set_merge_func(merge_chd5)
train_cfg.set_times(1)
# simplify_table("/高血压.xlsx")


if __name__ == "__main__":
    """
    预测类型：
    1. 题目得分
    2. 阴阳性
    题目
    """
    print("main")
