#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
import numpy
import pandas


def chd_group_type_get() -> numpy.ndarray:
    """
    获取未排序的 chd
    :return:
    """
    chd_gp = [
        [1, 2, 9],
        [3, 4, 5, 6, 7],
        [11],
        [8, 10],
        [12, 13, 14],
    ]
    return numpy.array(chd_gp)


def chd_sorted_group_type_get() -> numpy.ndarray:
    """
    获取组间排序后的 chd 顺序
    :return:
    """
    chd_gp = [
        [11],
        [8, 10],
        [12, 13, 14],
        [1, 2, 9],
        [3, 4, 5, 6, 7],
    ]
    return numpy.array(chd_gp)


def sub_df_get(df: pandas.DataFrame, idxs: list) -> pandas.DataFrame:
    line_size = df.index.size
    columns_size = len(idxs)
    np_zero = numpy.zeros(shape=(line_size, columns_size), dtype=int)
    new_columns = []
    for idx in idxs:
        cur_name = "CHD" + str(idx)
        new_columns.append(cur_name)
    ret = pandas.DataFrame(np_zero, new_columns, dtype=int)
    for idx in idxs:
        cur_name = "CHD" + str(idx)
        ret[cur_name] = df[cur_name]
    ret = pandas.DataFrame(ret, dtype=int)
    return ret


def chd_sorted_group_get(df: pandas.DataFrame) -> list[pandas.DataFrame]:
    group_type = chd_sorted_group_type_get()
    ret = []
    for i in range(5):
        df_i = sub_df_get(df, group_type[i])
        ret.append(df_i)
    return ret
