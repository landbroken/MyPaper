#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/20
# @Author  : LinYulong

g_merge_func = None  # 表格合并函数
g_times: int = 10000  # 表格取整计算放大倍数
g_cross_verify_times: int = 10  # n 折交叉验证
g_knn_k: int = 10  # knn 的 k
g_table_range_max: int = 5  # 表格归一化前范围最大值
g_knn_clf_list = []


def set_merge_func(func):
    global g_merge_func
    g_merge_func = func


def get_merge_func():
    global g_merge_func
    return g_merge_func


def set_times(param):
    global g_times
    g_times = param


def get_times():
    global g_times
    return g_times


def set_cross_verify_times(param):
    global g_cross_verify_times
    g_cross_verify_times = param


def get_cross_verify_times():
    global g_cross_verify_times
    return g_cross_verify_times


def set_knn_k(param):
    global g_knn_k
    g_knn_k = param


def get_knn_k():
    global g_knn_k
    return g_knn_k


def set_table_range_max(param):
    global g_table_range_max
    g_table_range_max = param


def get_table_range_max():
    global g_table_range_max
    return g_table_range_max


def set_knn_clf_list(param):
    global g_knn_clf_list
    g_knn_clf_list = param


def append_knn_clf_list(param):
    global g_knn_clf_list
    g_knn_clf_list.append(param)


def get_knn_clf_list():
    global g_knn_clf_list
    return g_knn_clf_list
