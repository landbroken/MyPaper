#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/31
# @Author  : LinYulong
import numpy
import pandas
from xgboost import XGBClassifier

from src.alg import knn_helper, math_helper
from src.train import train_cfg
from src.train.train import column_split
from src.train.train_result import TrainResult


def train_predict(np_test_data_set: numpy.ndarray, np_train_data_set: numpy.ndarray,
                  np_train_labels: numpy.ndarray) -> numpy.ndarray:
    model = XGBClassifier()  # 载入模型（模型命名为model)
    model.fit(np_train_data_set, np_train_labels)  # 训练模型（训练集）
    y_predict = model.predict(np_test_data_set)  # 模型预测（测试集），y_pred为预测结果

    y_predict = y_predict / train_cfg.get_times()
    return y_predict


def train_no_group_all(test_df: pandas.DataFrame, train_df: pandas.DataFrame, last_result: TrainResult):
    columns_size = train_df.columns.size
    rmse_columns = []
    err_columns = []
    r_columns = []
    r2_columns = []
    mae_columns = []
    rmsd_columns = []
    m_res_columns = []
    sd_res_columns = []
    cfg_train_times = train_cfg.get_times()
    for columns_idx in range(columns_size):
        # 去掉被预测列
        test_data_set, test_labels = column_split(test_df, columns_idx)
        train_data_set, train_labels = column_split(train_df, columns_idx)

        test_data_set = test_data_set * cfg_train_times
        test_data_set = pandas.DataFrame(test_data_set, dtype=int)
        test_labels = test_labels * cfg_train_times
        test_labels = pandas.DataFrame(test_labels, dtype=int)
        train_data_set = train_data_set * cfg_train_times
        train_data_set = pandas.DataFrame(train_data_set, dtype=int)
        train_labels = train_labels * cfg_train_times
        train_labels = pandas.DataFrame(train_labels, dtype=int)

        np_test_data_set = numpy.array(test_data_set)
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels).ravel()

        # 预测
        y_predict = train_predict(np_test_data_set, np_train_data_set, np_train_labels)
        np_test_labels: numpy.ndarray = numpy.array(test_labels) / train_cfg.get_times()

        # 性能度量
        last_result.append_single_result(y_predict, np_test_labels)

    # err_arr = numpy.array(err_columns)
    # err_percent = caculate_err_percent(err_arr)


    return numpy.array(rmse_columns)
