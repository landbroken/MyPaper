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

from src.alg import cross_verify
from src.train import train_cfg
from src.train.train import column_split
from src.train.train_result import TrainResult
from sklearn import preprocessing


def get_encoded_train_labels(np_train_labels: numpy.ndarray):
    le = preprocessing.LabelEncoder()
    le.fit(np_train_labels)
    # 可以查看一下 fit 以后的类别是什么
    # le_type = le.classes_
    # transform 以后，这一列数就变成了 [0,  n-1] 这个区间的数，即是  le.classes_ 中的索引
    encoded_train_labels = le.transform(np_train_labels)
    return encoded_train_labels


def train_predict(x_test: numpy.ndarray, x_train: numpy.ndarray,
                  y_train: numpy.ndarray) -> numpy.ndarray:
    # 因为 XGBClassifier 告警要有 use_label_encoder=False，所以需要这个预处理
    encoded_y_train = get_encoded_train_labels(y_train)
    model = XGBClassifier(use_label_encoder=False, eval_metric='rmse')  # 载入模型（模型命名为model)
    model.fit(x_train, encoded_y_train)  # 训练模型（训练集）
    y_predict = model.predict(x_test)  # 模型预测（测试集），y_pred为预测结果

    cfg_train_times = train_cfg.get_times()
    offset = 1  # 偏移，因为预处理的 labels 一定是 0,...n-1。所以要加偏移才是实际分数
    y_predict = y_predict / cfg_train_times + offset
    return y_predict


def train_no_group_all(test_df: pandas.DataFrame) -> TrainResult:
    columns_size = test_df.columns.size
    cfg_train_times = train_cfg.get_times()
    result_list = []
    for columns_idx in range(columns_size):
        # 去掉被预测列
        print("------------------------------")
        print("predict columns idx = " + str(columns_idx))
        test_data_set, test_labels = column_split(test_df, columns_idx)

        test_data_set = test_data_set * cfg_train_times
        test_data_set = pandas.DataFrame(test_data_set, dtype=int)
        test_labels_times = test_labels * cfg_train_times
        test_labels_times = pandas.DataFrame(test_labels_times, dtype=int)

        cross_verify_times = train_cfg.get_cross_verify_times()
        result: TrainResult = cross_verify.cross_verify_no_group_all(cross_verify_times, test_data_set,
                                                                     test_labels_times)
        result_list.append(result)
        print("----------end idx = " + str(columns_idx) + "--------------------")
    return result_list[0]
