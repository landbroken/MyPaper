#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/31
# @Author  : LinYulong

import numpy
import pandas
import xgboost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor
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


def train_predict_xgb_classifier(x_test: numpy.ndarray, x_train: numpy.ndarray,
                                 y_train: numpy.ndarray):
    # 因为 XGBClassifier 告警要有 use_label_encoder=False，所以需要这个预处理
    encoded_y_train = get_encoded_train_labels(y_train)
    """
    https://blog.csdn.net/qq_38735017/article/details/111203258
    eval_metric
    回归 rmse, mae
    分类 auc, error, merror, logloss, mlogloss
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='merror')  # 载入模型（模型命名为model)
    model.fit(x_train, encoded_y_train)  # 训练模型（训练集）
    y_predict = model.predict(x_test)  # 模型预测（测试集），y_pred为预测结果

    cfg_train_times = train_cfg.get_times()
    offset = 1  # 偏移，因为预处理的 labels 一定是 0,...n-1。所以要加偏移才是实际分数
    y_predict = y_predict / cfg_train_times + offset
    return y_predict, model


def train_predict_xgb_regressor(x_test: numpy.ndarray, x_train: numpy.ndarray,
                                y_train: numpy.ndarray):
    # 因为 XGBClassifier 告警要有 use_label_encoder=False，所以需要这个预处理
    encoded_y_train = get_encoded_train_labels(y_train)
    """
    https://blog.csdn.net/qq_38735017/article/details/111203258
    eval_metric
    回归 rmse, mae
    分类 auc, error, merror, logloss, mlogloss
    """
    model_r = xgboost.XGBRegressor(max_depth=3,
                                   learning_rate=0.1,
                                   n_estimators=100,
                                   objective='reg:squarederror',  # 此默认参数与 XGBClassifier 不同
                                   booster='gbtree',
                                   gamma=0,
                                   min_child_weight=1,
                                   subsample=1,
                                   colsample_bytree=1,
                                   reg_alpha=0,
                                   reg_lambda=1,
                                   random_state=0)
    model_r.fit(x_train, encoded_y_train, eval_metric='rmse')  # 训练模型（训练集）
    # model_r.save_model('xgb100.model')  # 保存模型
    y_predict = model_r.predict(x_test)  # 模型预测（测试集），y_pred为预测结果

    cfg_train_times = train_cfg.get_times()
    offset = 1  # 偏移，因为预处理的 labels 一定是 0,...n-1。所以要加偏移才是实际分数
    y_predict = y_predict / cfg_train_times + offset
    return y_predict, model_r


def train_predict_linear_discriminant_analysis(x_test: numpy.ndarray, x_train: numpy.ndarray,
                                               y_train: numpy.ndarray):
    model_lda = LinearDiscriminantAnalysis(n_components=2)
    model_lda.fit(x_train, y_train)

    y_predict = model_lda.predict(x_test)
    return y_predict, model_lda


def train_predict_random_forest_regressor(x_test: numpy.ndarray, x_train: numpy.ndarray,
                                          y_train: numpy.ndarray):
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)  # 训练模型（训练集）
    y_predict = model.predict(x_test)  # 模型预测（测试集），y_pred为预测结果
    return y_predict, model


def get_best_result(result_list: list):
    best_i = 0
    ret_result: TrainResult = result_list[best_i]
    for i in range(0, len(result_list)):
        tmp_result: TrainResult = result_list[i]
        if tmp_result.get_avg_r2() > ret_result.get_avg_r2():
            ret_result = tmp_result
            best_i = i
    print("best result idx = " + str(ret_result.get_id()))
    return ret_result


def train_no_group_all(test_df: pandas.DataFrame) -> TrainResult:
    columns_size = test_df.columns.size
    cfg_train_times = train_cfg.get_times()
    result_list = []
    for columns_idx in range(columns_size):
        # 去掉被预测列
        test_data_set, test_labels = column_split(test_df, columns_idx)

        test_data_set = test_data_set * cfg_train_times
        test_data_set = pandas.DataFrame(test_data_set, dtype=int)
        test_labels_times = test_labels * cfg_train_times
        test_labels_times = pandas.DataFrame(test_labels_times, dtype=int)

        cross_verify_times = train_cfg.get_cross_verify_times()
        result: TrainResult = cross_verify.cross_verify_no_group_all(cross_verify_times, test_data_set,
                                                                     test_labels_times,
                                                                     train_predict_xgb_regressor)
        result.set_id(columns_idx)
        result_list.append(result)

    print("------------------------------")
    ret_result = get_best_result(result_list)
    ret_result.print_average_result()
    print("------------------------------")
    return ret_result


def train_no_group_all_predict(begin_df: pandas.DataFrame, origin_df: pandas.DataFrame, train_result_list: list):
    train_result_list_len = len(train_result_list)
    x_test = begin_df.copy(deep=True)
    for idx in range(0, train_result_list_len):
        cur_train_result_idx = train_result_list_len - idx - 1
        cur_old_train_result: TrainResult = train_result_list[cur_train_result_idx]
        model = cur_old_train_result.get_model()
        old_id = cur_old_train_result.get_id()
        if model is None:
            print("model is None")
            raise Exception("model is None" + str(old_id))

        test_data_set, y_test = column_split(origin_df, old_id)

        y_predict = model.predict(x_test)  # 模型预测（测试集），y_pred为预测结果
        cfg_train_times = train_cfg.get_times()
        offset = 1  # 偏移，因为预处理的 labels 一定是 0,...n-1。所以要加偏移才是实际分数
        y_predict = y_predict / cfg_train_times + offset

        cur_new_train_result = TrainResult()
        cur_new_train_result.append_single_result(y_predict, y_test)
        cur_new_train_result.print_average_result()

        # 下一轮数据
        column_name = cur_old_train_result.get_name()
        try:
            x_test.insert(old_id, column_name, y_predict)
            print("--- insert {} ---".format(column_name))
        except ValueError:
            print("--- insert ValueError! {} ---".format(column_name))
            raise ValueError
        except TypeError:
            print("--- insert TypeError! {} ---".format(column_name))
            raise TypeError
        except BaseException:
            print("--- insert BaseException! {} ---".format(column_name))
            raise BaseException

    print("--- end predict ---")
