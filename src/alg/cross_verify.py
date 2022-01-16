#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import math
from enum import EnumMeta
from typing import Union

import numpy
import pandas

from src.alg import knn_helper
from src.alg.medicine_type import DiseaseCheckType
from src.train import train_cfg, train
from src.train.confusion_matrix import ConfusionMatrix, ConfusionMatrixHelper
from src.train.train_bad import train_predict
from src.train.train_result import TrainResult


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


def cross_verify_2(verify_cnt: int, df_feature: pandas.DataFrame, df_result: pandas.DataFrame, merged_columns_size: int,
                   params, fun):
    """
    n 折交叉验证
    :param merged_columns_size:
    :param verify_cnt: 交叉验证的折数
    :param df_result: 训练用结果
    :param df_feature: 预测用特征
    :param params: fun 会用到的参数集
    :param fun: 交叉验证时使用的算法执行函数
    :return:
    """
    test_size = math.ceil(df_feature.index.size / verify_cnt)  # 测试集大小
    real_verify_cnt = 0
    cm_list = []
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
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels).ravel()

        if numpy.size(np_test_data_set) <= 0:
            # n 折计算时， (ceil（total_cnt / n）) * n 可能超过 total_cnt
            print("zero")
            continue

        real_verify_cnt += 1

        # 使用 XX 算法拟合，得到模型
        knn_k = train_cfg.get_knn_k()
        clf = knn_helper.ski_fit(np_train_data_set, np_train_labels, knn_k)
        # 计算得到预测分值。注：样本过少，这里预测结果也是个位数
        np_predict_score: numpy.ndarray = clf.predict(np_test_data_set)

        # 预测分值划分成相应的阴阳性
        df_predict_score = pandas.DataFrame(np_predict_score)
        df_real_np = train.to_negative_and_positive_table(test_labels, merged_columns_size)
        df_predict_np = train.to_negative_and_positive_table(df_predict_score, merged_columns_size)

        # 计算混淆矩阵
        true_negative = 0  # 真阴性
        false_negative = 0  # 假阴性
        true_positive = 0  # 真阳性
        false_positive = 0  # 假阳性
        np_real_np = numpy.array(df_real_np)  # 真实阴阳性
        np_predict_np = numpy.array(df_predict_np)  # 预测阴阳性
        for predict_idx in range(np_predict_np.size):
            cur_real_np = np_real_np[predict_idx]
            cur_predict_np = np_predict_np[predict_idx]
            if cur_predict_np == DiseaseCheckType.positive.value:
                # 预测阳性
                if cur_real_np == cur_predict_np:
                    # 实际阳性
                    true_positive += 1
                else:
                    # 实际阴性
                    false_positive += 1
            else:
                # 预测阴性
                if cur_real_np == cur_predict_np:
                    # 实际阴性
                    true_negative += 1
                else:
                    # 实际阳性
                    false_negative += 1

        cm = ConfusionMatrix()
        if (true_positive + false_negative) != 0:
            cm.cal_sensitivity(true_positive, false_negative)
        else:
            # 没有患者时，阳性准确率为 0
            print("no positive tester")
            cm.set_tpr(0.0)
        if (true_negative + false_positive) != 0:
            cm.cal_specificity(true_negative, false_positive)
        else:
            # 没有患者时，阴性准确率为 0
            print("no negative tester")
            cm.set_tnr(0.0)
        cm_list.append(cm)
    cm_helper = ConfusionMatrixHelper(cm_list)
    avg_cm = cm_helper.avg()
    return avg_cm


def cross_verify_3(verify_cnt: int, df_feature: pandas.DataFrame, df_result: pandas.DataFrame, merged_columns_size: int,
                   params, fun):
    """
    n 折交叉验证
    :param merged_columns_size:
    :param verify_cnt: 交叉验证的折数
    :param df_result: 训练用结果
    :param df_feature: 预测用特征
    :param params: fun 会用到的参数集
    :param fun: 交叉验证时使用的算法执行函数
    :return:
    """
    test_size = math.ceil(df_feature.index.size / verify_cnt)  # 测试集大小
    cm_list = []
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
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels).ravel()

        if numpy.size(np_test_data_set) <= 0:
            # n 折计算时， (ceil（total_cnt / n）) * n 可能超过 total_cnt
            print("zero")
            continue

        # 使用 XX 算法拟合，得到模型
        knn_k = train_cfg.get_knn_k()
        clf = knn_helper.ski_fit(np_train_data_set, np_train_labels, knn_k)
        # 计算得到预测分值。注：样本过少，这里预测结果也是个位数
        np_predict_score: numpy.ndarray = clf.predict(np_test_data_set)

        # 预测分值划分成相应的阴阳性
        df_real_np = pandas.DataFrame(test_labels)
        df_predict_np = pandas.DataFrame(np_predict_score)

        # 计算混淆矩阵
        true_negative = 0  # 真阴性
        false_negative = 0  # 假阴性
        true_positive = 0  # 真阳性
        false_positive = 0  # 假阳性
        np_real_np = numpy.array(df_real_np)  # 真实阴阳性
        np_predict_np = numpy.array(df_predict_np)  # 预测阴阳性
        for predict_idx in range(np_predict_np.size):
            cur_real_np = np_real_np[predict_idx]
            cur_predict_np = np_predict_np[predict_idx]
            if cur_predict_np == DiseaseCheckType.positive.value:
                # 预测阳性
                if cur_real_np == cur_predict_np:
                    # 实际阳性
                    true_positive += 1
                else:
                    # 实际阴性
                    false_positive += 1
            else:
                # 预测阴性
                if cur_real_np == cur_predict_np:
                    # 实际阴性
                    true_negative += 1
                else:
                    # 实际阳性
                    false_negative += 1

        cm = ConfusionMatrix()
        if (true_positive + false_negative) != 0:
            cm.cal_sensitivity(true_positive, false_negative)
        else:
            # 没有患者时，阳性准确率为 0
            print("no positive tester")
            cm.set_tpr(0.0)
        if (true_negative + false_positive) != 0:
            cm.cal_specificity(true_negative, false_positive)
        else:
            # 没有患者时，阴性准确率为 0
            print("no negative tester")
            cm.set_tnr(0.0)
        cm_list.append(cm)
    cm_helper = ConfusionMatrixHelper(cm_list)
    avg_cm = cm_helper.avg()
    return avg_cm


def cross_verify_4(verify_cnt: int, df_feature: pandas.DataFrame, df_result: pandas.DataFrame):
    """
    n 折交叉验证
    :param verify_cnt: 交叉验证的折数
    :param df_result: 训练用结果
    :param df_feature: 预测用特征
    :return:
    """
    test_size = math.ceil(df_feature.index.size / verify_cnt)  # 测试集大小
    cm_list = []
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
        np_train_data_set = numpy.array(train_data_set)
        np_train_labels = numpy.array(train_labels).ravel()

        if numpy.size(np_test_data_set) <= 0:
            # n 折计算时， (ceil（total_cnt / n）) * n 可能超过 total_cnt
            print("zero")
            continue

        # 使用 XX 算法拟合，得到模型
        knn_k = train_cfg.get_knn_k()
        clf = knn_helper.ski_fit(np_train_data_set, np_train_labels, knn_k)
        # 计算得到预测分值。注：样本过少，这里预测结果也是个位数
        np_predict_score: numpy.ndarray = clf.predict(np_test_data_set)

        # 预测分值划分成相应的阴阳性
        df_real_np = pandas.DataFrame(test_labels)
        df_predict_np = pandas.DataFrame(np_predict_score)

        # 计算混淆矩阵
        true_negative = 0  # 真阴性
        false_negative = 0  # 假阴性
        true_positive = 0  # 真阳性
        false_positive = 0  # 假阳性
        np_real_np = numpy.array(df_real_np)  # 真实阴阳性
        np_predict_np = numpy.array(df_predict_np)  # 预测阴阳性
        for predict_idx in range(np_predict_np.size):
            cur_real_np = np_real_np[predict_idx]
            cur_predict_np = np_predict_np[predict_idx]
            if cur_predict_np == DiseaseCheckType.positive.value:
                # 预测阳性
                if cur_real_np == cur_predict_np:
                    # 实际阳性
                    true_positive += 1
                else:
                    # 实际阴性
                    false_positive += 1
            else:
                # 预测阴性
                if cur_real_np == cur_predict_np:
                    # 实际阴性
                    true_negative += 1
                else:
                    # 实际阳性
                    false_negative += 1

        cm = ConfusionMatrix()
        cm.cal_sensitivity(true_positive, false_negative)
        cm.cal_specificity(true_negative, false_positive)
        cm.cal_accuracy(true_negative, true_positive, false_negative, false_positive)
        cm.cal_recall(true_positive, false_negative)
        cm.cal_precision(true_positive, false_positive)
        cm.cal_f_measure(1)

        cm_list.append(cm)
    cm_helper = ConfusionMatrixHelper(cm_list)
    avg_cm = cm_helper.avg()
    return avg_cm


def cross_verify_no_group_all(verify_cnt: int, train_data: pandas.DataFrame, test_label: pandas.DataFrame):
    """
    n 折交叉验证
    :param verify_cnt: 交叉验证的折数
    :param train_data: 训练数据
    :param test_label: 训练数据标签
    :return:
    """
    test_size = math.ceil(test_label.index.size / verify_cnt)  # 测试集大小
    train_result = TrainResult()
    for i in range(verify_cnt):
        begin_line = 0 + test_size * i
        end_line = begin_line + test_size
        cfg_train_times = train_cfg.get_times()
        # 训练集
        x_train_part1 = train_data[:begin_line]
        x_train_part2 = train_data[end_line:]
        x_train_df = x_train_part1.append(x_train_part2) * cfg_train_times
        x_train = numpy.array(x_train_df)

        x_test_df = train_data[begin_line:end_line] * cfg_train_times
        x_test = numpy.array(x_test_df)

        # 测试集
        y_train_part1 = test_label[:begin_line]
        y_train_part2 = test_label[end_line:]
        y_train_df = y_train_part1.append(y_train_part2) * cfg_train_times
        y_train = numpy.array(y_train_df).ravel()

        y_test_df = test_label[begin_line:end_line] * cfg_train_times
        y_test = numpy.array(y_test_df).ravel()

        # 预测
        y_predict = train_predict(x_test, x_train, y_train)

        # 性能度量
        train_result.append_single_result(y_predict, y_test)

    train_result.print_average_result()
    return train_result
