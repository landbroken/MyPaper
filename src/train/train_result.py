#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/12/26
# @Author  : LinYulong
import numpy
from sklearn.metrics import accuracy_score  # 准确率

from src.alg import math_helper


class TrainResult:
    def __init__(self):
        self.rmse_columns_ = []
        self.err_columns_ = []
        self.r_columns_ = []
        self.r2_columns_ = []
        self.r2_avg_ = 0.0
        self.mae_columns_ = []
        self.rmsd_columns_ = []
        self.m_res_columns_ = []
        self.sd_res_columns_ = []
        self.accuracy_columns_ = []

    def append_single_result(self, y_predict: numpy.ndarray, y_test_labels: numpy.ndarray):
        # 求偏差，离散程度
        rmse = math_helper.root_mean_square_error(y_predict, y_test_labels)
        self.rmse_columns_.append(rmse)
        # 求绝对误差
        err_single = y_predict - y_test_labels
        self.err_columns_.append(err_single)
        #
        r = math_helper.my_pearson_correlation_coefficient(y_test_labels, y_predict)
        self.r_columns_.append(r)
        #
        r2 = math_helper.my_pearson_correlation_coefficient_square(y_test_labels, y_predict)
        self.r2_columns_.append(r2)
        #
        mae = math_helper.mean_absolute_error(y_test_labels, y_predict)
        self.mae_columns_.append(mae)
        #
        rmsd = math_helper.root_mean_square_error(y_test_labels, y_predict)
        self.rmsd_columns_.append(rmsd)
        #
        m_res = math_helper.my_mean_of_residuals(y_test_labels, y_predict)
        self.m_res_columns_.append(m_res)
        #
        sd_res = math_helper.my_standard_deviation_of_residuals(y_test_labels, y_predict)
        self.sd_res_columns_.append(sd_res)
        #
        accuracy = 0
        if isinstance(y_predict[0], int):
            accuracy = accuracy_score(y_test_labels, y_predict)  # 回归问题
        self.accuracy_columns_.append(accuracy)

    def print_average_result(self):
        r_avg = numpy.average(self.r_columns_)
        print("r avg = " + str(r_avg))
        r2_avg = self.get_avg_r2()
        print("r2 avg = " + str(r2_avg))
        mae_avg = numpy.average(self.mae_columns_)
        print("mae_avg = " + str(mae_avg))

        rmsd_avg = numpy.average(self.rmsd_columns_)
        print("rmsd_columns = " + str(rmsd_avg))

        m_res_avg = numpy.average(self.m_res_columns_)
        print("m_res_columns = " + str(m_res_avg))

        sd_res_avg = numpy.average(self.sd_res_columns_)
        print("sd_res_columns = " + str(sd_res_avg))

        accuracy_avg = numpy.average(self.accuracy_columns_)
        print("accuracy_avg = " + str(accuracy_avg))

    def get_avg_r2(self):
        r2_arr = numpy.array(self.r2_columns_)
        self.r2_avg_ = numpy.average(r2_arr)
        return self.r2_avg_
