#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/12/26
# @Author  : LinYulong
from typing import Optional, Union

import numpy
from sklearn.metrics import accuracy_score  # 准确率
from xgboost import XGBModel, XGBClassifier, XGBRegressor

from src.alg import math_helper


class TrainResult:
    def __init__(self):
        self.model_: Optional[Union[XGBClassifier, XGBRegressor]] = None  # 训练用模型
        self.id_: int = 0  # 结果所属数据标识
        self.name_: str = ""  # 结果所属数据标识，字符串
        self.rmse_columns_ = []
        self.rmse_avg_ = 0.0
        self.err_columns_ = []
        self.r_columns_ = []
        self.r_avg_ = 0.0
        self.r2_columns_ = []
        self.r2_avg_ = 0.0
        self.mae_columns_ = []
        self.rmsd_columns_ = []
        self.m_res_columns_ = []
        self.sd_res_columns_ = []
        self.accuracy_columns_ = []

    def set_model(self, model: Union[XGBClassifier, XGBRegressor]):
        self.model_ = model

    def get_model(self) -> Optional[Union[XGBClassifier, XGBRegressor]]:
        return self.model_

    def set_id(self, data_id: int):
        self.id_ = data_id

    def get_id(self) -> int:
        return self.id_

    def set_name(self, in_name: str):
        self.name_ = in_name

    def get_name(self):
        return self.name_

    def get_avg_rmse(self):
        rmse_arr = numpy.array(self.rmse_columns_)
        self.rmse_avg_ = numpy.average(rmse_arr)
        return self.rmse_avg_

    def get_avg_r(self):
        r_arr = numpy.array(self.r_columns_)
        self.r_avg_ = numpy.average(r_arr)
        return self.r_avg_

    def get_avg_r2(self):
        r2_arr = numpy.array(self.r2_columns_)
        self.r2_avg_ = numpy.average(r2_arr)
        return self.r2_avg_

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
        rmse_avg = self.get_avg_rmse()
        print("rmse avg = " + str(rmse_avg))

        r_avg = self.get_avg_r()
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
