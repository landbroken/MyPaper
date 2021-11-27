#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/14
# @Author  : LinYulong
# @Descriptor: 混淆矩阵

class ConfusionMatrix:
    tpr_ = 0.0
    tnr_ = 0.0
    accuracy_ = 0.0
    precision_ = 0.0
    recall_ = 0.0

    def set_tpr(self, tpr: float):
        self.tpr_ = tpr

    def get_tpr(self):
        return self.tpr_

    def set_tnr(self, tnr: float):
        self.tnr_ = tnr

    def get_tnr(self):
        return self.tnr_

    def cal_sensitivity(self, tp: int, fn: int) -> float:
        """
        灵敏度计算。
        灵敏度是指测试正确检测出患有这种疾病的患者的能力
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        :param tp: 真阳性人数
        :param fn: 假阴性人数
        :return:tpr
        """
        self.tpr_ = tp / (tp + fn * 1.0)  # tp rate
        return self.tpr_

    def cal_specificity(self, tn: int, fp: int) -> float:
        """
        特异性计算
        :param tn: 真阴性人数
        :param fp: 假阳性人数
        :return: tnr: 特异性/选择性/真阴性率
        """
        self.tnr_ = tn / (tn + fp * 1.0)
        return self.tnr_

    def cal_accuracy(self, tn: int, tp: int, fn: int, fp: int) -> float:
        """
        准确率计算，在所有样本中，预测正确的概率
        :param tn:
        :param tp:
        :param fn:
        :param fp:
        :return:
        """
        self.accuracy_ = (tp + tn) * 1.0 / (tp + fn + fp + tn)
        return self.accuracy_

    def cal_precision(self, tp: int, fp: int) -> float:
        """
        精确率，你认为的正样本中，有多少是真的正确的概率
        :param tp:
        :param fp:
        :return:
        """
        self.precision_ = tp * 1.0 / (tp + fp)
        return self.precision_

    def cal_recall(self, tp: int, fn: int) -> float:
        """
        召回率 正样本中有多少是被找了出来
        :param tp:
        :param fn:
        :return:
        """
        self.recall_ = tp * 1.0 / (tp + fn)
        return self.recall_


class ConfusionMatrixHelper:
    cm_list_: list[ConfusionMatrix] = []

    def __init__(self, cm_list: list[ConfusionMatrix]):
        self.cm_list_ = cm_list

    def avg(self) -> ConfusionMatrix:
        cnt = len(self.cm_list_)
        sum_tpr = 0
        sum_tnr = 0
        for item in self.cm_list_:
            sum_tpr += item.get_tpr()
            sum_tnr += item.get_tnr()
        avg_tpr = sum_tpr * 1.0 / cnt
        avg_tnr = sum_tnr * 1.0 / cnt
        ret_cm: ConfusionMatrix = ConfusionMatrix()
        ret_cm.set_tnr(avg_tnr)
        ret_cm.set_tpr(avg_tpr)
        return ret_cm
