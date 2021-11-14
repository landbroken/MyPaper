#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/14
# @Author  : LinYulong
# @Descriptor: 混淆矩阵

class ConfusionMatrix:
    def cal_sensitivity(self, tp: int, fp: int) -> float:
        """
        灵敏度计算。
        灵敏度是指测试正确检测出患有这种疾病的患者的能力
        https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        :param tp: 真阳性人数
        :param fp: 假阳性人数
        :return:tpr
        """
        tpr = tp / (tp + fp * 1.0)  # tp rate
        return tpr

    def cal_specificity(self, tn: int, fp: int) -> float:
        """
        特异性计算
        :param tn: 真阴性人数
        :param fp: 假阳性人数
        :return: tnr: 特异性/选择性/真阴性率
        """
        tnr = tn / (tn + fp * 1.0)
        return tnr
