#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
# @Description: 医学方面的算法

def cal_sensitivity(tp: int, fp: int) -> float:
    """
    灵敏度计算。
    灵敏度是指测试正确检测出患有这种疾病的患者的能力
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    :param tp: 真阳性人数
    :param fp: 假阳性人数
    :return:
    """
    tpr = tp / (tp + fp * 1.0)  # tp rate
    return tpr


def cal_specificity(tn: int, fp: int) -> float:
    """
    特异性计算
    :param tn: 真阴性人数
    :param fp: 假阳性人数
    :return: tnr: 特异性/选择性/真阴性率
    """
    tnr = tn / (tn + fp * 1.0)
    return tnr
