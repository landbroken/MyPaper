#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
# @Description 组内简化
import numpy
import pandas

from src.alg import cross_verify, math_helper
from src.excel import excel_helper
from src.train import train_cfg, train


def simplify_in_one_group(df: pandas.DataFrame):
    """
    单个题组内简化
    :param df:
    :return:
    """
    # 过滤掉不需要简化的题组。一个题组至少要有两题
    question_size = df.columns.size
    if question_size < 2:
        group_name = ""  # TODO
        print(group_name + "group only one question, do not need to simplify")
        return
    # 重要性排序
    # 选择重要性排名最高的 n 个特征题目
    for n in range(question_size - 1):
        pass
        # x 折交叉验证
        cross_verify_cnt = 10
        for x in range(cross_verify_cnt):
            # train_data, test_data =
            # 算法拟合
            # 预测
            # 预测分值划分成相应的阴阳性
            # 计算混淆矩阵
            pass
        # 计算平均值作为特征评价指标


def simplify_in_group_main(filepath: str):
    df_origin: pandas.DataFrame = excel_helper.read_resource(filepath)
    # 预处理为题组链表
    # question_group_list =
    # 题组排序
    df_group = df_origin
    group_size = df_group.columns.size
    for n in range(group_size):
        pass
        # 获取前置题组
        # TODO pre_group = df_importance.iloc[,]
        # 获取当前简化题组
        # TODO cur_group =
        # 转换为阴阳性表格
        # np_table =
        for np_type in range(2):
            # 选择前置题组中为阳/阴性的测评者
            # pre_idx =
            # 选择这些测评者的当前题组的数据作为实验样本集
            # cur_sample_group =
            pass


simplify_in_group_main("/冠心病.xlsx")
