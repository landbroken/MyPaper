#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
# @Description 组内简化
from enum import EnumMeta
from typing import Union

import numpy
import pandas

from src.alg import cross_verify
from src.alg.medicine_type import DiseaseCheckType
from src.excel import excel_helper
from src.train import chd_helper
from src.train.confusion_matrix import ConfusionMatrix


def get_group_result(df: pandas.DataFrame):
    ret = df["kind"]
    return ret


def merge_to_one_columns(df: pandas.DataFrame):
    """
    将表格按组合并。
    :param df:
    :return:
    """
    # 消除原表头和序号
    np_df = numpy.array(df)
    df_new = pandas.DataFrame(np_df, columns=df.columns)
    # 合并
    line_size = df_new.index.size
    column_size = df_new.columns.size
    old_columns = df_new.columns
    np_zero = numpy.zeros(shape=(line_size, 1), dtype=int)
    ret = pandas.DataFrame(np_zero, columns=['Group_Score'])
    for i in range(column_size):
        old_column_name = old_columns[i]
        ret["Group_Score"] += df_new[old_column_name]
    return ret


def simplify_in_one_group(df: pandas.DataFrame):
    """
    单个题组内简化
    :param df:
    :return:
    """
    # 过滤掉不需要简化的题组。一个题组至少要有两题
    question_size = df.columns.size
    if question_size < 2:
        group_first_name = df.columns[0]
        print(group_first_name + "group only one question, do not need to simplify")
        return
    # 重要性排序
    df_importance = df  # TODO
    # 选择重要性排名最高的 n 个特征题目
    for n in range(2, question_size):
        print("cur question is top " + str(n))
        # 注：只用一个题目去预测第二个题目时，那么相当于一个 y = f(x) 函数，一般不应该有特别强的关联性，所以大部分时候，拟合的效果应该非常差
        # 筛选特征题组和得题组
        df_feature = df_importance.iloc[:, 0:n]
        merged_columns_size = df_importance.columns.size
        df_labels = get_group_result(df_importance)  # 预测的是题组得分
        # x 折交叉验证
        cross_verify_cnt = 10
        avg_ret: ConfusionMatrix = cross_verify.cross_verify_4(cross_verify_cnt, df_feature, df_labels)
        if (avg_ret.get_tpr() > 0.8) and (avg_ret.get_tnr() > 0.8):
            print("can delete question num = " + str(question_size - n))
        elif avg_ret.get_tpr() > 0.8:
            print("tnr = " + str(avg_ret.get_tnr()))
        elif avg_ret.get_tnr() > 0.8:
            print("tpr = " + str(avg_ret.get_tnr()))
        else:
            print("tpr = " + str(avg_ret.get_tpr()) + ", tnr = " + str(avg_ret.get_tnr()))


def select_tester(np_list, np_type: DiseaseCheckType) -> list[int]:
    """
    选择前置题组中为阳/阴性的测评者
    :param np_list:
    :param np_type:
    :return: 列表，被选中的测评者编号
    """
    ret = []
    np_list_size = len(np_list)
    for i in range(np_list_size):
        if np_list[i] == np_type.value:
            ret.append(i)
            print("select idx = " + str(i))
    return ret


def cal_group_np(df: pandas.DataFrame):
    """
    计算题组的阴阳性
    :param df: 一个题组
    :return: 每个测评者的阴阳性
    """
    ret = []
    kind_df = df["kind"]
    for item in kind_df:
        tmp_type = DiseaseCheckType.unknown.value
        if item == DiseaseCheckType.positive.value:
            tmp_type = DiseaseCheckType.positive.value
        elif item == DiseaseCheckType.negative.value:
            tmp_type = DiseaseCheckType.negative.value
        ret.append(tmp_type)
    return ret


def get_cur_sample_group(df: pandas.DataFrame, pre_idx: list[int]) -> pandas.DataFrame:
    pre_idx_size = len(pre_idx)
    column_size = df.columns.size
    np_zero = numpy.zeros(shape=(0, column_size), dtype=int)
    old_columns = df.columns
    ret = pandas.DataFrame(np_zero, columns=old_columns, dtype=int)
    for i in range(pre_idx_size):
        begin_line = pre_idx[i]
        end_line = pre_idx[i] + 1
        # 测试集
        tmp_def = df[begin_line:end_line]
        ret = ret.append(tmp_def)
    return ret


def simplify_in_group_with_df(df_origin: pandas.DataFrame):
    simplify_in_one_group(df_origin)


def simplify_in_group_main(filepath: str):
    df_origin: pandas.DataFrame = excel_helper.read_resource(filepath)
    simplify_in_group_with_df(df_origin)


if __name__ == "__main__":
    simplify_in_group_main("/CHD.xlsx")
