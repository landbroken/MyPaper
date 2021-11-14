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

from src.alg import cross_verify
from src.alg.medicine_type import DiseaseCheckType
from src.excel import excel_helper
from src.train import chd_helper


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
    for n in range(1, question_size):
        # 注：只用一个题目去预测第二个题目时，那么相当于一个 y = f(x) 函数，一般不应该有特别强的关联性，所以大部分时候，拟合的效果应该非常差
        # 变成特征题组和得题组
        df_feature = df_importance.iloc[:, 0:n]
        df_other = df_importance.iloc[:, n:(n + 1)]
        for predict_idx in range(n, question_size):
            # x 折交叉验证
            cross_verify_cnt = 10
            cross_verify.cross_verify_2(cross_verify_cnt, df_feature, df_other, None)
            pass
        # 计算平均值作为特征评价指标


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
    return ret


def cal_group_np(df: pandas.DataFrame):
    """
    计算题组的阴阳性
    :param df: 一个题组
    :return: 每个测评者的阴阳性
    """
    ret = []
    columns_size = df.columns.size
    for index, row in df.iterrows():
        row_sum = row.sum()
        tmp_type = DiseaseCheckType.positive.value
        if row_sum > (2 * columns_size):
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
    # 预处理为排序后的题组链表
    sorted_group_list = chd_helper.chd_sorted_group_get(df_origin)
    group_size = len(sorted_group_list)
    for n in range(1, group_size):
        # 获取前置题组
        pre_group = sorted_group_list[n - 1]
        # 获取当前简化题组
        cur_group = sorted_group_list[n]
        # 计算前置题组的阴阳性
        pre_np_list = cal_group_np(pre_group)
        for np_type in DiseaseCheckType:
            # 选择前置题组中为阳/阴性的测评者
            pre_idx = select_tester(pre_np_list, np_type)
            # 选择这些测评者的当前题组的数据作为实验样本集
            cur_sample_group = get_cur_sample_group(cur_group, pre_idx)
            simplify_in_one_group(cur_sample_group)


def simplify_in_group_main(filepath: str):
    df_origin: pandas.DataFrame = excel_helper.read_resource(filepath)
    simplify_in_group_with_df(df_origin)


if __name__ == "__main__":
    simplify_in_group_main("/冠心病.xlsx")
