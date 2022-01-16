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

from src.alg import decision_tree_helper
from src.alg.medicine_type import DiseaseCheckType
from src.excel import excel_helper
from src.train import train_cfg, train_bad


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


def chd_get_df_feature(df: pandas.DataFrame) -> pandas.DataFrame:
    feature = df[
        [
            'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9', 'CHD10',
            'CHD11', 'CHD12', 'CHD13', 'CHD14'
        ]
    ]
    return feature


def hy_get_df_feature(df: pandas.DataFrame) -> pandas.DataFrame:
    return df


def get_df_importance(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    获取重要性排序后的
    :param df:
    :return:
    """
    feature = df[
        [
            'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9', 'CHD10',
            'CHD11', 'CHD12', 'CHD13', 'CHD14'
        ]
    ]
    result = df[['kind']]
    clf = decision_tree_helper.get_clf(feature, result, "../../output/simplify_no_group.dot")

    idxs = [14, 3, 6, 11, 12, 13, 2, 1, 4, 8, 5, 10, 7, 9]
    line_size = df.index.size
    columns_size = len(idxs)
    np_zero = numpy.zeros(shape=(line_size, columns_size + 1), dtype=int)
    new_columns = []
    for idx in idxs:
        cur_name = "CHD" + str(idx)
        new_columns.append(cur_name)
    new_columns.append("kind")
    ret = pandas.DataFrame(np_zero, columns=new_columns, dtype=int)
    for idx in idxs:
        cur_name = "CHD" + str(idx)
        ret[cur_name] = df[cur_name]
    ret["kind"] = df["kind"]
    ret = pandas.DataFrame(ret, dtype=int)
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
    df_train = chd_get_df_feature(df)
    # 简化计算表格
    question_size = df_train.columns.size
    min_err_percent = 0  # 初始化预测错误率
    for idx in range(0, question_size - 1):
        print("begin idx = " + str(idx))
        if min_err_percent > 0.5:
            # 错误率过高提前跳出
            break

        # 当前轮简化
        train_bad.train_no_group_all(df_train)
        print("end idx = " + str(idx))
        # min_idx = math_helper.min_id(rmse_arr)
        # min_err_percent = err_percent[min_idx]
        # print("cur rmse = " + str(rmse_arr[min_idx]) +
        #       ", cur err percent = " + str(min_err_percent) +
        #       ", idx = " + str(min_idx))
        # # 下一轮数据
        # next_df, min_df = train.column_split(df_train, min_idx)
        # df_train = next_df
        # question_size -= 1

    # 选择重要性排名最高的 n 个特征题目
    # for n in range(2, question_size):
    #     print("cur question is top " + str(n))
    #     # 注：只用一个题目去预测第二个题目时，那么相当于一个 y = f(x) 函数，一般不应该有特别强的关联性，所以大部分时候，拟合的效果应该非常差
    #     # 筛选特征题组和得题组
    #     df_feature = df_importance.iloc[:, 0:n]
    #     df_labels = get_group_result(df_importance)  # 预测的是题组得分
    #     # x 折交叉验证
    #     cross_verify_cnt = 10
    #     avg_ret: ConfusionMatrix = cross_verify.cross_verify_4(cross_verify_cnt, df_feature, df_labels)
    #     print("|accuracy    = {}|f1 score = {}".format(avg_ret.get_precision(), avg_ret.get_f_measure()))
    #     print("|tpr = {}".format(avg_ret.get_tpr()))
    #     print("|tnr = {}".format(avg_ret.get_tnr()))


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


def chd_sub_df_get(df: pandas.DataFrame) -> pandas.DataFrame:
    line_size = df.index.size
    columns_size = 14
    np_zero = numpy.zeros(shape=(line_size, columns_size + 1), dtype=int)
    new_columns = []
    for idx in range(1, columns_size + 1):
        cur_name = "CHD" + str(idx)
        new_columns.append(cur_name)
    new_columns.append("kind")
    ret = pandas.DataFrame(np_zero, columns=new_columns, dtype=int)
    for idx in range(1, columns_size + 1):
        cur_name = "CHD" + str(idx)
        ret[cur_name] = df[cur_name]
    ret["kind"] = df["kind"]
    ret = pandas.DataFrame(ret, dtype=int)
    return ret


def hy_sub_df_get(df: pandas.DataFrame) -> pandas.DataFrame:
    return df


def simplify_chd_in_group_main():
    df_origin: pandas.DataFrame = excel_helper.read_resource("/CHD.xlsx")
    df_train = chd_sub_df_get(df_origin)
    simplify_in_one_group(df_train)


def simplify_hy_in_group_main():
    df_origin: pandas.DataFrame = excel_helper.read_resource("/高血压.xlsx")
    df_train = hy_sub_df_get(df_origin)
    simplify_in_one_group(df_train)


if __name__ == "__main__":
    """
    简化：
    1.无题组
    2.预测题目得分
    3.最终同时预测阴阳性
    """
    train_cfg.set_times(1)
    train_cfg.set_cross_verify_times(10)  # 10 折交叉验证
    simplify_chd_in_group_main()
