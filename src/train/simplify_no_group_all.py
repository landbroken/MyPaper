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
from src.train import train_cfg, train_bad, train
from src.train.train_result import TrainResult


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
    cur_df = df_train
    train_result_list = []
    used_id = set()
    for idx in range(0, question_size - 1):
        print("--- begin idx = " + str(idx) + " ---")
        # 当前轮简化
        last_result: TrainResult = train_bad.train_no_group_all(cur_df)
        # 下一轮数据
        old_id = last_result.get_id()
        next_df, min_df = train.column_split(cur_df, old_id)
        cur_df = next_df
        print("--- end idx = {} ---".format(idx))
        # 提前跳出
        finish_param = last_result.get_avg_rmse()
        if finish_param > 0.3:
            # 错误率过高提前跳出
            print("break train and predict")
            break

        real_id = old_id
        while real_id in used_id:
            real_id = real_id + 1
        used_id.add(real_id)
        last_result.set_id(real_id)
        train_result_list.append(last_result)
        print("real delete column name = {}".format((real_id + 1)))
        # 用当前轮剩余数据，一直预测到全部题组，并打印过程偏差
        # if idx == 0:
        #     continue  # 跳过第一次
        train_bad.train_no_group_all_predict(cur_df, df_train, train_result_list)
        print("--- predict ---")

    print("=== end simplify in one group ===")


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
