#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong
import pandas

from src.excel import excel_helper
from src.alg import cross_verify, math_helper
from train import train


def simplify_table(filepath: str):
    min_err_percent = 0
    df: pandas.DataFrame = excel_helper.read_resource(filepath)
    # 简化计算表格
    question_size = df.columns.size
    while question_size > 1:
        if min_err_percent > 0.4:
            # 错误率过高提前跳出
            break
        # 当前轮简化
        rmse_arr, err_percent = cross_verify.cross_verify(10, df, train.train)
        min_idx = math_helper.min_id(rmse_arr)
        min_err_percent = err_percent[min_idx]
        print("cur rmse = " + str(rmse_arr[min_idx]) +
              ", cur err percent = " + str(min_err_percent) +
              ", idx = " + str(min_idx))
        # 下一轮数据
        next_df, min_df = train.column_split(df, min_idx)
        df = next_df
        question_size -= 1


simplify_table("/冠心病.xlsx")
# simplify_table("/高血压.xlsx")


if __name__ == "__main__":
    print("main")
