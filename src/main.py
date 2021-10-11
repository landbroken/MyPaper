#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

from src.excel import excel_helper
from src.alg import knn_helper
from src.alg import cross_verify
from train import train

df_chd = excel_helper.read_resource("/冠心病.xlsx")
df_ht = excel_helper.read_resource("/高血压.xlsx")

cross_verify.cross_verify(10, df_chd, train.train)

if __name__ == "__main__":
    print("begin")
