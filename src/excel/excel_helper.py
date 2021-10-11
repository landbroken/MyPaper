#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import pandas
import os
import matplotlib.pyplot as plt


def read_resource(filepath: str):
    cur_file_path = os.path.dirname(__file__)  # 获取当前文件的绝对路径
    root_path = os.path.dirname(os.path.dirname(cur_file_path))
    resources_path = root_path + "/resources"
    full_path = resources_path + filepath
    df = pandas.read_excel(full_path, sheet_name="Sheet1")
    return df
