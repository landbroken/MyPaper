#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/11/7
# @Author  : LinYulong
from enum import Enum


class DiseaseCheckType(Enum):
    unknown = 0  # 缺失值
    negative = 1  # 阴性
    positive = 2  # 阳性
