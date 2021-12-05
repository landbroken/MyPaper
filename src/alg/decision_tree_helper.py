#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/27
# @Author  : LinYulong
# @Descriptor:
# 参考
# https://www.cnblogs.com/listenfwind/p/11310924.html
# https://zhuanlan.zhihu.com/p/266880465?ivk_sa=1024320u

import pandas
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split


def get_watermelon_data() -> pandas.DataFrame:
    titles_columns = \
        ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    df = pd.DataFrame([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],  # 1
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],  # 5

        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],  # 6
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],  # 10

        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],  # 15

        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否'],  # 17
    ],
        columns=titles_columns)
    return df
