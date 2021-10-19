#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/12
# @Author  : LinYulong
import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame

df = pd.DataFrame([['Snow', 'M', 22], ['Tyrion', 'M', 32], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                  columns=['name', 'gender', 'age'])
print(df)

df2 = df.iloc[:, 0:2]  # 全部行，[0,2)列
print(df2)

df3 = df.iloc[:, 0]  # 全部行，[0,0]列
print(df3)

df4 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], columns=['CHD1', 'CHD2', 'CHD3'])
np5 = numpy.zeros(shape=(4, 2))
df5 = pd.DataFrame(np5, columns=['1', '2'])
df5["1"] = df4["CHD1"] + df4["CHD2"]
df5["2"] = df4["CHD3"]
print(df5)
