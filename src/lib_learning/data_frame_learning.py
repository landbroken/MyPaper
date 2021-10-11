#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/12
# @Author  : LinYulong

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

