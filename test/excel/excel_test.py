#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:22
# @Author  : LinYulong

import unittest

import numpy

from src.excel.excel_helper import *


class MyTestCase(unittest.TestCase):
    def test_write_resource(self):
        input_val = "test.xlsx"
        test_index = ['BeiJing', 'NanJing', 'XiAn', 'TianJing', 'ChongQing']
        test_columns = ['One', 'Two', 'Three', 'Four', 'Five', 'Six']
        df = pandas.DataFrame(numpy.arange(30).reshape(5, 6), index=test_index, columns=test_columns)
        write_resource(input_val, df)


if __name__ == '__main__':
    unittest.main()
