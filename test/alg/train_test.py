#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/12
# @Author  : LinYulong

import unittest

import numpy
import pandas as pd

from src.train import train


class MyTestCase(unittest.TestCase):
    def test_root_mean_square_error(self):
        predict = numpy.array([1.0, 2.0, 3.0, 4.0])
        real = numpy.array([2.0, 2.0, 2.0, 2.0])
        ret = train.root_mean_square_error(predict, real)
        expect = ((1.0 + 0.0 + 1.0 + 4.0) / 4.0) ** 0.5
        self.assertAlmostEqual(ret, expect, delta=0.001)

    def test_to_negative_and_positive_table(self):
        df = pd.DataFrame(
            [
                [4, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 4, 2, 2, 2, 2],
                [3, 4, 4, 4, 4, 2],
                [4, 5, 1, 2, 2, 2],

                [3, 2, 2, 2, 2, 3],
                [2, 2, 2, 3, 3, 3],
                [4, 3, 2, 3, 3, 4],
                [1, 4, 3, 3, 3, 2],
                [2, 2, 2, 4, 4, 4],
            ],
            columns=['CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6'])
        ret_df = train.to_negative_and_positive_table(df)
        self.assertEqual(ret_df.iloc[0, 0], 1)
        self.assertEqual(ret_df.iloc[1, 1], 1)
        self.assertEqual(ret_df.iloc[2, 2], 0)
        self.assertEqual(ret_df.iloc[3, 3], 1)
        self.assertEqual(ret_df.iloc[4, 4], 0)
        self.assertEqual(ret_df.iloc[5, 5], 1)


if __name__ == '__main__':
    unittest.main()
