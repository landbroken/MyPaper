#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/14
# @Author  : LinYulong

import unittest

import numpy
import numpy as np

from src.alg import math_helper


class MyTestCase(unittest.TestCase):
    def test_min_id(self):
        arr = numpy.array([5.0, 1.0, 2.0, 3.0, 4.0])
        ret = math_helper.min_id(arr)
        self.assertAlmostEqual(ret, 1)

    def test_mean_absolute_error(self):
        y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
        y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])
        ret = math_helper.mean_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(ret, 1.9286, delta=0.0001)

    def test_mean_absolute_error_002(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 5.0, 6.0])
        ret = math_helper.mean_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(ret, 0.4, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
