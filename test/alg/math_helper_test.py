#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/14
# @Author  : LinYulong

import unittest

import numpy

from src.alg import math_helper


class MyTestCase(unittest.TestCase):
    def test_min_id(self):
        arr = numpy.array([5.0, 1.0, 2.0, 3.0, 4.0])
        ret = math_helper.min_id(arr)
        self.assertAlmostEqual(ret, 1)


if __name__ == '__main__':
    unittest.main()
