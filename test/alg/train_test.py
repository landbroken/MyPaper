#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/12
# @Author  : LinYulong

import unittest

import numpy

from src.train import train
from src.train.train import TrainCfg


class MyTestCase(unittest.TestCase):
    def test_root_mean_square_error(self):
        predict = numpy.array([1.0, 2.0, 3.0, 4.0])
        real = numpy.array([2.0, 2.0, 2.0, 2.0])
        ret = train.root_mean_square_error(predict, real)
        expect = ((1.0 + 0.0 + 1.0 + 4.0) / 4.0) ** 0.5
        self.assertAlmostEqual(ret, expect, delta=0.001)


if __name__ == '__main__':
    unittest.main()
