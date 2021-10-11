#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import unittest

import numpy
from numpy import array

from src.alg import knn_helper


class MyTestCase(unittest.TestCase):
    def test_classify(self):
        labels = ['A', 'A', 'B', 'B']
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        result = knn_helper.classify([0, 0], group, labels, 3)
        self.assertEqual(result, 'B')

    def test_classify(self):
        wait_predict = numpy.array([[0, 0], [1, 1]])
        labels = numpy.array(['A', 'A', 'B', 'B'])
        group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        result = knn_helper.ski_classify(wait_predict, group, labels, 3)
        self.assertEqual(result[0], 'B')
        self.assertEqual(result[1], 'A')


if __name__ == '__main__':
    unittest.main()
