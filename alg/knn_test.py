#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved
#
# @Time    : 2021/10/9 23:10
# @Author  : LinYulong

import unittest

from numpy import array

from knn_helper import KNN


class MyTestCase(unittest.TestCase):
    def test_something(self):
        labels = ['A', 'A', 'B', 'B']
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        alg = KNN()
        result = alg.classify([0, 0], group, labels, 3)
        self.assertEqual(result, 'B')


if __name__ == '__main__':
    unittest.main()
