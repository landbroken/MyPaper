#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/14
# @Author  : LinYulong
import numpy


def min_id(arr: numpy.ndarray):
    min_idx = 0
    min_val = arr[0]
    for i in range(arr.size):
        if min_val > arr[i]:
            min_idx = i
            min_val = arr[i]

    return min_idx
