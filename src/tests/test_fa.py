# coding=utf-8
# 探索性因子分析
from __future__ import print_function, division, unicode_literals
from src.psy import Factor, data


def test_fa():
    # score = data['lsat.dat']
    score = [
            [14, 13, 28, 14, 22, 39],
            [10, 14, 15, 14, 34, 35],
            [11, 12, 19, 13, 24, 39],
            [7, 7, 7, 9, 20, 23],
            [13, 12, 24, 12, 26, 38],  # 5
            [19, 14, 22, 16, 23, 37],
            [20, 16, 26, 21, 38, 69],
            [9, 10, 14, 9, 31, 46],
            [9, 8, 15, 13, 14, 46],
            [9, 9, 12, 10, 23, 46],
        ]
    factor = Factor(score, 3)
    load = factor.loadings()
    print(load)


if __name__ == '__main__':
    test_fa()
