import unittest

import pandas
from pandas import DataFrame


class MyTestCase(unittest.TestCase):
    def test_something(self):
        iq_scores = [
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
        df_iq_scores: DataFrame = pandas.DataFrame(iq_scores, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
        ret_corr: DataFrame = df_iq_scores.corr()
        ret_12 = ret_corr.iloc[1][0]
        self.assertAlmostEqual(ret_12, 0.8343, delta=0.0001)  # add assertion here


if __name__ == '__main__':
    unittest.main()
