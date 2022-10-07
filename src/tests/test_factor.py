import math
import unittest

import numpy as np
import pandas
from pandas import DataFrame


class MyTestCase(unittest.TestCase):
    def test_something(self):
        """
        测试因子分析计算
        """
        # 原始值
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

        # 相关系数矩阵计算
        ret_corr: DataFrame = df_iq_scores.corr()
        ret_12 = ret_corr.iloc[1][0]
        self.assertAlmostEqual(ret_12, 0.8343, delta=0.0001)  # add assertion here

        # 特征根计算
        mat: np.ndarray = ret_corr.to_numpy()
        eigen_value, feature_vector = np.linalg.eig(mat)
        # 从大到小是 0.414696, 0.86211, 0.060208, 0.25686, 0.10675, 0.02523
        eigen_value_sorted = eigen_value
        eigen_value_sorted.sort()
        eigen_value_sorted = eigen_value_sorted[::-1]
        eigen_value_expected = [4.14696, 0.86211, 0.60208, 0.25686, 0.10675, 0.02523]
        for i in range(0, 5):
            self.assertAlmostEqual(eigen_value_sorted[i], eigen_value_expected[i], delta=0.00001)
        print(feature_vector)

        # 因子负荷系数
        factor_loads = []
        for j in range(0, 6):
            f12 = []
            for i in range(0, 2):
                cur_aij = math.sqrt(eigen_value[i]) * feature_vector[j][i]
                f12.append(cur_aij)
            factor_loads.append(f12)

        print(factor_loads)

        # 共同度
        hi_2 = []
        for item in factor_loads:
            cur_hi_2 = 0
            for fi in item:
                cur_hi_2 = cur_hi_2 + fi * fi
            hi_2.append(cur_hi_2)
        print(hi_2)


if __name__ == '__main__':
    unittest.main()
