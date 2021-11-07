import unittest

from src.train import chd_helper
from src.train import simplify_in_group


class MyTestCase(unittest.TestCase):
    def test_simplify_in_group_with_df(self):
        chd_ut_data = chd_helper.get_ut_data()
        simplify_in_group.simplify_in_group_with_df(chd_ut_data)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
