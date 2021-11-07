import unittest

from src.train import chd_helper


class MyTestCase(unittest.TestCase):
    def test_chd_sorted_group_get(self):
        chd_ut_data = chd_helper.get_ut_data()
        groups = chd_helper.chd_sorted_group_get(chd_ut_data)
        self.assertEqual(len(groups), 5)


if __name__ == '__main__':
    unittest.main()
