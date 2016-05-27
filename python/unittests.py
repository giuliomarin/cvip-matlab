import random
import unittest
import sys

import numpy as np

"""Simple test harness"""
class TestImages(unittest.TestCase):

    def test_white(self):
        self.assertTrue(1 == 1)

    def test_black(self):
        self.assertFalse(1 == 0)


if __name__ == '__main__':
    unittest.main()
