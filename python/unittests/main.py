import unittest
import sys
sys.path.append('../')
from cvip import *
import numpy as np

"""Simple test"""
class TestDataIO(unittest.TestCase):

    def test_imread(self):
        img, type = dataio.imread('../../samples/images/depth.png')
        self.assertTrue(type == 1)
        img, type = dataio.imread('../../samples/images/left.png')
        self.assertTrue(type == 0)
        self.assertRaises(IOError, dataio.imread32f, '../../samples/images/left.png')

if __name__ == '__main__':
    unittest.main()
