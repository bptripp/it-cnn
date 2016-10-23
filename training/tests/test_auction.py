__author__ = 'bptripp'

import time
import unittest
import numpy as np
from bertsekas.auction import auction

class MyTestCase(unittest.TestCase):
    def test_small_examples_vs_matlab(self):
        """
        Correct assignments are from Matlab implementation.
        """
        # Note A is a benefit matrix (higher is better)
        A = np.array([[5, 9, 2], [10, 3, 2], [8, 7, 4]])
        assignments, prices = auction(A)
        self.assertTrue(np.allclose(assignments, [1, 0, 2]))

        A = np.array([[1, 5, 9, 2], [10, 3, 2, 1], [8, 7, 4, 1], [1, 1, 1, 1]])
        assignments, prices = auction(A)
        self.assertTrue(np.allclose(assignments, [2, 0, 1, 3]))

    def test_performance(self):
        A = 10000.0 * np.random.randn(1000,1000)
        start_time = time.time()
        assignments = auction(A)
        end_time= time.time()
        self.assertLess(end_time - start_time, .75)

    def test_vs_munkres(self):
        pass



if __name__ == '__main__':
    unittest.main()
