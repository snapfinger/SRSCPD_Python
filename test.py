import unittest
import numpy as np
from numpy.testing import assert_equal

from utils import *


class TestUtils(unittest.TestCase):
    def test_katriRao(self):
        U = [[[1, 2], [3, 4]], \
            [[11, 22], [33, 44], [55, 66]]]
        assert_equal(KrProd(U), \
                        np.array([[11, 44], [33, 88], [55, 132], \
                         [33, 88], [99, 176], [165, 264]]))


    def test_matricize(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = 2 * a
        c = 3 * a
        d = np.stack([a, b, c], axis=2)
        assert_equal(matricize(d, 1), np.array([[1, 4, 7, 2, 8, 14, 3, 12, 21], \
                                                [2, 5, 8, 4, 10, 16, 6, 15, 24], \
                                                [3, 6, 9, 6, 12, 18, 9, 18, 27]]))


    def test_nblockDiag(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(nBlockDiag(a, 2), np.array([[1, 2, 0, 0], [3, 4, 0, 0], \
                                                 [0, 0, 1, 2], [0, 0, 3, 4]]))


if __name__ == "__main__":
    unittest.main()
