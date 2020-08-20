import unittest
import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

from utils import *
from morlet import *


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


    def test_cpFull(self):
        U = [np.array([[0.38631935, 0.38631606], [0.92236509, 0.92236647]]), \
             np.array([[0.42866664, 0.42866762], [0.56630685, 0.56630698], [0.70394705, 0.70394635]]), \
             np.array([[0.26726112, 0.26726136], [0.53452262, 0.53452235], [0.80178367, 0.80178378]])]

        lambs = np.array([[17.78789995], [17.78789995]])
        # test saveMemory option
        out1 = cpFull(U, lambs, True)
        assert_almost_equal(out1[:, :, 0], np.array([[1.5745, 2.0801, 2.5857], [3.7594, 4.9664, 6.1735]]), decimal=4)
        assert_almost_equal(out1[:, :, 1], np.array([[3.1491, 4.1602, 5.1714], [7.5187, 9.9329, 12.3471]]), decimal=4)
        assert_almost_equal(out1[:, :, 2], np.array([[4.7236, 6.2403, 7.7570], [11.2781, 14.8993, 18.5206]]), decimal=4)

        # test w/o saveMemory option
        out2 = cpFull(U, lambs, False)
        assert_almost_equal(out2[:, :, 0], np.array([[1.5745, 2.0801, 2.5857], [3.7594, 4.9664, 6.1735]]), decimal=4)
        assert_almost_equal(out2[:, :, 1], np.array([[3.1491, 4.1602, 5.1714], [7.5187, 9.9329, 12.3471]]), decimal=4)
        assert_almost_equal(out2[:, :, 2], np.array([[4.7236, 6.2403, 7.7570], [11.2781, 14.8993, 18.5206]]), decimal=4)


class TestTransform(unittest.TestCase):
    def test_morlet(self):
        opt = cMorletTransformS()
        testdata1 = np.array([[10, 12, 13], [24, 25, 26]])
        testtime1 = np.array([1, 2, 3])
        M1, timeDs1 = cMorletTransformS(testdata1, testtime1, opt)
        assert_equal(timeDs1, np.array([1, 2, 3]))
        assert_almost_equal(M1[:, :, 9], np.array([[.0100, .0115, .0078], [.0227, .0236, .0156]]), decimal=4)
        assert_almost_equal(M1[:, :, 99], np.array([[.0105, .0126, .0137], [.0252, .0263, .0273]]), decimal=4)


if __name__ == "__main__":
    unittest.main()
