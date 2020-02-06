from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from numpy import nan
from numpy.random import normal
from k2sc.utils import *

# execfile('../../src/utils.py')

class TestMedSig(unittest.TestCase):
    """Test the median and sigma calculation
    """
    def setUp(self):
        pass

    def test_medsig(self):
        arr = normal(14, 5, 50000)
        m,s = medsig(arr)
        npt.assert_almost_equal(m, 14, decimal=1)
        npt.assert_almost_equal(s,  5, decimal=1)


class TestSigmaClip(unittest.TestCase):
    """Test the iterative sigma clipping
    """
    def setUp(self):
        self.npt = 100
        self.badi = [6, 23, 34, 55, 76]
        self.badv = [-2, 1.6, 31, nan, -5]
        self.arr  = normal(1, 0.1, self.npt)
        self.arr[self.badi] = [-2, 1.8, 31, nan, -5]


    def test_combined(self):
        tmask = np.ones(self.npt, np.bool)
        tmask[self.badi] = 0
        m = sigma_clip(self.arr)
        npt.assert_array_equal(m,tmask)


    def test_separated(self):
        tinf = np.ones(self.npt, np.bool)
        thig = np.ones(self.npt, np.bool)
        tlow = np.ones(self.npt, np.bool)

        tinf[55] = 0
        thig[[23,34]] = 0
        tlow[[6,76]] = 0

        minf, mlow, mhig = sigma_clip(self.arr, separate_masks=True)
        npt.assert_array_equal(minf,tinf)
        npt.assert_array_equal(mlow,tlow)
        npt.assert_array_equal(mhig,thig)


if __name__ == '__main__':
    unittest.main()
