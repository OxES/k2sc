from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from numpy import nan
from numpy.random import normal

import sys
sys.path.append('../../src')

from k2sc.kernels import *
from k2sc.gp import *

class TestSplitGP(unittest.TestCase):
    """Test the SplitGP class
    """
    def setUp(self):
        self.kernel = BasicKernel()
        self.gp = SplitGP(self.kernel)

    def test_blah(self):
        pass
        #npt.assert_almost_equal(m, 14, decimal=1)
        #npt.assert_almost_equal(s,  5, decimal=1)
