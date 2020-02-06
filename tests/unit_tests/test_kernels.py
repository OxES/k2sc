from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from numpy import nan
from numpy.random import normal

import sys
sys.path.append('../../src')
from k2sc.kernels import *

class TestSplitGP(unittest.TestCase):
    """Test the SplitGP class
    """
    def setUp(self):
        pass

    def test_init_basic(self):
        BasicKernel()

    def test_init_basic_ep(self):
        BasicKernelEP()

    def test_init_quasiperiodic(self):
        QuasiPeriodicKernel()

    def test_init_quasiperiodic_ep(self):
        QuasiPeriodicKernelEP()

if __name__ == '__main__':
    unittest.main()
