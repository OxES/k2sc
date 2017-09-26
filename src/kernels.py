""" Module defining Gaussian process kernels for k2sc. 

    Copyright (C) 2016  Suzanne Aigrain

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division, absolute_import 

import math as m
import numpy as np

from numpy import array, asarray, s_, log, pi, inf

from george.kernels import ExpSquaredKernel as ESK
from george.kernels import ExpSine2Kernel as ESn2K
from george.kernels import ExpKernel as EK

from .priors import UniformPrior as UP
from .priors import NormalPrior  as NP
from .priors import LogNormPrior as LP

class DtKernel(object):
    names  = []
    pv0    = array([])
    ndim   = 0
    npar   = 0
    priors = []

    def __init__(self, p0=None, **kwargs):
        self._pv = asarray(p0) if p0 is not None else self.pv0
        self._pm = self.map_pv(self.pv0)
        self._define_kernel()
        self._nk1 = len(self._k1)
        self._nk2 = len(self._k2)
        self._sk1 = s_[:self._nk1]
        self._sk2 = s_[self._nk1:-1]
        self.lims = np.transpose([p.lims for p in self.priors])

        assert len(self._k) == self.npar - 1, "Expected and true number of kernel parameters do not match"
        assert len(self.names) == self.npar, "Number of parameter names doesn't match the number of parameters"
        assert len(self.priors) == self.npar, "Number of parameter priors doesn't match the number of parameters" 
        assert self.pv0.size == self.npar, "Number of parameter vector elements doesn't match the number of parameters"

    def _define_kernel(self):
        raise NotImplementedError()
        self._k1 = None
        self._k2 = None
        self._k  = None


    def map_pv(self, pv):
        raise NotImplementedError()

    
    def set_pv(self, pv):
        self._pv = pv.copy()
        pp = self.map_pv(pv)
        self._k1.set_parameter_vector(log(pp[self._sk1]))
        self._k2.set_parameter_vector(log(pp[self._sk2]))
        self._k.set_parameter_vector(log(pp[:-1]))

    def ln_prior(self, pv):
        return sum([p.logpdf(v) for p,v in zip(self.priors,pv)])
        

class BasicKernel(DtKernel):
    name  = "BasicKernel"
    eq    = 'At*ESK(1/St) + Ap*ESK(1/Sx)*ESK(1/Sy)'
    names = 'time_amplitude time_iscale xy_amplitude x_iscale y_iscale white_noise '.split()
    pv0   = array([-6.0, 0.25, -5.4, 21, 21, -3])
    ndim  = 3
    npar  = 6
    priors = [UP(   -7,    1),              ## 0 -- log10 time amplitude
              LP( 0.25, 1.25, lims=[0,1]),  ## 1 -- inverse time scale
              UP(   -7,    0),              ## 2 -- xy log10 amplitude
              NP(   17,    8, lims=[0,70]), ## 3 -- inverse x scale
              NP(   17,    8, lims=[0,70]), ## 4 -- inverse y scale
              UP(   -6,    0)]              ## 5 -- log10 white noise
    bounds = [[-5,-3],[0.01,0.6],[-5,-3],[2,20],[2,20],[-4,-2]] 
    
    def _define_kernel(self):
        pv = self.map_pv(self._pv)
        self._k1 = pv[0] * ESK(1/pv[1], ndim=3, axes=0)
        self._k2 = pv[2] * ESK(1/pv[3], ndim=3, axes=1) * ESK(1/pv[4], ndim=3, axes=2)
        self._k   = self._k1 + self._k2

    def map_pv(self, pv):
        self._pm = pp = pv.copy()
        pp[0] = 10**pv[0]
        pp[1] =  1./pv[1]
        pp[2] = 10**pv[2]
        pp[3] =  1./pv[3]
        pp[4] =  1./pv[4]
        pp[5] = 10**pv[5]
        return self._pm
    

class BasicKernelEP(BasicKernel):
    name  = "BasicKernelEP"
    eq    = 'At*ESK(1/St) + Ap*EK(1/Sx)*EK(1/Sy)'
    priors = [UP(  -7,    1),              ## 0 -- log10 time amplitude
              LP(0.25, 1.25, lims=[0,1]),  ## 1 -- inverse time scale
              UP(  -7,    0),              ## 2 -- xy log10 amplitude
              NP(  10,   15, lims=[0,70]), ## 3 -- inverse x scale
              NP(  10,   15, lims=[0,70]), ## 4 -- inverse y scale
              UP(  -6,    0)]              ## 5 -- glo10 white noise

    def _define_kernel(self):
        pv = self.map_pv(self._pv)
        self._k1 = pv[0] * ESK(1/pv[1], ndim=3, axes=0)
        self._k2 = pv[2] * EK(1/pv[3], ndim=3, axes=1) * EK(1/pv[4], ndim=3, axes=2)
        self._k   = self._k1 + self._k2


class PeriodicKernel(DtKernel):
    name  = 'PeriodicKernel'
    names = 'time_amplitude inv_time_scale time_period xy_amplitude x_scale y_scale tl_amplitude tl_scale white_noise '.split()
    pv0   = array([1, 1, 5, 1, 0.25, 0.25, 0.01, 100, 0.01])
    ndim  = 3
    npar  = 9

    def __init__(self, p0=None, period=5, **kwargs):
        super(PeriodicKernel, self).__init__(p0, **kwargs)
        self.period = period
        self._pv[2] = period
        self.set_pv(self._pv)

    def _define_kernel(self):
        pv = self.map_pv(self._pv)
        self._k1 = pv[0] * ESn2K(gamma=pv[1], log_period=log(pv[2]), ndim=3, axes=0)
        self._k2 = ( pv[3] * ESK(pv[4], ndim=3, axes=1) * ESK(pv[5], ndim=3, axes=2)
                   + pv[6] * ESK(pv[7], ndim=3, axes=0))
        self._k   = self._k1 + self._k2


class QuasiPeriodicKernel(BasicKernel):
    name  = 'QuasiPeriodicKernel'
    names = 'time_amplitude time_iscale time_period time_evolution xy_amplitude x_iscale y_iscale white_noise '.split()
    pv0   = array([-6.0, 0.25, 10, 0.01, -5.4, 21, 21, -3])
    ndim  = 3
    npar  = 8
    priors = [UP(   -6,    1),               ## 0 -- time log10 amplitude
              LP( 0.25, 1.25, lims=[0,2]),   ## 1 -- inverse time scale
              UP( 0.04,   45),               ## 2 -- period
              LP( 0.25, 1.25, lims=[0,2]),   ## 3 -- time Evolution
              UP(   -6,    0),               ## 4 -- xy log10 amplitude
              NP(   17,    8, lims=[0,70]),  ## 5 -- inverse x scale
              NP(   17,    8, lims=[0,70]),  ## 6 -- inverse y scale
              UP(   -6,    0)]               ## 7 -- log10 white noise

    bounds = [[-5,-2],[0.1,1],[0,20],[0.01,1],[-5,-2],[2,50],[2,50],[-4,-2]]

    def __init__(self, p0=None, period=5, evolution_scale=100, **kwargs):
        super(QuasiPeriodicKernel, self).__init__(p0, **kwargs)
        self.period = period
        self._pv[2] = period
        self._pv[3] = evolution_scale
        self.set_pv(self._pv)

    def _define_kernel(self):
        pv = self.map_pv(self._pv)
        self._k1 = pv[0] * ESn2K(gamma=1/pv[1], log_period=log(pv[2]), ndim=3, axes=0) * ESK(1/pv[3], ndim=3, axes=0)
        self._k2 = pv[4] * ESK(1/pv[5], ndim=3, axes=1) * ESK(1/pv[6], ndim=3, axes=2)
        self._k  = self._k1 + self._k2

    def map_pv(self, pv):
        self._pm = pp = pv.copy()
        pp[0] = 10**pv[0]
        pp[1] =  1./pv[1]
        pp[3] =  1./pv[3]
        pp[4] = 10**pv[4]
        pp[5] =  1./pv[5]
        pp[6] =  1./pv[6]
        pp[7] = 10**pv[7]
        return self._pm


class QuasiPeriodicKernelEP(QuasiPeriodicKernel):
    name  = "QuasiPeriodicKernelEP"
    eq    = ''
    priors = [UP(   -6,    1),               ## 0 -- time log10 amplitude
              LP( 0.25, 1.25, lims=[0,2]),   ## 1 -- inverse time scale
              UP( 0.25,   45),               ## 2 -- period
              LP( 0.25, 1.25, lims=[0,2]),   ## 3 -- time Evolution
              UP(   -6,    0),               ## 4 -- xy log10 amplitude
              NP(   10,   15, lims=[0,70]),  ## 5 -- inverse x scale
              NP(   10,   15, lims=[0,70]),  ## 6 -- inverse y scale
              UP(   -6,   0)]                ## 7 -- log10 white noise

    def _define_kernel(self):
        pv = self.map_pv(self._pv)
        self._k1 = pv[0] * ESn2K(gamma=1/pv[1], log_period=log(pv[2]), ndim=3, axes=0) * ESK(1/pv[3], ndim=3, axes=0)
        self._k2 = pv[4] * EK(1/pv[5], ndim=3, axes=1) * EK(1/pv[6], ndim=3, axes=2)
        self._k  = self._k1 + self._k2
    
        
kernels = dict(basic         = BasicKernel,
               basic_ep      = BasicKernelEP,
               periodic      = PeriodicKernel,
               quasiperiodic = QuasiPeriodicKernel,
               quasiperiodic_ep = QuasiPeriodicKernelEP)
