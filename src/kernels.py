from __future__ import division

import math as m
import numpy as np

from numpy import array, asarray, s_, log, pi, inf

from george.kernels import ExpSquaredKernel as ESK
from george.kernels import ExpSine2Kernel as ESn2K
from george.kernels import ExpKernel as EK

class Prior(object):
    def __init__(self):
        raise NotImplementedError()
    def logpdf(self, x):
        raise NotImplementedError()


class UniformPrior(Prior):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.C = 1./(b-a)
        self.lnC = m.log(self.C)
        self.lims = [a,b]

    def logpdf(self, x):
        if x > self.a and x < self.b:
            return self.lnC
        else:
            return -inf

class NormalPrior(Prior):
    def __init__(self, mean, std, lims=None):
        self.lims = np.array(lims)
        self.a, self.b = lims
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ m.sqrt(2.*m.pi*std*std)
        self._lf1 = m.log(self._f1)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._f1 * np.exp(-(x-self.mean)**2 * self._f2), 1e-80)
        else:
            return self._f1 * m.exp(-(x-self.mean)**2 * self._f2) if self.lims[0] < x < self.lims[1] else 1e-80

    def logpdf(self, x):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._lf1 - (x-self.mean)**2 * self._f2, -np.inf)
        else:
            return self._lf1 -(x-self.mean)**2*self._f2 if self.a < x < self.b else -np.inf


class LogNormPrior(Prior):
    def __init__(self, mu, sigma, lims=None):
        self.mu = mu
        self.sigma = sigma
        self.C = -m.log(sigma*m.sqrt(2*pi))
        self.lims = lims if lims is not None else [0,inf]

    def logpdf(self, x):
        if (x <= self.lims[0]) or (x > self.lims[1]):
            return -inf
        mu, sigma  = self.mu, self.sigma
        lnx = m.log(x)
        return -lnx + self.C - ((lnx*lnx - mu*lnx + mu*mu)/(2.*sigma*sigma))


class DtKernel(object):
    names  = []
    pv0    = array([])
    ndim   = 0
    npar   = 0
    priors = []

    def __init__(self, p0=None, **kwargs):
        self._pv = asarray(p0) if p0 is not None else self.pv0
        self._pm = None
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
        self._k1[:] = log(pp[self._sk1])
        self._k2[:] = log(pp[self._sk2])
        self._k[:]  = log(pp[:-1])

    def ln_prior(self, pv):
        return sum([p.logpdf(v) for p,v in zip(self.priors,pv)])
        

class BasicKernel(DtKernel):
    name  = "BasicKernel"
    eq    = 'At*ESK(1/St) + Ap*ESK(1/Sx)*ESK(1/Sy)'
    names = 'time_amplitude time_iscale xy_amplitude x_iscale y_iscale white_noise '.split()
    pv0   = array([1e-4, 0.25, 1e-4, 4, 4, 0.01])
    ndim  = 3
    npar  = 6
    priors = [UniformPrior(-7, 1),
              LogNormPrior(0.25, 1.25, lims=[0,1]),
              UniformPrior(-7, 0),
              NormalPrior( 17, 8, lims=[0,70]),
              NormalPrior( 17, 8, lims=[0,70]),
              UniformPrior(-6, 0)]
    bounds = [[-5,-3],[0.01,0.6],[-5,-3],[2,20],[2,20],[-4,-2]] 
    
    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESK(1./pv[1], ndim=3, dim=0)
        self._k2 = pv[2] * ESK(1./pv[3], ndim=3, dim=1) * ESK(1./pv[4], ndim=3, dim=2)
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

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESK(1./pv[1], ndim=3, dim=0)
        self._k2 = pv[2] * EK(1./pv[3], ndim=3, dim=1) * EK(1./pv[4], ndim=3, dim=2)
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
        pv = self._pv
        self._k1 = pv[0] * ESn2K(pv[1], pv[2], ndim=3, dim=0)
        self._k2 = ( pv[3] * ESK(pv[4], ndim=3, dim=1) * ESK(pv[5], ndim=3, dim=2)
                   + pv[6] * ESK(pv[7], ndim=3, dim=0))
        self._k   = self._k1 + self._k2


class QuasiPeriodicKernel(BasicKernel):
    name  = 'QuasiPeriodicKernel'
    names = 'time_amplitude time_scale time_period time_evolution xy_amplitude x_scale y_scale white_noise '.split()
    pv0   = array([1, 0.25, 10, 0.01, 1, 0.25, 0.25, 0.01])
    ndim  = 3
    npar  = 8

    priors = [UniformPrior(-6,  1),                    ## 0 Time log10 amplitude
              LogNormPrior( 0.25, 1.25, lims=[0,2]),   ## 1 Inverse time scale
              UniformPrior( 0.10, 25),                 ## 2 Period
              LogNormPrior( 0.25, 1.25, lims=[0,2]),   ## 3 Time Evolution
              UniformPrior(-6,  0),                    ## 4 XY log10 amplitude
              NormalPrior( 17, 8, lims=[0,70]),        ## 5 inverse X scale
              NormalPrior( 17, 8, lims=[0,70]),        ## 6 inverse Y scale
              UniformPrior(-6,  0)]                    ## 7 White noise

    bounds = [[-5,-2],[0.1,1],[0,20],[0.01,1],[-5,-2],[2,50],[2,50],[-4,-2]]

    def __init__(self, p0=None, period=5, evolution_scale=100, **kwargs):
        super(QuasiPeriodicKernel, self).__init__(p0, **kwargs)
        self.period = period
        self._pv[2] = period
        self._pv[3] = evolution_scale
        self.set_pv(self._pv)

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESn2K(1./pv[1], pv[2], ndim=3, dim=0) * ESK(1./pv[3], ndim=3, dim=0)
        self._k2 = pv[4] * ESK(1./pv[5], ndim=3, dim=1) * ESK(1./pv[6], ndim=3, dim=2)
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

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESn2K(1./pv[1], pv[2], ndim=3, dim=0) * ESK(1./pv[3], ndim=3, dim=0)
        self._k2 = pv[4] * EK(1./pv[5], ndim=3, dim=1) * EK(1./pv[6], ndim=3, dim=2)
        self._k  = self._k1 + self._k2
    
        
kernels = dict(basic         = BasicKernel,
               basic_ep      = BasicKernelEP,
               periodic      = PeriodicKernel,
               quasiperiodic = QuasiPeriodicKernel,
               quasiperiodic_ep = QuasiPeriodicKernelEP)
