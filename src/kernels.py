import math as m

from numpy import array, asarray, s_, log, pi, inf

from george.kernels import ExpSquaredKernel as ESK
from george.kernels import ExpSine2Kernel as ESn2K

##TODO: Add priors

class Prior(object):
    def __init__(self):
        raise NotImplementedError()
    def logpdf(self, x):
        raise NotImplementedError()


class UniformPrior(Prior):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.C = 1./(b-1)
        self.lnC = m.log(self.C)

    def logpdf(self, x):
        if x > self.a and x < self.b:
            return self.lnC
        else:
            return -inf


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

    def __init__(self, p0=None):
        self._pv = asarray(p0) if p0 is not None else self.pv0
        self._define_kernel()
        self._nk1 = len(self._k1)
        self._nk2 = len(self._k2)
        self._sk1 = s_[:self._nk1]
        self._sk2 = s_[self._nk1:-1]

        assert len(self._k) == self.npar - 1, "Expected and true number of kernel parameters do not match"
        assert len(self.names) == self.npar, "Number of parameter names doesn't match the number of parameters"
        assert len(self.priors) == self.npar, "Number of parameter priors doesn't match the number of parameters" 
        assert self.pv0.size == self.npar, "Number of parameter vector elements doesn't match the number of parameters"

    def _define_kernel(self):
        raise NotImplementedError()
        self._k1 = None
        self._k2 = None
        self._k  = None

    def set_pv(self, pv):
        self._pv = pv.copy()
        self._k1[:] = log(pv[self._sk1])
        self._k2[:] = log(pv[self._sk2])
        self._k[:]  = log(pv[:-1])

    def ln_prior(self, pv):
        return sum([p.logpdf(v) for p,v in zip(self.priors,pv)])


class BasicKernel(DtKernel):
    name  = 'BasicKernel'
    eq    = 'At*ESK(St) + Ap*ESK(Sx)*ESK(Sy)'
    names = 'time_amplitude time_scale xy_amplitude x_scale y_scale white_noise '.split()
    pv0   = array([1, 4, 1, 0.5, 0.5, 0.01])
    ndim  = 3
    npar  = 6
    priors = [UniformPrior(0,10),
              LogNormPrior(8, 2, lims=[0.5,inf]),
              UniformPrior(0,10),
              UniformPrior(1e-4,10),
              UniformPrior(1e-4,10),
              UniformPrior(0,10)]
    bounds = [[0,2],[0.1,10],[0,2],[1e-2,3],[1e-2,3],[0.01,1]]

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESK(pv[1], ndim=3, dim=0)
        self._k2 = pv[2] * ESK(pv[3], ndim=3, dim=1) * ESK(pv[4], ndim=3, dim=2)
        self._k   = self._k1 + self._k2


class BasicKernelInvScale(DtKernel):
    name  = "BasicKernelInvScale"
    eq    = 'At*ESK(1/St) + Ap*ESK(1/Sx)*ESK(1/Sy)'
    names = 'time_amplitude time_iscale xy_amplitude x_iscale y_iscale white_noise '.split()
    pv0   = array([1, 0.25, 1, 4, 4, 0.01])
    ndim  = 3
    npar  = 6
    priors = [UniformPrior(0,10),
              LogNormPrior(0.25, 1.25, lims=[0,2]),
              UniformPrior(0,10),
              LogNormPrior(6, 1),
              LogNormPrior(6, 1),
              UniformPrior(0,10)]
    bounds = [[0,2],[0.01,1],[0,2],[2,50],[2,50],[0.01,1]]

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESK(1./pv[1], ndim=3, dim=0)
        self._k2 = pv[2] * ESK(1./pv[3], ndim=3, dim=1) * ESK(1./pv[4], ndim=3, dim=2)
        self._k   = self._k1 + self._k2


class BasicKernel2(DtKernel):
    name  = 'BasicKernel2'
    names = 'time_amplitude time_scale xy_amplitude x_scale y_scale lt_amplitude lt_scale white_noise '.split()
    pv0   = array([1, 4, 1, 0.25, 0.25, 1, 100, 0.01])
    ndim  = 3
    npar  = 8

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESK(pv[1], ndim=3, dim=0)
        self._k2 = pv[2] * ESK(pv[3], ndim=3, dim=1) * ESK(pv[4], ndim=3, dim=2) + pv[5] * ESK(pv[6], ndim=3, dim=0)
        self._k   = self._k1 + self._k2


class PeriodicKernel(DtKernel):
    name  = 'PeriodicKernel'
    names = 'time_amplitude inv_time_scale time_period xy_amplitude x_scale y_scale tl_amplitude tl_scale white_noise '.split()
    pv0   = array([1, 1, 5, 1, 0.25, 0.25, 0.01, 100, 0.01])
    ndim  = 3
    npar  = 9

    def __init__(self, p0=None, period=5):
        super(PeriodicKernel, self).__init__(p0)
        self._pv[2] = period
        self.set_pv(self._pv)

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESn2K(pv[1], pv[2], ndim=3, dim=0)
        self._k2 = ( pv[3] * ESK(pv[4], ndim=3, dim=1) * ESK(pv[5], ndim=3, dim=2)
                   + pv[6] * ESK(pv[7], ndim=3, dim=0))
        self._k   = self._k1 + self._k2


# class QuasiPeriodicKernel(BasicKernel):
#     name  = 'QuasiPeriodicKernel'
#     names = 'time_amplitude time_scale time_period time_evolution xy_amplitude x_scale y_scale white_noise '.split()
#     pv0   = array([1, 0.25, 10, 100, 1, 0.25, 0.25, 0.01])
#     ndim  = 3
#     npar  = 8

#     priors = [UniformPrior(0,10),    ## Time amplitude
#               LogNormPrior(8, 2, lims=[0.5,inf]),    ## Time scale
#               UniformPrior(0,20),    ## Period
#               UniformPrior(0,500),   ## Evolution
#               UniformPrior(0,10),    ## XY amplitude
#               UniformPrior(1e-4,10), ## X scale
#               UniformPrior(1e-4,10), ## Y scale
#               UniformPrior(0,10)]    ## White noise

#     bounds = [[0,2],[0.1,10],[0,20],[0,500],[0,2],[1e-2,3],[1e-2,3],[0.01,1]]

#     def __init__(self, p0=None, period=5, evolution_scale=100):
#         super(QuasiPeriodicKernel, self).__init__(p0)
#         self._pv[2] = period
#         self._pv[3] = evolution_scale
#         self.set_pv(self._pv)

#     def _define_kernel(self):
#         pv = self._pv
#         self._k1 = pv[0] * ESn2K(1./pv[1], pv[2], ndim=3, dim=0) * ESK(pv[3], ndim=3, dim=0)
#         self._k2 = pv[4] * ESK(pv[5], ndim=3, dim=1) * ESK(pv[6], ndim=3, dim=2)
#         self._k  = self._k1 + self._k2


class QuasiPeriodicKernel(BasicKernel):
    name  = 'QuasiPeriodicKernel'
    names = 'time_amplitude time_scale time_period time_evolution xy_amplitude x_scale y_scale white_noise '.split()
    pv0   = array([1, 0.25, 10, 100, 1, 0.25, 0.25, 0.01])
    ndim  = 3
    npar  = 8

    priors = [UniformPrior(0, 5),    ## Time amplitude
              LogNormPrior(0.25, 1.25, lims=[0,2]),    ## Inverse time scale
              UniformPrior(0,25),    ## Period
              UniformPrior(0,500),   ## Evolution
              UniformPrior(0, 5),    ## XY amplitude
              LogNormPrior(6, 1),    ## inverse X scale
              LogNormPrior(6, 1),    ## inverse Y scale
              UniformPrior(0,10)]    ## White noise

    bounds = [[0,2],[0.1,10],[0,20],[0,500],[0,2],[1e-2,3],[1e-2,3],[0.01,1]]

    def __init__(self, p0=None, period=5, evolution_scale=100):
        super(QuasiPeriodicKernel, self).__init__(p0)
        self._pv[2] = period
        self._pv[3] = evolution_scale
        self.set_pv(self._pv)

    def _define_kernel(self):
        pv = self._pv
        self._k1 = pv[0] * ESn2K(pv[1], pv[2], ndim=3, dim=0) * ESK(pv[3], ndim=3, dim=0)
        self._k2 = pv[4] * ESK(1./pv[5], ndim=3, dim=1) * ESK(1./pv[6], ndim=3, dim=2)
        self._k  = self._k1 + self._k2
