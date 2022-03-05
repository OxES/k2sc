import numpy as np
import scipy.linalg as sla

from numpy import (array, asarray, array_equal, argmin, abs, diag, dot, identity, log, median, s_, where, zeros, diag_indices)
from tinygp import GaussianProcess as GP
from tinygp import kernels, transforms


import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

jax.config.update("jax_enable_x64", True)


class TinyGP(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self._gp = GP(kernel._k)
        self._y     = None       ## Cached inputs
        self._x     = None       ## Cached values
        # self._dirty = True       ## Flag telling if the arrays are up to date


    # @property
    # def is_dirty(self):
    #     return self._dirty

    # def set_dirty(self, is_dirty=True):
    #     self._dirty = is_dirty

    def set_pv(self, pv=None):
        if pv is not None and not array_equal(pv, self.kernel._pv):
            kernel, _, _ = self.kernel.set_pv(pv)
            return kernel
            # self._gp.kernel = self.kernel._k
            # self.set_dirty()

    def set_inputs(self, x=None):
        if x is not None and not array_equal(x, self._x):
            self._x = x
            # self.set_dirty()

    def _covariance_matrix(self, x1, x2=None, pv=None, separate=False):
        kernel, k1, k2 = self.set_pv(pv)
        if separate:
            return (k1.get_value(x1, x2),
                    k2.get_value(x1, x2))
        else:
            return self.kernel._k.get_value(x1, x2)
    
    def build_gp(self,params):
        kernel, k1, k2 = self.set_pv(params)
        return GaussianProcess(kernel, self._x, diag=jnp.exp(params[-1]))

    @partial(jit,static_argnums=0)
    def lnlikelihood(self, pv, y):
        return -self.build_gp(pv).log_probability(y)

    def predict(self, x):
        gp = self.build_gp(self.pv)
        return gp.predict(self._y, x, mean_only=mean_only)

    def predict_components(self, pv, y, x1, x2=None):
        gp = self.build_gp(self.pv)
        mu_time = gp.condition(y, kernel=gp.kernel.kernel1).gp.loc
        mu_pos = gp.condition(y, kernel=gp.kernel.kernel2).gp.loc
        return mu_time, mu_pos


class SplitGP(TinyGP):
    def __init__(self, kernel, splits=None):
        super(SplitGP, self).__init__(kernel)
        self.splits = splits
        self._K     = None
        self._L     = None
        self._alpha = None

    def _get_mask(self, t1, t2, splits=None):
        """Creates a covariance matrix mask based on a list of splits in time."""
        splits = self.splits if splits is None else splits
        mask = zeros((t1.size,t2.size), np.bool)

        o1,o2 = 0, 0
        for split in splits:
            e1 = argmin(abs(t1-split))
            e2 = argmin(abs(t2-split))
            if (e1 < t1.size-1) & (e2 < t2.size-1):
                mask[o1:e1+1,o2:e2+1] = 1     
                o1,o2 = e1+1,e2+1
        mask[o1:,o2:] = 1
        return mask

    def _covariance_matrix(self, x1, x2=None, separate=False, add_wn=False):
        K1 = self.kernel._k1.get_value(x1, x2)
        K2 = self.kernel._k2.get_value(x1, x2)

        x2 = x2 if x2 is not None else x1
        if self.splits is not None:
            t1 = x1[:,0].flatten()
            t2 = x2[:,0].flatten()
            K2 *= self._get_mask(t1, t2, self.splits)

        if separate:
            return K1, K2
        else:
            if add_wn:
                return K1 + K2 + self.kernel._pm[-1]**2 * identity(K1.shape[0])
            else:
                return K1 + K2

    def lnlikelihood(self, pv, y=None, x=None):
        self.compute(x, pv)
        self._compute_alpha(y)
        return -(log(diag(self._L[0])).sum() + 0.5 * dot(y,self._alpha))


    def predict_components(self, x):
        self._compute_alpha(self._y)
        K1, K2 = self._covariance_matrix(x, self._x, separate=True, add_wn=False)
        b = sla.cho_solve(self._L,self._y)
        mu_time = dot(K1, b)
        mu_pos  = dot(K2, b)
        m0 = median(mu_pos)
        mu_pos  -= m0
        mu_time += m0

        return mu_time, mu_pos
