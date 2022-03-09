import numpy as np
import scipy.linalg as sla

from numpy import (array, asarray, array_equal, argmin, abs, diag, dot, identity, log, median, s_, where, zeros, diag_indices)

from tinygp import GaussianProcess
from tinygp import kernels, transforms
import tinygp

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

jax.config.update("jax_enable_x64", True)

# A kernel restrict to specific "times"
@tinygp.helpers.dataclass
class Blocked(tinygp.kernels.Kernel):
    base_kernel: tinygp.kernels.Kernel
    t_min: jnp.ndarray
    t_max: jnp.ndarray

    def evaluate(self, X1, X2):
        t1, *_ = X1
        t2, *_ = X2
        return jnp.where(
            jnp.logical_and(
                jnp.logical_and(self.t_min < t1, t1 < self.t_max),
                jnp.logical_and(self.t_min < t2, t2 < self.t_max),                
            ),
            self.base_kernel.evaluate(X1, X2),
            0
        )


class TinyGP(object):
    def __init__(self, kernel, flux):
        self.kernel = kernel
        self._y     = flux       ## Cached inputs
        self._x     = None       ## Cached values
        # self._dirty = True       ## Flag telling if the arrays are up to date


    # @property
    # def is_dirty(self):
    #     return self._dirty

    # def set_dirty(self, is_dirty=True):
    #     self._dirty = is_dirty

    def set_pv(self, pv):
        kernel, k1, k2 = self.kernel.set_pv(pv)
        return kernel, k1, k2
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
    
    def build_gp(self,params, inputs):
        kernel, k1, k2 = self.set_pv(params)
        return GaussianProcess(kernel, inputs, diag=10**(params[-1]))

    @partial(jit,static_argnums=0)
    def lnlikelihood(self, pv, y, inputs):
        return -self.build_gp(pv,inputs).log_probability(y)

    def predict(self, pv, inputs):
        gp = self.build_gp(pv,inputs)
        return gp.predict(self._y, x)

    def predict_components(self, pv, inputs):
        gp = self.build_gp(pv,inputs)
        mu_time = gp.condition(self._y, kernel=gp.kernel.kernel1).gp.loc
        mu_pos = gp.condition(self._y, kernel=gp.kernel.kernel2).gp.loc
        return mu_time, mu_pos


class SplitGP(TinyGP):
    def __init__(self, kernel, flux, tmin, tmax, splits):
        super(SplitGP, self).__init__(kernel,flux)
        self.splits = splits
        self._t_min = tmin
        self._t_max = tmax


    def build_gp(self, params, inputs):
        _, k1, _k2 = self.set_pv(params)

        embed = [self._t_min,*self.splits,self._t_max]

        k2 = 0
        for t_min, t_max in zip(embed, embed[1:]):
            k2 += Blocked(_k2,t_min,t_max)
        
        kernel = k1 + k2
    
        return GaussianProcess(3*kernel, inputs, diag=10**(params[-1]))

    @partial(jit,static_argnums=0)
    def lnlikelihood(self, pv, y, inputs):
        return -self.build_gp(pv,inputs).log_probability(y)

