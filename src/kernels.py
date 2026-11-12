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

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import tinygp
from tinygp.kernels import Exp, ExpSineSquared, ExpSquared
from tinygp.kernels.distance import Distance
from tinygp.transforms import Subspace

from ._jax_env import as_f64, ensure_float64
from .priors import (
    LogNormPrior,
    NormalPrior,
    Prior,
    UniformPrior,
    build_prior_spec,
    eval_ln_prior_jax,
)

ensure_float64()

UP = UniformPrior
NP = NormalPrior
LP = LogNormPrior

MAX_SPLITS = 16

KERNEL_BASIC = 0
KERNEL_BASIC_EP = 1
KERNEL_QUASIPERIODIC = 2
KERNEL_QUASIPERIODIC_EP = 3
KERNEL_PERIODIC = 4

KERNEL_IDS: dict[str, int] = {
    "BasicKernel": KERNEL_BASIC,
    "basic": KERNEL_BASIC,
    "BasicKernelEP": KERNEL_BASIC_EP,
    "basic_ep": KERNEL_BASIC_EP,
    "QuasiPeriodicKernel": KERNEL_QUASIPERIODIC,
    "quasiperiodic": KERNEL_QUASIPERIODIC,
    "QuasiPeriodicKernelEP": KERNEL_QUASIPERIODIC_EP,
    "quasiperiodic_ep": KERNEL_QUASIPERIODIC_EP,
    "PeriodicKernel": KERNEL_PERIODIC,
    "periodic": KERNEL_PERIODIC,
}


_DEFAULT_INF_SPLITS = as_f64(np.full((MAX_SPLITS,), np.inf, dtype=np.float64))


def pad_splits(splits, max_splits: int = MAX_SPLITS) -> jax.Array:
    """Normalize None or 1D split thresholds into a float64 JAX array padded with +inf."""
    if splits is None:
        if max_splits == MAX_SPLITS:
            return _DEFAULT_INF_SPLITS
        return as_f64(np.full((max_splits,), np.inf, dtype=np.float64))
    if not isinstance(splits, jax.Array):
        arr_np = np.asarray(splits, dtype=np.float64).ravel()
        n = arr_np.size
        if n == 0:
            if max_splits == MAX_SPLITS:
                return _DEFAULT_INF_SPLITS
            return as_f64(np.full((max_splits,), np.inf, dtype=np.float64))
        target = max(max_splits, ((n + max_splits - 1) // max_splits) * max_splits)
        out = np.full((target,), np.inf, dtype=np.float64)
        out[:n] = np.sort(arr_np)
        return as_f64(out)
    arr = as_f64(splits).ravel()
    n = arr.shape[0]
    if n == 0:
        return as_f64(np.full((max_splits,), np.inf, dtype=np.float64))
    target = max(max_splits, ((n + max_splits - 1) // max_splits) * max_splits)
    arr_sorted = jnp.sort(arr)
    if n == target:
        return arr_sorted
    pad = jnp.full((target - n,), jnp.inf, dtype=jnp.float64)
    return jnp.concatenate([arr_sorted, pad], axis=0)


class SafeL1Distance(Distance):
    """Zero-bias, autodiff-safe L1 distance for 0-D scalar and 1-D vector coordinates."""

    def distance(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        diff = jnp.asarray(X1) - jnp.asarray(X2)
        is_zero = diff == 0.0
        safe_diff = jnp.where(is_zero, jnp.ones_like(diff), diff)
        return jnp.sum(jnp.where(is_zero, jnp.zeros_like(diff), jnp.abs(safe_diff)))

    def squared_distance(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        return jnp.square(self.distance(X1, X2))


class PiecewiseSpatialKernel(tinygp.kernels.Kernel):
    """Piecewise-constant-in-time spatial covariance kernel (guaranteed PSD)."""

    base_kernel: tinygp.kernels.Kernel
    splits_padded: jax.Array

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        seg1 = jnp.sum(X1[0] > self.splits_padded)
        seg2 = jnp.sum(X2[0] > self.splits_padded)
        val = self.base_kernel.evaluate(X1, X2)
        return jnp.where(seg1 == seg2, val, jnp.zeros_like(val))


def build_tinygp_kernels(
    pv: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int | str,
) -> tuple[tinygp.kernels.Kernel, tinygp.kernels.Kernel, jax.Array]:
    """Build stateless (k_time, k_pos, diag_variance) tinygp kernels for a hyperparameter vector."""
    p = as_f64(pv)
    kid = KERNEL_IDS[kernel_id] if isinstance(kernel_id, str) else int(kernel_id)
    safe_l1 = SafeL1Distance()

    if kid == KERNEL_BASIC:
        k1 = (10.0 ** p[0]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[1])))
        k2_base = (
            (10.0 ** p[2])
            * Subspace(1, ExpSquared(scale=1.0 / jnp.sqrt(p[3])))
            * Subspace(2, ExpSquared(scale=1.0 / jnp.sqrt(p[4])))
        )
        k2 = PiecewiseSpatialKernel(k2_base, splits_padded)
        diag = 10.0 ** (2.0 * p[5])
        return k1, k2, diag

    if kid == KERNEL_BASIC_EP:
        k1 = (10.0 ** p[0]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[1])))
        k2_base = (
            (10.0 ** p[2])
            * Subspace(1, Exp(scale=1.0 / jnp.sqrt(p[3]), distance=safe_l1))
            * Subspace(2, Exp(scale=1.0 / jnp.sqrt(p[4]), distance=safe_l1))
        )
        k2 = PiecewiseSpatialKernel(k2_base, splits_padded)
        diag = 10.0 ** (2.0 * p[5])
        return k1, k2, diag

    if kid == KERNEL_QUASIPERIODIC:
        k1 = (10.0 ** p[0]) * Subspace(
            0,
            ExpSineSquared(scale=p[2], gamma=1.0 / p[1], distance=safe_l1)
            * ExpSquared(scale=1.0 / jnp.sqrt(p[3])),
        )
        k2_base = (
            (10.0 ** p[4])
            * Subspace(1, ExpSquared(scale=1.0 / jnp.sqrt(p[5])))
            * Subspace(2, ExpSquared(scale=1.0 / jnp.sqrt(p[6])))
        )
        k2 = PiecewiseSpatialKernel(k2_base, splits_padded)
        diag = 10.0 ** (2.0 * p[7])
        return k1, k2, diag

    if kid == KERNEL_QUASIPERIODIC_EP:
        k1 = (10.0 ** p[0]) * Subspace(
            0,
            ExpSineSquared(scale=p[2], gamma=1.0 / p[1], distance=safe_l1)
            * ExpSquared(scale=1.0 / jnp.sqrt(p[3])),
        )
        k2_base = (
            (10.0 ** p[4])
            * Subspace(1, Exp(scale=1.0 / jnp.sqrt(p[5]), distance=safe_l1))
            * Subspace(2, Exp(scale=1.0 / jnp.sqrt(p[6]), distance=safe_l1))
        )
        k2 = PiecewiseSpatialKernel(k2_base, splits_padded)
        diag = 10.0 ** (2.0 * p[7])
        return k1, k2, diag

    if kid == KERNEL_PERIODIC:
        k1 = (10.0 ** p[0]) * Subspace(
            0,
            ExpSineSquared(scale=p[2], gamma=1.0 / p[1], distance=safe_l1),
        ) + (10.0 ** p[3]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[4])))
        k2_base = (
            (10.0 ** p[5])
            * Subspace(1, ExpSquared(scale=1.0 / jnp.sqrt(p[6])))
            * Subspace(2, ExpSquared(scale=1.0 / jnp.sqrt(p[7])))
        )
        k2 = PiecewiseSpatialKernel(k2_base, splits_padded)
        diag = 10.0 ** (2.0 * p[8])
        return k1, k2, diag

    raise ValueError(f"Unknown kernel_id: {kernel_id!r}")


class DtKernel:
    """Base class for K2SC additive time + spatial Gaussian Process kernels."""

    name: ClassVar[str] = ""
    kernel_id: ClassVar[int] = -1
    eq: ClassVar[str] = ""
    names: ClassVar[list[str]] = []
    pv0: ClassVar[np.ndarray] = np.array([], dtype=np.float64)
    ndim: ClassVar[int] = 0
    npar: ClassVar[int] = 0
    priors: ClassVar[list[Prior]] = []
    bounds: ClassVar[list[list[float]]] = []

    def __init__(self, p0=None, **kwargs):
        base_pv = np.array(self.pv0, dtype=np.float64, copy=True)
        self._pv = np.array(p0, dtype=np.float64, copy=True) if p0 is not None else base_pv
        self.lims = np.transpose([p.lims for p in self.priors]).astype(np.float64)
        self.prior_spec = build_prior_spec(self.priors)

        assert len(self.names) == self.npar, (
            "Number of parameter names doesn't match the number of parameters"
        )
        assert len(self.priors) == self.npar, (
            "Number of parameter priors doesn't match the number of parameters"
        )
        assert np.asarray(self.pv0).size == self.npar, (
            "Number of parameter vector elements doesn't match the number of parameters"
        )
        assert self._pv.size == self.npar, (
            "Supplied initial parameter vector length doesn't match npar"
        )

        self._pm = self.map_pv(self._pv)

    def map_pv(self, pv) -> np.ndarray:
        raise NotImplementedError()

    def set_pv(self, pv) -> None:
        self._pv = np.array(pv, dtype=np.float64, copy=True)
        self._pm = self.map_pv(self._pv)

    def build_tinygp_kernels(self, pv=None, splits_padded=None):
        p = self._pv if pv is None else pv
        s_pad = pad_splits(splits_padded)
        return build_tinygp_kernels(p, s_pad, self.kernel_id)

    def ln_prior(self, pv) -> jax.Array:
        return eval_ln_prior_jax(as_f64(pv), self.prior_spec)


class BasicKernel(DtKernel):
    name: ClassVar[str] = "BasicKernel"
    kernel_id: ClassVar[int] = KERNEL_BASIC
    eq: ClassVar[str] = "At*ESK(1/St) + Ap*ESK(1/Sx)*ESK(1/Sy)"
    names: ClassVar[list[str]] = [
        "time_amplitude",
        "time_iscale",
        "xy_amplitude",
        "x_iscale",
        "y_iscale",
        "white_noise",
    ]
    pv0: ClassVar[np.ndarray] = np.array(
        [-6.0, 0.25, -5.4, 21.0, 21.0, -3.0], dtype=np.float64
    )
    ndim: ClassVar[int] = 3
    npar: ClassVar[int] = 6
    priors: ClassVar[list[Prior]] = [
        UP(-7, 1),  ## 0 -- log10 time amplitude
        LP(0.25, 1.25, lims=[0, 1]),  ## 1 -- inverse time scale
        UP(-7, 0),  ## 2 -- xy log10 amplitude
        NP(17, 8, lims=[0, 70]),  ## 3 -- inverse x scale
        NP(17, 8, lims=[0, 70]),  ## 4 -- inverse y scale
        UP(-6, 0),  ## 5 -- log10 white noise
    ]
    bounds: ClassVar[list[list[float]]] = [
        [-5, -3],
        [0.01, 0.6],
        [-5, -3],
        [2, 20],
        [2, 20],
        [-4, -2],
    ]

    def map_pv(self, pv) -> np.ndarray:
        pp = np.array(pv, dtype=np.float64, copy=True)
        pp[0] = 10.0 ** pv[0]
        pp[1] = 1.0 / pv[1]
        pp[2] = 10.0 ** pv[2]
        pp[3] = 1.0 / pv[3]
        pp[4] = 1.0 / pv[4]
        pp[5] = 10.0 ** pv[5]
        self._pm = pp
        return self._pm


class BasicKernelEP(BasicKernel):
    name: ClassVar[str] = "BasicKernelEP"
    kernel_id: ClassVar[int] = KERNEL_BASIC_EP
    eq: ClassVar[str] = "At*ESK(1/St) + Ap*EK(1/Sx)*EK(1/Sy)"
    priors: ClassVar[list[Prior]] = [
        UP(-7, 1),  ## 0 -- log10 time amplitude
        LP(0.25, 1.25, lims=[0, 1]),  ## 1 -- inverse time scale
        UP(-7, 0),  ## 2 -- xy log10 amplitude
        NP(10, 15, lims=[0, 70]),  ## 3 -- inverse x scale
        NP(10, 15, lims=[0, 70]),  ## 4 -- inverse y scale
        UP(-6, 0),  ## 5 -- log10 white noise
    ]


class PeriodicKernel(DtKernel):
    name: ClassVar[str] = "PeriodicKernel"
    kernel_id: ClassVar[int] = KERNEL_PERIODIC
    eq: ClassVar[str] = "At*ESn2K(1/St, P) + Atl*ESK(1/Stl) + Ap*ESK(1/Sx)*ESK(1/Sy)"
    names: ClassVar[list[str]] = [
        "time_amplitude",
        "time_iscale",
        "time_period",
        "tl_amplitude",
        "tl_iscale",
        "xy_amplitude",
        "x_iscale",
        "y_iscale",
        "white_noise",
    ]
    pv0: ClassVar[np.ndarray] = np.array(
        [-6.0, 0.25, 10.0, -6.0, 0.01, -5.4, 21.0, 21.0, -3.0], dtype=np.float64
    )
    ndim: ClassVar[int] = 3
    npar: ClassVar[int] = 9
    priors: ClassVar[list[Prior]] = [
        UP(-6, 1),  ## 0 -- log10 periodic time amplitude
        LP(0.25, 1.25, lims=[0, 2]),  ## 1 -- inverse harmonic complexity (1/Gamma)
        UP(0.04, 45),  ## 2 -- period
        UP(-7, 1),  ## 3 -- log10 long-term time amplitude
        LP(0.25, 1.25, lims=[0, 1]),  ## 4 -- inverse long-term time scale
        UP(-6, 0),  ## 5 -- log10 xy amplitude
        NP(17, 8, lims=[0, 70]),  ## 6 -- inverse x scale
        NP(17, 8, lims=[0, 70]),  ## 7 -- inverse y scale
        UP(-6, 0),  ## 8 -- log10 white noise
    ]
    bounds: ClassVar[list[list[float]]] = [
        [-5, -2],
        [0.1, 1.0],
        [0.05, 25.0],
        [-5, -3],
        [0.005, 0.5],
        [-5, -2],
        [2, 50],
        [2, 50],
        [-4, -2],
    ]

    def __init__(self, p0=None, period=None, **kwargs):
        super().__init__(p0=p0, **kwargs)
        if period is not None:
            self._pv[2] = float(period)
        self.period = float(self._pv[2])
        self.set_pv(self._pv)

    def map_pv(self, pv) -> np.ndarray:
        pp = np.array(pv, dtype=np.float64, copy=True)
        pp[0] = 10.0 ** pv[0]
        pp[1] = 1.0 / pv[1]
        pp[2] = pv[2]
        pp[3] = 10.0 ** pv[3]
        pp[4] = 1.0 / pv[4]
        pp[5] = 10.0 ** pv[5]
        pp[6] = 1.0 / pv[6]
        pp[7] = 1.0 / pv[7]
        pp[8] = 10.0 ** pv[8]
        self._pm = pp
        return self._pm


class QuasiPeriodicKernel(BasicKernel):
    name: ClassVar[str] = "QuasiPeriodicKernel"
    kernel_id: ClassVar[int] = KERNEL_QUASIPERIODIC
    eq: ClassVar[str] = "At*ESn2K(1/St, P)*ESK(1/Se) + Ap*ESK(1/Sx)*ESK(1/Sy)"
    names: ClassVar[list[str]] = [
        "time_amplitude",
        "time_iscale",
        "time_period",
        "time_evolution",
        "xy_amplitude",
        "x_iscale",
        "y_iscale",
        "white_noise",
    ]
    pv0: ClassVar[np.ndarray] = np.array(
        [-6.0, 0.25, 10.0, 0.01, -5.4, 21.0, 21.0, -3.0], dtype=np.float64
    )
    ndim: ClassVar[int] = 3
    npar: ClassVar[int] = 8
    priors: ClassVar[list[Prior]] = [
        UP(-6, 1),  ## 0 -- time log10 amplitude
        LP(0.25, 1.25, lims=[0, 2]),  ## 1 -- inverse harmonic complexity (1/Gamma)
        UP(0.04, 45),  ## 2 -- period
        LP(0.25, 1.25, lims=[0, 2]),  ## 3 -- inverse squared evolution timescale
        UP(-6, 0),  ## 4 -- xy log10 amplitude
        NP(17, 8, lims=[0, 70]),  ## 5 -- inverse x scale
        NP(17, 8, lims=[0, 70]),  ## 6 -- inverse y scale
        UP(-6, 0),  ## 7 -- log10 white noise
    ]
    bounds: ClassVar[list[list[float]]] = [
        [-5, -2],
        [0.1, 1.0],
        [0.05, 25.0],
        [0.01, 1.0],
        [-5, -2],
        [2, 50],
        [2, 50],
        [-4, -2],
    ]

    def __init__(
        self,
        p0=None,
        period=None,
        evolution_iscale=None,
        evolution_scale=None,
        **kwargs,
    ):
        super().__init__(p0=p0, **kwargs)
        if period is not None:
            self._pv[2] = float(period)
        if evolution_iscale is not None:
            self._pv[3] = float(evolution_iscale)
        elif evolution_scale is not None:
            self._pv[3] = 1.0 / (float(evolution_scale) ** 2)
        self.period = float(self._pv[2])
        self.evolution_iscale = float(self._pv[3])
        self.set_pv(self._pv)

    def map_pv(self, pv) -> np.ndarray:
        pp = np.array(pv, dtype=np.float64, copy=True)
        pp[0] = 10.0 ** pv[0]
        pp[1] = 1.0 / pv[1]
        pp[2] = pv[2]
        pp[3] = 1.0 / pv[3]
        pp[4] = 10.0 ** pv[4]
        pp[5] = 1.0 / pv[5]
        pp[6] = 1.0 / pv[6]
        pp[7] = 10.0 ** pv[7]
        self._pm = pp
        return self._pm


class QuasiPeriodicKernelEP(QuasiPeriodicKernel):
    name: ClassVar[str] = "QuasiPeriodicKernelEP"
    kernel_id: ClassVar[int] = KERNEL_QUASIPERIODIC_EP
    eq: ClassVar[str] = "At*ESn2K(1/St, P)*ESK(1/Se) + Ap*EK(1/Sx)*EK(1/Sy)"
    priors: ClassVar[list[Prior]] = [
        UP(-6, 1),  ## 0 -- time log10 amplitude
        LP(0.25, 1.25, lims=[0, 2]),  ## 1 -- inverse harmonic complexity (1/Gamma)
        UP(0.25, 45),  ## 2 -- period
        LP(0.25, 1.25, lims=[0, 2]),  ## 3 -- inverse squared evolution timescale
        UP(-6, 0),  ## 4 -- xy log10 amplitude
        NP(10, 15, lims=[0, 70]),  ## 5 -- inverse x scale
        NP(10, 15, lims=[0, 70]),  ## 6 -- inverse y scale
        UP(-6, 0),  ## 7 -- log10 white noise
    ]
    bounds: ClassVar[list[list[float]]] = [
        [-5, -2],
        [0.1, 1.0],
        [0.26, 25.0],
        [0.01, 1.0],
        [-5, -2],
        [2, 50],
        [2, 50],
        [-4, -2],
    ]


kernels: dict[str, type[DtKernel]] = {
    "basic": BasicKernel,
    "basic_ep": BasicKernelEP,
    "periodic": PeriodicKernel,
    "quasiperiodic": QuasiPeriodicKernel,
    "quasiperiodic_ep": QuasiPeriodicKernelEP,
}
