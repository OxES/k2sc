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

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tinygp
from tinygp.kernels.distance import Distance

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


_VALID_KERNEL_IDS = (
    KERNEL_BASIC,
    KERNEL_BASIC_EP,
    KERNEL_QUASIPERIODIC,
    KERNEL_QUASIPERIODIC_EP,
    KERNEL_PERIODIC,
)
_WHITE_NOISE_IDX: dict[int, int] = {
    KERNEL_BASIC: 5,
    KERNEL_BASIC_EP: 5,
    KERNEL_QUASIPERIODIC: 7,
    KERNEL_QUASIPERIODIC_EP: 7,
    KERNEL_PERIODIC: 8,
}


def _normalize_kernel_id(kernel_id: int | str) -> int:
    if isinstance(kernel_id, str):
        if kernel_id not in KERNEL_IDS:
            raise ValueError(f"Unknown kernel_id: {kernel_id!r}")
        return KERNEL_IDS[kernel_id]
    kid = int(kernel_id)
    if kid not in _VALID_KERNEL_IDS:
        raise ValueError(f"Unknown kernel_id: {kernel_id!r}")
    return kid


def _safe_abs(diff: jax.Array) -> jax.Array:
    """Zero-bias, 1st- and 2nd-order autodiff-safe elementwise |diff|."""
    is_zero = diff == 0.0
    safe_diff = jnp.where(is_zero, jnp.ones_like(diff), diff)
    return jnp.where(is_zero, jnp.zeros_like(diff), jnp.abs(safe_diff))


class SafeL1Distance(Distance):
    """Zero-bias, autodiff-safe L1 distance for 0-D scalar and 1-D vector coordinates."""

    def distance(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        diff = jnp.asarray(X1) - jnp.asarray(X2)
        return jnp.sum(_safe_abs(diff))

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

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        return self.base_kernel.evaluate_diag(X)

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            return self.base_kernel(X1)
        seg1 = jnp.sum(X1[:, 0:1] > self.splits_padded[None, :], axis=-1)
        seg2 = jnp.sum(X2[:, 0:1] > self.splits_padded[None, :], axis=-1)
        val = self.base_kernel(X1, X2)
        return jnp.where(seg1[:, None] == seg2[None, :], val, jnp.zeros_like(val))


class _VectorizedSum(tinygp.kernels.Sum):
    """Sum of two tinygp kernels preserving vectorized 1-D diagonal and 2-D matrix calls."""

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        return self.kernel1.evaluate_diag(X) + self.kernel2.evaluate_diag(X)

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            return self.kernel1(X1) + self.kernel2(X1)
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)


def _eval_time_diag_scalar(p: jax.Array, kernel_id: int) -> jax.Array:
    if kernel_id == KERNEL_PERIODIC:
        return (10.0 ** p[0]) + (10.0 ** p[3])
    return 10.0 ** p[0]


def _eval_time_from_diff(p: jax.Array, diff_t: jax.Array, kernel_id: int) -> jax.Array:
    dt2 = jnp.square(diff_t)
    if kernel_id in (KERNEL_BASIC, KERNEL_BASIC_EP):
        return (10.0 ** p[0]) * jnp.exp(-0.5 * p[1] * dt2)
    dt = _safe_abs(diff_t)
    s2 = jnp.square(jnp.sin(jnp.pi * dt / p[2]))
    if kernel_id in (KERNEL_QUASIPERIODIC, KERNEL_QUASIPERIODIC_EP):
        return (10.0 ** p[0]) * jnp.exp(-s2 / p[1] - 0.5 * p[3] * dt2)
    if kernel_id == KERNEL_PERIODIC:
        return (10.0 ** p[0]) * jnp.exp(-s2 / p[1]) + (10.0 ** p[3]) * jnp.exp(
            -0.5 * p[4] * dt2
        )
    raise ValueError(f"Unknown kernel_id: {kernel_id!r}")


def _spatial_param_indices(kernel_id: int) -> tuple[int, int, int, bool]:
    if kernel_id == KERNEL_BASIC:
        return 2, 3, 4, False
    if kernel_id == KERNEL_BASIC_EP:
        return 2, 3, 4, True
    if kernel_id == KERNEL_QUASIPERIODIC:
        return 4, 5, 6, False
    if kernel_id == KERNEL_QUASIPERIODIC_EP:
        return 4, 5, 6, True
    if kernel_id == KERNEL_PERIODIC:
        return 5, 6, 7, False
    raise ValueError(f"Unknown kernel_id: {kernel_id!r}")


def _eval_spatial_diag_scalar(p: jax.Array, kernel_id: int) -> jax.Array:
    amp_idx, _, _, _ = _spatial_param_indices(kernel_id)
    return 10.0 ** p[amp_idx]


def _eval_spatial_from_diffs(
    p: jax.Array,
    diff_x: jax.Array,
    diff_y: jax.Array,
    same_seg: jax.Array,
    kernel_id: int,
) -> jax.Array:
    amp_idx, x_idx, y_idx, is_ep = _spatial_param_indices(kernel_id)
    amp = 10.0 ** p[amp_idx]
    if is_ep:
        dx = _safe_abs(diff_x)
        dy = _safe_abs(diff_y)
        val = amp * jnp.exp(-jnp.sqrt(p[x_idx]) * dx - jnp.sqrt(p[y_idx]) * dy)
    else:
        val = amp * jnp.exp(
            -0.5 * (p[x_idx] * jnp.square(diff_x) + p[y_idx] * jnp.square(diff_y))
        )
    return jnp.where(same_seg, val, jnp.zeros_like(val))


class K2SCTimeKernel(tinygp.kernels.Kernel):
    """Vectorized time-component tinygp kernel for all five K2SC kernel families."""

    pv: jax.Array
    kernel_id: int = eqx.field(static=True)

    def __init__(self, pv: jax.Array, kernel_id: int | str):
        self.pv = as_f64(pv)
        self.kernel_id = _normalize_kernel_id(kernel_id)

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        return _eval_time_from_diff(self.pv, X1[0] - X2[0], self.kernel_id)

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        del X
        return _eval_time_diag_scalar(self.pv, self.kernel_id)

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            return jnp.broadcast_to(
                _eval_time_diag_scalar(self.pv, self.kernel_id), (X1.shape[0],)
            )
        diff_t = X1[:, 0:1] - X2[None, :, 0]
        return _eval_time_from_diff(self.pv, diff_t, self.kernel_id)

    def __add__(self, other: tinygp.kernels.Kernel | Any) -> tinygp.kernels.Kernel:
        if isinstance(other, tinygp.kernels.Kernel):
            return _VectorizedSum(self, other)
        return super().__add__(other)


class K2SCSpatialKernel(tinygp.kernels.Kernel):
    """Vectorized piecewise-spatial tinygp kernel with O((N + M) * S) split indexing."""

    pv: jax.Array
    splits_padded: jax.Array
    kernel_id: int = eqx.field(static=True)

    def __init__(self, pv: jax.Array, splits_padded: jax.Array, kernel_id: int | str):
        self.pv = as_f64(pv)
        self.splits_padded = as_f64(splits_padded)
        self.kernel_id = _normalize_kernel_id(kernel_id)

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        seg1 = jnp.sum(X1[0] > self.splits_padded)
        seg2 = jnp.sum(X2[0] > self.splits_padded)
        return _eval_spatial_from_diffs(
            self.pv,
            X1[1] - X2[1],
            X1[2] - X2[2],
            seg1 == seg2,
            self.kernel_id,
        )

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        del X
        return _eval_spatial_diag_scalar(self.pv, self.kernel_id)

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            return jnp.broadcast_to(
                _eval_spatial_diag_scalar(self.pv, self.kernel_id), (X1.shape[0],)
            )
        seg1 = jnp.sum(X1[:, 0:1] > self.splits_padded[None, :], axis=-1)
        seg2 = jnp.sum(X2[:, 0:1] > self.splits_padded[None, :], axis=-1)
        same_seg = seg1[:, None] == seg2[None, :]
        diff_x = X1[:, 1:2] - X2[None, :, 1]
        diff_y = X1[:, 2:3] - X2[None, :, 2]
        return _eval_spatial_from_diffs(
            self.pv, diff_x, diff_y, same_seg, self.kernel_id
        )

    def __add__(self, other: tinygp.kernels.Kernel | Any) -> tinygp.kernels.Kernel:
        if isinstance(other, tinygp.kernels.Kernel):
            return _VectorizedSum(self, other)
        return super().__add__(other)


class K2SCKernel(tinygp.kernels.Kernel):
    """Vectorized fused additive time + piecewise-spatial tinygp kernel for K2SC."""

    pv: jax.Array
    splits_padded: jax.Array
    kernel_id: int = eqx.field(static=True)

    def __init__(self, pv: jax.Array, splits_padded: jax.Array, kernel_id: int | str):
        self.pv = as_f64(pv)
        self.splits_padded = as_f64(splits_padded)
        self.kernel_id = _normalize_kernel_id(kernel_id)

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        k1 = _eval_time_from_diff(self.pv, X1[0] - X2[0], self.kernel_id)
        seg1 = jnp.sum(X1[0] > self.splits_padded)
        seg2 = jnp.sum(X2[0] > self.splits_padded)
        k2 = _eval_spatial_from_diffs(
            self.pv,
            X1[1] - X2[1],
            X1[2] - X2[2],
            seg1 == seg2,
            self.kernel_id,
        )
        return k1 + k2

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        del X
        return _eval_time_diag_scalar(
            self.pv, self.kernel_id
        ) + _eval_spatial_diag_scalar(self.pv, self.kernel_id)

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            diag_val = _eval_time_diag_scalar(
                self.pv, self.kernel_id
            ) + _eval_spatial_diag_scalar(self.pv, self.kernel_id)
            return jnp.broadcast_to(diag_val, (X1.shape[0],))
        diff_t = X1[:, 0:1] - X2[None, :, 0]
        k1 = _eval_time_from_diff(self.pv, diff_t, self.kernel_id)
        seg1 = jnp.sum(X1[:, 0:1] > self.splits_padded[None, :], axis=-1)
        seg2 = jnp.sum(X2[:, 0:1] > self.splits_padded[None, :], axis=-1)
        same_seg = seg1[:, None] == seg2[None, :]
        diff_x = X1[:, 1:2] - X2[None, :, 1]
        diff_y = X1[:, 2:3] - X2[None, :, 2]
        k2 = _eval_spatial_from_diffs(
            self.pv, diff_x, diff_y, same_seg, self.kernel_id
        )
        return k1 + k2


def build_tinygp_kernels(
    pv: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int | str,
) -> tuple[K2SCTimeKernel, K2SCSpatialKernel, jax.Array]:
    """Build stateless vectorized (k_time, k_pos, diag_variance) tinygp kernels."""
    p = as_f64(pv)
    s_pad = as_f64(splits_padded)
    kid = _normalize_kernel_id(kernel_id)
    k1 = K2SCTimeKernel(p, kid)
    k2 = K2SCSpatialKernel(p, s_pad, kid)
    diag = 10.0 ** (2.0 * p[_WHITE_NOISE_IDX[kid]])
    return k1, k2, diag


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
