""" Module defining priors.

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

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._jax_env import as_f64, ensure_float64

ensure_float64()

PRIOR_UNIFORM = 0
PRIOR_NORMAL = 1
PRIOR_LOGNORM = 2


class PriorSpec(NamedTuple):
    """Dynamic JAX PyTree encoding a vector of 1D independent parameter priors."""

    kind: jax.Array
    vmin: jax.Array
    vmax: jax.Array
    mu: jax.Array
    sigma: jax.Array


@jax.jit
def _uniform_logpdf_core(x_arr: jax.Array, vmin: float, vmax: float, ln_c: float) -> jax.Array:
    in_bounds = (x_arr >= vmin) & (x_arr <= vmax) & jnp.isfinite(x_arr)
    return jnp.where(in_bounds, jnp.full_like(x_arr, ln_c), -jnp.inf)


@jax.jit
def _normal_logpdf_core(
    x_arr: jax.Array, vmin: float, vmax: float, mu: float, lf1: float, f2: float
) -> jax.Array:
    in_bounds = (x_arr >= vmin) & (x_arr <= vmax) & jnp.isfinite(x_arr)
    safe_x = jnp.where(jnp.isfinite(x_arr), x_arr, jnp.full_like(x_arr, mu))
    val = lf1 - jnp.square(safe_x - mu) * f2
    return jnp.where(in_bounds, val, -jnp.inf)


@jax.jit
def _lognorm_logpdf_core(
    x_arr: jax.Array, vmin: float, vmax: float, mu: float, sigma: float, c_const: float
) -> jax.Array:
    is_pos_finite = (x_arr > 0.0) & jnp.isfinite(x_arr)
    in_bounds = is_pos_finite & (x_arr > vmin) & (x_arr <= vmax)
    safe_x = jnp.where(is_pos_finite, x_arr, jnp.ones_like(x_arr))
    lnx = jnp.log(safe_x)
    val = -lnx + c_const - 0.5 * jnp.square((lnx - mu) / sigma)
    return jnp.where(in_bounds, val, -jnp.inf)


class Prior:
    """Abstract base class for 1D parameter priors."""

    def __init__(self):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.logpdf(x)


class UniformPrior(Prior):
    """Continuous uniform prior on [vmin, vmax]."""

    def __init__(self, vmin: float, vmax: float):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        if self.vmax <= self.vmin:
            raise ValueError("UniformPrior requires vmax > vmin.")
        self.C = 1.0 / (self.vmax - self.vmin)
        self.lnC = math.log(self.C)
        self.lims = np.array([self.vmin, self.vmax], dtype=np.float64)

    def logpdf(self, x) -> jax.Array:
        return _uniform_logpdf_core(as_f64(x), self.vmin, self.vmax, self.lnC)


class NormalPrior(Prior):
    """Truncated or unbounded Gaussian prior on [vmin, vmax]."""

    def __init__(self, mu: float, sigma: float, lims=None):
        lims_pair = (-np.inf, np.inf) if lims is None else lims
        self.vmin = float(lims_pair[0])
        self.vmax = float(lims_pair[1])
        if self.vmax <= self.vmin:
            raise ValueError("NormalPrior requires lims[1] > lims[0].")
        self.lims = np.array([self.vmin, self.vmax], dtype=np.float64)
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.sigma <= 0.0:
            raise ValueError("NormalPrior requires sigma > 0.")
        self._f1 = 1.0 / math.sqrt(2.0 * math.pi * self.sigma * self.sigma)
        self._lf1 = math.log(self._f1)
        self._f2 = 1.0 / (2.0 * self.sigma * self.sigma)

    def logpdf(self, x) -> jax.Array:
        return _normal_logpdf_core(
            as_f64(x), self.vmin, self.vmax, self.mu, self._lf1, self._f2
        )


class LogNormPrior(Prior):
    """Log-normal prior on (max(0, vmin), vmax] with autodiff-safe boundary evaluation."""

    def __init__(self, mu: float, sigma: float, lims=None):
        lims_pair = (0.0, np.inf) if lims is None else lims
        self.vmin = float(lims_pair[0])
        self.vmax = float(lims_pair[1])
        if self.vmax <= self.vmin:
            raise ValueError("LogNormPrior requires lims[1] > lims[0].")
        self.lims = np.array([self.vmin, self.vmax], dtype=np.float64)
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.sigma <= 0.0:
            raise ValueError("LogNormPrior requires sigma > 0.")
        self.C = -math.log(self.sigma * math.sqrt(2.0 * math.pi))
        self._B = 2.0 * (self.sigma**2)

    def logpdf(self, x) -> jax.Array:
        return _lognorm_logpdf_core(
            as_f64(x), self.vmin, self.vmax, self.mu, self.sigma, self.C
        )


def build_prior_spec(priors: Sequence[Prior]) -> PriorSpec:
    """Compile a sequence of Prior instances into a traced JAX PriorSpec PyTree."""
    kinds: list[int] = []
    vmins: list[float] = []
    vmaxs: list[float] = []
    mus: list[float] = []
    sigmas: list[float] = []

    for p in priors:
        if isinstance(p, UniformPrior):
            kinds.append(PRIOR_UNIFORM)
            vmins.append(p.vmin)
            vmaxs.append(p.vmax)
            mus.append(0.0)
            sigmas.append(1.0)
        elif isinstance(p, NormalPrior):
            kinds.append(PRIOR_NORMAL)
            vmins.append(p.vmin)
            vmaxs.append(p.vmax)
            mus.append(p.mu)
            sigmas.append(p.sigma)
        elif isinstance(p, LogNormPrior):
            kinds.append(PRIOR_LOGNORM)
            vmins.append(p.vmin)
            vmaxs.append(p.vmax)
            mus.append(p.mu)
            sigmas.append(p.sigma)
        else:
            raise TypeError(f"Unsupported prior type: {type(p)!r}")

    return PriorSpec(
        kind=jnp.asarray(kinds, dtype=jnp.int32),
        vmin=jnp.asarray(vmins, dtype=jnp.float64),
        vmax=jnp.asarray(vmaxs, dtype=jnp.float64),
        mu=jnp.asarray(mus, dtype=jnp.float64),
        sigma=jnp.asarray(sigmas, dtype=jnp.float64),
    )


@jax.jit
def _eval_ln_prior_core(pv_arr: jax.Array, prior_spec: PriorSpec) -> jax.Array:
    kind = prior_spec.kind
    vmin = prior_spec.vmin
    vmax = prior_spec.vmax
    mu = prior_spec.mu
    sigma = jnp.where(prior_spec.sigma > 0.0, prior_spec.sigma, 1.0)

    span = vmax - vmin
    safe_span = jnp.where((kind == PRIOR_UNIFORM) & (span > 0.0) & jnp.isfinite(span), span, 1.0)
    u_val = -jnp.log(safe_span)

    safe_norm_x = jnp.where(jnp.isfinite(pv_arr), pv_arr, mu)
    log_norm_const = -jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
    n_val = log_norm_const - 0.5 * jnp.square((safe_norm_x - mu) / sigma)

    is_pos_finite = (pv_arr > 0.0) & jnp.isfinite(pv_arr)
    safe_pos_x = jnp.where(is_pos_finite, pv_arr, jnp.ones_like(pv_arr))
    lnx = jnp.log(safe_pos_x)
    ln_val = -lnx + log_norm_const - 0.5 * jnp.square((lnx - mu) / sigma)

    val = jnp.select(
        [kind == PRIOR_UNIFORM, kind == PRIOR_NORMAL, kind == PRIOR_LOGNORM],
        [u_val, n_val, ln_val],
        default=-jnp.inf,
    )
    in_bounds = jnp.select(
        [kind == PRIOR_UNIFORM, kind == PRIOR_NORMAL, kind == PRIOR_LOGNORM],
        [
            (pv_arr >= vmin) & (pv_arr <= vmax) & jnp.isfinite(pv_arr),
            (pv_arr >= vmin) & (pv_arr <= vmax) & jnp.isfinite(pv_arr),
            is_pos_finite & (pv_arr > vmin) & (pv_arr <= vmax),
        ],
        default=False,
    )
    return jnp.where(jnp.all(in_bounds), jnp.sum(val), -jnp.inf)


def eval_ln_prior_jax(pv: jax.Array, prior_spec: PriorSpec) -> jax.Array:
    """Evaluate total log-prior density in pure branchless JAX (autodiff-safe everywhere)."""
    return _eval_ln_prior_core(as_f64(pv), prior_spec)
