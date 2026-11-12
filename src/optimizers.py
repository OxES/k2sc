"""Global and gradient-based JAX/SciPy/Optimistix hyperparameter optimizers for K2SC."""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from scipy.stats import qmc, truncnorm

from ._jax_env import as_f64, ensure_float64
from .gp import (
    PrecomputedPairwise,
    _eval_gp_lnlike_and_grad_precomputed_raw,
    _eval_gp_lnlike_precomputed_raw,
    eval_gp_lnlike_jax,
    precompute_pairwise_jax,
)
from .kernels import (
    KERNEL_PERIODIC,
    KERNEL_QUASIPERIODIC,
    KERNEL_QUASIPERIODIC_EP,
    DtKernel,
    pad_splits,
)
from .priors import (
    LogNormPrior,
    NormalPrior,
    PriorSpec,
    UniformPrior,
    eval_ln_prior_jax,
)

ensure_float64()

_PERIODIC_KERNEL_IDS = (
    KERNEL_QUASIPERIODIC,
    KERNEL_QUASIPERIODIC_EP,
    KERNEL_PERIODIC,
)
_VMAP_CHUNK_SIZE = 64
_DE_HAS_RNG_KWARG = "rng" in inspect.signature(differential_evolution).parameters


@dataclass
class OptimizationResult:
    """Structured result from GP hyperparameter optimization."""

    pv: np.ndarray
    fun: float
    grad_norm: float
    success: bool
    nfev: int
    njev: int
    runtime_sec: float
    method: str
    message: str


def _sanitize_pv_bounds(
    pv: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    pv_arr = as_f64(pv)
    low_arr = as_f64(lims_low)
    high_arr = as_f64(lims_high)

    in_bounds = jnp.all(
        jnp.isfinite(pv_arr) & (pv_arr > low_arr) & (pv_arr <= high_arr)
    )

    # Well-conditioned interior fallback when pv is out-of-bounds or non-finite,
    # avoiding NaN cotangents in the unselected branch of jnp.where under grad/vmap.
    both_fin = jnp.isfinite(low_arr) & jnp.isfinite(high_arr)
    fallback = jnp.where(
        both_fin,
        0.5 * (low_arr + high_arr),
        jnp.where(
            jnp.isfinite(low_arr),
            low_arr + 1.0,
            jnp.where(jnp.isfinite(high_arr), high_arr - 1.0, 0.0),
        ),
    )
    pv_eval = jnp.where(in_bounds, pv_arr, fallback)
    safe_pv = jnp.clip(pv_eval, low_arr + 1e-12, high_arr - 1e-12)
    return in_bounds, pv_eval, safe_pv, low_arr, high_arr


def _neglnposterior_impl(
    pv: jax.Array,
    X_fit: jax.Array,
    y_fit: jax.Array,
    splits_padded: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> jax.Array:
    in_bounds, _, safe_pv, _, _ = _sanitize_pv_bounds(pv, lims_low, lims_high)
    ln_p = eval_ln_prior_jax(safe_pv, prior_spec)
    ln_l = eval_gp_lnlike_jax(safe_pv, X_fit, y_fit, splits_padded, kernel_id)
    raw_nll = -(ln_p + ln_l)
    return jnp.where(in_bounds & jnp.isfinite(raw_nll), raw_nll, jnp.inf)


_jit_neglnposterior = functools.partial(jax.jit, static_argnames=("kernel_id",))(
    _neglnposterior_impl
)


def _neglnposterior_precomputed_impl(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> jax.Array:
    in_bounds, _, safe_pv, _, _ = _sanitize_pv_bounds(pv, lims_low, lims_high)
    ln_p = eval_ln_prior_jax(safe_pv, prior_spec)
    ln_l = _eval_gp_lnlike_precomputed_raw(safe_pv, precomp, y_fit, kernel_id)
    raw_nll = -(ln_p + ln_l)
    return jnp.where(in_bounds & jnp.isfinite(raw_nll), raw_nll, jnp.inf)


_jit_neglnposterior_precomputed = functools.partial(
    jax.jit, static_argnames=("kernel_id",)
)(_neglnposterior_precomputed_impl)


def _val_and_grad_precomputed_impl(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> tuple[jax.Array, jax.Array]:
    in_bounds, pv_eval, safe_pv, low_arr, high_arr = _sanitize_pv_bounds(
        pv, lims_low, lims_high
    )
    ln_p, grad_ln_p = jax.value_and_grad(eval_ln_prior_jax)(safe_pv, prior_spec)
    ln_l, grad_ln_l = _eval_gp_lnlike_and_grad_precomputed_raw(
        safe_pv, precomp, y_fit, kernel_id
    )
    raw_nll = -(ln_p + ln_l)
    clip_active = (pv_eval >= low_arr + 1e-12) & (pv_eval <= high_arr - 1e-12)
    raw_grad = jnp.where(clip_active, -(grad_ln_p + grad_ln_l), 0.0)
    ok = in_bounds & jnp.isfinite(raw_nll) & jnp.all(jnp.isfinite(raw_grad))
    val = jnp.where(ok, raw_nll, jnp.inf)
    safe_grad = jnp.where(ok, raw_grad, jnp.zeros_like(raw_grad))
    return val, safe_grad


_jit_val_and_grad_precomputed = functools.partial(
    jax.jit, static_argnames=("kernel_id",)
)(_val_and_grad_precomputed_impl)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_val_and_grad(
    pv: jax.Array,
    X_fit: jax.Array,
    y_fit: jax.Array,
    splits_padded: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> tuple[jax.Array, jax.Array]:
    precomp = precompute_pairwise_jax(X_fit, splits_padded)
    return _val_and_grad_precomputed_impl(
        pv, precomp, y_fit, lims_low, lims_high, prior_spec, kernel_id
    )


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_vmap_neglnposterior(
    pv_batch: jax.Array,
    X_fit: jax.Array,
    y_fit: jax.Array,
    splits_padded: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> jax.Array:
    precomp = precompute_pairwise_jax(X_fit, splits_padded)
    return jax.vmap(
        _neglnposterior_precomputed_impl,
        in_axes=(0, None, None, None, None, None, None),
    )(pv_batch, precomp, y_fit, lims_low, lims_high, prior_spec, kernel_id)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_vmap_neglnposterior_precomputed(
    pv_batch: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> jax.Array:
    return jax.vmap(
        _neglnposterior_precomputed_impl,
        in_axes=(0, None, None, None, None, None, None),
    )(pv_batch, precomp, y_fit, lims_low, lims_high, prior_spec, kernel_id)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_map_neglnposterior_precomputed(
    pv_batch: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    lims_low: jax.Array,
    lims_high: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
) -> jax.Array:
    return jax.lax.map(
        lambda pv: _neglnposterior_precomputed_impl(
            pv, precomp, y_fit, lims_low, lims_high, prior_spec, kernel_id
        ),
        pv_batch,
    )


def _parse_lims_arrays(lims) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(lims, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D lims array, got shape {arr.shape}.")
    if arr.shape[0] == 2 and (arr.shape[1] != 2 or np.all(arr[0] <= arr[1])):
        return arr[0].copy(), arr[1].copy()
    if arr.shape[1] == 2:
        return arr[:, 0].copy(), arr[:, 1].copy()
    raise ValueError(f"Expected lims of shape (2, npar) or (npar, 2), got {arr.shape}.")


def compute_closed_lims(lims) -> list[tuple[float, float]]:
    """Construct closed interior box bounds [(a_k + eps_k, b_k - eps_k)] from prior limits."""
    low, high = _parse_lims_arrays(lims)
    closed: list[tuple[float, float]] = []
    for a_k, b_k in zip(low, high, strict=True):
        if np.isfinite(a_k) and np.isfinite(b_k):
            span = float(b_k - a_k)
            if span <= 0.0:
                raise ValueError(f"Invalid parameter limits ({a_k}, {b_k}).")
            eps_k = 1e-7 * min(1.0, span)
            closed.append((float(a_k + eps_k), float(b_k - eps_k)))
        elif np.isfinite(a_k):
            closed.append((float(a_k + 1e-7), float(np.inf)))
        elif np.isfinite(b_k):
            closed.append((float(-np.inf), float(b_k - 1e-7)))
        else:
            closed.append((float(-np.inf), float(np.inf)))
    return closed


def _finite_box_bounds(
    closed_lims: list[tuple[float, float]],
    kernel: DtKernel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite (low, high) arrays for sampling / DE when any prior bound is +/-inf."""
    c_arr = np.asarray(closed_lims, dtype=np.float64)
    low = c_arr[:, 0].copy()
    high = c_arr[:, 1].copy()
    k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
    for k in range(low.size):
        if not np.isfinite(low[k]):
            low[k] = float(k_bounds[k, 0] - 10.0) if k < k_bounds.shape[0] else -10.0
        if not np.isfinite(high[k]):
            high[k] = float(k_bounds[k, 1] + 10.0) if k < k_bounds.shape[0] else 10.0
    return low, high


def _eval_candidates_chunked(
    candidates: np.ndarray,
    X_fit_jax: jax.Array,
    y_fit_jax: jax.Array,
    splits_padded: jax.Array,
    lims_low_jax: jax.Array,
    lims_high_jax: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
    chunk_size: int = _VMAP_CHUNK_SIZE,
    precomp: PrecomputedPairwise | None = None,
) -> np.ndarray:
    """Evaluate (M, npar) candidate parameter vectors in static-shape chunks (lax.map on CPU, vmap on GPU/TPU)."""
    cand_arr = np.asarray(candidates, dtype=np.float64)
    n_total, _ = cand_arr.shape
    if n_total == 0:
        return np.empty(0, dtype=np.float64)

    pc = (
        precompute_pairwise_jax(X_fit_jax, splits_padded)
        if precomp is None
        else precomp
    )
    if n_total == 1:
        val_single = _jit_neglnposterior_precomputed(
            as_f64(cand_arr[0]),
            pc,
            y_fit_jax,
            lims_low_jax,
            lims_high_jax,
            prior_spec,
            kernel_id,
        )
        return np.array([float(val_single)], dtype=np.float64)

    chunk = max(1, int(chunk_size))
    # On CPU, lax.map keeps each (N, N) covariance matrix inside L2 cache
    # (~3.9 MB at N=700) and calls single-matrix LAPACK dpotrf directly,
    # while static chunk shapes compile at most once per kernel_id.
    batch_fn = (
        _jit_map_neglnposterior_precomputed
        if jax.default_backend() == "cpu"
        else _jit_vmap_neglnposterior_precomputed
    )
    out = np.empty(n_total, dtype=np.float64)
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        batch = cand_arr[start:end]
        rem = end - start
        if rem < chunk:
            pad_rows = np.repeat(batch[:1], chunk - rem, axis=0)
            batch = np.concatenate([batch, pad_rows], axis=0)
        vals = batch_fn(
            as_f64(batch),
            pc,
            y_fit_jax,
            lims_low_jax,
            lims_high_jax,
            prior_spec,
            kernel_id,
        )
        out[start:end] = np.asarray(vals, dtype=np.float64)[:rem]
    return out


def _sample_truncated_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    u = rng.uniform(1e-9, 1.0 - 1e-9, size=size)
    samples = truncnorm.ppf(u, a, b, loc=mu, scale=sigma)
    if not np.all(np.isfinite(samples)):
        # Extreme tail fallback
        samples = np.where(np.isfinite(samples), samples, np.clip(mu, low, high))
    return np.clip(np.asarray(samples, dtype=np.float64), low, high)


def _generate_candidates(
    kernel: DtKernel,
    closed_lims: list[tuple[float, float]],
    rng: np.random.Generator,
    n_candidates: int = 256,
    pv0: np.ndarray | None = None,
    ls_min_period: float = 0.05,
    ls_max_period: float = 25.0,
    y_var: float | None = None,
) -> np.ndarray:
    """Deterministic Sobol + prior/harmonic-stratified candidate design."""
    npar = kernel.npar
    c_low, c_high = _finite_box_bounds(closed_lims, kernel)

    base_pv0 = np.asarray(kernel._pv if pv0 is None else pv0, dtype=np.float64).ravel()
    pv0_clipped = np.clip(base_pv0, c_low, c_high)

    if n_candidates <= 1:
        return pv0_clipped[np.newaxis, :]

    n_rem = n_candidates - 1
    n_sobol = round(0.70 * n_rem)
    n_prior = n_rem - n_sobol

    k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
    b_low = np.clip(k_bounds[:, 0], c_low, c_high)
    b_high = np.clip(k_bounds[:, 1], c_low, c_high)
    bad_box = b_high <= b_low
    b_low = np.where(bad_box, c_low, b_low)
    b_high = np.where(bad_box, c_high, b_high)

    parts = [pv0_clipped[np.newaxis, :]]

    if n_sobol > 0:
        m_pow2 = int(np.ceil(np.log2(max(1, n_sobol))))
        sobol = qmc.Sobol(d=npar, scramble=True, seed=rng)
        u_sobol = sobol.random_base2(m=m_pow2)[:n_sobol]
        sobol_pts = np.clip(b_low + u_sobol * (b_high - b_low), c_low, c_high)
        if kernel.kernel_id in _PERIODIC_KERNEL_IDS:
            p_lo_s = max(float(ls_min_period), float(b_low[2]), float(c_low[2]))
            p_hi_s = min(max(p_lo_s * 1.5, float(b_high[2])), float(c_high[2]))
            if p_hi_s > p_lo_s:
                sobol_pts[:, 2] = np.clip(
                    np.exp(np.log(p_lo_s) + u_sobol[:, 2] * (np.log(p_hi_s) - np.log(p_lo_s))),
                    c_low[2],
                    c_high[2],
                )
        if kernel.kernel_id in (KERNEL_QUASIPERIODIC, KERNEL_QUASIPERIODIC_EP):
            # Map Sobol coordinate 3 log-uniformly across [max(1e-4, c_low[3]), b_high[3]]
            # so multi-period spot evolution timescales (tau_evol >= P) are well sampled.
            ev_lo = max(1e-4, float(c_low[3]))
            ev_hi = max(ev_lo * 10.0, float(b_high[3]))
            sobol_pts[:, 3] = np.clip(
                np.exp(np.log(ev_lo) + u_sobol[:, 3] * (np.log(ev_hi) - np.log(ev_lo))),
                c_low[3],
                c_high[3],
            )
        parts.append(sobol_pts)

    if n_prior > 0:
        prior_pts = np.empty((n_prior, npar), dtype=np.float64)
        is_periodic_kernel = kernel.kernel_id in _PERIODIC_KERNEL_IDS
        is_qp_kernel = kernel.kernel_id in (KERNEL_QUASIPERIODIC, KERNEL_QUASIPERIODIC_EP)
        log10_yvar = (
            float(np.clip(np.log10(max(y_var, 1e-10)), -6.5, -1.0))
            if (y_var is not None and np.isfinite(y_var) and y_var > 0.0)
            else -5.0
        )

        # Draw period column first when periodic so evolution scale can couple coherently
        if is_periodic_kernel:
            lo_p = float(c_low[2])
            hi_p = float(c_high[2])
            p_seed = float(base_pv0[2])
            if not np.isfinite(p_seed) or p_seed <= 0.0:
                p_seed = float(getattr(kernel, "period", 10.0))
            p_min = max(float(ls_min_period), lo_p)
            p_max = min(float(ls_max_period), hi_p)
            if p_max <= p_min:
                p_min, p_max = lo_p, hi_p

            n1 = round(0.50 * n_prior)
            n2 = round(0.20 * n_prior)
            n3 = round(0.20 * n_prior)
            n4 = max(0, n_prior - n1 - n2 - n3)
            while n1 + n2 + n3 + n4 > n_prior:
                if n1 > 1:
                    n1 -= 1
                elif n2 > 0:
                    n2 -= 1
                elif n3 > 0:
                    n3 -= 1
                else:
                    break

            p_draws = []
            if n1 > 0:
                p_draws.append(
                    np.clip(
                        rng.normal(p_seed, max(1e-4, 0.08 * p_seed), size=n1),
                        lo_p,
                        hi_p,
                    )
                )
            if n2 > 0:
                p_draws.append(
                    np.clip(
                        rng.normal(0.5 * p_seed, max(1e-4, 0.05 * p_seed), size=n2),
                        lo_p,
                        hi_p,
                    )
                )
            if n3 > 0:
                p_draws.append(
                    np.clip(
                        rng.normal(2.0 * p_seed, max(1e-4, 0.10 * p_seed), size=n3),
                        lo_p,
                        hi_p,
                    )
                )
            if n4 > 0:
                log_p = rng.uniform(np.log(p_min), np.log(p_max), size=n4)
                p_draws.append(np.clip(np.exp(log_p), lo_p, hi_p))

            p_col = np.concatenate(p_draws, axis=0)
            rng.shuffle(p_col)
            prior_pts[:, 2] = p_col

        for k, prior in enumerate(kernel.priors):
            if is_periodic_kernel and k == 2:
                continue
            lo_k = float(c_low[k])
            hi_k = float(c_high[k])

            if k == npar - 1:
                # Stratify white-noise log10(sigma_n) across [-4.5, -2.5] so lengthscale
                # gradients remain informative during initial screening and polishing.
                wn_lo = max(-4.5, lo_k)
                wn_hi = min(-2.5, hi_k)
                if wn_hi <= wn_lo:
                    wn_lo, wn_hi = lo_k, hi_k
                u_strat = (np.arange(n_prior, dtype=np.float64) + rng.random(n_prior)) / n_prior
                rng.shuffle(u_strat)
                prior_pts[:, k] = np.clip(wn_lo + u_strat * (wn_hi - wn_lo), lo_k, hi_k)
            elif is_qp_kernel and k == 3:
                # Ensure coherent multi-cycle evolution timescales (tau_evol in [1.5*P, 8*P])
                # so cycle-to-cycle ExpSineSquared correlations are not damped out at lag P.
                n_coh = round(0.75 * n_prior)
                n_broad = n_prior - n_coh
                tau_coh = prior_pts[:n_coh, 2] * np.exp(rng.uniform(np.log(1.5), np.log(8.0), size=n_coh))
                ev_coh = np.clip(1.0 / np.square(np.maximum(tau_coh, 0.5)), lo_k, hi_k)
                ln_lo = np.log(max(lo_k, 1e-12))
                ln_hi = np.log(max(hi_k, lo_k + 1e-9))
                ev_broad = np.clip(
                    np.exp(_sample_truncated_normal(rng, prior.mu, prior.sigma, ln_lo, ln_hi, n_broad)),
                    lo_k,
                    hi_k,
                )
                prior_pts[:, k] = np.concatenate([ev_coh, ev_broad], axis=0)
            elif isinstance(prior, NormalPrior):
                prior_pts[:, k] = _sample_truncated_normal(
                    rng, prior.mu, prior.sigma, lo_k, hi_k, n_prior
                )
            elif isinstance(prior, LogNormPrior):
                ln_lo = np.log(max(lo_k, 1e-12))
                ln_hi = np.log(max(hi_k, lo_k + 1e-9))
                ln_samples = _sample_truncated_normal(
                    rng, prior.mu, prior.sigma, ln_lo, ln_hi, n_prior
                )
                prior_pts[:, k] = np.clip(np.exp(ln_samples), lo_k, hi_k)
            elif isinstance(prior, UniformPrior):
                # Mix empirical variance-anchored draws with broad uniform box draws
                n_emp = n_prior // 2
                n_uni = n_prior - n_emp
                emp_draws = np.clip(
                    rng.normal(log10_yvar, 1.0, size=n_emp),
                    lo_k,
                    hi_k,
                )
                uni_draws = rng.uniform(b_low[k], b_high[k], size=n_uni)
                col = np.concatenate([emp_draws, uni_draws], axis=0)
                rng.shuffle(col)
                prior_pts[:, k] = np.clip(col, lo_k, hi_k)
            else:
                prior_pts[:, k] = rng.uniform(lo_k, hi_k, size=n_prior)

        if is_periodic_kernel and n_prior >= 3:
            # Guarantee explicit well-conditioned harmonic anchors at {P, 0.5*P, 2*P}
            p_seed_anchor = float(base_pv0[2])
            if not np.isfinite(p_seed_anchor) or p_seed_anchor <= 0.0:
                p_seed_anchor = float(getattr(kernel, "period", 10.0))
            for h_idx, mult in enumerate((1.0, 0.5, 2.0)):
                anchor = pv0_clipped.copy()
                p_h = float(np.clip(mult * p_seed_anchor, c_low[2], c_high[2]))
                anchor[0] = float(np.clip(log10_yvar, c_low[0], c_high[0]))
                anchor[1] = float(np.clip(0.35, c_low[1], c_high[1]))
                anchor[2] = p_h
                if is_qp_kernel:
                    tau_h = max(3.0 * p_h, 1.0)
                    anchor[3] = float(np.clip(1.0 / (tau_h * tau_h), c_low[3], c_high[3]))
                    xy_idx = 4
                else:
                    # PeriodicKernel: long-term smooth amplitude & inverse scale at indices 3, 4
                    anchor[3] = float(np.clip(log10_yvar - 0.5, c_low[3], c_high[3]))
                    anchor[4] = float(np.clip(0.01, c_low[4], c_high[4]))
                    xy_idx = 5
                anchor[xy_idx] = float(np.clip(log10_yvar, c_low[xy_idx], c_high[xy_idx]))
                anchor[-1] = float(np.clip(-3.8, c_low[-1], c_high[-1]))
                prior_pts[h_idx] = anchor

        parts.append(prior_pts)

    return np.concatenate(parts, axis=0)


def _select_diverse_top_k(
    candidates: np.ndarray,
    energies: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    top_k: int = 5,
    min_dist: float = 0.15,
    is_periodic: bool = False,
) -> np.ndarray:
    """Greedy normalized-box distance basin selector with period-harmonic diversity."""
    cand_arr = np.asarray(candidates, dtype=np.float64)
    en_arr = np.asarray(energies, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(en_arr) & (en_arr < 1e29))
    if finite_idx.size == 0:
        return cand_arr[: min(max(1, top_k), cand_arr.shape[0])].copy()

    order = finite_idx[np.argsort(en_arr[finite_idx], kind="mergesort")]
    span = np.maximum(norm_high - norm_low, 1e-6)
    p_norm = (cand_arr - norm_low) / span

    target_k = min(max(1, int(top_k)), order.size)
    selected: list[int] = []

    if is_periodic and cand_arr.shape[1] > 2:
        # Pass 1: retain distinct log-period basins (e.g. P, P/2, 2P) first
        log_p = np.log(np.maximum(cand_arr[:, 2], 1e-6))
        for idx_np in order:
            idx = int(idx_np)
            if all(
                abs(float(log_p[idx] - log_p[s])) > 0.18
                and float(np.linalg.norm(p_norm[idx] - p_norm[s])) > min_dist
                for s in selected
            ):
                selected.append(idx)
                if len(selected) == target_k:
                    break

    if len(selected) < target_k:
        selected_set = set(selected)
        for idx_np in order:
            idx = int(idx_np)
            if idx not in selected_set and all(
                float(np.linalg.norm(p_norm[idx] - p_norm[s])) > min_dist
                for s in selected
            ):
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == target_k:
                    break

    if len(selected) < target_k:
        selected_set = set(selected)
        for idx_np in order:
            idx = int(idx_np)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == target_k:
                    break

    return cand_arr[selected].copy()


def _make_scipy_host_callback(
    X_fit_jax: jax.Array,
    y_fit_jax: jax.Array,
    splits_padded: jax.Array,
    lims_low_jax: jax.Array,
    lims_high_jax: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
    precomp: PrecomputedPairwise | None = None,
):
    """Build host value-and-gradient callback sanitizing non-finite values to (1e30, 0)."""
    pc = (
        precompute_pairwise_jax(X_fit_jax, splits_padded)
        if precomp is None
        else precomp
    )
    counts = {"nfev": 0, "njev": 0}
    last_eval: dict[str, Any] = {"x": None, "val": float("nan"), "grad": None}

    def _eval_exact(x_arr: np.ndarray) -> tuple[float, np.ndarray]:
        if last_eval["x"] is not None and np.array_equal(x_arr, last_eval["x"]):
            return float(last_eval["val"]), last_eval["grad"].copy()
        counts["nfev"] += 1
        counts["njev"] += 1
        if not np.all(np.isfinite(x_arr)):
            return float("inf"), np.zeros_like(x_arr, dtype=np.float64)
        val_j, grad_j = _jit_val_and_grad_precomputed(
            as_f64(x_arr),
            pc,
            y_fit_jax,
            lims_low_jax,
            lims_high_jax,
            prior_spec,
            kernel_id,
        )
        val = float(val_j)
        grad = np.asarray(grad_j, dtype=np.float64)
        last_eval["x"] = x_arr.copy()
        last_eval["val"] = val
        last_eval["grad"] = grad.copy()
        return val, grad

    def _val_and_grad_host(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        val, grad = _eval_exact(x_arr)
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e30, np.zeros_like(x_arr, dtype=np.float64)
        return val, grad

    _val_and_grad_host.eval_exact = _eval_exact  # type: ignore[attr-defined]
    return _val_and_grad_host, counts


def polish_lbfgsb(
    x0: np.ndarray,
    X_fit_jax: jax.Array,
    y_fit_jax: jax.Array,
    splits_padded: jax.Array,
    lims_low_jax: jax.Array,
    lims_high_jax: jax.Array,
    prior_spec: PriorSpec,
    kernel_id: int,
    closed_lims: list[tuple[float, float]],
    ftol: float = 1e-9,
    gtol: float = 1e-6,
    maxiter: int = 200,
    precomp: PrecomputedPairwise | None = None,
) -> OptimizationResult:
    """Run bounded exact-gradient SciPy L-BFGS-B polish from a starting point x0."""
    t0 = time.perf_counter()
    c_arr = np.asarray(closed_lims, dtype=np.float64)
    x0_clipped = np.clip(np.asarray(x0, dtype=np.float64), c_arr[:, 0], c_arr[:, 1])

    pc = (
        precompute_pairwise_jax(X_fit_jax, splits_padded)
        if precomp is None
        else precomp
    )
    host_vg, counts = _make_scipy_host_callback(
        X_fit_jax,
        y_fit_jax,
        splits_padded,
        lims_low_jax,
        lims_high_jax,
        prior_spec,
        kernel_id,
        precomp=pc,
    )

    init_val, init_grad = host_vg.eval_exact(x0_clipped)  # type: ignore[attr-defined]

    res = minimize(
        host_vg,
        x0_clipped,
        method="L-BFGS-B",
        jac=True,
        bounds=closed_lims,
        options={"ftol": ftol, "gtol": gtol, "maxiter": maxiter},
    )

    cand_x = np.clip(np.asarray(res.x, dtype=np.float64), c_arr[:, 0], c_arr[:, 1])
    fin_val, fin_grad = host_vg.eval_exact(cand_x)  # type: ignore[attr-defined]

    if np.isfinite(init_val) and (not np.isfinite(fin_val) or init_val < fin_val):
        best_x = x0_clipped
        best_val = init_val
        best_grad = init_grad
    else:
        best_x = cand_x
        best_val = fin_val
        best_grad = fin_grad

    grad_norm = _box_projected_grad_norm(best_x, best_grad, c_arr[:, 0], c_arr[:, 1])
    msg = res.message if isinstance(res.message, str) else str(res.message)
    success = bool(np.isfinite(best_val) and (res.success or grad_norm < 1e-2))

    return OptimizationResult(
        pv=best_x.copy(),
        fun=best_val,
        grad_norm=grad_norm,
        success=success,
        nfev=counts["nfev"],
        njev=counts["njev"],
        runtime_sec=time.perf_counter() - t0,
        method="L-BFGS-B",
        message=msg,
    )


def _box_projected_grad_norm(
    pv: np.ndarray,
    grad: np.ndarray,
    c_low: np.ndarray,
    c_high: np.ndarray,
) -> float:
    if not np.all(np.isfinite(grad)):
        return float("inf")
    proj = pv - np.clip(pv - grad, c_low, c_high)
    return float(np.linalg.norm(proj))


class MultiStartLBFGSBOptimizer:
    """Sobol + harmonic-stratified candidate screening with multi-start L-BFGS-B polishing."""

    name = "multistart"

    def __init__(
        self,
        n_candidates: int = 256,
        top_k: int = 5,
        min_dist: float = 0.15,
        maxiter: int = 200,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        ls_min_period: float = 0.05,
        ls_max_period: float = 25.0,
    ):
        self.n_candidates = int(n_candidates)
        self.top_k = int(top_k)
        self.min_dist = float(min_dist)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.ls_min_period = float(ls_min_period)
        self.ls_max_period = float(ls_max_period)

    def optimize(
        self,
        X_fit,
        y_fit,
        splits,
        kernel: DtKernel,
        pv0=None,
        seed: int | np.random.Generator | None = 0,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        X_jax = as_f64(X_fit)
        y_jax = as_f64(y_fit)
        splits_pad = pad_splits(splits)
        precomp = precompute_pairwise_jax(X_jax, splits_pad)
        lims_low_jax = as_f64(kernel.lims[0])
        lims_high_jax = as_f64(kernel.lims[1])
        closed_lims = compute_closed_lims(kernel.lims)
        c_low, c_high = _finite_box_bounds(closed_lims, kernel)

        y_np = np.asarray(y_fit, dtype=np.float64)
        y_var = float(np.nanvar(y_np)) if y_np.size > 0 else None
        is_per = kernel.kernel_id in _PERIODIC_KERNEL_IDS

        candidates = _generate_candidates(
            kernel,
            closed_lims,
            rng,
            n_candidates=max(1, self.n_candidates),
            pv0=pv0,
            ls_min_period=self.ls_min_period,
            ls_max_period=self.ls_max_period,
            y_var=y_var,
        )
        energies = _eval_candidates_chunked(
            candidates,
            X_jax,
            y_jax,
            splits_pad,
            lims_low_jax,
            lims_high_jax,
            kernel.prior_spec,
            kernel.kernel_id,
            precomp=precomp,
        )
        total_nfev = int(candidates.shape[0])
        total_njev = 0

        k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
        norm_low = np.where(np.isfinite(k_bounds[:, 0]), k_bounds[:, 0], c_low)
        norm_high = np.where(np.isfinite(k_bounds[:, 1]), k_bounds[:, 1], c_high)
        seeds = _select_diverse_top_k(
            candidates,
            energies,
            norm_low,
            norm_high,
            top_k=self.top_k,
            min_dist=self.min_dist,
            is_periodic=is_per,
        )

        best_res: OptimizationResult | None = None
        for x_seed in seeds:
            pol = polish_lbfgsb(
                x_seed,
                X_jax,
                y_jax,
                splits_pad,
                lims_low_jax,
                lims_high_jax,
                kernel.prior_spec,
                kernel.kernel_id,
                closed_lims,
                ftol=self.ftol,
                gtol=self.gtol,
                maxiter=self.maxiter,
                precomp=precomp,
            )
            total_nfev += pol.nfev
            total_njev += pol.njev
            if best_res is None or pol.fun < best_res.fun:
                best_res = pol

        assert best_res is not None
        return OptimizationResult(
            pv=best_res.pv,
            fun=best_res.fun,
            grad_norm=best_res.grad_norm,
            success=bool(np.isfinite(best_res.fun) and (best_res.success or best_res.grad_norm < 1.0)),
            nfev=total_nfev,
            njev=total_njev,
            runtime_sec=time.perf_counter() - t0,
            method=self.name,
            message=best_res.message,
        )


class VectorizedDEOptimizer:
    """Vectorized SciPy Differential Evolution backed by jax.vmap + top-3 L-BFGS-B polish."""

    name = "de"

    def __init__(
        self,
        npop: int = 60,
        niter: int = 50,
        max_time: float = 300.0,
        top_k: int = 3,
        min_dist: float = 0.15,
        maxiter_polish: int = 200,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        ls_min_period: float = 0.05,
        ls_max_period: float = 25.0,
        *,
        de_npop: int | None = None,
        de_niter: int | None = None,
        de_max_time: float | None = None,
    ):
        self.npop = max(5, int(de_npop if de_npop is not None else npop))
        self.niter = max(1, int(de_niter if de_niter is not None else niter))
        self.max_time = float(de_max_time if de_max_time is not None else max_time)
        self.top_k = max(1, int(top_k))
        self.min_dist = float(min_dist)
        self.maxiter_polish = int(maxiter_polish)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.ls_min_period = float(ls_min_period)
        self.ls_max_period = float(ls_max_period)

    def optimize(
        self,
        X_fit,
        y_fit,
        splits,
        kernel: DtKernel,
        pv0=None,
        seed: int | np.random.Generator | None = 0,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        X_jax = as_f64(X_fit)
        y_jax = as_f64(y_fit)
        splits_pad = pad_splits(splits)
        precomp = precompute_pairwise_jax(X_jax, splits_pad)
        lims_low_jax = as_f64(kernel.lims[0])
        lims_high_jax = as_f64(kernel.lims[1])
        closed_lims = compute_closed_lims(kernel.lims)
        c_low, c_high = _finite_box_bounds(closed_lims, kernel)
        finite_bounds = list(zip(c_low.tolist(), c_high.tolist(), strict=True))

        y_np = np.asarray(y_fit, dtype=np.float64)
        y_var = float(np.nanvar(y_np)) if y_np.size > 0 else None
        is_per = kernel.kernel_id in _PERIODIC_KERNEL_IDS

        seeded_pop = _generate_candidates(
            kernel,
            finite_bounds,
            rng,
            n_candidates=self.npop,
            pv0=pv0,
            ls_min_period=self.ls_min_period,
            ls_max_period=self.ls_max_period,
            y_var=y_var,
        )

        eval_state = {"nfev": 0, "gen": 0, "stop_reason": ""}
        archive_x: list[np.ndarray] = []
        archive_en: list[np.ndarray] = []
        t_de_start = time.perf_counter()

        def _de_vec_objective(x: np.ndarray):
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 1:
                eval_state["nfev"] += 1
                vals = _eval_candidates_chunked(
                    x_arr[np.newaxis, :],
                    X_jax,
                    y_jax,
                    splits_pad,
                    lims_low_jax,
                    lims_high_jax,
                    kernel.prior_spec,
                    kernel.kernel_id,
                    precomp=precomp,
                )
                v = float(vals[0])
                v_clean = v if np.isfinite(v) else 1e30
                archive_x.append(x_arr[np.newaxis, :].copy())
                archive_en.append(np.array([v_clean], dtype=np.float64))
                return v_clean
            # SciPy vectorized DE passes x with shape (npar, S)
            pv_batch = np.ascontiguousarray(x_arr.T, dtype=np.float64)
            eval_state["nfev"] += int(pv_batch.shape[0])
            vals = _eval_candidates_chunked(
                pv_batch,
                X_jax,
                y_jax,
                splits_pad,
                lims_low_jax,
                lims_high_jax,
                kernel.prior_spec,
                kernel.kernel_id,
                precomp=precomp,
            )
            vals_clean = np.where(np.isfinite(vals), vals, 1e30)
            archive_x.append(pv_batch.copy())
            archive_en.append(vals_clean.copy())
            return vals_clean

        def _de_callback(intermediate_result: OptimizeResult) -> bool:
            eval_state["gen"] += 1
            pop_energies = np.asarray(
                intermediate_result.population_energies, dtype=np.float64
            )
            elapsed = time.perf_counter() - t_de_start
            if elapsed > self.max_time:
                eval_state["stop_reason"] = "DE stopped at max_time budget."
                return True
            if (
                eval_state["gen"] > 3
                and np.all(np.isfinite(pop_energies))
                and float(np.ptp(pop_energies)) < 3.0
            ):
                eval_state["stop_reason"] = "DE converged (ptp(energies) < 3.0)."
                return True
            return False

        de_kwargs: dict[str, Any] = {
            "bounds": finite_bounds,
            "strategy": "rand1bin",
            "maxiter": self.niter,
            "vectorized": True,
            "updating": "deferred",
            "polish": False,
            "init": seeded_pop,
            "callback": _de_callback,
            "tol": 0.0,
            "atol": 0.0,
        }
        if _DE_HAS_RNG_KWARG:
            de_kwargs["rng"] = rng
        else:
            de_kwargs["seed"] = rng

        de_res = differential_evolution(_de_vec_objective, **de_kwargs)

        pop = np.asarray(de_res.population, dtype=np.float64)
        pop_en = np.asarray(de_res.population_energies, dtype=np.float64)
        pool_x = np.concatenate([pop, *archive_x], axis=0) if archive_x else pop
        pool_en = np.concatenate([pop_en, *archive_en], axis=0) if archive_en else pop_en
        k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
        norm_low = np.where(np.isfinite(k_bounds[:, 0]), k_bounds[:, 0], c_low)
        norm_high = np.where(np.isfinite(k_bounds[:, 1]), k_bounds[:, 1], c_high)
        seeds = _select_diverse_top_k(
            pool_x,
            pool_en,
            norm_low,
            norm_high,
            top_k=self.top_k,
            min_dist=self.min_dist,
            is_periodic=is_per,
        )

        total_nfev = int(eval_state["nfev"])
        total_njev = 0
        best_res: OptimizationResult | None = None
        for x_seed in seeds:
            pol = polish_lbfgsb(
                x_seed,
                X_jax,
                y_jax,
                splits_pad,
                lims_low_jax,
                lims_high_jax,
                kernel.prior_spec,
                kernel.kernel_id,
                closed_lims,
                ftol=self.ftol,
                gtol=self.gtol,
                maxiter=self.maxiter_polish,
                precomp=precomp,
            )
            total_nfev += pol.nfev
            total_njev += pol.njev
            if best_res is None or pol.fun < best_res.fun:
                best_res = pol

        assert best_res is not None
        msg = eval_state["stop_reason"] or str(de_res.message)
        return OptimizationResult(
            pv=best_res.pv,
            fun=best_res.fun,
            grad_norm=best_res.grad_norm,
            success=bool(np.isfinite(best_res.fun) and (best_res.success or best_res.grad_norm < 1.0)),
            nfev=total_nfev,
            njev=total_njev,
            runtime_sec=time.perf_counter() - t0,
            method=self.name,
            message=msg,
        )


def _unconstrained_to_pv_jax(
    z: jax.Array,
    closed_low: jax.Array,
    closed_high: jax.Array,
) -> jax.Array:
    """Map unconstrained z in R^D to theta in closed_lims without Jacobian term."""
    both_fin = jnp.isfinite(closed_low) & jnp.isfinite(closed_high)
    only_low = jnp.isfinite(closed_low) & ~jnp.isfinite(closed_high)
    only_high = ~jnp.isfinite(closed_low) & jnp.isfinite(closed_high)

    s_low = jnp.where(jnp.isfinite(closed_low), closed_low, 0.0)
    s_high = jnp.where(jnp.isfinite(closed_high), closed_high, 1.0)
    sig = jax.nn.sigmoid(z)

    val_both = s_low + (s_high - s_low) * sig
    val_low = s_low + jax.nn.softplus(z)
    val_high = s_high - jax.nn.softplus(-z)
    return jnp.select(
        [both_fin, only_low, only_high],
        [val_both, val_low, val_high],
        default=z,
    )


def _pv_to_unconstrained_np(
    pv: np.ndarray,
    closed_low: np.ndarray,
    closed_high: np.ndarray,
) -> np.ndarray:
    """Inverse logit/softplus map from theta in closed_lims to unconstrained z."""
    theta = np.asarray(pv, dtype=np.float64)
    z = np.empty_like(theta, dtype=np.float64)
    for k in range(theta.shape[-1]):
        lo = float(closed_low[k])
        hi = float(closed_high[k])
        tk = theta[..., k]
        if np.isfinite(lo) and np.isfinite(hi):
            u = np.clip((tk - lo) / (hi - lo), 1e-7, 1.0 - 1e-7)
            z[..., k] = np.log(u / (1.0 - u))
        elif np.isfinite(lo):
            d = np.maximum(tk - lo, 1e-7)
            z[..., k] = np.where(d > 20.0, d, np.log(np.expm1(d)))
        elif np.isfinite(hi):
            d = np.maximum(hi - tk, 1e-7)
            z[..., k] = -np.where(d > 20.0, d, np.log(np.expm1(d)))
        else:
            z[..., k] = tk
    return z


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _lnlike_precomputed_optx(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    kernel_id: int,
) -> jax.Array:
    return _eval_gp_lnlike_precomputed_raw(pv, precomp, y_fit, kernel_id)


def _lnlike_precomputed_optx_fwd(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    kernel_id: int,
):
    val, grad_pv = _eval_gp_lnlike_and_grad_precomputed_raw(
        pv, precomp, y_fit, kernel_id
    )
    return val, grad_pv


def _lnlike_precomputed_optx_bwd(kernel_id: int, grad_pv: jax.Array, g: jax.Array):
    del kernel_id
    return (g * grad_pv, None, None)


_lnlike_precomputed_optx.defvjp(
    _lnlike_precomputed_optx_fwd, _lnlike_precomputed_optx_bwd
)


_OPTX_VMAP_SOLVER_CACHE: dict[tuple[int, int, float, float], Any] = {}


def _get_optx_batch_solver(
    kernel_id: int,
    max_steps: int,
    rtol: float,
    atol: float,
):
    key = (int(kernel_id), int(max_steps), float(rtol), float(atol))
    cached = _OPTX_VMAP_SOLVER_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import optimistix as optx
    except ImportError as exc:
        raise ImportError(
            "OptimistixOptimizer requires optional dependency 'optimistix'. "
            'Install it via `pip install "k2sc[optimistix]"`.'
        ) from exc

    solver = optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol))

    def _loss_z(z: jax.Array, args) -> jax.Array:
        precomp, y_fit, lims_low, lims_high, c_low, c_high, prior_spec = args
        pv = _unconstrained_to_pv_jax(z, c_low, c_high)
        in_bounds, _, safe_pv, _, _ = _sanitize_pv_bounds(pv, lims_low, lims_high)
        ln_p = eval_ln_prior_jax(safe_pv, prior_spec)
        ln_l = _lnlike_precomputed_optx(safe_pv, precomp, y_fit, kernel_id)
        raw_nll = -(ln_p + ln_l)
        val = jnp.where(in_bounds & jnp.isfinite(raw_nll), raw_nll, jnp.inf)
        # Arrest unconstrained z runaway beyond |z| > 14 (where sigmoid saturates within
        # <1e-6 of closed_lims) while remaining identically 0 throughout |z| <= 14.
        both_fin = jnp.isfinite(c_low) & jnp.isfinite(c_high)
        z_excess = jnp.where(both_fin, jnp.maximum(0.0, jnp.abs(z) - 14.0), 0.0)
        wall = 1e-8 * jnp.sum(jnp.square(z_excess))
        return jnp.where(jnp.isfinite(val), val + wall, jnp.asarray(1e30, dtype=jnp.float64))

    def _solve_one(z0: jax.Array, args):
        sol = optx.minimise(
            _loss_z,
            solver,
            z0,
            args=args,
            max_steps=max_steps,
            throw=False,
        )
        c_low, c_high = args[4], args[5]
        pv_opt = _unconstrained_to_pv_jax(sol.value, c_low, c_high)
        is_ok = sol.result == optx.RESULTS.successful
        num_steps = sol.stats.get("num_steps", jnp.asarray(0, dtype=jnp.int32))
        return pv_opt, is_ok, num_steps

    @jax.jit
    def _solve_batch(
        z0_batch: jax.Array,
        precomp: PrecomputedPairwise,
        y_fit: jax.Array,
        lims_low: jax.Array,
        lims_high: jax.Array,
        c_low: jax.Array,
        c_high: jax.Array,
        prior_spec: PriorSpec,
    ):
        args = (precomp, y_fit, lims_low, lims_high, c_low, c_high, prior_spec)
        return jax.vmap(_solve_one, in_axes=(0, None))(z0_batch, args)

    _OPTX_VMAP_SOLVER_CACHE[key] = _solve_batch
    return _solve_batch


class OptimistixOptimizer:
    """Pure-JAX optimistix.BFGS optimizer over unconstrained logit/sigmoid coordinates."""

    name = "optimistix"

    def __init__(
        self,
        n_candidates: int = 256,
        top_k: int = 5,
        min_dist: float = 0.15,
        max_steps: int = 250,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        ls_min_period: float = 0.05,
        ls_max_period: float = 25.0,
    ):
        # Validate optimistix availability eagerly when constructed.
        try:
            import optimistix as _optx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OptimistixOptimizer requires optional dependency 'optimistix'. "
                'Install it via `pip install "k2sc[optimistix]"`.'
            ) from exc

        self.n_candidates = int(n_candidates)
        self.top_k = int(top_k)
        self.min_dist = float(min_dist)
        self.max_steps = int(max_steps)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.ls_min_period = float(ls_min_period)
        self.ls_max_period = float(ls_max_period)

    def optimize(
        self,
        X_fit,
        y_fit,
        splits,
        kernel: DtKernel,
        pv0=None,
        seed: int | np.random.Generator | None = 0,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        X_jax = as_f64(X_fit)
        y_jax = as_f64(y_fit)
        splits_pad = pad_splits(splits)
        precomp = precompute_pairwise_jax(X_jax, splits_pad)
        lims_low_jax = as_f64(kernel.lims[0])
        lims_high_jax = as_f64(kernel.lims[1])
        closed_lims = compute_closed_lims(kernel.lims)
        c_arr = np.asarray(closed_lims, dtype=np.float64)
        c_low_raw = c_arr[:, 0]
        c_high_raw = c_arr[:, 1]
        c_low_fin, c_high_fin = _finite_box_bounds(closed_lims, kernel)

        y_np = np.asarray(y_fit, dtype=np.float64)
        y_var = float(np.nanvar(y_np)) if y_np.size > 0 else None
        is_per = kernel.kernel_id in _PERIODIC_KERNEL_IDS

        candidates = _generate_candidates(
            kernel,
            closed_lims,
            rng,
            n_candidates=max(1, self.n_candidates),
            pv0=pv0,
            ls_min_period=self.ls_min_period,
            ls_max_period=self.ls_max_period,
            y_var=y_var,
        )
        energies = _eval_candidates_chunked(
            candidates,
            X_jax,
            y_jax,
            splits_pad,
            lims_low_jax,
            lims_high_jax,
            kernel.prior_spec,
            kernel.kernel_id,
            precomp=precomp,
        )

        k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
        norm_low = np.where(np.isfinite(k_bounds[:, 0]), k_bounds[:, 0], c_low_fin)
        norm_high = np.where(np.isfinite(k_bounds[:, 1]), k_bounds[:, 1], c_high_fin)
        seeds = _select_diverse_top_k(
            candidates,
            energies,
            norm_low,
            norm_high,
            top_k=self.top_k,
            min_dist=self.min_dist,
            is_periodic=is_per,
        )
        # Pad seeds to fixed batch size top_k if fewer than top_k candidates existed
        if seeds.shape[0] < self.top_k:
            pad_s = np.repeat(seeds[:1], self.top_k - seeds.shape[0], axis=0)
            seeds_batch = np.concatenate([seeds, pad_s], axis=0)
        else:
            seeds_batch = seeds

        z0_batch = _pv_to_unconstrained_np(seeds_batch, c_low_raw, c_high_raw)
        solve_batch = _get_optx_batch_solver(
            kernel.kernel_id, self.max_steps, self.rtol, self.atol
        )
        pv_opts_j, ok_flags_j, num_steps_j = solve_batch(
            as_f64(z0_batch),
            precomp,
            y_jax,
            lims_low_jax,
            lims_high_jax,
            as_f64(c_low_raw),
            as_f64(c_high_raw),
            kernel.prior_spec,
        )
        pv_opts = np.asarray(pv_opts_j, dtype=np.float64)
        ok_flags = np.asarray(ok_flags_j, dtype=bool)
        steps_total = int(np.sum(np.asarray(num_steps_j, dtype=np.int64)))

        best_pv = seeds_batch[0].copy()
        best_val = float("inf")
        best_grad = np.full_like(best_pv, np.nan)
        best_ok = False

        # Include best screened point as a safeguard
        all_check = np.concatenate([pv_opts, seeds[:1]], axis=0)
        all_ok = np.concatenate([ok_flags, np.array([False])], axis=0)
        for cand_pv, cand_ok in zip(all_check, all_ok, strict=True):
            val_j, grad_j = _jit_val_and_grad_precomputed(
                as_f64(cand_pv),
                precomp,
                y_jax,
                lims_low_jax,
                lims_high_jax,
                kernel.prior_spec,
                kernel.kernel_id,
            )
            v = float(val_j)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_pv = np.asarray(cand_pv, dtype=np.float64).copy()
                best_grad = np.asarray(grad_j, dtype=np.float64)
                best_ok = bool(cand_ok)

        grad_norm = _box_projected_grad_norm(best_pv, best_grad, c_low_raw, c_high_raw)
        success = bool(np.isfinite(best_val) and (best_ok or grad_norm < 1.0))
        msg = (
            "Optimistix BFGS converged."
            if best_ok
            else "Optimistix BFGS returned best-so-far iterate."
        )
        nfev = int(candidates.shape[0]) + steps_total + int(all_check.shape[0])
        njev = steps_total + int(all_check.shape[0])

        return OptimizationResult(
            pv=best_pv,
            fun=best_val,
            grad_norm=grad_norm,
            success=success,
            nfev=nfev,
            njev=njev,
            runtime_sec=time.perf_counter() - t0,
            method=self.name,
            message=msg,
        )


OptimistixBFGSOptimizer = OptimistixOptimizer


def get_optimizer(name: str | Any = "auto", **kwargs):
    """Instantiate a K2SC hyperparameter optimizer by name."""
    if not isinstance(name, str):
        if hasattr(name, "optimize") and callable(name.optimize):
            return name
        raise TypeError(f"Expected optimizer name str or optimizer instance, got {type(name)!r}.")

    key = name.strip().lower()
    if key == "jaxopt":
        raise ValueError(
            "Optimizer 'jaxopt' is deprecated upstream and is not supported in k2sc v2. "
            "Please use 'optimistix', 'multistart' (default), or 'de' instead."
        )
    if key in ("auto", "multistart"):
        allowed = {
            "n_candidates",
            "top_k",
            "min_dist",
            "maxiter",
            "ftol",
            "gtol",
            "ls_min_period",
            "ls_max_period",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return MultiStartLBFGSBOptimizer(**filtered)
    if key in ("de", "differential_evolution"):
        allowed = {
            "npop",
            "niter",
            "max_time",
            "top_k",
            "min_dist",
            "maxiter_polish",
            "ftol",
            "gtol",
            "ls_min_period",
            "ls_max_period",
            "de_npop",
            "de_niter",
            "de_max_time",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return VectorizedDEOptimizer(**filtered)
    if key == "optimistix":
        allowed = {
            "n_candidates",
            "top_k",
            "min_dist",
            "max_steps",
            "rtol",
            "atol",
            "ls_min_period",
            "ls_max_period",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return OptimistixOptimizer(**filtered)

    raise ValueError(
        f"Unknown optimizer {name!r}. Choose from 'auto', 'multistart', 'de', or 'optimistix'."
    )
