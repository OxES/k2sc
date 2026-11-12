"""Unified tinygp + JAX Gaussian Process solver for K2SC."""

from __future__ import annotations

import functools
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tinygp

from ._jax_env import as_f64, ensure_float64
from .kernels import (
    _WHITE_NOISE_IDX,
    KERNEL_BASIC,
    KERNEL_BASIC_EP,
    KERNEL_PERIODIC,
    KERNEL_QUASIPERIODIC,
    KERNEL_QUASIPERIODIC_EP,
    DtKernel,
    K2SCKernel,
    _normalize_kernel_id,
    _safe_abs,
    build_tinygp_kernels,
    pad_splits,
)

ensure_float64()

_JITTER_LADDER = (1e-12, 1e-10, 1e-8)
_LN10 = float(np.log(10.0))


class PrecomputedPairwise(NamedTuple):
    """Precomputed pairwise distance and split-segment tensors for a fixed training set."""

    dt: jax.Array
    dt2: jax.Array
    dx: jax.Array
    dx2: jax.Array
    dy: jax.Array
    dy2: jax.Array
    same_seg: jax.Array
    eye_n: jax.Array


class _BlockMaskedKernel(tinygp.kernels.Kernel):
    """Internal kernel wrapper decoupling masked indices into an identity block."""

    base_kernel: tinygp.kernels.Kernel

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        m = X1[-1] * X2[-1]
        val = self.base_kernel.evaluate(X1[:-1], X2[:-1])
        return jnp.where(m > 0.0, m * val, jnp.zeros_like(val))

    def evaluate_diag(self, X: jax.Array) -> jax.Array:
        m2 = X[-1] * X[-1]
        val = self.base_kernel.evaluate_diag(X[:-1])
        return jnp.where(m2 > 0.0, m2 * val, jnp.zeros_like(val))

    def __call__(self, X1: jax.Array, X2: jax.Array | None = None) -> jax.Array:
        if X2 is None:
            m2 = jnp.square(X1[:, -1])
            val = self.base_kernel(X1[:, :-1])
            return jnp.where(m2 > 0.0, m2 * val, jnp.zeros_like(val))
        m = X1[:, -1:] * X2[None, :, -1]
        val = self.base_kernel(X1[:, :-1], X2[:, :-1])
        return jnp.where(m > 0.0, m * val, jnp.zeros_like(val))


def _shift_time_origin(
    X: jax.Array,
    splits_padded: jax.Array,
    t_ref: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    X_rel = X.at[:, 0].add(-t_ref)
    splits_rel = jnp.where(jnp.isfinite(splits_padded), splits_padded - t_ref, splits_padded)
    return X_rel, splits_rel


@jax.jit
def precompute_pairwise_jax(
    X_fit: jax.Array,
    splits_padded: jax.Array,
) -> PrecomputedPairwise:
    """Precompute pairwise coordinate differences and split block mask for fixed X_fit."""
    X_arr = as_f64(X_fit)
    s_arr = as_f64(splits_padded)
    n = X_arr.shape[0]
    if n > 0:
        t_ref = jnp.where(
            jnp.isfinite(X_arr[0, 0]),
            X_arr[0, 0],
            jnp.asarray(0.0, dtype=jnp.float64),
        )
    else:
        t_ref = jnp.asarray(0.0, dtype=jnp.float64)

    X_rel, splits_rel = _shift_time_origin(X_arr, s_arr, t_ref)
    t_rel = X_rel[:, 0]
    dt = _safe_abs(t_rel[:, None] - t_rel[None, :])
    dt2 = jnp.square(dt)
    dx = _safe_abs(X_rel[:, 1:2] - X_rel[None, :, 1])
    dx2 = jnp.square(dx)
    dy = _safe_abs(X_rel[:, 2:3] - X_rel[None, :, 2])
    dy2 = jnp.square(dy)
    seg = jnp.sum(t_rel[:, None] > splits_rel[None, :], axis=-1)
    same_seg = seg[:, None] == seg[None, :]
    eye_n = jnp.eye(n, dtype=jnp.float64)
    return PrecomputedPairwise(
        dt=dt,
        dt2=dt2,
        dx=dx,
        dx2=dx2,
        dy=dy,
        dy2=dy2,
        same_seg=same_seg,
        eye_n=eye_n,
    )


def _assemble_precomputed_components(
    p: jax.Array,
    precomp: PrecomputedPairwise,
    kid: int,
):
    if kid == KERNEL_BASIC:
        K1 = (10.0 ** p[0]) * jnp.exp(-0.5 * p[1] * precomp.dt2)
        K2_raw = (10.0 ** p[2]) * jnp.exp(
            -0.5 * (p[3] * precomp.dx2 + p[4] * precomp.dy2)
        )
        K2 = jnp.where(precomp.same_seg, K2_raw, 0.0)
        diag = 10.0 ** (2.0 * p[5])
        return K1, None, K2, None, None, None, diag

    if kid == KERNEL_BASIC_EP:
        K1 = (10.0 ** p[0]) * jnp.exp(-0.5 * p[1] * precomp.dt2)
        K2_raw = (10.0 ** p[2]) * jnp.exp(
            -jnp.sqrt(p[3]) * precomp.dx - jnp.sqrt(p[4]) * precomp.dy
        )
        K2 = jnp.where(precomp.same_seg, K2_raw, 0.0)
        diag = 10.0 ** (2.0 * p[5])
        return K1, None, K2, None, None, None, diag

    if kid == KERNEL_QUASIPERIODIC:
        u = jnp.pi * precomp.dt / p[2]
        s = jnp.sin(u)
        c = jnp.cos(u)
        s2 = jnp.square(s)
        K1 = (10.0 ** p[0]) * jnp.exp(-s2 / p[1] - 0.5 * p[3] * precomp.dt2)
        K2_raw = (10.0 ** p[4]) * jnp.exp(
            -0.5 * (p[5] * precomp.dx2 + p[6] * precomp.dy2)
        )
        K2 = jnp.where(precomp.same_seg, K2_raw, 0.0)
        diag = 10.0 ** (2.0 * p[7])
        return K1, None, K2, s, c, s2, diag

    if kid == KERNEL_QUASIPERIODIC_EP:
        u = jnp.pi * precomp.dt / p[2]
        s = jnp.sin(u)
        c = jnp.cos(u)
        s2 = jnp.square(s)
        K1 = (10.0 ** p[0]) * jnp.exp(-s2 / p[1] - 0.5 * p[3] * precomp.dt2)
        K2_raw = (10.0 ** p[4]) * jnp.exp(
            -jnp.sqrt(p[5]) * precomp.dx - jnp.sqrt(p[6]) * precomp.dy
        )
        K2 = jnp.where(precomp.same_seg, K2_raw, 0.0)
        diag = 10.0 ** (2.0 * p[7])
        return K1, None, K2, s, c, s2, diag

    if kid == KERNEL_PERIODIC:
        u = jnp.pi * precomp.dt / p[2]
        s = jnp.sin(u)
        c = jnp.cos(u)
        s2 = jnp.square(s)
        K1a = (10.0 ** p[0]) * jnp.exp(-s2 / p[1])
        K1b = (10.0 ** p[3]) * jnp.exp(-0.5 * p[4] * precomp.dt2)
        K2_raw = (10.0 ** p[5]) * jnp.exp(
            -0.5 * (p[6] * precomp.dx2 + p[7] * precomp.dy2)
        )
        K2 = jnp.where(precomp.same_seg, K2_raw, 0.0)
        diag = 10.0 ** (2.0 * p[8])
        return K1a, K1b, K2, s, c, s2, diag

    raise ValueError(f"Unknown kernel_id: {kid!r}")


def _eval_gp_lnlike_precomputed_raw(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    kernel_id: int | str,
) -> jax.Array:
    """Fast forward Gaussian log-likelihood using precomputed pairwise tensors."""
    p = as_f64(pv)
    y = as_f64(y_fit)
    kid = _normalize_kernel_id(kernel_id)
    n = y.shape[0]
    if n == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)

    K1a, K1b, K2, _, _, _, diag = _assemble_precomputed_components(p, precomp, kid)
    K1 = K1a if K1b is None else (K1a + K1b)
    K = K1 + K2 + diag * precomp.eye_n

    L = jax.scipy.linalg.cholesky(K, lower=True)
    z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
    diag_L = jnp.diag(L)
    valid_L = jnp.all(jnp.isfinite(L)) & jnp.all(diag_L > 0.0)
    safe_diag_L = jnp.where(valid_L, diag_L, jnp.ones_like(diag_L))
    lp = (
        -0.5 * jnp.dot(z, z)
        - jnp.sum(jnp.log(safe_diag_L))
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )
    return jnp.where(valid_L & jnp.isfinite(lp), lp, -jnp.inf)


eval_gp_lnlike_precomputed_jax = functools.partial(
    jax.jit, static_argnames=("kernel_id",)
)(_eval_gp_lnlike_precomputed_raw)


def _eval_gp_lnlike_and_grad_precomputed_raw(
    pv: jax.Array,
    precomp: PrecomputedPairwise,
    y_fit: jax.Array,
    kernel_id: int | str,
) -> tuple[jax.Array, jax.Array]:
    """Exact analytic Gaussian log-likelihood and gradient via Cholesky sensitivity W."""
    p = as_f64(pv)
    y = as_f64(y_fit)
    kid = _normalize_kernel_id(kernel_id)
    n = y.shape[0]
    if n == 0:
        return jnp.asarray(0.0, dtype=jnp.float64), jnp.zeros_like(p)

    K1a, K1b, K2, s, c, s2, diag = _assemble_precomputed_components(p, precomp, kid)
    K1 = K1a if K1b is None else (K1a + K1b)
    K = K1 + K2 + diag * precomp.eye_n

    L = jax.scipy.linalg.cholesky(K, lower=True)
    z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
    diag_L = jnp.diag(L)
    valid_L = jnp.all(jnp.isfinite(L)) & jnp.all(diag_L > 0.0)
    safe_diag_L = jnp.where(valid_L, diag_L, jnp.ones_like(diag_L))
    lp = (
        -0.5 * jnp.dot(z, z)
        - jnp.sum(jnp.log(safe_diag_L))
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )

    alpha = jax.scipy.linalg.solve_triangular(L, z, lower=True, trans=1)
    K_inv_raw = jax.scipy.linalg.cho_solve((L, True), precomp.eye_n)
    K_inv = 0.5 * (K_inv_raw + K_inv_raw.T)
    W = 0.5 * (jnp.outer(alpha, alpha) - K_inv)
    tr_W = jnp.trace(W)
    WK2 = W * K2
    g_wn = 2.0 * _LN10 * diag * tr_W

    if kid == KERNEL_BASIC:
        WK1 = W * K1a
        grad = jnp.stack(
            [
                _LN10 * jnp.sum(WK1),
                -0.5 * jnp.sum(WK1 * precomp.dt2),
                _LN10 * jnp.sum(WK2),
                -0.5 * jnp.sum(WK2 * precomp.dx2),
                -0.5 * jnp.sum(WK2 * precomp.dy2),
                g_wn,
            ]
        )
    elif kid == KERNEL_BASIC_EP:
        WK1 = W * K1a
        grad = jnp.stack(
            [
                _LN10 * jnp.sum(WK1),
                -0.5 * jnp.sum(WK1 * precomp.dt2),
                _LN10 * jnp.sum(WK2),
                (-0.5 / jnp.sqrt(p[3])) * jnp.sum(WK2 * precomp.dx),
                (-0.5 / jnp.sqrt(p[4])) * jnp.sum(WK2 * precomp.dy),
                g_wn,
            ]
        )
    elif kid == KERNEL_QUASIPERIODIC:
        assert s is not None and c is not None and s2 is not None
        WK1 = W * K1a
        grad = jnp.stack(
            [
                _LN10 * jnp.sum(WK1),
                (1.0 / jnp.square(p[1])) * jnp.sum(WK1 * s2),
                (2.0 * jnp.pi / (p[1] * jnp.square(p[2])))
                * jnp.sum(WK1 * (s * c * precomp.dt)),
                -0.5 * jnp.sum(WK1 * precomp.dt2),
                _LN10 * jnp.sum(WK2),
                -0.5 * jnp.sum(WK2 * precomp.dx2),
                -0.5 * jnp.sum(WK2 * precomp.dy2),
                g_wn,
            ]
        )
    elif kid == KERNEL_QUASIPERIODIC_EP:
        assert s is not None and c is not None and s2 is not None
        WK1 = W * K1a
        grad = jnp.stack(
            [
                _LN10 * jnp.sum(WK1),
                (1.0 / jnp.square(p[1])) * jnp.sum(WK1 * s2),
                (2.0 * jnp.pi / (p[1] * jnp.square(p[2])))
                * jnp.sum(WK1 * (s * c * precomp.dt)),
                -0.5 * jnp.sum(WK1 * precomp.dt2),
                _LN10 * jnp.sum(WK2),
                (-0.5 / jnp.sqrt(p[5])) * jnp.sum(WK2 * precomp.dx),
                (-0.5 / jnp.sqrt(p[6])) * jnp.sum(WK2 * precomp.dy),
                g_wn,
            ]
        )
    elif kid == KERNEL_PERIODIC:
        assert K1b is not None and s is not None and c is not None and s2 is not None
        WK1a = W * K1a
        WK1b = W * K1b
        grad = jnp.stack(
            [
                _LN10 * jnp.sum(WK1a),
                (1.0 / jnp.square(p[1])) * jnp.sum(WK1a * s2),
                (2.0 * jnp.pi / (p[1] * jnp.square(p[2])))
                * jnp.sum(WK1a * (s * c * precomp.dt)),
                _LN10 * jnp.sum(WK1b),
                -0.5 * jnp.sum(WK1b * precomp.dt2),
                _LN10 * jnp.sum(WK2),
                -0.5 * jnp.sum(WK2 * precomp.dx2),
                -0.5 * jnp.sum(WK2 * precomp.dy2),
                g_wn,
            ]
        )
    else:
        raise ValueError(f"Unknown kernel_id: {kid!r}")

    ok = valid_L & jnp.isfinite(lp) & jnp.all(jnp.isfinite(grad))
    safe_lp = jnp.where(valid_L & jnp.isfinite(lp), lp, -jnp.inf)
    safe_grad = jnp.where(ok, grad, jnp.zeros_like(grad))
    return safe_lp, safe_grad


eval_gp_lnlike_and_grad_precomputed_jax = functools.partial(
    jax.jit, static_argnames=("kernel_id",)
)(_eval_gp_lnlike_and_grad_precomputed_raw)


def _eval_gp_lnlike_raw(
    pv: jax.Array,
    X_fit: jax.Array,
    y_fit: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int,
) -> jax.Array:
    """Pure-JAX normalized Gaussian log-likelihood (including -0.5 * M * ln(2*pi))."""
    X_arr = as_f64(X_fit)
    y_arr = as_f64(y_fit)
    p_arr = as_f64(pv)
    s_arr = as_f64(splits_padded)

    if X_arr.shape[0] > 0:
        t_ref = jnp.where(
            jnp.isfinite(X_arr[0, 0]),
            X_arr[0, 0],
            jnp.asarray(0.0, dtype=jnp.float64),
        )
    else:
        t_ref = jnp.asarray(0.0, dtype=jnp.float64)
    X_rel, splits_rel = _shift_time_origin(X_arr, s_arr, t_ref)
    kid = _normalize_kernel_id(kernel_id)
    k_tot = K2SCKernel(p_arr, splits_rel, kid)
    diag = 10.0 ** (2.0 * p_arr[_WHITE_NOISE_IDX[kid]])
    gp = tinygp.GaussianProcess(k_tot, X_rel, diag=diag)
    lp = gp.log_probability(y_arr)
    return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)


eval_gp_lnlike_jax = functools.partial(jax.jit, static_argnames=("kernel_id",))(
    _eval_gp_lnlike_raw
)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def eval_gp_lnlike_masked_jax(
    pv: jax.Array,
    X: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int,
) -> jax.Array:
    """Static-shape block-identity masked log-likelihood matching sliced X[mask], y[mask]."""
    X_arr = as_f64(X)
    y_arr = as_f64(y)
    p_arr = as_f64(pv)
    s_arr = as_f64(splits_padded)
    mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
    m = mask_bool.astype(jnp.float64)

    idx0 = jnp.argmax(mask_bool)
    has_valid = jnp.any(mask_bool)
    raw_t0 = X_arr[idx0, 0]
    t_ref = jnp.where(has_valid & jnp.isfinite(raw_t0), raw_t0, jnp.asarray(0.0, dtype=jnp.float64))

    X_clean_raw = jnp.where(mask_bool[:, None], X_arr, 0.0)
    X_rel, splits_rel = _shift_time_origin(X_clean_raw, s_arr, t_ref)
    X_clean = jnp.where(mask_bool[:, None], X_rel, 0.0)
    y_clean = jnp.where(mask_bool, y_arr, 0.0)

    kid = _normalize_kernel_id(kernel_id)
    k_tot = K2SCKernel(p_arr, splits_rel, kid)
    diag = 10.0 ** (2.0 * p_arr[_WHITE_NOISE_IDX[kid]])
    X_aug = jnp.concatenate([X_clean, m[:, None]], axis=1)
    diag_aug = m * diag + (1.0 - m)
    gp = tinygp.GaussianProcess(_BlockMaskedKernel(k_tot), X_aug, diag=diag_aug)

    n_masked = X_arr.shape[0] - jnp.sum(m)
    lp = gp.log_probability(y_clean) + 0.5 * n_masked * jnp.log(2.0 * jnp.pi)
    return jnp.where(has_valid & jnp.isfinite(lp), lp, -jnp.inf)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def predict_components_masked_jax(
    pv: jax.Array,
    X: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int,
    X_test: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Static-shape block-identity masked component prediction (zero JIT recompilation)."""
    X_arr = as_f64(X)
    y_arr = as_f64(y)
    p_arr = as_f64(pv)
    s_arr = as_f64(splits_padded)
    mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
    m = mask_bool.astype(jnp.float64)

    idx0 = jnp.argmax(mask_bool)
    has_valid = jnp.any(mask_bool)
    raw_t0 = X_arr[idx0, 0]
    t_ref = jnp.where(has_valid & jnp.isfinite(raw_t0), raw_t0, jnp.asarray(0.0, dtype=jnp.float64))

    X_clean_raw = jnp.where(mask_bool[:, None], X_arr, 0.0)
    X_rel, splits_rel = _shift_time_origin(X_clean_raw, s_arr, t_ref)
    X_clean = jnp.where(mask_bool[:, None], X_rel, 0.0)
    y_clean = jnp.where(mask_bool, y_arr, 0.0)

    kid = _normalize_kernel_id(kernel_id)
    k1, k2, diag = build_tinygp_kernels(p_arr, splits_rel, kid)
    k_tot = K2SCKernel(p_arr, splits_rel, kid)
    X_aug = jnp.concatenate([X_clean, m[:, None]], axis=1)
    diag_aug = m * diag + (1.0 - m)
    gp = tinygp.GaussianProcess(_BlockMaskedKernel(k_tot), X_aug, diag=diag_aug)

    alpha_aug = gp.solver.solve_triangular(gp.solver.solve_triangular(y_clean), transpose=True)
    alpha_masked = jnp.where(mask_bool, alpha_aug, 0.0)

    X_q_raw = X_arr if X_test is None else as_f64(X_test)
    valid_q = jnp.all(jnp.isfinite(X_q_raw), axis=1)
    X_q_safe_raw = jnp.where(valid_q[:, None], X_q_raw, 0.0)
    X_q_rel = X_q_safe_raw.at[:, 0].add(-t_ref)
    X_q_safe = jnp.where(valid_q[:, None], X_q_rel, 0.0)

    mu_time_raw = k1(X_q_safe, X_clean) @ alpha_masked
    mu_pos_raw = k2(X_q_safe, X_clean) @ alpha_masked

    mu_time_raw = jnp.where(valid_q, mu_time_raw, jnp.nan)
    mu_pos_raw = jnp.where(valid_q, mu_pos_raw, jnp.nan)

    m0 = jnp.nanmedian(mu_pos_raw)
    mu_pos = mu_pos_raw - m0
    mu_time = mu_time_raw + m0
    return mu_time, mu_pos


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_build_gp(
    pv: jax.Array,
    X_jax: jax.Array,
    splits_pad: jax.Array,
    jitter: jax.Array,
    kernel_id: int,
) -> tuple[
    tinygp.GaussianProcess,
    tinygp.kernels.Kernel,
    tinygp.kernels.Kernel,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    if X_jax.shape[0] > 0:
        t_ref = jnp.where(
            jnp.isfinite(X_jax[0, 0]),
            X_jax[0, 0],
            jnp.asarray(0.0, dtype=jnp.float64),
        )
    else:
        t_ref = jnp.asarray(0.0, dtype=jnp.float64)
    X_rel, splits_rel = _shift_time_origin(X_jax, splits_pad, t_ref)
    k1, k2, diag_scalar = build_tinygp_kernels(pv, splits_rel, kernel_id)
    k_tot = K2SCKernel(pv, splits_rel, kernel_id)
    k0_diag = k_tot(X_rel)
    diag_vec = jnp.where(
        jitter > 0.0,
        diag_scalar + jitter * k0_diag,
        jnp.broadcast_to(diag_scalar, (X_rel.shape[0],)),
    )
    gp = tinygp.GaussianProcess(k_tot, X_rel, diag=diag_vec)
    return gp, k1, k2, X_rel, splits_rel, t_ref, diag_scalar


@jax.jit
def _jit_solve_tril_alpha(scale_tril: jax.Array, y_jax: jax.Array) -> jax.Array:
    z = jax.scipy.linalg.solve_triangular(scale_tril, y_jax, lower=True)
    return jax.scipy.linalg.solve_triangular(scale_tril, z, lower=True, trans=1)


@jax.jit
def _jit_lnlike_from_tril(scale_tril: jax.Array, y_jax: jax.Array) -> jax.Array:
    z = jax.scipy.linalg.solve_triangular(scale_tril, y_jax, lower=True)
    norm = jnp.sum(jnp.log(jnp.diag(scale_tril))) + 0.5 * scale_tril.shape[0] * jnp.log(
        2.0 * jnp.pi
    )
    lp = -0.5 * jnp.sum(jnp.square(z)) - norm
    return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_cov_matrices(
    pv: jax.Array,
    X1: jax.Array,
    X2: jax.Array,
    splits_pad: jax.Array,
    t_ref: jax.Array,
    kernel_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    X1_rel, splits_rel = _shift_time_origin(X1, splits_pad, t_ref)
    X2_rel = X2.at[:, 0].add(-t_ref)
    k1, k2, diag_scalar = build_tinygp_kernels(pv, splits_rel, kernel_id)
    return k1(X1_rel, X2_rel), k2(X1_rel, X2_rel), diag_scalar


@functools.partial(jax.jit, static_argnames=("kernel_id",))
def _jit_cross_and_components(
    pv: jax.Array,
    X_rel: jax.Array,
    splits_rel: jax.Array,
    t_ref: jax.Array,
    X_q: jax.Array,
    alpha: jax.Array,
    kernel_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    valid_q = jnp.all(jnp.isfinite(X_q), axis=1)
    X_q_safe = jnp.where(valid_q[:, None], X_q, 0.0)
    X_q_rel = jnp.where(valid_q[:, None], X_q_safe.at[:, 0].add(-t_ref), 0.0)

    k1, k2, diag_scalar = build_tinygp_kernels(pv, splits_rel, kernel_id)
    Ks1 = k1(X_q_rel, X_rel)
    Ks2 = k2(X_q_rel, X_rel)
    Ks = Ks1 + Ks2

    mu = jnp.where(valid_q, Ks @ alpha, jnp.nan)
    mu_time_raw = jnp.where(valid_q, Ks1 @ alpha, jnp.nan)
    mu_pos_raw = jnp.where(valid_q, Ks2 @ alpha, jnp.nan)
    m0 = jnp.nanmedian(mu_pos_raw)
    kss_diag = k1(X_q_rel) + k2(X_q_rel) + diag_scalar
    return mu, mu_time_raw + m0, mu_pos_raw - m0, Ks, kss_diag, valid_q


@jax.jit
def _jit_pred_err_from_tril(
    scale_tril: jax.Array,
    Ks: jax.Array,
    kss_diag: jax.Array,
    valid_q: jax.Array,
) -> jax.Array:
    V = jax.scipy.linalg.solve_triangular(scale_tril, Ks.T, lower=True)
    obs_var = jnp.maximum(0.0, kss_diag - jnp.sum(jnp.square(V), axis=0))
    return jnp.where(valid_q, jnp.sqrt(obs_var), jnp.nan)


class SplitGP:
    """Unified additive time + piecewise-spatial Gaussian Process backed by tinygp."""

    def __init__(self, kernel: DtKernel, splits=None):
        self.kernel = kernel
        self._splits = splits
        self._y: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._dirty: bool = True
        self._t_ref: float = 0.0
        self._t_ref_jax: jax.Array = jnp.asarray(0.0, dtype=jnp.float64)
        self._jitter: float = 0.0
        self._gp: tinygp.GaussianProcess | None = None
        self._k1: tinygp.kernels.Kernel | None = None
        self._k2: tinygp.kernels.Kernel | None = None
        self._k: tinygp.kernels.Kernel | None = None
        self._diag_scalar: float = 0.0
        self._X_rel: jax.Array | None = None
        self._splits_rel: jax.Array | None = None
        self._pv_jax: jax.Array | None = None
        self._alpha: jax.Array | None = None

    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, value) -> None:
        self._splits = value
        self.set_dirty(True)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def set_dirty(self, is_dirty: bool = True) -> None:
        self._dirty = bool(is_dirty)
        if self._dirty:
            self._alpha = None

    def set_pv(self, pv=None) -> None:
        if pv is not None:
            pv_arr = np.asarray(pv, dtype=np.float64)
            if self._gp is None or not np.array_equal(pv_arr, self.kernel._pv):
                self.kernel.set_pv(pv_arr)
                self.set_dirty(True)

    def set_inputs(self, x=None) -> None:
        if x is not None:
            x_arr = np.asarray(as_f64(x), dtype=np.float64)
            if self._x is None or not np.array_equal(x_arr, self._x):
                self._x = x_arr.copy()
                self.set_dirty(True)

    def compute(self, x=None, pv=None, jitter: float = 0.0) -> None:
        self.set_pv(pv)
        self.set_inputs(x)
        if self._x is None:
            raise ValueError("Cannot compute GP without input coordinates x.")

        if self.is_dirty or self._gp is None or jitter != self._jitter:
            X_jax = as_f64(self._x)
            pv_jax = as_f64(self.kernel._pv)
            splits_pad = pad_splits(self._splits)
            jitter_jax = jnp.asarray(float(jitter), dtype=jnp.float64)
            gp, k1, k2, X_rel, splits_rel, t_ref_jax, diag_scalar = _jit_build_gp(
                pv_jax,
                X_jax,
                splits_pad,
                jitter_jax,
                self.kernel.kernel_id,
            )
            self._gp = gp
            self._k1 = k1
            self._k2 = k2
            self._k = gp.kernel
            self._X_rel = X_rel
            self._splits_rel = splits_rel
            self._pv_jax = pv_jax
            self._t_ref_jax = t_ref_jax
            self._t_ref = float(t_ref_jax)
            self._diag_scalar = float(diag_scalar)
            self._jitter = float(jitter)
            self._alpha = None
            self.set_dirty(False)

    def _solve_alpha(self, y_jax: jax.Array) -> jax.Array:
        assert self._gp is not None
        return _jit_solve_tril_alpha(self._gp.solver.scale_tril, y_jax)

    def _compute_alpha(self, y=None, allow_jitter_fallback: bool = False) -> jax.Array:
        if y is not None:
            y_arr = np.asarray(as_f64(y), dtype=np.float64)
            if self._y is None or not np.array_equal(y_arr, self._y):
                self._y = y_arr.copy()
                self._alpha = None
        if self._y is None:
            raise ValueError("Trying to calculate alpha without y given.")
        if self.is_dirty or self._gp is None:
            self.compute()

        if self._alpha is None:
            y_jax = as_f64(self._y)
            alpha = self._solve_alpha(y_jax)
            alpha_np = np.asarray(alpha)
            if (
                allow_jitter_fallback
                and not np.all(np.isfinite(alpha_np))
                and np.all(np.isfinite(self._y))
                and self._x is not None
                and np.all(np.isfinite(self._x))
            ):
                for eta in _JITTER_LADDER:
                    self.compute(jitter=eta)
                    alpha = self._solve_alpha(y_jax)
                    if np.all(np.isfinite(np.asarray(alpha))):
                        warnings.warn(
                            f"GP Cholesky solve produced NaN at zero jitter; "
                            f"succeeded with relative diagonal jitter {eta:.0e}.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                        break
            self._alpha = alpha
        return self._alpha

    def _covariance_matrix(
        self,
        x1,
        x2=None,
        pv=None,
        separate: bool = False,
        add_wn: bool = False,
    ):
        self.set_pv(pv)
        X1 = as_f64(x1)
        X2 = X1 if x2 is None else as_f64(x2)

        if self._x is not None and not self.is_dirty:
            t_ref_jax = self._t_ref_jax
        elif X1.shape[0] > 0 and np.isfinite(float(X1[0, 0])):
            t_ref_jax = jnp.asarray(float(X1[0, 0]), dtype=jnp.float64)
        else:
            t_ref_jax = jnp.asarray(0.0, dtype=jnp.float64)

        splits_pad = pad_splits(self._splits)
        K1_j, K2_j, diag_scalar = _jit_cov_matrices(
            as_f64(self.kernel._pv),
            X1,
            X2,
            splits_pad,
            t_ref_jax,
            self.kernel.kernel_id,
        )
        K1 = np.asarray(K1_j, dtype=np.float64)
        K2 = np.asarray(K2_j, dtype=np.float64)

        if separate:
            return K1, K2
        K = K1 + K2
        if add_wn:
            if K.shape[0] != K.shape[1]:
                raise ValueError("add_wn=True requires a square covariance matrix.")
            K = K + float(diag_scalar) * np.eye(K.shape[0], dtype=np.float64)
        return K

    def lnlikelihood(self, pv, y=None, x=None) -> float:
        if self._jitter != 0.0:
            self.set_dirty(True)
        self.compute(x=x, pv=pv, jitter=0.0)
        y_target = y if y is not None else self._y
        if y_target is None:
            raise ValueError("No target vector y provided to lnlikelihood.")
        y_jax = as_f64(y_target)
        self._y = np.asarray(y_jax, dtype=np.float64).copy()
        self._alpha = None
        assert self._gp is not None
        lp = float(_jit_lnlike_from_tril(self._gp.solver.scale_tril, y_jax))
        return lp if np.isfinite(lp) else -np.inf

    def negll(self, pv, y=None, x=None) -> float:
        return -self.lnlikelihood(pv, y=y, x=x)

    def predict(self, x, mean_only: bool = True, y=None):
        alpha = self._compute_alpha(y=y, allow_jitter_fallback=True)
        assert self._gp is not None
        assert self._X_rel is not None
        assert self._splits_rel is not None
        assert self._pv_jax is not None

        X_q = as_f64(x)
        mu_j, _, _, Ks, kss_diag, valid_q = _jit_cross_and_components(
            self._pv_jax,
            self._X_rel,
            self._splits_rel,
            self._t_ref_jax,
            X_q,
            alpha,
            self.kernel.kernel_id,
        )
        mu = np.asarray(mu_j, dtype=np.float64)
        if mean_only:
            return mu

        err_j = _jit_pred_err_from_tril(self._gp.solver.scale_tril, Ks, kss_diag, valid_q)
        return mu, np.asarray(err_j, dtype=np.float64)

    def predict_components(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Predict additive (mu_time, mu_pos) components with zero-median spatial trend.

        Supports both:
        - ``predict_components(x)`` (using cached training inputs and targets), and
        - ``predict_components(pv, y, x1, x2=None)`` (legacy 4-argument form).
        """
        if len(args) in (3, 4) or "pv" in kwargs or "x1" in kwargs:
            if len(args) >= 3:
                pv = args[0]
                y = args[1]
                x1 = args[2]
                x2 = args[3] if len(args) == 4 else kwargs.get("x2", None)
            else:
                pv = kwargs["pv"]
                y = kwargs["y"]
                x1 = kwargs["x1"]
                x2 = kwargs.get("x2", None)
            self.compute(x=x1, pv=pv, jitter=0.0)
            self._compute_alpha(y=y, allow_jitter_fallback=True)
            x_query = x1 if x2 is None else x2
        elif len(args) == 1 and not kwargs:
            x_query = args[0]
        elif "x" in kwargs and len(args) == 0:
            x_query = kwargs["x"]
        else:
            raise TypeError(
                "predict_components expects (x) or (pv, y, x1, x2=None)."
            )

        alpha = self._compute_alpha(allow_jitter_fallback=True)
        assert self._X_rel is not None
        assert self._splits_rel is not None
        assert self._pv_jax is not None

        X_q = as_f64(x_query)
        _, mt_j, mp_j, _, _, _ = _jit_cross_and_components(
            self._pv_jax,
            self._X_rel,
            self._splits_rel,
            self._t_ref_jax,
            X_q,
            alpha,
            self.kernel.kernel_id,
        )
        return np.asarray(mt_j, dtype=np.float64), np.asarray(mp_j, dtype=np.float64)


GeorgeGP = SplitGP
