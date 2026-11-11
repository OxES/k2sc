"""Unit tests for JAX-traceable, autodiff-safe priors in k2sc.priors."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from k2sc.kernels import kernels
from k2sc.priors import (
    LogNormPrior,
    NormalPrior,
    Prior,
    UniformPrior,
    build_prior_spec,
    eval_ln_prior_jax,
)
from scipy import integrate, stats


def test_base_prior_abstract():
    with pytest.raises(NotImplementedError):
        Prior()


def test_uniform_prior_values_and_grads():
    p = UniformPrior(-6.0, 1.0)
    expected = -math.log(7.0)

    val_in = p(0.0)
    assert val_in.dtype == jnp.float64
    assert val_in.shape == ()
    npt.assert_allclose(float(val_in), expected, rtol=1e-15, atol=1e-15)

    arr = np.array([-7.0, -6.0, -2.5, 1.0, 2.0, np.nan, np.inf, -np.inf])
    out = np.asarray(p.logpdf(arr))
    assert out.shape == arr.shape
    npt.assert_allclose(out[[1, 2, 3]], expected, rtol=1e-15, atol=1e-15)
    assert np.all(out[[0, 4, 5, 6, 7]] == -np.inf)

    grad_fn = jax.jit(jax.grad(lambda x: p.logpdf(x)))
    hess_fn = jax.jit(jax.hessian(lambda x: p.logpdf(x)))
    for x0 in (-7.0, -6.0, -2.5, 1.0, 3.0):
        g = float(grad_fn(jnp.array(x0, dtype=jnp.float64)))
        h = float(hess_fn(jnp.array(x0, dtype=jnp.float64)))
        assert g == 0.0
        assert h == 0.0

    with pytest.raises(ValueError, match="vmax > vmin"):
        UniformPrior(2.0, 2.0)


def test_normal_prior_default_lims_values_and_grads():
    # Default lims=None previously crashed with TypeError unpacking None
    p_unbounded = NormalPrior(mu=17.0, sigma=8.0)
    assert np.isneginf(p_unbounded.lims[0]) and np.isposinf(p_unbounded.lims[1])

    xs = np.linspace(-25.0, 60.0, 17, dtype=np.float64)
    npt.assert_allclose(
        np.asarray(p_unbounded.logpdf(xs)),
        stats.norm.logpdf(xs, loc=17.0, scale=8.0),
        rtol=1e-14,
        atol=1e-14,
    )

    p_trunc = NormalPrior(mu=17.0, sigma=8.0, lims=[0.0, 70.0])
    npt.assert_allclose(
        float(p_trunc(21.0)),
        float(stats.norm.logpdf(21.0, loc=17.0, scale=8.0)),
        rtol=1e-14,
        atol=1e-14,
    )
    assert np.isfinite(float(p_trunc(0.0)))
    assert np.isfinite(float(p_trunc(70.0)))
    for bad_x in (-1.0, 75.0, -np.inf, np.inf, np.nan):
        assert float(p_trunc(bad_x)) == -np.inf

    grad_fn = jax.jit(jax.grad(lambda x: p_trunc.logpdf(x)))
    g_in = float(grad_fn(jnp.array(21.0, dtype=jnp.float64)))
    npt.assert_allclose(g_in, -(21.0 - 17.0) / (8.0**2), rtol=1e-14, atol=1e-14)
    for out_x in (-5.0, -0.01, 70.01, 80.0):
        g_out = float(grad_fn(jnp.array(out_x, dtype=jnp.float64)))
        assert not np.isnan(g_out)
        assert g_out == 0.0


def test_lognorm_prior_fixed_algebra_and_double_where_grads():
    mu, sigma = 0.25, 1.25
    p_full = LogNormPrior(mu=mu, sigma=sigma)

    xs = np.array([0.05, 0.25, 0.8, 1.5, 3.2], dtype=np.float64)
    expected = stats.lognorm.logpdf(xs, s=sigma, scale=np.exp(mu))
    npt.assert_allclose(np.asarray(p_full.logpdf(xs)), expected, rtol=1e-14, atol=1e-14)

    # Confirm the fixed -2*mu*lnx cross-term restores unit normalization over (0, inf)
    norm_integral, _ = integrate.quad(lambda x: math.exp(float(p_full(x))), 0.0, np.inf)
    npt.assert_allclose(norm_integral, 1.0, rtol=1e-10, atol=1e-10)

    # Bounded LogNormPrior on (0, 2]: x <= 0 is -inf, x == 2.0 is finite, x > 2.0 is -inf
    p_bounded = LogNormPrior(mu=mu, sigma=sigma, lims=[0.0, 2.0])
    assert float(p_bounded(-1.0)) == -np.inf
    assert float(p_bounded(0.0)) == -np.inf
    assert np.isfinite(float(p_bounded(2.0)))
    assert float(p_bounded(2.01)) == -np.inf
    assert float(p_bounded(np.inf)) == -np.inf
    assert float(p_bounded(np.nan)) == -np.inf

    grad_fn = jax.jit(jax.grad(lambda x: p_bounded.logpdf(x)))
    hess_fn = jax.jit(jax.hessian(lambda x: p_bounded.logpdf(x)))

    x_in = 0.65
    g_expected = -(1.0 + (math.log(x_in) - mu) / (sigma**2)) / x_in
    npt.assert_allclose(
        float(grad_fn(jnp.array(x_in, dtype=jnp.float64))),
        g_expected,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isfinite(float(hess_fn(jnp.array(x_in, dtype=jnp.float64))))

    for x_bad in (-2.0, -1e-15, 0.0, 2.5):
        g_val = float(grad_fn(jnp.array(x_bad, dtype=jnp.float64)))
        h_val = float(hess_fn(jnp.array(x_bad, dtype=jnp.float64)))
        assert not np.isnan(g_val) and g_val == 0.0
        assert not np.isnan(h_val) and h_val == 0.0


def test_prior_spec_and_eval_ln_prior_jax_across_all_kernels():
    jit_ln_prior = jax.jit(eval_ln_prior_jax)
    vmap_val_grad = jax.jit(
        jax.vmap(jax.value_and_grad(eval_ln_prior_jax), in_axes=(0, None))
    )

    for k_cls in kernels.values():
        k = k_cls()
        spec = build_prior_spec(k.priors)

        pv_valid = k.pv0.copy()
        ref_sum = sum(float(pr(v)) for pr, v in zip(k.priors, pv_valid))
        got = float(jit_ln_prior(jnp.asarray(pv_valid), spec))
        npt.assert_allclose(got, ref_sum, rtol=1e-14, atol=1e-14)
        npt.assert_allclose(float(k.ln_prior(pv_valid)), ref_sum, rtol=1e-14, atol=1e-14)

        # Batch with valid, negative LogNorm slot, below lower bound, and NaN
        pv_neg_lognorm = pv_valid.copy()
        pv_neg_lognorm[1] = -0.5
        pv_zero_lognorm = pv_valid.copy()
        pv_zero_lognorm[1] = 0.0
        pv_above_upper = pv_valid.copy()
        pv_above_upper[0] = 999.0
        pv_with_nan = pv_valid.copy()
        pv_with_nan[2] = np.nan

        batch = jnp.asarray(
            np.stack([pv_valid, pv_neg_lognorm, pv_zero_lognorm, pv_above_upper, pv_with_nan]),
            dtype=jnp.float64,
        )
        vals, grads = vmap_val_grad(batch, spec)
        assert np.isfinite(float(vals[0]))
        assert np.all(np.isneginf(np.asarray(vals[1:])))
        assert np.all(np.isfinite(np.asarray(grads)))
        npt.assert_array_equal(np.asarray(grads[1:]), 0.0)
