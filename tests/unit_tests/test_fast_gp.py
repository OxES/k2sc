"""Unit tests for vectorized tinygp kernels, analytic GP gradients, and static-chunk screening."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import tinygp
from k2sc import configure_single_thread_env
from k2sc.detrender import Detrender
from k2sc.gp import (
    PrecomputedPairwise,
    _shift_time_origin,
    eval_gp_lnlike_and_grad_precomputed_jax,
    eval_gp_lnlike_precomputed_jax,
    precompute_pairwise_jax,
)
from k2sc.kernels import (
    KERNEL_BASIC,
    KERNEL_BASIC_EP,
    KERNEL_PERIODIC,
    KERNEL_QUASIPERIODIC,
    KERNEL_QUASIPERIODIC_EP,
    BasicKernel,
    K2SCKernel,
    K2SCSpatialKernel,
    K2SCTimeKernel,
    QuasiPeriodicKernel,
    SafeL1Distance,
    build_tinygp_kernels,
    kernels,
    pad_splits,
)
from k2sc.optimizers import (
    MultiStartLBFGSBOptimizer,
    _eval_candidates_chunked,
    _finite_box_bounds,
    _generate_candidates,
    _jit_map_neglnposterior_precomputed,
    _jit_neglnposterior_precomputed,
    _jit_val_and_grad,
    _jit_val_and_grad_precomputed,
    _jit_vmap_neglnposterior_precomputed,
    _lnlike_precomputed_optx,
    _make_scipy_host_callback,
    _neglnposterior_impl,
    _pv_to_unconstrained_np,
    _sanitize_pv_bounds,
    _select_diverse_top_k,
    _unconstrained_to_pv_jax,
    compute_closed_lims,
)
from k2sc.priors import eval_ln_prior_jax
from scipy.optimize import minimize
from tinygp.kernels import Exp, ExpSineSquared, ExpSquared
from tinygp.transforms import Subspace


class _RefPiecewiseSpatialKernel(tinygp.kernels.Kernel):
    """Reference point-pair Subspace + piecewise split kernel using tinygp's default vmap."""

    base_kernel: tinygp.kernels.Kernel
    splits_padded: jax.Array

    def evaluate(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        seg1 = jnp.sum(X1[0] > self.splits_padded)
        seg2 = jnp.sum(X2[0] > self.splits_padded)
        val = self.base_kernel.evaluate(X1, X2)
        return jnp.where(seg1 == seg2, val, jnp.zeros_like(val))


def _build_reference_subspace_kernels(
    pv: jax.Array,
    splits_padded: jax.Array,
    kernel_id: int,
) -> tuple[tinygp.kernels.Kernel, tinygp.kernels.Kernel, jax.Array]:
    """Explicit reference tinygp Subspace + ExpSquared/Exp/ExpSineSquared kernels."""
    p = jnp.asarray(pv, dtype=jnp.float64)
    s_pad = jnp.asarray(splits_padded, dtype=jnp.float64)
    safe_l1 = SafeL1Distance()

    if kernel_id == KERNEL_BASIC:
        k1 = (10.0 ** p[0]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[1])))
        k2_base = (
            (10.0 ** p[2])
            * Subspace(1, ExpSquared(scale=1.0 / jnp.sqrt(p[3])))
            * Subspace(2, ExpSquared(scale=1.0 / jnp.sqrt(p[4])))
        )
        return k1, _RefPiecewiseSpatialKernel(k2_base, s_pad), 10.0 ** (2.0 * p[5])

    if kernel_id == KERNEL_BASIC_EP:
        k1 = (10.0 ** p[0]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[1])))
        k2_base = (
            (10.0 ** p[2])
            * Subspace(1, Exp(scale=1.0 / jnp.sqrt(p[3]), distance=safe_l1))
            * Subspace(2, Exp(scale=1.0 / jnp.sqrt(p[4]), distance=safe_l1))
        )
        return k1, _RefPiecewiseSpatialKernel(k2_base, s_pad), 10.0 ** (2.0 * p[5])

    if kernel_id == KERNEL_QUASIPERIODIC:
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
        return k1, _RefPiecewiseSpatialKernel(k2_base, s_pad), 10.0 ** (2.0 * p[7])

    if kernel_id == KERNEL_QUASIPERIODIC_EP:
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
        return k1, _RefPiecewiseSpatialKernel(k2_base, s_pad), 10.0 ** (2.0 * p[7])

    if kernel_id == KERNEL_PERIODIC:
        k1 = (10.0 ** p[0]) * Subspace(
            0,
            ExpSineSquared(scale=p[2], gamma=1.0 / p[1], distance=safe_l1),
        ) + (10.0 ** p[3]) * Subspace(0, ExpSquared(scale=1.0 / jnp.sqrt(p[4])))
        k2_base = (
            (10.0 ** p[5])
            * Subspace(1, ExpSquared(scale=1.0 / jnp.sqrt(p[6])))
            * Subspace(2, ExpSquared(scale=1.0 / jnp.sqrt(p[7])))
        )
        return k1, _RefPiecewiseSpatialKernel(k2_base, s_pad), 10.0 ** (2.0 * p[8])

    raise ValueError(f"Unknown kernel_id: {kernel_id}")


def _nontrivial_pv(kname: str) -> np.ndarray:
    """Return an interior, asymmetric parameter vector exercising all kernel terms."""
    mapping = {
        "basic": np.array([-4.7, 0.31, -4.3, 14.5, 22.5, -3.4], dtype=np.float64),
        "basic_ep": np.array([-4.8, 0.27, -4.4, 12.0, 19.5, -3.3], dtype=np.float64),
        "quasiperiodic": np.array(
            [-4.6, 0.42, 7.35, 0.035, -4.2, 15.5, 23.0, -3.5], dtype=np.float64
        ),
        "quasiperiodic_ep": np.array(
            [-4.5, 0.38, 6.85, 0.045, -4.1, 11.5, 18.5, -3.4], dtype=np.float64
        ),
        "periodic": np.array(
            [-4.7, 0.36, 8.15, -5.1, 0.025, -4.3, 16.0, 21.5, -3.0], dtype=np.float64
        ),
    }
    return mapping[kname].copy()


@pytest.mark.parametrize("kname", list(kernels.keys()))
@pytest.mark.parametrize("use_splits", [False, True])
def test_vectorized_tinygp_kernels_diagonal_rectangular_and_conditioned_parity(
    canonical_gp_dataset, kname: str, use_splits: bool
):
    """Verify 1-D diagonal kernel(X) and 2-D rectangular kernel(X1, X2) vs reference Subspace."""
    X_full = jnp.asarray(canonical_gp_dataset["X"], dtype=jnp.float64)
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"], dtype=jnp.float64)
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"], dtype=jnp.float64)
    splits = canonical_gp_dataset["splits"] if use_splits else ()

    t_ref = X_tr[0, 0]
    X_tr_rel, splits_rel = _shift_time_origin(X_tr, pad_splits(splits), t_ref)
    X_full_rel, _ = _shift_time_origin(X_full, pad_splits(splits), t_ref)

    k_inst = kernels[kname]()
    kid = k_inst.kernel_id
    pv = jnp.asarray(_nontrivial_pv(kname), dtype=jnp.float64)

    k1_fast, k2_fast, diag_fast = build_tinygp_kernels(pv, splits_rel, kid)
    k_tot_fast = K2SCKernel(pv, splits_rel, kid)
    k_sum_fast = k1_fast + k2_fast
    assert isinstance(k1_fast, K2SCTimeKernel)
    assert isinstance(k2_fast, K2SCSpatialKernel)

    k1_ref, k2_ref, diag_ref = _build_reference_subspace_kernels(pv, splits_rel, kid)
    k_tot_ref = k1_ref + k2_ref

    npt.assert_allclose(float(diag_fast), float(diag_ref), rtol=1e-14, atol=1e-16)

    # CRITICAL tinygp contract: kernel(X) (X2=None) MUST return 1-D diagonal of shape (N,)
    d1_fast = k1_fast(X_tr_rel)
    d2_fast = k2_fast(X_tr_rel)
    dtot_fast = k_tot_fast(X_tr_rel)
    dsum_fast = k_sum_fast(X_tr_rel)
    assert d1_fast.shape == (X_tr_rel.shape[0],)
    assert d2_fast.shape == (X_tr_rel.shape[0],)
    assert dtot_fast.shape == (X_tr_rel.shape[0],)
    assert dsum_fast.shape == (X_tr_rel.shape[0],)

    npt.assert_allclose(np.asarray(d1_fast), np.asarray(k1_ref(X_tr_rel)), rtol=1e-12, atol=1e-12)
    npt.assert_allclose(np.asarray(d2_fast), np.asarray(k2_ref(X_tr_rel)), rtol=1e-12, atol=1e-12)
    npt.assert_allclose(
        np.asarray(dtot_fast), np.asarray(k_tot_ref(X_tr_rel)), rtol=1e-12, atol=1e-12
    )
    npt.assert_allclose(
        np.asarray(dsum_fast), np.asarray(k_tot_ref(X_tr_rel)), rtol=1e-12, atol=1e-12
    )

    # Scalar evaluate and evaluate_diag parity
    for i, j in ((0, 0), (0, 17), (12, 49)):
        npt.assert_allclose(
            float(k_tot_fast.evaluate(X_tr_rel[i], X_tr_rel[j])),
            float(k_tot_ref.evaluate(X_tr_rel[i], X_tr_rel[j])),
            rtol=1e-12,
            atol=1e-12,
        )
    npt.assert_allclose(
        float(k_tot_fast.evaluate_diag(X_tr_rel[5])),
        float(k_tot_ref.evaluate_diag(X_tr_rel[5])),
        rtol=1e-12,
        atol=1e-12,
    )

    # Rectangular 2-D matrix kernel(X1, X2) of shape (N, M)
    K1_rect_fast = k1_fast(X_full_rel, X_tr_rel)
    K2_rect_fast = k2_fast(X_full_rel, X_tr_rel)
    Ktot_rect_fast = k_tot_fast(X_full_rel, X_tr_rel)
    Ksum_rect_fast = k_sum_fast(X_full_rel, X_tr_rel)
    assert K1_rect_fast.shape == (X_full_rel.shape[0], X_tr_rel.shape[0])
    assert K2_rect_fast.shape == (X_full_rel.shape[0], X_tr_rel.shape[0])
    assert Ktot_rect_fast.shape == (X_full_rel.shape[0], X_tr_rel.shape[0])

    K1_rect_ref = k1_ref(X_full_rel, X_tr_rel)
    K2_rect_ref = k2_ref(X_full_rel, X_tr_rel)
    Ktot_rect_ref = k_tot_ref(X_full_rel, X_tr_rel)

    npt.assert_allclose(np.asarray(K1_rect_fast), np.asarray(K1_rect_ref), rtol=1e-12, atol=1e-12)
    npt.assert_allclose(np.asarray(K2_rect_fast), np.asarray(K2_rect_ref), rtol=1e-12, atol=1e-12)
    npt.assert_allclose(
        np.asarray(Ktot_rect_fast), np.asarray(Ktot_rect_ref), rtol=1e-12, atol=1e-12
    )
    npt.assert_allclose(
        np.asarray(Ksum_rect_fast), np.asarray(Ktot_rect_ref), rtol=1e-12, atol=1e-12
    )

    # Full tinygp.GaussianProcess log_probability and condition() compatibility
    gp_fast = tinygp.GaussianProcess(k_tot_fast, X_tr_rel, diag=diag_fast)
    gp_ref = tinygp.GaussianProcess(k_tot_ref, X_tr_rel, diag=diag_ref)
    npt.assert_allclose(
        float(gp_fast.log_probability(y_tr)),
        float(gp_ref.log_probability(y_tr)),
        rtol=1e-11,
        atol=1e-11,
    )
    _, cond_fast = gp_fast.condition(y_tr, X_full_rel[:12])
    _, cond_ref = gp_ref.condition(y_tr, X_full_rel[:12])
    npt.assert_allclose(
        np.asarray(cond_fast.loc), np.asarray(cond_ref.loc), rtol=1e-11, atol=1e-11
    )
    npt.assert_allclose(
        np.asarray(cond_fast.variance),
        np.asarray(cond_ref.variance),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("kname", list(kernels.keys()))
@pytest.mark.parametrize("use_splits", [False, True])
def test_analytic_lnlike_and_grad_precomputed_vs_reference_autodiff_and_fd(
    canonical_gp_dataset, kname: str, use_splits: bool
):
    """Verify precomputed forward & analytic gradient vs reference tinygp AD and central FD."""
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"], dtype=jnp.float64)
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"], dtype=jnp.float64)
    splits = canonical_gp_dataset["splits"] if use_splits else ()
    s_pad = pad_splits(splits)

    kid = kernels[kname].kernel_id
    pv_np = _nontrivial_pv(kname)
    pv = jnp.asarray(pv_np, dtype=jnp.float64)

    precomp = precompute_pairwise_jax(X_tr, s_pad)
    assert isinstance(precomp, PrecomputedPairwise)
    assert precomp.dt.shape == (X_tr.shape[0], X_tr.shape[0])

    lp_fwd = eval_gp_lnlike_precomputed_jax(pv, precomp, y_tr, kid)
    lp_ana, grad_ana = eval_gp_lnlike_and_grad_precomputed_jax(pv, precomp, y_tr, kid)

    def _ref_gp_lnlike(p_vec: jax.Array) -> jax.Array:
        X_rel, splits_rel = _shift_time_origin(X_tr, s_pad, X_tr[0, 0])
        _, _, diag_r = build_tinygp_kernels(p_vec, splits_rel, kid)
        gp_r = tinygp.GaussianProcess(K2SCKernel(p_vec, splits_rel, kid), X_rel, diag=diag_r)
        return gp_r.log_probability(y_tr)

    lp_ref, grad_ref = jax.jit(jax.value_and_grad(_ref_gp_lnlike))(pv)

    npt.assert_allclose(float(lp_fwd), float(lp_ref), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(float(lp_ana), float(lp_ref), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(
        np.asarray(grad_ana), np.asarray(grad_ref), rtol=1e-11, atol=1e-11
    )

    # Independent central finite-difference check across every hyperparameter
    grad_fd = np.zeros_like(pv_np)
    for idx in range(pv_np.size):
        eps = 1e-6 * max(abs(float(pv_np[idx])), 0.1)
        dp = np.zeros_like(pv_np)
        dp[idx] = eps
        lp_plus = float(
            eval_gp_lnlike_precomputed_jax(jnp.asarray(pv_np + dp), precomp, y_tr, kid)
        )
        lp_minus = float(
            eval_gp_lnlike_precomputed_jax(jnp.asarray(pv_np - dp), precomp, y_tr, kid)
        )
        grad_fd[idx] = (lp_plus - lp_minus) / (2.0 * eps)

    npt.assert_allclose(np.asarray(grad_ana), grad_fd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kname", list(kernels.keys()))
def test_jit_val_and_grad_posterior_priors_and_oob_safety(
    canonical_gp_dataset, kname: str, monkeypatch
):
    """Verify full _jit_val_and_grad (with Uniform/Normal/LogNorm priors) vs jax.value_and_grad."""
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    k_inst = kernels[kname]()
    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=k_inst,
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=17,
    )
    # PrecomputedPairwise cache must be lazy so prediction-only Detrender passes skip O(N^2) allocs
    assert dt._precomp_tr is None
    assert dt._precomp_splits_key is None
    X_tr, y_tr, s_pad, lims_low, lims_high = dt._objective_tensors(training=True)
    precomp = dt._training_precomp()
    assert dt._precomp_tr is precomp

    if kname == "basic":
        import k2sc.optimizers as opt_mod

        recompute_calls = {"n": 0}
        orig_precomp = opt_mod.precompute_pairwise_jax

        def _counted_precomp(*args, **kwargs):
            recompute_calls["n"] += 1
            return orig_precomp(*args, **kwargs)

        with monkeypatch.context() as mp:
            mp.setattr(opt_mod, "precompute_pairwise_jax", _counted_precomp)
            dt.optimize(optimizer="multistart", n_candidates=8, top_k=1, maxiter=2, seed=0)
        assert recompute_calls["n"] == 0
        assert dt._precomp_tr is precomp

    pv = jnp.asarray(_nontrivial_pv(kname), dtype=jnp.float64)
    val_fast, grad_fast = _jit_val_and_grad(
        pv, X_tr, y_tr, s_pad, lims_low, lims_high, k_inst.prior_spec, k_inst.kernel_id
    )
    val_pre, grad_pre = _jit_val_and_grad_precomputed(
        pv, precomp, y_tr, lims_low, lims_high, k_inst.prior_spec, k_inst.kernel_id
    )
    val_ad, grad_ad = jax.jit(jax.value_and_grad(dt.neglnposterior_jax))(pv)

    npt.assert_allclose(float(val_fast), float(val_ad), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(float(val_pre), float(val_ad), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(np.asarray(grad_fast), np.asarray(grad_ad), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(np.asarray(grad_pre), np.asarray(grad_ad), rtol=1e-11, atol=1e-11)

    # Out-of-bounds and non-finite inputs must return (+inf, zeros) cleanly
    for bad_pv in (
        pv.at[0].set(-99.0),
        pv.at[1].set(0.0),
        pv.at[-1].set(99.0),
        pv.at[2].set(jnp.nan),
    ):
        v_bad, g_bad = _jit_val_and_grad(
            bad_pv,
            X_tr,
            y_tr,
            s_pad,
            lims_low,
            lims_high,
            k_inst.prior_spec,
            k_inst.kernel_id,
        )
        assert np.isposinf(float(v_bad))
        npt.assert_array_equal(np.asarray(g_bad), np.zeros(k_inst.npar))

    # In-place list mutations on dt.gp.splits (.append, item assignment, .clear)
    # and reassignment to None must invalidate cached _precomp_tr automatically.
    dt.gp.splits = [2326.0]
    val_one_split = dt.neglnposterior(np.asarray(pv), training=True)
    dt.gp.splits.append(2354.0)
    val_two_splits = dt.neglnposterior(np.asarray(pv), training=True)
    expected_two_splits = -(
        float(dt.kernel.ln_prior(pv))
        + dt.gp.lnlikelihood(
            np.asarray(pv), dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    assert not np.isclose(val_one_split, val_two_splits, rtol=0.0, atol=1e-9)
    npt.assert_allclose(val_two_splits, expected_two_splits, rtol=1e-12, atol=1e-12)

    dt.gp.splits.clear()
    val_cleared = dt.neglnposterior(np.asarray(pv), training=True)
    dt.gp.splits = None
    val_unsplit = dt.neglnposterior(np.asarray(pv), training=True)
    expected_unsplit = -(
        float(dt.kernel.ln_prior(pv))
        + dt.gp.lnlikelihood(
            np.asarray(pv), dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    npt.assert_allclose(val_cleared, expected_unsplit, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(val_unsplit, expected_unsplit, rtol=1e-12, atol=1e-12)


def test_singular_cholesky_and_overflowing_grad_rejected_cleanly():
    """Singular covariance or overflowing analytic gradients must be rejected as infeasible."""
    X_dup = np.zeros((16, 3), dtype=np.float64)
    X_dup[:, 0] = 2300.0
    y_dup = np.linspace(-1e-3, 1e-3, 16, dtype=np.float64)
    precomp_dup = precompute_pairwise_jax(jnp.asarray(X_dup), pad_splits(None))
    pv_sing = jnp.array([1.0, 0.25, 0.0, 21.0, 21.0, -9.0], dtype=jnp.float64)

    lp = eval_gp_lnlike_precomputed_jax(
        pv_sing, precomp_dup, jnp.asarray(y_dup), KERNEL_BASIC
    )
    lp2, grad2 = eval_gp_lnlike_and_grad_precomputed_jax(
        pv_sing, precomp_dup, jnp.asarray(y_dup), KERNEL_BASIC
    )
    assert float(lp) == -np.inf
    assert float(lp2) == -np.inf
    npt.assert_array_equal(np.asarray(grad2), np.zeros(6))

    # Construct a well-conditioned covariance where forward triangular solve z stays
    # within float64 (lp finite) while back-substitution alpha / outer(alpha, alpha)
    # overflows float64 (||z||^2 < 1.8e308, while alpha_0^2 > 1.8e308 -> inf!).
    t_grid = np.linspace(2300.0, 2315.0, 16, dtype=np.float64)
    X_well = np.column_stack([t_grid, np.linspace(-0.2, 0.2, 16), np.linspace(0.2, -0.2, 16)])
    y_huge = np.zeros(16, dtype=np.float64)
    y_huge[0] = 1.0e150

    k_basic = BasicKernel()
    s_pad = pad_splits(None)
    X_well_jax = jnp.asarray(X_well, dtype=jnp.float64)
    y_huge_jax = jnp.asarray(y_huge, dtype=jnp.float64)
    precomp_well = precompute_pairwise_jax(X_well_jax, s_pad)
    pv_small_noise = jnp.array([-5.5, 0.5, -5.5, 20.0, 20.0, -5.0], dtype=jnp.float64)

    # Confirm forward log-likelihood alone is finite, proving this exercises the grad-overflow guard
    lp_only = float(
        eval_gp_lnlike_precomputed_jax(pv_small_noise, precomp_well, y_huge_jax, KERNEL_BASIC)
    )
    assert np.isfinite(lp_only)

    lp_vg, grad_vg = eval_gp_lnlike_and_grad_precomputed_jax(
        pv_small_noise, precomp_well, y_huge_jax, KERNEL_BASIC
    )
    assert float(lp_vg) == -np.inf
    npt.assert_array_equal(np.asarray(grad_vg), np.zeros(6))

    lims_low = jnp.asarray(k_basic.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(k_basic.lims[1], dtype=jnp.float64)
    nll_vg, nll_grad = _jit_val_and_grad_precomputed(
        pv_small_noise,
        precomp_well,
        y_huge_jax,
        lims_low,
        lims_high,
        k_basic.prior_spec,
        KERNEL_BASIC,
    )
    assert np.isposinf(float(nll_vg))
    npt.assert_array_equal(np.asarray(nll_grad), np.zeros(6))

    host_cb, _ = _make_scipy_host_callback(
        X_well_jax,
        y_huge_jax,
        s_pad,
        lims_low,
        lims_high,
        k_basic.prior_spec,
        KERNEL_BASIC,
        precomp=precomp_well,
    )
    val_host, grad_host = host_cb(np.asarray(pv_small_noise))
    assert val_host == 1e30
    npt.assert_array_equal(grad_host, np.zeros(6))


def test_optx_custom_vjp_batch_parity_and_infeasible_grad_safety(canonical_gp_dataset):
    """Verify @jax.custom_vjp _lnlike_precomputed_optx under vmap on interior & infeasible batches."""
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"], dtype=jnp.float64)
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"], dtype=jnp.float64)
    s_pad = pad_splits(canonical_gp_dataset["splits"])
    precomp = precompute_pairwise_jax(X_tr, s_pad)

    kernel = QuasiPeriodicKernel(period=7.3)
    lims_low = jnp.asarray(kernel.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(kernel.lims[1], dtype=jnp.float64)
    closed_lims = compute_closed_lims(kernel.lims)
    c_arr = np.asarray(closed_lims, dtype=np.float64)
    c_low = jnp.asarray(c_arr[:, 0], dtype=jnp.float64)
    c_high = jnp.asarray(c_arr[:, 1], dtype=jnp.float64)

    def _make_loss_z_pair(
        kid: int,
        l_low: jax.Array,
        l_high: jax.Array,
        cl_low: jax.Array,
        cl_high: jax.Array,
        prior_spec,
    ):
        def _loss_z_optx(z: jax.Array, precomp_arg: PrecomputedPairwise, y_arg: jax.Array):
            pv = _unconstrained_to_pv_jax(z, cl_low, cl_high)
            in_bounds, _, safe_pv, _, _ = _sanitize_pv_bounds(pv, l_low, l_high)
            ln_p = eval_ln_prior_jax(safe_pv, prior_spec)
            ln_l = _lnlike_precomputed_optx(safe_pv, precomp_arg, y_arg, kid)
            raw_nll = -(ln_p + ln_l)
            val = jnp.where(in_bounds & jnp.isfinite(raw_nll), raw_nll, jnp.inf)
            both_fin = jnp.isfinite(cl_low) & jnp.isfinite(cl_high)
            z_excess = jnp.where(both_fin, jnp.maximum(0.0, jnp.abs(z) - 14.0), 0.0)
            wall = 1e-8 * jnp.sum(jnp.square(z_excess))
            return jnp.where(jnp.isfinite(val), val + wall, jnp.asarray(1e30, dtype=jnp.float64))

        def _loss_z_ref(z: jax.Array, X_arg: jax.Array, s_arg: jax.Array, y_arg: jax.Array):
            pv = _unconstrained_to_pv_jax(z, cl_low, cl_high)
            val = _neglnposterior_impl(pv, X_arg, y_arg, s_arg, l_low, l_high, prior_spec, kid)
            both_fin = jnp.isfinite(cl_low) & jnp.isfinite(cl_high)
            z_excess = jnp.where(both_fin, jnp.maximum(0.0, jnp.abs(z) - 14.0), 0.0)
            wall = 1e-8 * jnp.sum(jnp.square(z_excess))
            return jnp.where(jnp.isfinite(val), val + wall, jnp.asarray(1e30, dtype=jnp.float64))

        return _loss_z_optx, _loss_z_ref

    loss_qp_optx, loss_qp_ref = _make_loss_z_pair(
        kernel.kernel_id, lims_low, lims_high, c_low, c_high, kernel.prior_spec
    )
    batch_vg_optx = jax.jit(jax.vmap(jax.value_and_grad(loss_qp_optx), in_axes=(0, None, None)))
    batch_vg_ref = jax.jit(
        jax.vmap(jax.value_and_grad(loss_qp_ref), in_axes=(0, None, None, None))
    )

    rng = np.random.default_rng(77)
    base_pv = _nontrivial_pv("quasiperiodic")
    rel_step = rng.uniform(-0.1, 0.1, size=(8, base_pv.size))
    cands_8 = np.clip(
        base_pv[None, :] * (1.0 + rel_step),
        c_arr[:, 0] + 1e-3,
        c_arr[:, 1] - 1e-3,
    )
    z_np = _pv_to_unconstrained_np(cands_8, c_arr[:, 0], c_arr[:, 1])
    # Exercise both |z| <= 14 interior rows and the |z| > 14 quadratic saturation wall
    z_np[-1, 0] = 15.5
    z_batch = jnp.asarray(z_np, dtype=jnp.float64)

    vals_optx, grads_optx = batch_vg_optx(z_batch, precomp, y_tr)
    vals_ref, grads_ref = batch_vg_ref(z_batch, X_tr, s_pad, y_tr)
    npt.assert_allclose(np.asarray(vals_optx), np.asarray(vals_ref), rtol=1e-11, atol=1e-11)
    npt.assert_allclose(np.asarray(grads_optx), np.asarray(grads_ref), rtol=1e-11, atol=1e-11)

    # Infeasible Cholesky-singular and analytic-gradient-overflowing batches must yield (1e30, 0)
    k_basic = BasicKernel()
    b_low = jnp.asarray(k_basic.lims[0], dtype=jnp.float64)
    b_high = jnp.asarray(k_basic.lims[1], dtype=jnp.float64)
    b_closed = np.asarray(compute_closed_lims(k_basic.lims), dtype=np.float64)
    bc_low = jnp.asarray(b_closed[:, 0], dtype=jnp.float64)
    bc_high = jnp.asarray(b_closed[:, 1], dtype=jnp.float64)
    loss_basic_optx, _ = _make_loss_z_pair(
        KERNEL_BASIC, b_low, b_high, bc_low, bc_high, k_basic.prior_spec
    )
    batch_basic_vg = jax.jit(
        jax.vmap(jax.value_and_grad(loss_basic_optx), in_axes=(0, None, None))
    )

    X_dup = np.zeros((16, 3), dtype=np.float64)
    X_dup[:, 0] = 2300.0
    y_dup = jnp.asarray(np.linspace(-1e-3, 1e-3, 16, dtype=np.float64))
    precomp_dup = precompute_pairwise_jax(jnp.asarray(X_dup), pad_splits(None))
    pv_sing_batch = np.vstack(
        [
            np.array([1.0, 0.25, 0.0, 21.0, 21.0, -9.0], dtype=np.float64),
            np.array([0.5, 0.50, -0.5, 18.0, 18.0, -9.2], dtype=np.float64),
        ]
    )
    direct_lnlike_vg = jax.jit(
        jax.vmap(
            jax.value_and_grad(
                lambda p, pc, y: _lnlike_precomputed_optx(p, pc, y, KERNEL_BASIC)
            ),
            in_axes=(0, None, None),
        )
    )
    lp_sing, glp_sing = direct_lnlike_vg(
        jnp.asarray(pv_sing_batch, dtype=jnp.float64), precomp_dup, y_dup
    )
    assert np.all(np.isneginf(np.asarray(lp_sing)))
    npt.assert_array_equal(np.asarray(glp_sing), np.zeros((2, 6)))

    b_low_sing = b_low.at[5].set(-10.0)
    b_closed_sing = b_closed.copy()
    b_closed_sing[5, 0] = -10.0 + 1e-12
    sing_prior = k_basic.prior_spec._replace(
        vmin=k_basic.prior_spec.vmin.at[5].set(-10.0)
    )
    loss_sing_optx, _ = _make_loss_z_pair(
        KERNEL_BASIC,
        b_low_sing,
        b_high,
        jnp.asarray(b_closed_sing[:, 0], dtype=jnp.float64),
        bc_high,
        sing_prior,
    )
    batch_sing_vg = jax.jit(
        jax.vmap(jax.value_and_grad(loss_sing_optx), in_axes=(0, None, None))
    )
    z_sing = _pv_to_unconstrained_np(pv_sing_batch, b_closed_sing[:, 0], b_closed_sing[:, 1])
    z_sing[1, 0] = 16.0
    v_sing, g_sing = batch_sing_vg(jnp.asarray(z_sing, dtype=jnp.float64), precomp_dup, y_dup)
    npt.assert_array_equal(np.asarray(v_sing), np.full(2, 1e30))
    npt.assert_array_equal(np.asarray(g_sing), np.zeros((2, 6)))

    t_grid = np.linspace(2300.0, 2315.0, 16, dtype=np.float64)
    X_well = np.column_stack([t_grid, np.linspace(-0.2, 0.2, 16), np.linspace(0.2, -0.2, 16)])
    y_huge = np.zeros(16, dtype=np.float64)
    y_huge[0] = 1.0e150
    precomp_well = precompute_pairwise_jax(jnp.asarray(X_well), pad_splits(None))
    pv_ovf_batch = np.vstack(
        [
            np.array([-5.5, 0.5, -5.5, 20.0, 20.0, -5.0], dtype=np.float64),
            np.array([-5.0, 0.4, -5.0, 19.0, 19.0, -5.5], dtype=np.float64),
        ]
    )
    z_ovf = _pv_to_unconstrained_np(pv_ovf_batch, b_closed[:, 0], b_closed[:, 1])
    z_ovf[1, 2] = -16.0
    v_ovf, g_ovf = batch_basic_vg(
        jnp.asarray(z_ovf, dtype=jnp.float64), precomp_well, jnp.asarray(y_huge, dtype=jnp.float64)
    )
    npt.assert_array_equal(np.asarray(v_ovf), np.full(2, 1e30))
    npt.assert_array_equal(np.asarray(g_ovf), np.zeros((2, 6)))


def test_eval_candidates_chunked_static_shapes_and_vmap_map_parity(canonical_gp_dataset):
    """Verify exact parity across scalar, static-chunk lax.map, and vmap batch evaluators."""
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"], dtype=jnp.float64)
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"], dtype=jnp.float64)
    y_np = np.asarray(canonical_gp_dataset["y_tr"], dtype=np.float64)
    s_pad = pad_splits(canonical_gp_dataset["splits"])
    precomp = precompute_pairwise_jax(X_tr, s_pad)
    X_rel, splits_rel = _shift_time_origin(X_tr, s_pad, X_tr[0, 0])

    kernel = QuasiPeriodicKernel(period=7.3)
    lims_low = jnp.asarray(kernel.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(kernel.lims[1], dtype=jnp.float64)
    closed_lims = compute_closed_lims(kernel.lims)
    rng = np.random.default_rng(1234)
    cands_19 = _generate_candidates(
        kernel, closed_lims, rng, n_candidates=19, y_var=float(np.var(y_np))
    )
    # Inject an out-of-bounds row to verify +inf parity across vmap, lax.map, and chunked slicing
    cands_19[7, 0] = -99.0
    is_cpu = jax.default_backend() == "cpu"

    for n_sub in (1, 5, 19):
        sub = cands_19[:n_sub]
        sub_jax = jnp.asarray(sub, dtype=jnp.float64)
        vals_chunked = _eval_candidates_chunked(
            sub,
            X_tr,
            y_tr,
            s_pad,
            lims_low,
            lims_high,
            kernel.prior_spec,
            kernel.kernel_id,
            chunk_size=8,
            precomp=precomp,
        )
        vals_vmap = np.asarray(
            _jit_vmap_neglnposterior_precomputed(
                sub_jax,
                precomp,
                y_tr,
                lims_low,
                lims_high,
                kernel.prior_spec,
                kernel.kernel_id,
            ),
            dtype=np.float64,
        )
        vals_map = np.asarray(
            _jit_map_neglnposterior_precomputed(
                sub_jax,
                precomp,
                y_tr,
                lims_low,
                lims_high,
                kernel.prior_spec,
                kernel.kernel_id,
            ),
            dtype=np.float64,
        )
        assert vals_chunked.shape == (n_sub,)
        # Static-chunk padding/slicing must match the active backend's unpadded XLA kernel to 1e-13
        vals_same_backend = vals_map if is_cpu else vals_vmap
        npt.assert_allclose(vals_chunked, vals_same_backend, rtol=1e-13, atol=1e-13)
        assert np.array_equal(np.isposinf(vals_map), np.isposinf(vals_vmap))

    # Cross-lowering check (sequential LAPACK lax.map vs batched XLA vmap Cholesky):
    # bound each finite candidate's discrepancy by its empirical condition number cond(K_cand).
    cands_19_jax = jnp.asarray(cands_19, dtype=jnp.float64)
    vals_map_19 = np.asarray(
        _jit_map_neglnposterior_precomputed(
            cands_19_jax, precomp, y_tr, lims_low, lims_high, kernel.prior_spec, kernel.kernel_id
        ),
        dtype=np.float64,
    )
    vals_vmap_19 = np.asarray(
        _jit_vmap_neglnposterior_precomputed(
            cands_19_jax, precomp, y_tr, lims_low, lims_high, kernel.prior_spec, kernel.kernel_id
        ),
        dtype=np.float64,
    )
    eps_f64 = float(np.finfo(np.float64).eps)
    n_pts = int(X_tr.shape[0])
    eye_n = jnp.eye(n_pts, dtype=jnp.float64)
    for idx in range(cands_19.shape[0]):
        if np.isposinf(vals_map_19[idx]):
            assert np.isposinf(vals_vmap_19[idx])
            continue
        row_pv = jnp.asarray(cands_19[idx], dtype=jnp.float64)
        _, _, diag_cand = build_tinygp_kernels(row_pv, splits_rel, kernel.kernel_id)
        K_cand = np.asarray(
            K2SCKernel(row_pv, splits_rel, kernel.kernel_id)(X_rel, X_rel) + diag_cand * eye_n,
            dtype=np.float64,
        )
        cond_k = float(np.linalg.cond(K_cand))
        quad_fit = float(y_np @ np.linalg.solve(K_cand, y_np))
        scale = max(float(n_pts), abs(float(vals_map_19[idx])), quad_fit)
        tol_row = max(1e-12, 32.0 * cond_k * eps_f64 * scale)
        assert abs(float(vals_map_19[idx]) - float(vals_vmap_19[idx])) <= tol_row

    val_scalar = float(
        _jit_neglnposterior_precomputed(
            jnp.asarray(cands_19[0], dtype=jnp.float64),
            precomp,
            y_tr,
            lims_low,
            lims_high,
            kernel.prior_spec,
            kernel.kernel_id,
        )
    )
    npt.assert_allclose(
        _eval_candidates_chunked(
            cands_19[:1],
            X_tr,
            y_tr,
            s_pad,
            lims_low,
            lims_high,
            kernel.prior_spec,
            kernel.kernel_id,
            chunk_size=8,
            precomp=precomp,
        )[0],
        val_scalar,
        rtol=1e-13,
        atol=1e-13,
    )


def test_multistart_lbfgsb_harmonic_trap_map_parity(canonical_gp_dataset):
    """Seeded at 2*P_true (14.6 d), default MultiStartLBFGSBOptimizer must recover P_true=7.3 d."""
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    splits = canonical_gp_dataset["splits"]

    # Initialize on the 2 * P_true (14.6 d) harmonic trap under default tolerances (ftol=1e-9, gtol=1e-6)
    kernel = QuasiPeriodicKernel(period=14.6)
    opt = MultiStartLBFGSBOptimizer(n_candidates=128, top_k=5)
    res_fast = opt.optimize(X_tr, y_tr, splits, kernel, seed=2026)

    assert res_fast.success
    assert np.isfinite(res_fast.fun)
    assert abs(float(res_fast.pv[2]) - 7.3) / 7.3 < 0.05

    # Reference uncached tinygp reverse-mode AD L-BFGS-B polish from the exact same 5 diverse seeds
    rng = np.random.default_rng(2026)
    X_jax = jnp.asarray(X_tr, dtype=jnp.float64)
    y_jax = jnp.asarray(y_tr, dtype=jnp.float64)
    s_pad = pad_splits(splits)
    precomp = precompute_pairwise_jax(X_jax, s_pad)
    lims_low = jnp.asarray(kernel.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(kernel.lims[1], dtype=jnp.float64)
    closed_lims = compute_closed_lims(kernel.lims)
    c_low, c_high = _finite_box_bounds(closed_lims, kernel)
    cands = _generate_candidates(
        kernel,
        closed_lims,
        rng,
        n_candidates=128,
        y_var=float(np.nanvar(y_tr)),
    )
    energies = _eval_candidates_chunked(
        cands,
        X_jax,
        y_jax,
        s_pad,
        lims_low,
        lims_high,
        kernel.prior_spec,
        kernel.kernel_id,
        precomp=precomp,
    )
    k_bounds = np.asarray(kernel.bounds, dtype=np.float64)
    norm_low = np.where(np.isfinite(k_bounds[:, 0]), k_bounds[:, 0], c_low)
    norm_high = np.where(np.isfinite(k_bounds[:, 1]), k_bounds[:, 1], c_high)
    seeds = _select_diverse_top_k(
        cands,
        energies,
        norm_low,
        norm_high,
        top_k=5,
        min_dist=0.15,
        is_periodic=True,
    )

    ref_vg_jit = functools.partial(jax.jit, static_argnames=("kernel_id",))(
        jax.value_and_grad(_neglnposterior_impl)
    )
    c_arr = np.asarray(closed_lims, dtype=np.float64)
    best_ref_fun = float("inf")
    uncached_same_grad_nfev = int(cands.shape[0])
    best_same_grad_fun = float("inf")
    for x_seed in seeds:
        x0_clip = np.clip(np.asarray(x_seed, dtype=np.float64), c_arr[:, 0], c_arr[:, 1])

        def _ref_cb(x_vec: np.ndarray) -> tuple[float, np.ndarray]:
            vj, gj = ref_vg_jit(
                jnp.asarray(x_vec, dtype=jnp.float64),
                X_jax,
                y_jax,
                s_pad,
                lims_low,
                lims_high,
                kernel.prior_spec,
                kernel.kernel_id,
            )
            vf = float(vj)
            gf = np.asarray(gj, dtype=np.float64)
            if not np.isfinite(vf) or not np.all(np.isfinite(gf)):
                return 1e30, np.zeros_like(x_vec, dtype=np.float64)
            return vf, gf

        init_v, _ = _ref_cb(x0_clip)
        mres = minimize(
            _ref_cb,
            x0_clip,
            method="L-BFGS-B",
            jac=True,
            bounds=closed_lims,
            options={"ftol": 1e-9, "gtol": 1e-6, "maxiter": 200},
        )
        fin_v, _ = _ref_cb(np.clip(np.asarray(mres.x, dtype=np.float64), c_arr[:, 0], c_arr[:, 1]))
        best_ref_fun = min(best_ref_fun, init_v, fin_v)

        # Deterministic deduplication check: uncached callback using the *identical*
        # analytic _jit_val_and_grad_precomputed follows the exact same L-BFGS-B iterates,
        # so consecutive-point caching saves >= 1 boundary call per seed (at x0_clip).
        def _same_grad_uncached_cb(x_vec: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal uncached_same_grad_nfev
            uncached_same_grad_nfev += 1
            vj, gj = _jit_val_and_grad_precomputed(
                jnp.asarray(x_vec, dtype=jnp.float64),
                precomp,
                y_jax,
                lims_low,
                lims_high,
                kernel.prior_spec,
                kernel.kernel_id,
            )
            vf = float(vj)
            gf = np.asarray(gj, dtype=np.float64)
            if not np.isfinite(vf) or not np.all(np.isfinite(gf)):
                return 1e30, np.zeros_like(x_vec, dtype=np.float64)
            return vf, gf

        init_sg, _ = _same_grad_uncached_cb(x0_clip)
        mres_sg = minimize(
            _same_grad_uncached_cb,
            x0_clip,
            method="L-BFGS-B",
            jac=True,
            bounds=closed_lims,
            options={"ftol": 1e-9, "gtol": 1e-6, "maxiter": 200},
        )
        fin_sg, _ = _same_grad_uncached_cb(
            np.clip(np.asarray(mres_sg.x, dtype=np.float64), c_arr[:, 0], c_arr[:, 1])
        )
        best_same_grad_fun = min(best_same_grad_fun, init_sg, fin_sg)

    npt.assert_allclose(res_fast.fun, best_ref_fun, rtol=0.0, atol=1e-6)
    npt.assert_allclose(res_fast.fun, best_same_grad_fun, rtol=0.0, atol=1e-12)
    assert res_fast.nfev <= uncached_same_grad_nfev - len(seeds)


def test_configure_single_thread_env_idempotence_and_fresh_subprocess(monkeypatch):
    """Verify stdlib-only import ordering, valid OpenXLA flags, and JAX_ENABLE_X64=0 override."""
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for var in (*thread_vars, "XLA_FLAGS"):
        monkeypatch.delenv(var, raising=False)

    configure_single_thread_env()
    for var in thread_vars:
        assert os.environ[var] == "1"
    xla_first = os.environ["XLA_FLAGS"]
    assert xla_first == "--xla_cpu_multi_thread_eigen=false"
    assert "intra_op_parallelism_threads" not in xla_first

    # Second call must be strictly idempotent (no duplicate flags)
    configure_single_thread_env()
    assert os.environ["XLA_FLAGS"] == xla_first

    # Explicit caller overrides and existing unrelated XLA_FLAGS must be preserved
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    configure_single_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
    )
    configure_single_thread_env()
    assert os.environ["XLA_FLAGS"].count("--xla_cpu_multi_thread_eigen") == 1

    # Fresh subprocess verification with JAX_ENABLE_X64=0 pre-set in os.environ:
    # importing k2sc MUST override JAX_ENABLE_X64 to 'True' without importing numpy or jax,
    # and subsequent jax.devices() + jnp.zeros(1) must initialize in 64-bit mode cleanly.
    src_dir = Path(__file__).resolve().parents[2] / "src"
    sub_code = (
        "import importlib.util, os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ.get('JAX_ENABLE_X64') == '0'\n"
        f"src_dir = Path({str(src_dir)!r})\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'k2sc', src_dir / '__init__.py', submodule_search_locations=[str(src_dir)]\n"
        ")\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['k2sc'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'configure_single_thread_env' in mod.__all__\n"
        "assert os.environ['JAX_ENABLE_X64'] == 'True'\n"
        "assert 'numpy' not in sys.modules\n"
        "assert 'jax' not in sys.modules\n"
        "from k2sc import configure_single_thread_env\n"
        "configure_single_thread_env()\n"
        "assert 'numpy' not in sys.modules\n"
        "assert 'jax' not in sys.modules\n"
        "assert os.environ['OMP_NUM_THREADS'] == '1'\n"
        "assert os.environ['XLA_FLAGS'] == '--xla_cpu_multi_thread_eigen=false'\n"
        "import jax\n"
        "assert jax.config.jax_enable_x64 is True\n"
        "assert len(jax.devices()) >= 1\n"
        "import jax.numpy as jnp\n"
        "assert jnp.zeros(1).dtype == jnp.float64\n"
    )
    sub_env = {
        k: v
        for k, v in os.environ.items()
        if k not in (*thread_vars, "XLA_FLAGS")
    }
    sub_env["JAX_ENABLE_X64"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        env=sub_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
