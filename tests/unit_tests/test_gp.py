"""Comprehensive unit, numerical parity, autodiff, and conditioning tests for k2sc.gp."""

from __future__ import annotations

import ast
import functools
import sys
import types
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import k2sc
import numpy as np
import numpy.testing as npt
import pytest
import reference_gp_numpy as ref
from k2sc._jax_env import as_f64
from k2sc.gp import (
    GeorgeGP,
    SplitGP,
    _eval_gp_lnlike_raw,
    eval_gp_lnlike_jax,
    eval_gp_lnlike_masked_jax,
    predict_components_masked_jax,
)
from k2sc.kernels import (
    BasicKernel,
    BasicKernelEP,
    QuasiPeriodicKernelEP,
    kernels,
    pad_splits,
)
from k2sc.priors import eval_ln_prior_jax

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "gp_golden.npz"


def test_reference_oracle_has_zero_jax_or_tinygp_imports():
    ref_path = Path(ref.__file__).resolve()
    tree = ast.parse(ref_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "jax" not in imported_roots
    assert "tinygp" not in imported_roots


@pytest.mark.parametrize("kname", list(kernels.keys()))
def test_2x2_closed_form_cramers_rule_algebra(kname):
    X_2x2 = np.array([[2301.25, -0.15, 0.10], [2303.75, 0.20, -0.05]], dtype=np.float64)
    y_2x2 = np.array([1.1e-3, -0.7e-3], dtype=np.float64)
    k = kernels[kname]()
    pv = k.pv0.copy()
    pv[-1] = -3.2

    K_cramer, lnlike_cramer, alpha_cramer = ref.gp_2x2_cramers_oracle(
        kname, pv, X_2x2, y_2x2, splits=None
    )
    gp = SplitGP(k, splits=None)
    lnlike_gp = gp.lnlikelihood(pv, y_2x2, X_2x2)
    alpha_gp = np.asarray(gp._compute_alpha(y_2x2))
    K_gp = gp._covariance_matrix(X_2x2, pv=pv, add_wn=True)

    npt.assert_allclose(K_gp, K_cramer, rtol=1e-14, atol=1e-16)
    npt.assert_allclose(alpha_gp, alpha_cramer, rtol=1e-13, atol=1e-13)
    npt.assert_allclose(lnlike_gp, lnlike_cramer, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kname", list(kernels.keys()))
@pytest.mark.parametrize("split_mode", ["none", "empty", "two_splits"])
def test_float64_parity_all_five_kernels_vs_reference_numpy(
    canonical_gp_dataset, kname, split_mode
):
    X_test = canonical_gp_dataset["X"]
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]

    if split_mode == "none":
        splits = None
    elif split_mode == "empty":
        splits = []
    else:
        splits = canonical_gp_dataset["splits"]

    k = kernels[kname]()
    pv = k.pv0.copy()
    pv[0] = -5.2
    pv[-1] = -3.1

    gp = SplitGP(k, splits=splits)
    K1_got, K2_got = gp._covariance_matrix(X_tr, pv=pv, separate=True)
    K_got = gp._covariance_matrix(X_tr, pv=pv, add_wn=True)
    K1_ref, K2_ref, diag_ref = ref.kernel_matrices_numpy(kname, pv, X_tr, X_tr, splits=splits)
    K_ref = K1_ref + K2_ref + diag_ref * np.eye(X_tr.shape[0], dtype=np.float64)
    npt.assert_allclose(K1_got, K1_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(K2_got, K2_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(K_got, K_ref, rtol=1e-12, atol=1e-12)

    lnlike_got = gp.lnlikelihood(pv, y_tr, X_tr)
    negll_got = gp.negll(pv, y_tr)
    lnlike_jax = float(
        eval_gp_lnlike_jax(pv, X_tr, y_tr, pad_splits(splits), k.kernel_id)
    )
    lnlike_ref = ref.gp_lnlikelihood_numpy(kname, pv, X_tr, y_tr, splits=splits, include_const=True)

    npt.assert_allclose(lnlike_got, lnlike_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(lnlike_jax, lnlike_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(negll_got, -lnlike_ref, rtol=1e-12, atol=1e-12)

    mu_only = gp.predict(X_test, mean_only=True)
    mu_got, err_got = gp.predict(X_test, mean_only=False)
    mu_time_got, mu_pos_got = gp.predict_components(X_test)
    mu_ref, err_ref, mu_time_ref, mu_pos_ref = ref.gp_predict_numpy(
        kname, pv, X_tr, y_tr, X_test, splits=splits
    )

    npt.assert_allclose(mu_only, mu_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mu_got, mu_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(err_got, err_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mu_time_got, mu_time_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mu_pos_got, mu_pos_ref, rtol=1e-12, atol=1e-12)

    # Zero-median spatial component & exact additive conservation
    npt.assert_allclose(np.nanmedian(mu_pos_got), 0.0, rtol=0.0, atol=1e-15)
    npt.assert_allclose(mu_time_got + mu_pos_got, mu_got, rtol=1e-12, atol=1e-12)

    # Legacy 4-argument and 3-argument predict_components(pv, y, x1, x2=None)
    mt_4arg, mp_4arg = gp.predict_components(pv, y_tr, X_tr, X_test)
    npt.assert_allclose(mt_4arg, mu_time_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mp_4arg, mu_pos_ref, rtol=1e-12, atol=1e-12)

    mt_3arg, mp_3arg = gp.predict_components(pv, y_tr, X_tr)
    _, _, mt_tr_ref, mp_tr_ref = ref.gp_predict_numpy(kname, pv, X_tr, y_tr, X_tr, splits=splits)
    npt.assert_allclose(mt_3arg, mt_tr_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mp_3arg, mp_tr_ref, rtol=1e-12, atol=1e-12)


def test_georgegp_alias_and_none_vs_empty_splits_equivalence(canonical_gp_dataset):
    assert GeorgeGP is SplitGP
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    X_test = canonical_gp_dataset["X"]
    pv = BasicKernel.pv0.copy()

    gp_none = GeorgeGP(BasicKernel())
    gp_empty = SplitGP(BasicKernel(), splits=[])

    ll_none = gp_none.lnlikelihood(pv, y_tr, X_tr)
    ll_empty = gp_empty.lnlikelihood(pv, y_tr, X_tr)
    npt.assert_allclose(ll_none, ll_empty, rtol=1e-15, atol=1e-15)

    mt_none, mp_none = gp_none.predict_components(X_test)
    mt_empty, mp_empty = gp_empty.predict_components(X_test)
    npt.assert_allclose(mt_none, mt_empty, rtol=1e-15, atol=1e-15)
    npt.assert_allclose(mp_none, mp_empty, rtol=1e-15, atol=1e-15)


def test_high_bjd_time_origin_shifting_precision(canonical_gp_dataset):
    """Full BJD (~2.458e6 d) must match BKJD to 1e-12 via consistent t - t_ref shifting."""
    bjd_offset = 2_458_000.0
    X_tr_bkjd = canonical_gp_dataset["X_tr"].copy()
    X_test_bkjd = canonical_gp_dataset["X"].copy()
    y_tr = canonical_gp_dataset["y_tr"]

    # Quantize BKJD timestamps to the exact BJD grid so X_bjd = X_bkjd + bjd_offset
    # has zero pre-input rounding difference while still having magnitude ~2.458e6.
    X_tr_bkjd[:, 0] = (X_tr_bkjd[:, 0] + bjd_offset) - bjd_offset
    X_test_bkjd[:, 0] = (X_test_bkjd[:, 0] + bjd_offset) - bjd_offset
    splits_bkjd = [(s + bjd_offset) - bjd_offset for s in canonical_gp_dataset["splits"]]

    X_tr_bjd = X_tr_bkjd.copy()
    X_tr_bjd[:, 0] += bjd_offset
    X_test_bjd = X_test_bkjd.copy()
    X_test_bjd[:, 0] += bjd_offset
    splits_bjd = [s + bjd_offset for s in splits_bkjd]

    k = QuasiPeriodicKernelEP(period=7.3)
    pv = k._pv.copy()
    pv[0] = -4.9
    pv[-1] = -3.1

    gp_bkjd = SplitGP(k, splits=splits_bkjd)
    gp_bjd = SplitGP(QuasiPeriodicKernelEP(period=7.3), splits=splits_bjd)

    ll_bkjd = gp_bkjd.lnlikelihood(pv, y_tr, X_tr_bkjd)
    ll_bjd = gp_bjd.lnlikelihood(pv, y_tr, X_tr_bjd)
    npt.assert_allclose(ll_bjd, ll_bkjd, rtol=1e-12, atol=1e-12)

    mu_bkjd, err_bkjd = gp_bkjd.predict(X_test_bkjd, mean_only=False)
    mu_bjd, err_bjd = gp_bjd.predict(X_test_bjd, mean_only=False)
    npt.assert_allclose(mu_bjd, mu_bkjd, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(err_bjd, err_bkjd, rtol=1e-12, atol=1e-12)

    mt_bkjd, mp_bkjd = gp_bkjd.predict_components(X_test_bkjd)
    mt_bjd, mp_bjd = gp_bjd.predict_components(X_test_bjd)
    npt.assert_allclose(mt_bjd, mt_bkjd, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(mp_bjd, mp_bkjd, rtol=1e-12, atol=1e-12)


def test_legacy_george03_golden_bijection():
    """Verify float64 parity with historical george>=0.3 outputs via p1_fixed = -1/ln(p1_legacy)."""
    assert _FIXTURE_PATH.exists(), f"Missing golden fixture {_FIXTURE_PATH}"
    with np.load(_FIXTURE_PATH, allow_pickle=False) as data:
        assert "tinygp_version" in data
        assert "provenance_note" in data
        X_fit = data["X_fit"]
        y_fit = data["y_fit"]
        X_test = data["X_test"]
        splits = data["splits"]
        m_fit = X_fit.shape[0]
        norm_const = 0.5 * m_fit * np.log(2.0 * np.pi)

        for kname in ("basic", "basic_ep", "quasiperiodic", "quasiperiodic_ep"):
            pv_legacy = data[f"{kname}_pv_legacy"].copy()
            pv_fixed = pv_legacy.copy()
            if kname.startswith("quasiperiodic"):
                assert 0.0 < pv_legacy[1] < 1.0
                pv_fixed[1] = -1.0 / np.log(pv_legacy[1])

            for mode_label, sp in (("unsplit", None), ("split", splits)):
                prefix = f"{kname}_{mode_label}"
                gp = SplitGP(kernels[kname](), splits=sp)
                K1_got, K2_got = gp._covariance_matrix(X_fit, pv=pv_fixed, separate=True)
                npt.assert_allclose(K1_got, data[f"{prefix}_K1_tr"], rtol=1e-12, atol=1e-12)
                npt.assert_allclose(K2_got, data[f"{prefix}_K2_tr"], rtol=1e-12, atol=1e-12)

                ll_got = gp.lnlikelihood(pv_fixed, y_fit, X_fit)
                npt.assert_allclose(
                    ll_got, float(data[f"{prefix}_lnlike_george"]), rtol=1e-12, atol=1e-12
                )
                npt.assert_allclose(
                    ll_got,
                    float(data[f"{prefix}_lnlike_splitgp_no_const"]) - norm_const,
                    rtol=1e-12,
                    atol=1e-12,
                )

                mu_got, err_got = gp.predict(X_test, mean_only=False)
                mt_got, mp_got = gp.predict_components(X_test)
                npt.assert_allclose(mu_got, data[f"{prefix}_mu"], rtol=1e-12, atol=1e-12)
                npt.assert_allclose(err_got, data[f"{prefix}_err"], rtol=1e-12, atol=1e-12)
                npt.assert_allclose(mt_got, data[f"{prefix}_mu_time"], rtol=1e-12, atol=1e-12)
                npt.assert_allclose(mp_got, data[f"{prefix}_mu_pos"], rtol=1e-12, atol=1e-12)


def test_live_george_parity(canonical_gp_dataset):
    george = pytest.importorskip(
        "george", reason="Optional live C++ george parity test (george not installed)"
    )
    from george.kernels import ExpKernel, ExpSine2Kernel, ExpSquaredKernel

    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    pv = np.array([-5.0, 0.4, 6.2, 0.04, -4.8, 16.0, 20.0, -3.5], dtype=np.float64)

    k1_g = (10.0 ** pv[0]) * ExpSine2Kernel(
        gamma=1.0 / pv[1], log_period=np.log(pv[2]), ndim=3, axes=0
    ) * ExpSquaredKernel(1.0 / pv[3], ndim=3, axes=0)
    k2_g = (10.0 ** pv[4]) * ExpKernel(1.0 / pv[5], ndim=3, axes=1) * ExpKernel(
        1.0 / pv[6], ndim=3, axes=2
    )
    gp_g = george.GP(k1_g + k2_g)
    gp_g.compute(X_tr, yerr=10.0 ** pv[7], sort=False)

    ll_george = gp_g.log_likelihood(y_tr)
    npt.assert_allclose(
        ref.gp_lnlikelihood_numpy("quasiperiodic_ep", pv, X_tr, y_tr, splits=None),
        ll_george,
        rtol=1e-12,
        atol=1e-12,
    )
    gp_k2sc = SplitGP(QuasiPeriodicKernelEP(), splits=None)
    npt.assert_allclose(
        gp_k2sc.lnlikelihood(pv, y_tr, X_tr),
        ll_george,
        rtol=1e-12,
        atol=1e-12,
    )


def test_block_identity_static_masking_parity_and_zero_recompilation(canonical_gp_dataset):
    X = canonical_gp_dataset["X"].copy()
    y = canonical_gp_dataset["y"].copy()
    splits = canonical_gp_dataset["splits"]
    s_pad = pad_splits(splits)

    # mask1 selects the 64 training cadences (index 0 is False -> tests masked leading NaN)
    mask1 = np.isin(X[:, 0], canonical_gp_dataset["X_tr"][:, 0])
    assert int(np.sum(mask1)) == 64
    assert not mask1[0]
    mask2 = mask1.copy()
    mask2[10:40] = False  # Simulate second-pass outlier clipping removing additional points
    assert int(np.sum(mask2)) < int(np.sum(mask1))

    # Inject NaNs at masked rows (including index 0!) to prove NaN isolation
    X_dirty = X.copy()
    y_dirty = y.copy()
    X_dirty[~mask1] = np.nan
    y_dirty[~mask1] = np.nan

    k = QuasiPeriodicKernelEP(period=7.3)
    pv = k._pv.copy()
    pv[0] = -4.9
    pv[-1] = -3.2

    predict_components_masked_jax.clear_cache()
    eval_gp_lnlike_masked_jax.clear_cache()

    # Pass 1 (M1 = 64): compare directly against SplitGP on sliced X[mask1]
    gp_sub = SplitGP(QuasiPeriodicKernelEP(period=7.3), splits=splits)
    ll_sub1 = gp_sub.lnlikelihood(pv, y[mask1], X[mask1])
    mt_sub1, mp_sub1 = gp_sub.predict_components(X)

    ll_masked1 = float(
        eval_gp_lnlike_masked_jax(pv, X_dirty, y_dirty, mask1, s_pad, k.kernel_id)
    )
    mt_masked1, mp_masked1 = predict_components_masked_jax(
        pv, X_dirty, y_dirty, mask1, s_pad, k.kernel_id, X
    )
    npt.assert_allclose(ll_masked1, ll_sub1, rtol=1e-13, atol=1e-13)
    npt.assert_allclose(np.asarray(mt_masked1), mt_sub1, rtol=1e-13, atol=1e-13)
    npt.assert_allclose(np.asarray(mp_masked1), mp_sub1, rtol=1e-13, atol=1e-13)

    # Pass 2 (M2 < M1): compare against reference NumPy oracle on sliced X[mask2]
    ll_sub2 = ref.gp_lnlikelihood_numpy(
        "quasiperiodic_ep", pv, X[mask2], y[mask2], splits=splits, include_const=True
    )
    _, _, mt_sub2, mp_sub2 = ref.gp_predict_numpy(
        "quasiperiodic_ep", pv, X[mask2], y[mask2], X, splits=splits
    )
    ll_masked2 = float(
        eval_gp_lnlike_masked_jax(pv, X_dirty, y_dirty, mask2, s_pad, k.kernel_id)
    )
    mt_masked2, mp_masked2 = predict_components_masked_jax(
        pv, X_dirty, y_dirty, mask2, s_pad, k.kernel_id, X
    )
    npt.assert_allclose(ll_masked2, ll_sub2, rtol=1e-13, atol=1e-13)
    npt.assert_allclose(np.asarray(mt_masked2), mt_sub2, rtol=1e-13, atol=1e-13)
    npt.assert_allclose(np.asarray(mp_masked2), mp_sub2, rtol=1e-13, atol=1e-13)

    # Varying mask counts M1 != M2 on static (N=128, 3) compiled XLA graph only ONCE
    assert predict_components_masked_jax._cache_size() == 1
    assert eval_gp_lnlike_masked_jax._cache_size() == 1


_AD_VG = jax.jit(jax.value_and_grad(_eval_gp_lnlike_raw), static_argnames=("kernel_id",))
_AD_HESS = jax.jit(jax.hessian(_eval_gp_lnlike_raw), static_argnames=("kernel_id",))
_AD_GRAD_X = jax.jit(jax.grad(_eval_gp_lnlike_raw, argnums=1), static_argnames=("kernel_id",))
_AD_IDX4 = np.array([0, 14, 34, 56])
_AD_DUP4 = np.array([0, 5, 6, 36])


@pytest.mark.parametrize("kname", list(kernels.keys()))
def test_autodiff_grad_and_hessian_vs_finite_differences(canonical_gp_dataset, kname):
    X_sub = canonical_gp_dataset["X_tr"][_AD_IDX4]
    y_sub = canonical_gp_dataset["y_tr"][_AD_IDX4]
    s_pad = pad_splits(canonical_gp_dataset["splits"])

    k = kernels[kname]()
    pv = k.pv0.copy()
    pv[0] = -4.7
    pv[-1] = -3.4
    kid = k.kernel_id

    val, grad_ad = _AD_VG(jnp.asarray(pv), X_sub, y_sub, s_pad, kid)
    hess_ad = _AD_HESS(jnp.asarray(pv), X_sub, y_sub, s_pad, kid)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad_ad)))
    assert np.all(np.isfinite(np.asarray(hess_ad)))

    # Central finite-difference gradient and Hessian checks with scale-proportional step
    grad_fd = np.zeros_like(pv)
    hess_fd = np.zeros((pv.size, pv.size), dtype=np.float64)
    for i in range(pv.size):
        eps_i = 1e-6 * max(abs(float(pv[i])), 0.1)
        dp = np.zeros_like(pv)
        dp[i] = eps_i
        f_p, g_p = _AD_VG(jnp.asarray(pv + dp), X_sub, y_sub, s_pad, kid)
        f_m, g_m = _AD_VG(jnp.asarray(pv - dp), X_sub, y_sub, s_pad, kid)
        grad_fd[i] = (float(f_p) - float(f_m)) / (2.0 * eps_i)
        hess_fd[:, i] = (np.asarray(g_p) - np.asarray(g_m)) / (2.0 * eps_i)

    npt.assert_allclose(np.asarray(grad_ad), grad_fd, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(np.asarray(hess_ad), hess_fd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("k_cls", [BasicKernelEP, QuasiPeriodicKernelEP])
def test_duplicate_coordinate_rows_ep_kernels_no_nan_grads(canonical_gp_dataset, k_cls):
    """Duplicate off-diagonal rows X_i == X_j on L1 Exp kernels must never emit NaN in grad/hessian."""
    X_dup = canonical_gp_dataset["X_tr"][_AD_DUP4].copy()
    y_dup = canonical_gp_dataset["y_tr"][_AD_DUP4].copy()
    # Exact full 3D duplicate row and exact spatial (x, y) duplicate row in same split segment
    X_dup[1] = X_dup[0]
    X_dup[2, 1:] = X_dup[0, 1:]

    k = k_cls()
    pv = k.pv0.copy()
    pv[-1] = -3.3
    kid = k.kernel_id
    s_pad = pad_splits(canonical_gp_dataset["splits"])

    val, grad_pv = _AD_VG(jnp.asarray(pv), X_dup, y_dup, s_pad, kid)
    hess_pv = _AD_HESS(jnp.asarray(pv), X_dup, y_dup, s_pad, kid)
    grad_X = _AD_GRAD_X(jnp.asarray(pv), jnp.asarray(X_dup), y_dup, s_pad, kid)

    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad_pv)))
    assert np.all(np.isfinite(np.asarray(hess_pv)))
    assert np.all(np.isfinite(np.asarray(grad_X)))

    # Verify pv gradient and Hessian match central finite differences even with duplicate rows
    grad_fd = np.zeros_like(pv)
    hess_fd = np.zeros((pv.size, pv.size), dtype=np.float64)
    for i in range(pv.size):
        eps_i = 1e-6 * max(abs(float(pv[i])), 0.1)
        dp = np.zeros_like(pv)
        dp[i] = eps_i
        f_p, g_p = _AD_VG(jnp.asarray(pv + dp), X_dup, y_dup, s_pad, kid)
        f_m, g_m = _AD_VG(jnp.asarray(pv - dp), X_dup, y_dup, s_pad, kid)
        grad_fd[i] = (float(f_p) - float(f_m)) / (2.0 * eps_i)
        hess_fd[:, i] = (np.asarray(g_p) - np.asarray(g_m)) / (2.0 * eps_i)

    npt.assert_allclose(np.asarray(grad_pv), grad_fd, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(np.asarray(hess_pv), hess_fd, rtol=1e-6, atol=1e-6)


def test_float32_pre_import_coercion_and_x64_enforcement(canonical_gp_dataset):
    X32_np = canonical_gp_dataset["X_tr"].astype(np.float32)
    y32_np = canonical_gp_dataset["y_tr"].astype(np.float32)
    pv32_np = BasicKernel.pv0.astype(np.float32)

    # Simulate user having disabled jax_enable_x64 globally before calling into k2sc
    jax.config.update("jax_enable_x64", False)
    try:
        x32_jax = jnp.asarray(X32_np)
        assert x32_jax.dtype == jnp.float32
        promoted = as_f64(x32_jax)
        assert jax.config.jax_enable_x64 is True
        assert promoted.dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", True)

    gp = SplitGP(BasicKernel(), splits=None)
    ll = gp.lnlikelihood(pv32_np, y32_np, X32_np)
    assert np.isfinite(ll)
    assert gp._x.dtype == np.float64
    assert gp._y.dtype == np.float64

    # Thread-local context lock-out must raise RuntimeError rather than silently running float32
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64"):
        as_f64([1.0, 2.0])


def test_extreme_low_white_noise_conditioning_and_jitter_fallback():
    """At extreme corner A_t=10, sigma_n=1e-6 with duplicate points, Cholesky failure must not NaN."""
    X_ill = np.zeros((64, 3), dtype=np.float64)
    X_ill[:, 0] = 2300.0  # Identical timestamps & positions -> rank-1 K_0 = 10 * ones((64, 64))
    rng = np.random.default_rng(123)
    y_ill = rng.normal(0.0, 1e-3, size=64)

    # Set A_t = 10^1 = 10, A_xy = 1, sigma_n = 10^-9 (variance 10^-18 < eps_f64 * 11) -> singular K!
    pv_singular = np.array([1.0, 0.25, 0.0, 21.0, 21.0, -9.0], dtype=np.float64)
    s_pad = pad_splits(None)

    ll_jax = float(eval_gp_lnlike_jax(pv_singular, X_ill, y_ill, s_pad, BasicKernel.kernel_id))
    assert not np.isnan(ll_jax)
    assert ll_jax == -np.inf

    gp = SplitGP(BasicKernel(), splits=None)
    ll_oo = gp.lnlikelihood(pv_singular, y_ill, X_ill)
    assert ll_oo == -np.inf
    assert gp.negll(pv_singular, y_ill) == np.inf

    # Full-curve predict / predict_components adaptive relative jitter fallback
    gp_pred = SplitGP(BasicKernel(), splits=None)
    gp_pred.compute(X_ill, pv_singular)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu, err = gp_pred.predict(X_ill, mean_only=False, y=y_ill)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(err))

    gp_comp = SplitGP(BasicKernel(), splits=None)
    gp_comp.compute(X_ill, pv_singular)
    with warnings.catch_warnings(record=True) as caught_comp:
        warnings.simplefilter("always")
        mt, mp = gp_comp.predict_components(pv_singular, y_ill, X_ill, X_ill)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught_comp)
    assert np.all(np.isfinite(mt))
    assert np.all(np.isfinite(mp))


def test_package_metadata_lazy_exports_and_jit_neglnposterior_contract(
    canonical_gp_dataset, monkeypatch
):
    assert k2sc.name == "K2SC"
    assert k2sc.__version__ == "2.0.0"
    with pytest.raises(AttributeError):
        _ = k2sc.nonexistent_symbol_xyz

    # Verify PEP 562 lazy resolution for pipeline.{detrend, K2SCResult} & corrector.K2SCCorrector
    fake_pipeline = types.ModuleType("k2sc.pipeline")
    fake_pipeline.detrend = lambda *a, **kw: "ok_detrend"  # type: ignore[attr-defined]
    fake_pipeline.K2SCResult = type("K2SCResult", (), {})  # type: ignore[attr-defined]
    fake_corrector = types.ModuleType("k2sc.corrector")
    fake_corrector.K2SCCorrector = type("K2SCCorrector", (), {})  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "k2sc.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "k2sc.corrector", fake_corrector)
    assert k2sc.detrend() == "ok_detrend"
    assert k2sc.K2SCResult is fake_pipeline.K2SCResult
    assert k2sc.K2SCCorrector is fake_corrector.K2SCCorrector

    # Verify PR 4 _jit_neglnposterior contract (vmap + value_and_grad in & out of bounds)
    @functools.partial(jax.jit, static_argnames=("kernel_id",))
    def jit_neglnposterior(pv, X_fit, y_fit, splits_padded, lims_low, lims_high, prior_spec, kernel_id):
        in_bounds = jnp.all((pv > lims_low) & (pv <= lims_high))
        safe_pv = jnp.clip(pv, lims_low + 1e-12, lims_high - 1e-12)
        ln_p = eval_ln_prior_jax(safe_pv, prior_spec)
        ln_l = eval_gp_lnlike_jax(safe_pv, X_fit, y_fit, splits_padded, kernel_id)
        raw_nll = -(ln_p + ln_l)
        return jnp.where(in_bounds & jnp.isfinite(raw_nll), raw_nll, jnp.inf)

    k = QuasiPeriodicKernelEP(period=7.3)
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"])
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"])
    s_pad = pad_splits(canonical_gp_dataset["splits"])
    lims_low = jnp.asarray(k.lims[0])
    lims_high = jnp.asarray(k.lims[1])

    pv_ok = jnp.clip(jnp.asarray(k._pv), lims_low + 1e-7, lims_high - 1e-7)
    pv_oob = pv_ok.at[1].set(-0.5)  # Negative LogNorm + inverse gamma parameter!
    batch = jnp.stack([pv_ok, pv_oob])

    vmap_vg = jax.jit(
        jax.vmap(
            jax.value_and_grad(jit_neglnposterior),
            in_axes=(0, None, None, None, None, None, None, None),
        ),
        static_argnames=("kernel_id",),
    )
    vals, grads = vmap_vg(
        batch, X_tr, y_tr, s_pad, lims_low, lims_high, k.prior_spec, k.kernel_id
    )
    assert np.isfinite(float(vals[0]))
    assert np.isposinf(float(vals[1]))
    assert np.all(np.isfinite(np.asarray(grads[0])))
    npt.assert_array_equal(np.asarray(grads[1]), 0.0)
