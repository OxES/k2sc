"""Unit tests for Detrender (src/detrender.py) and deprecated DiffEvol shim (src/de.py)."""

from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import k2sc.detrender as detrender_mod
import matplotlib.pyplot as plt
import numpy as np
import pytest
from k2sc.de import DiffEvol
from k2sc.detrender import Detrender
from k2sc.dtdata import DtData
from k2sc.kernels import (
    BasicKernel,
    QuasiPeriodicKernelEP,
    kernels,
)
from k2sc.optimizers import OptimizationResult


def test_detrender_module_clean_imports():
    """Verify src/detrender.py neither calls mpl.use('Agg') nor imports numpy.linalg.linalg."""
    src_text = inspect.getsource(detrender_mod)
    assert "mpl.use" not in src_text
    assert "numpy.linalg.linalg" not in src_text
    assert ".ptp()" not in src_text


def test_detrender_training_set_nblocks_bspan_issue_19():
    """Verify tr_nblocks=6, tr_bspan=50 produces 6 blocks of >=50 consecutive points (issue 19)."""
    n_pts = 1000
    t = np.arange(n_pts, dtype=np.float64)
    x = 0.1 * np.sin(0.05 * t)
    y = 0.1 * np.cos(0.05 * t)
    flux = 1000.0 + 0.5 * x - 0.3 * y
    inputs = np.column_stack([t, x, y])

    dt = Detrender(
        flux,
        inputs,
        tr_nrandom=0,
        tr_nblocks=6,
        tr_bspan=50,
        seed=19,
    )
    tr_t = dt.tr_data.unmasked_time
    assert tr_t.size == 6 * 50

    # Decompose sorted integer timestamps into contiguous consecutive runs (diff == 1.0)
    breaks = np.flatnonzero(np.diff(tr_t) > 1.0) + 1
    runs = np.split(tr_t, breaks)
    run_lengths = [r.size for r in runs]

    # Every contiguous run must be a multiple of 50 (never 50 tiny blocks of length 6!)
    assert len(runs) <= 6
    for length in run_lengths:
        assert length >= 50
        assert length % 50 == 0
    assert sum(run_lengths) == 300


def test_detrender_seed_reproducibility_and_global_rng_isolation(canonical_gp_dataset):
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    np.random.seed(31415)
    state_before = np.random.get_state()

    dt1 = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        tr_nrandom=16,
        tr_nblocks=3,
        tr_bspan=16,
        seed=99,
    )
    dt2 = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        tr_nrandom=16,
        tr_nblocks=3,
        tr_bspan=16,
        seed=99,
    )

    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]

    np.testing.assert_array_equal(
        dt1.tr_data.unmasked_inputs, dt2.tr_data.unmasked_inputs
    )
    np.testing.assert_array_equal(dt1.tr_data.unmasked_flux, dt2.tr_data.unmasked_flux)


def test_detrender_neglnposterior_bounds_and_jax_grad(canonical_gp_dataset):
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=BasicKernel(),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=7,
    )
    pv = dt.kernel._pv.copy()

    nll_tr = dt.neglnposterior(pv, training=True)
    assert isinstance(nll_tr, float)
    assert np.isfinite(nll_tr)

    # Cross-check against explicit SplitGP.lnlikelihood + kernel.ln_prior
    expected_tr = -(
        float(dt.kernel.ln_prior(pv))
        + dt.gp.lnlikelihood(
            pv, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    np.testing.assert_allclose(nll_tr, expected_tr, rtol=1e-12, atol=1e-12)

    nll_full = dt.neglnposterior(pv, training=False)
    expected_full = -(
        float(dt.kernel.ln_prior(pv))
        + dt.gp.lnlikelihood(pv, dt.data.masked_normalised_flux, dt.data.masked_inputs)
    )
    np.testing.assert_allclose(nll_full, expected_full, rtol=1e-12, atol=1e-12)

    # Out-of-bounds checks
    pv_oob_low = pv.copy()
    pv_oob_low[1] = 0.0
    assert np.isposinf(dt.neglnposterior(pv_oob_low))

    pv_oob_high = pv.copy()
    pv_oob_high[0] = 5.0
    assert np.isposinf(dt.neglnposterior(pv_oob_high))

    # Pure-JAX jit + grad works on both neglnposterior_jax and neglnposterior
    grad_in = np.asarray(jax.jit(jax.grad(dt.neglnposterior_jax))(jnp.asarray(pv)))
    grad_in_alias = np.asarray(jax.jit(jax.grad(dt.neglnposterior))(jnp.asarray(pv)))
    assert np.all(np.isfinite(grad_in))
    assert np.linalg.norm(grad_in) > 0.0
    np.testing.assert_allclose(grad_in, grad_in_alias, rtol=1e-14, atol=1e-14)

    grad_oob = np.asarray(
        jax.jit(jax.grad(dt.neglnposterior_jax))(jnp.asarray(pv_oob_low))
    )
    np.testing.assert_allclose(grad_oob, 0.0, atol=0.0)


def test_detrender_train_lbfgsb_powell_and_optimize(canonical_gp_dataset):
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=BasicKernel(),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=11,
    )
    init_pv = dt.kernel.pv0.copy()
    init_nll = dt.neglnposterior(init_pv)

    # Default L-BFGS-B local training
    pv_lbfgs, ok_lbfgs = dt.train(pv0=init_pv)
    assert isinstance(dt.opt_result, OptimizationResult)
    assert dt.opt_result.method == "L-BFGS-B"
    assert ok_lbfgs is dt.opt_result.success
    assert ok_lbfgs
    assert dt.neglnposterior(pv_lbfgs) <= init_nll
    np.testing.assert_allclose(dt.tr_pv, pv_lbfgs)

    # Explicit Powell local training
    pv_powell, ok_powell = dt.train(pv0=init_pv, method="Powell", maxiter=35)
    assert isinstance(dt.opt_result, OptimizationResult)
    assert dt.opt_result.method == "Powell"
    assert isinstance(ok_powell, bool)
    assert dt.neglnposterior(pv_powell) <= init_nll

    # Global + local optimize()
    opt_res = dt.optimize(
        optimizer="auto", seed=5, n_candidates=64, top_k=2, maxiter=80
    )
    assert isinstance(opt_res, OptimizationResult)
    assert opt_res.success
    assert opt_res.fun <= dt.neglnposterior(pv_lbfgs) + 1e-4
    np.testing.assert_allclose(dt.tr_pv, opt_res.pv)


def test_detrender_predict_compute_components_and_detrend_spatial(canonical_gp_dataset):
    X = canonical_gp_dataset["X"]
    flux = 5000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=BasicKernel(),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=21,
    )
    pv, _ = dt.train()

    # covariance_matrix convenience method
    K_tot = dt.covariance_matrix(pv)
    K1, K2 = dt.covariance_matrix(pv, separate=True)
    assert K_tot.shape == (dt.tr_data.nptm, dt.tr_data.nptm)
    np.testing.assert_allclose(K_tot, K1 + K2, rtol=1e-13, atol=1e-13)

    # predict (mean only & with uncertainty)
    mu = dt.predict(pv)
    mu2, std2 = dt.predict(pv, mean_only=False)
    assert mu.shape == (X.shape[0],)
    np.testing.assert_allclose(mu, mu2, rtol=1e-13, atol=1e-13)
    assert np.all(std2 >= 0.0)

    # predict(components=True) defaults to unmasked_inputs (length N)
    tt_full, tp_full = dt.predict(pv, components=True)
    assert tt_full.shape == (X.shape[0],)
    assert tp_full.shape == (X.shape[0],)
    np.testing.assert_allclose(np.nanmedian(tp_full), dt.data.flux_median, rtol=1e-12)

    # compute_components defaults to masked_inputs (length M == dt.data.nptm).
    # Note: zero-median spatial centring uses median over the queried point set, so
    # total predicted trend (mt + mp) and median-subtracted components match identically.
    mt_masked, mp_masked = dt.compute_components(pv)
    assert mt_masked.shape == (dt.data.nptm,)
    assert mp_masked.shape == (dt.data.nptm,)
    np.testing.assert_allclose(np.nanmedian(mp_masked), dt.data.flux_median, rtol=1e-12)
    # Tolerances here are 1e-10: detrend_spatial wiring correctness, not machine-precision GP
    # arithmetic. Cross-platform Cholesky accumulation (ARM macOS vs x86 Linux, Python 3.11
    # vs 3.12/3.13) introduces ~1–2 ULP (≈1e-12) differences; that is a known BLAS ordering
    # effect. Machine-precision GP tests live in test_gp.py and pass at 1e-12 on all platforms.
    np.testing.assert_allclose(
        mt_masked + mp_masked,
        tt_full[mask] + tp_full[mask],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        mt_masked - np.nanmedian(mt_masked),
        tt_full[mask] - np.nanmedian(tt_full[mask]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        mp_masked - np.nanmedian(mp_masked),
        tp_full[mask] - np.nanmedian(tp_full[mask]),
        rtol=1e-10,
        atol=1e-10,
    )

    # Explicit custom inputs to compute_components
    mt_sub, mp_sub = dt.compute_components(pv, inputs=X[:10])
    assert mt_sub.shape == (10,)
    assert mp_sub.shape == (10,)

    # detrend_spatial preserves masked cadences and fills excluded cadences with NaN
    fd = dt.detrend_spatial(pv)
    assert fd.shape == (X.shape[0],)
    assert np.all(np.isfinite(fd[mask]))
    assert np.all(np.isnan(fd[~mask]))
    expected_fd_masked = flux[mask] - mp_masked + np.nanmedian(mp_masked)
    np.testing.assert_allclose(fd[mask], expected_fd_masked, rtol=1e-10, atol=1e-10)


def test_detrender_plotting_smoke(canonical_gp_dataset, tmp_path: Path):
    X = canonical_gp_dataset["X"]
    flux = 2500.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=BasicKernel(),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=33,
    )
    pv = dt.kernel._pv.copy()
    report_path = tmp_path / "epic_report.png"

    try:
        ax_xy0 = dt.plot_xy(pv=None, plot_wireframe=True)
        assert ax_xy0 is not None
        plt.close(ax_xy0.figure)

        ax_xy1 = dt.plot_xy(pv=pv)
        assert ax_xy1 is not None
        plt.close(ax_xy1.figure)

        ax_t0 = dt.plot_t(pv=None)
        assert ax_t0 is not None
        plt.close(ax_t0.figure)

        ax_t1 = dt.plot_t(pv=pv)
        assert ax_t1 is not None
        assert ax_t1.get_xlabel() == "Time"
        plt.close(ax_t1.figure)

        fig_rep = dt.plot_report(pv=pv, tid=201367065, fname=report_path, maxpt=50)
        assert fig_rep is not None
        assert report_path.exists()
        assert report_path.stat().st_size > 0
        plt.close(fig_rep)
    finally:
        plt.close("all")


def test_diffevol_deprecation_and_global_rng_non_pollution():
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]], dtype=np.float64)
    target = np.array([1.2, -0.7, 2.1], dtype=np.float64)

    def sphere(x):
        return float(np.sum((x - target) ** 2))

    np.random.seed(2026)
    state_before = np.random.get_state()

    with pytest.warns(DeprecationWarning, match="DiffEvol is deprecated"):
        de1 = DiffEvol(sphere, bounds, npop=18, F=0.6, C=0.8, seed=123)
    loc1, val1 = de1.optimize(25)

    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]

    with pytest.warns(DeprecationWarning, match="DiffEvol is deprecated"):
        de2 = DiffEvol(sphere, bounds, npop=18, F=0.6, C=0.8, seed=123)
    loc2, val2 = de2.optimize(25)

    np.testing.assert_allclose(loc1, loc2, rtol=0.0, atol=0.0)
    assert val1 == val2
    assert val1 == de1.minimum_value
    np.testing.assert_allclose(loc1, de1.minimum_location)
    assert isinstance(de1.minimum_index, int)
    assert val1 < 0.05


@pytest.mark.parametrize("kname", list(kernels.keys()))
def test_detrender_neglnposterior_pv0_exact_parity_all_five_kernels(
    canonical_gp_dataset, kname: str
):
    """Detrender.neglnposterior(pv0) must match -(ln_prior + lnlikelihood) to 1e-12 on all 5 kernels."""
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=kernels[kname](),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=5,
    )
    pv0 = dt.kernel.pv0.copy()
    got_tr = dt.neglnposterior(pv0, training=True)
    expected_tr = -(
        float(dt.kernel.ln_prior(pv0))
        + dt.gp.lnlikelihood(
            pv0, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    assert np.isfinite(got_tr)
    np.testing.assert_allclose(got_tr, expected_tr, rtol=1e-12, atol=1e-12)

    got_full = dt.neglnposterior(pv0, training=False)
    expected_full = -(
        float(dt.kernel.ln_prior(pv0))
        + dt.gp.lnlikelihood(pv0, dt.data.masked_normalised_flux, dt.data.masked_inputs)
    )
    assert np.isfinite(got_full)
    np.testing.assert_allclose(got_full, expected_full, rtol=1e-12, atol=1e-12)


def test_detrender_two_column_inputs_match_three_column_zero_y(canonical_gp_dataset):
    """Detrender on 2-column [t, x] must match 3-column [t, x, 0.0] identically."""
    X2 = canonical_gp_dataset["X"][:, :2].copy()
    X3 = np.column_stack([X2, np.zeros(X2.shape[0], dtype=np.float64)])
    flux = 2000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt2 = Detrender(
        flux,
        X2,
        mask=mask,
        splits=splits,
        tr_nrandom=24,
        tr_nblocks=2,
        tr_bspan=12,
        seed=13,
    )
    dt3 = Detrender(
        flux,
        X3,
        mask=mask,
        splits=splits,
        tr_nrandom=24,
        tr_nblocks=2,
        tr_bspan=12,
        seed=13,
    )

    pv = dt2.kernel.pv0.copy()
    np.testing.assert_allclose(
        dt2.neglnposterior(pv), dt3.neglnposterior(pv), rtol=1e-13, atol=1e-13
    )
    pv2, ok2 = dt2.train(maxiter=30)
    pv3, ok3 = dt3.train(maxiter=30)
    assert ok2 and ok3
    np.testing.assert_allclose(pv2, pv3, rtol=1e-10, atol=1e-10)

    mt2, mp2 = dt2.compute_components(pv2)
    mt3, mp3 = dt3.compute_components(pv3)
    np.testing.assert_allclose(mt2, mt3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mp2, mp3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        dt2.detrend_spatial(pv2),
        dt3.detrend_spatial(pv3),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


def test_detrender_cache_invalidation_on_tr_data_and_kernel_mutations(
    canonical_gp_dataset,
):
    """Mutating dt.tr_data.mask, replacing dt.tr_data, or swapping dt.kernel must refresh caches."""
    X = canonical_gp_dataset["X"]
    flux = 1500.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = canonical_gp_dataset["splits"]

    dt = Detrender(
        flux,
        X,
        mask=mask,
        splits=splits,
        kernel=BasicKernel(),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=23,
    )
    pv_b = dt.kernel.pv0.copy()
    val_initial = dt.neglnposterior(pv_b, training=True)

    # 1. Mutate dt.tr_data.mask via setter
    new_mask = dt.tr_data.mask.copy()
    new_mask[:12] = False
    dt.tr_data.mask = new_mask
    val_masked = dt.neglnposterior(pv_b, training=True)
    expected_masked = -(
        float(dt.kernel.ln_prior(pv_b))
        + dt.gp.lnlikelihood(
            pv_b, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    assert not np.isclose(val_masked, val_initial, atol=1e-3)
    np.testing.assert_allclose(val_masked, expected_masked, rtol=1e-12, atol=1e-12)

    # 2. Mutate dt.tr_data._mask in place
    dt.tr_data._mask[12:18] = False
    val_inplace = dt.neglnposterior(pv_b, training=True)
    expected_inplace = -(
        float(dt.kernel.ln_prior(pv_b))
        + dt.gp.lnlikelihood(
            pv_b, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    assert not np.isclose(val_inplace, val_masked, atol=1e-3)
    np.testing.assert_allclose(val_inplace, expected_inplace, rtol=1e-12, atol=1e-12)

    # 3. Replace dt.tr_data object completely
    dt.tr_data = DtData(flux[:30], X[:30])
    val_replaced = dt.neglnposterior(pv_b, training=True)
    expected_replaced = -(
        float(dt.kernel.ln_prior(pv_b))
        + dt.gp.lnlikelihood(
            pv_b, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    np.testing.assert_allclose(val_replaced, expected_replaced, rtol=1e-12, atol=1e-12)

    # 4. Swap dt.kernel to QuasiPeriodicKernelEP
    dt.kernel = QuasiPeriodicKernelEP(period=7.3)
    pv_qp = dt.kernel._pv.copy()
    val_swapped = dt.neglnposterior(pv_qp, training=True)
    expected_swapped = -(
        float(dt.kernel.ln_prior(pv_qp))
        + dt.gp.lnlikelihood(
            pv_qp, dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs
        )
    )
    assert dt.gp.kernel is dt.kernel
    assert np.isfinite(val_swapped)
    np.testing.assert_allclose(val_swapped, expected_swapped, rtol=1e-12, atol=1e-12)


def test_detrender_empty_mask_plots_do_not_crash(canonical_gp_dataset, tmp_path: Path):
    """All-masked DtData (nptm == 0) must render empty plot_xy/plot_t/plot_report cleanly."""
    X = canonical_gp_dataset["X"]
    flux = 1000.0 * (1.0 + canonical_gp_dataset["y"])
    empty_mask = np.zeros(X.shape[0], dtype=bool)
    dt = Detrender(flux, X, mask=empty_mask, seed=0)
    assert dt.data.nptm == 0
    pv = dt.kernel.pv0.copy()
    out_png = tmp_path / "empty_report.png"

    try:
        ax_xy = dt.plot_xy(pv=pv)
        assert ax_xy is not None
        plt.close(ax_xy.figure)

        ax_t0 = dt.plot_t(pv=None)
        assert ax_t0 is not None
        plt.close(ax_t0.figure)

        ax_t1 = dt.plot_t(pv=pv)
        assert ax_t1 is not None
        plt.close(ax_t1.figure)

        fig = dt.plot_report(pv=pv, tid=201367065, fname=out_png)
        assert fig is not None
        assert out_png.exists() and out_png.stat().st_size > 0
        plt.close(fig)
    finally:
        plt.close("all")


def test_segmented_detrender_full_workflow(canonical_gp_dataset, tmp_path: Path):
    """Deprecated SegmentedDetrender must support neglnposterior, train, optimize, predict, plot_report."""
    import importlib

    with pytest.deprecated_call():
        import k2sc.segdetrender as seg_mod

        importlib.reload(seg_mod)

    X = canonical_gp_dataset["X"]
    flux = 2000.0 * (1.0 + canonical_gp_dataset["y"])
    mask = canonical_gp_dataset["mask"]
    splits = list(canonical_gp_dataset["splits"])

    with pytest.deprecated_call():
        sdet = seg_mod.SegmentedDetrender(
            flux,
            X,
            splits=splits,
            mask=mask,
            tr_nrandom=24,
            tr_bspan=12,
            tr_nblocks=2,
            seed=17,
        )

    pv0 = sdet.kernel.pv0.copy()
    nll = sdet.neglnposterior(pv0)
    assert isinstance(nll, float) and np.isfinite(nll)

    pv_tr, ok = sdet.train(maxiter=25)
    assert ok and np.all(np.isfinite(pv_tr))

    opt_res = sdet.optimize(
        optimizer="multistart", n_candidates=16, top_k=2, maxiter=25, seed=3
    )
    assert isinstance(opt_res, OptimizationResult)
    assert np.isfinite(opt_res.fun)

    mt, mp = sdet.predict(opt_res.pv, components=True)
    assert mt.shape == (X.shape[0],)
    assert mp.shape == (X.shape[0],)

    # Legacy predict(pv) without components=True must still work
    mu_legacy = sdet.predict(opt_res.pv)
    assert mu_legacy.shape == (sdet.data.nptm,)

    rep_path = tmp_path / "seg_report.png"
    try:
        fig = sdet.plot_report(opt_res.pv, tid="SEG-TEST", fname=rep_path, maxpt=40)
        assert fig is not None
        assert rep_path.exists() and rep_path.stat().st_size > 0
        plt.close(fig)
    finally:
        plt.close("all")


def test_split_and_gap_aware_training_set_blocks_and_stratification():
    """Blocks must never straddle splits or large timestamp gaps, and nrandom must stratify segments."""
    # Construct 3 segments:
    # - Seg 0: 120 pts in [0, 119]
    # - Split at t = 119.5
    # - Seg 1: 120 pts in [120, 239]
    # - Large 100-day gap (> 10 * median_dt = 10)
    # - Seg 2: 12 pts in [340, 351] (too short for bspan=25, must still receive stratified nrandom!)
    t0 = np.arange(0.0, 120.0, 1.0)
    t1 = np.arange(120.0, 240.0, 1.0)
    t2 = np.arange(340.0, 352.0, 1.0)
    t = np.concatenate([t0, t1, t2])
    inputs = np.column_stack([t, np.sin(0.1 * t), np.cos(0.1 * t)])
    flux = 1000.0 + 0.5 * inputs[:, 1]
    splits = [119.5]

    data = DtData(flux, inputs)

    # 1. Pure block sampling (nrandom=0): NO contiguous run may cross split 119.5 or the gap [239, 340]
    for seed in range(10):
        tr_b = data.create_training_set(
            nrandom=0, nblocks=4, bspan=25, seed=seed, splits=splits
        )
        tr_t = tr_b.unmasked_time
        assert tr_t.size == 100
        # Decompose into contiguous unit-step runs; every run must lie inside a single segment
        breaks = np.flatnonzero(np.diff(tr_t) > 1.0) + 1
        for run in np.split(tr_t, breaks):
            # Must not straddle split 119.5
            assert not (np.any(run <= 119.5) and np.any(run > 119.5))
            # Must not straddle the 100-day gap
            assert not (np.any(run <= 239.0) and np.any(run >= 340.0))
            assert run.size >= 25 and run.size % 25 == 0

    # 2. Mixed blocks + stratified random draws: short segment 2 (12 pts) must NEVER be starved!
    for seed in range(10):
        tr_m = data.create_training_set(
            nrandom=30, nblocks=2, bspan=25, seed=seed, splits=splits
        )
        tr_t = tr_m.unmasked_time
        assert tr_t.size == 80
        assert np.count_nonzero(tr_t <= 119.5) >= 1
        assert np.count_nonzero((tr_t > 119.5) & (tr_t <= 239.0)) >= 1
        assert np.count_nonzero(tr_t >= 340.0) >= 1

    # 3. Verify Detrender.__init__ wires splits into create_training_set automatically
    dt = Detrender(
        flux,
        inputs,
        splits=splits,
        tr_nrandom=0,
        tr_nblocks=4,
        tr_bspan=25,
        seed=42,
    )
    dt_t = dt.tr_data.unmasked_time
    breaks = np.flatnonzero(np.diff(dt_t) > 1.0) + 1
    for run in np.split(dt_t, breaks):
        assert not (np.any(run <= 119.5) and np.any(run > 119.5))
        assert not (np.any(run <= 239.0) and np.any(run >= 340.0))
        assert run.size >= 25 and run.size % 25 == 0
