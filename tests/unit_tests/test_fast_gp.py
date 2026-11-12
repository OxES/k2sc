"""Unit tests for vectorized tinygp kernels, analytic GP gradients, and two-stage L-BFGS-B."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import tinygp
from k2sc._jax_env import configure_single_thread_env
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
    _jit_val_and_grad,
    _jit_val_and_grad_precomputed,
    compute_closed_lims,
    polish_lbfgsb,
)
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
    canonical_gp_dataset, kname: str
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
    X_tr, y_tr, s_pad, lims_low, lims_high = dt._objective_tensors(training=True)
    precomp = dt._training_precomp()

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

    # Reassigning dt.gp.splits invalidates cached _precomp_tr automatically
    dt.gp.splits = None
    val_unsplit = dt.neglnposterior(np.asarray(pv), training=True)
    expected_unsplit = -(
        float(dt.kernel.ln_prior(pv))
        + dt.gp.lnlikelihood(np.asarray(pv), dt.tr_data.masked_normalised_flux, dt.tr_data.masked_inputs)
    )
    npt.assert_allclose(val_unsplit, expected_unsplit, rtol=1e-12, atol=1e-12)


def test_singular_cholesky_precomputed_returns_neginf_and_zero_grad():
    """Singular rank-1 covariance must return (-inf, zeros) without NaN propagation."""
    X_dup = np.zeros((16, 3), dtype=np.float64)
    X_dup[:, 0] = 2300.0
    y_dup = np.linspace(-1e-3, 1e-3, 16, dtype=np.float64)
    precomp = precompute_pairwise_jax(jnp.asarray(X_dup), pad_splits(None))
    pv_sing = jnp.array([1.0, 0.25, 0.0, 21.0, 21.0, -9.0], dtype=jnp.float64)

    lp = eval_gp_lnlike_precomputed_jax(pv_sing, precomp, jnp.asarray(y_dup), KERNEL_BASIC)
    lp2, grad2 = eval_gp_lnlike_and_grad_precomputed_jax(
        pv_sing, precomp, jnp.asarray(y_dup), KERNEL_BASIC
    )
    assert float(lp) == -np.inf
    assert float(lp2) == -np.inf
    npt.assert_array_equal(np.asarray(grad2), np.zeros(6))


def test_multistart_two_stage_lbfgsb_map_convergence_and_reduced_nfev(
    canonical_gp_dataset,
):
    """Two-stage MultiStartLBFGSBOptimizer must recover the exact MAP basin with fewer nfev."""
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    splits = canonical_gp_dataset["splits"]

    kernel = QuasiPeriodicKernel(period=7.3)
    opt = MultiStartLBFGSBOptimizer(
        n_candidates=128, top_k=5, maxiter=200, ftol=1e-15, gtol=1e-6
    )
    res_two_stage = opt.optimize(X_tr, y_tr, splits, kernel, seed=2026)

    assert res_two_stage.success
    assert np.isfinite(res_two_stage.fun)
    assert res_two_stage.grad_norm < 1e-3
    assert abs(float(res_two_stage.pv[2]) - 7.3) / 7.3 < 0.05

    # Compare against exhaustive full 200-iter single-stage polish on the exact same 5 seeds
    from k2sc.optimizers import (
        _eval_candidates_chunked,
        _generate_candidates,
        _select_diverse_top_k,
    )

    rng = np.random.default_rng(2026)
    X_jax = jnp.asarray(X_tr, dtype=jnp.float64)
    y_jax = jnp.asarray(y_tr, dtype=jnp.float64)
    s_pad = pad_splits(splits)
    precomp = precompute_pairwise_jax(X_jax, s_pad)
    lims_low = jnp.asarray(kernel.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(kernel.lims[1], dtype=jnp.float64)
    closed_lims = compute_closed_lims(kernel.lims)
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
    seeds = _select_diverse_top_k(
        cands,
        energies,
        k_bounds[:, 0],
        k_bounds[:, 1],
        top_k=5,
        min_dist=0.15,
        is_periodic=True,
    )

    single_stage_nfev = int(cands.shape[0])
    best_single_fun = float("inf")
    for x_seed in seeds:
        pol = polish_lbfgsb(
            x_seed,
            X_jax,
            y_jax,
            s_pad,
            lims_low,
            lims_high,
            kernel.prior_spec,
            kernel.kernel_id,
            closed_lims,
            ftol=1e-15,
            gtol=1e-6,
            maxiter=200,
            precomp=precomp,
        )
        single_stage_nfev += pol.nfev
        best_single_fun = min(best_single_fun, pol.fun)

    npt.assert_allclose(res_two_stage.fun, best_single_fun, rtol=1e-5, atol=1e-4)
    assert res_two_stage.nfev < single_stage_nfev


def test_configure_single_thread_env_idempotence(monkeypatch):
    """Verify configure_single_thread_env sets defaults, preserves overrides, and is idempotent."""
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
    assert "--xla_cpu_multi_thread_eigen=false" in xla_first
    assert "intra_op_parallelism_threads=1" in xla_first

    # Second call must be strictly idempotent (no duplicate flags)
    configure_single_thread_env()
    assert os.environ["XLA_FLAGS"] == xla_first

    # Explicit caller overrides and existing unrelated XLA_FLAGS must be preserved
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    configure_single_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=1 "
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    configure_single_thread_env()
    assert (
        os.environ["XLA_FLAGS"].count("--xla_cpu_multi_thread_eigen") == 1
    )
