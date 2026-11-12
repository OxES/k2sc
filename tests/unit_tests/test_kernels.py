"""Unit tests for tinygp kernels, SafeL1Distance, PiecewiseSpatialKernel, and stateless DtKernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import reference_gp_numpy as ref
import tinygp
from k2sc.gp import SplitGP
from k2sc.kernels import (
    KERNEL_IDS,
    MAX_SPLITS,
    BasicKernel,
    BasicKernelEP,
    PeriodicKernel,
    PiecewiseSpatialKernel,
    QuasiPeriodicKernel,
    QuasiPeriodicKernelEP,
    SafeL1Distance,
    build_tinygp_kernels,
    kernels,
    pad_splits,
)


def test_tinygp_version_and_jax_x64():
    parts = tuple(int(tok) for tok in tinygp.__version__.split(".")[:2])
    assert parts >= (0, 3), f"Expected tinygp >= 0.3.0, got {tinygp.__version__}"
    assert jax.config.jax_enable_x64 is True


def test_safe_l1_distance_shapes_and_grads():
    dist = SafeL1Distance()

    # 0-D scalar inputs (as passed by tinygp.transforms.Subspace(axis=int, ...))
    d_pos = dist.distance(jnp.array(2.0), jnp.array(5.0))
    assert d_pos.shape == ()
    assert d_pos.dtype == jnp.float64
    npt.assert_allclose(float(d_pos), 3.0, rtol=0.0, atol=1e-16)

    d_zero = dist.distance(jnp.array(0.0), jnp.array(0.0))
    assert d_zero.shape == ()
    assert float(d_zero) == 0.0

    d2_pos = dist.squared_distance(jnp.array(2.0), jnp.array(5.0))
    assert d2_pos.shape == ()
    npt.assert_allclose(float(d2_pos), 9.0, rtol=0.0, atol=1e-16)

    # 1-D vector metric consistency: squared_distance == distance**2 (L1^2, NOT L2^2)
    v1 = jnp.array([1.0, -2.0], dtype=jnp.float64)
    v2 = jnp.array([4.0, 3.0], dtype=jnp.float64)
    assert dist.distance(v1, v2).shape == ()
    npt.assert_allclose(float(dist.distance(v1, v2)), 8.0, rtol=0.0, atol=1e-16)
    npt.assert_allclose(
        float(dist.squared_distance(v1, v2)),
        float(dist.distance(v1, v2) ** 2),
        rtol=0.0,
        atol=1e-16,
    )
    assert float(dist.squared_distance(v1, v2)) == 64.0

    # Zero-lag autodiff safety (1st and 2nd order, 0-D scalar and 1-D vector)
    g0 = jax.jit(jax.grad(lambda x: dist.distance(x, jnp.array(0.0))))(jnp.array(0.0))
    h0 = jax.jit(jax.hessian(lambda x: dist.distance(x, jnp.array(0.0))))(jnp.array(0.0))
    assert g0.shape == () and float(g0) == 0.0
    assert h0.shape == () and float(h0) == 0.0

    gv0 = jax.jit(jax.grad(lambda x: dist.distance(x, v1)))(v1)
    assert not np.any(np.isnan(np.asarray(gv0)))
    npt.assert_array_equal(np.asarray(gv0), np.zeros_like(v1))


def test_pad_splits_and_none_vs_empty_equivalence():
    s_none = pad_splits(None)
    s_empty = pad_splits([])
    assert s_none.shape == (MAX_SPLITS,)
    assert s_empty.shape == (MAX_SPLITS,)
    assert s_none.dtype == jnp.float64
    assert jnp.all(jnp.isposinf(s_none))
    npt.assert_array_equal(np.asarray(s_none), np.asarray(s_empty))

    s_two = pad_splits([2354.0, 2326.0])
    assert s_two.shape == (MAX_SPLITS,)
    npt.assert_allclose(np.asarray(s_two[:2]), [2326.0, 2354.0])
    assert np.all(np.isposinf(np.asarray(s_two[2:])))

    many = np.linspace(1.0, 20.0, 20)
    s_many = pad_splits(many)
    assert s_many.shape == (2 * MAX_SPLITS,)
    npt.assert_allclose(np.asarray(s_many[:20]), many)
    assert np.all(np.isposinf(np.asarray(s_many[20:])))


def test_piecewise_spatial_kernel_psd_diag_tracing_and_masked_boundary(canonical_gp_dataset):
    X = canonical_gp_dataset["X"]
    X_fit = canonical_gp_dataset["X_fit"]
    splits = canonical_gp_dataset["splits"]

    base_xy = (
        1e-5
        * tinygp.transforms.Subspace(1, tinygp.kernels.ExpSquared(scale=0.25))
        * tinygp.transforms.Subspace(2, tinygp.kernels.ExpSquared(scale=0.25))
    )
    k_pos = PiecewiseSpatialKernel(base_xy, pad_splits(splits))

    # Dynamic traced leaf check: splits_padded must be a dynamic JAX PyTree leaf
    leaves = jax.tree_util.tree_leaves(k_pos)
    assert any(isinstance(leaf, jax.Array) and leaf.shape == (MAX_SPLITS,) for leaf in leaves)

    @jax.jit
    def eval_k2_trace(X_in: jax.Array, s_pad: jax.Array) -> jax.Array:
        kp = PiecewiseSpatialKernel(base_xy, s_pad)
        return jnp.sum(kp(X_in, X_in))

    _ = eval_k2_trace(jnp.asarray(X[:16]), pad_splits(None))
    _ = eval_k2_trace(jnp.asarray(X[:16]), pad_splits([2326.0]))
    _ = eval_k2_trace(jnp.asarray(X[:16]), pad_splits(splits))
    assert eval_k2_trace._cache_size() == 1

    @jax.jit
    def eval_kp_mat(kp_in: PiecewiseSpatialKernel, X1: jax.Array, X2: jax.Array) -> jax.Array:
        return kp_in(X1, X2)

    @jax.jit
    def eval_kp_diag(kp_in: PiecewiseSpatialKernel, X_in: jax.Array) -> jax.Array:
        return jax.vmap(kp_in.evaluate_diag)(X_in)

    # evaluate_diag(X) == diag(evaluate(X, X)) without overriding evaluate_diag
    X_j = jnp.asarray(X)
    diag_direct = np.asarray(eval_kp_diag(k_pos, X_j))
    diag_from_mat = np.diag(np.asarray(eval_kp_mat(k_pos, X_j, X_j)))
    npt.assert_allclose(diag_direct, diag_from_mat, rtol=1e-15, atol=1e-15)

    # PSD check across normal splits, out-of-range splits, and unsorted query rows
    rng = np.random.default_rng(7)
    perm = rng.permutation(X.shape[0])
    X_perm_j = jnp.asarray(X[perm])
    for sp in (None, [], splits, [1000.0], [9999.0], [1000.0, 2326.0, 9999.0]):
        kp = PiecewiseSpatialKernel(base_xy, pad_splits(sp))
        K2_perm = np.asarray(eval_kp_mat(kp, X_perm_j, X_perm_j))
        eigvals = np.linalg.eigvalsh(0.5 * (K2_perm + K2_perm.T))
        assert np.min(eigvals) >= -1e-13

    # N != M masked-boundary test: every unmasked cadence i in X_fit has exact self-covariance
    # K2(X_fit[i], X_fit[i]) == base_xy(X_fit[i], X_fit[i]) inside K2(X, X_fit)
    K2_rect = np.asarray(eval_kp_mat(k_pos, X_j, jnp.asarray(X_fit)))
    mask_idx = np.flatnonzero(canonical_gp_dataset["mask"])
    for col_j, row_i in enumerate(mask_idx):
        npt.assert_allclose(K2_rect[row_i, col_j], 1e-5, rtol=1e-15, atol=1e-15)


def test_all_five_kernels_init_bounds_lims_and_pv0_immutability():
    expected_classes = {
        "basic": BasicKernel,
        "basic_ep": BasicKernelEP,
        "periodic": PeriodicKernel,
        "quasiperiodic": QuasiPeriodicKernel,
        "quasiperiodic_ep": QuasiPeriodicKernelEP,
    }
    assert kernels == expected_classes

    snapshots = {name: cls.pv0.copy() for name, cls in expected_classes.items()}

    # Exercise custom keyword overrides and custom p0 vectors
    qp_default = QuasiPeriodicKernel()
    assert qp_default.period == 10.0
    assert qp_default.evolution_iscale == 0.01

    qp_override = QuasiPeriodicKernel(period=4.2, evolution_scale=20.0)
    assert qp_override.period == 4.2
    npt.assert_allclose(qp_override.evolution_iscale, 1.0 / 400.0, rtol=1e-15)
    npt.assert_allclose(qp_override._pv[3], 1.0 / 400.0, rtol=1e-15)

    qp_iscale = QuasiPeriodicKernel(period=6.5, evolution_iscale=0.09)
    assert qp_iscale.period == 6.5
    assert qp_iscale.evolution_iscale == 0.09

    custom_p0 = np.array([-5.1, 0.4, 7.7, 0.05, -4.6, 15.0, 19.0, -3.2], dtype=np.float64)
    custom_p0_copy = custom_p0.copy()
    qp_from_p0 = QuasiPeriodicKernel(p0=custom_p0)
    # Caller-supplied p0 must not be clobbered when period/evolution are None
    npt.assert_allclose(qp_from_p0._pv, custom_p0_copy, rtol=0.0, atol=0.0)
    qp_from_p0._pv[0] = -1.23
    npt.assert_allclose(custom_p0, custom_p0_copy, rtol=0.0, atol=0.0)

    pk_override = PeriodicKernel(period=3.75)
    assert pk_override.period == 3.75
    assert pk_override._pv[2] == 3.75

    # Class-level pv0 arrays must remain completely untouched
    for name, cls in expected_classes.items():
        npt.assert_array_equal(cls.pv0, snapshots[name])
        inst = cls()
        assert inst.name in KERNEL_IDS
        assert name in KERNEL_IDS
        assert np.isfinite(float(inst.ln_prior(inst._pv)))

        bounds = np.asarray(inst.bounds, dtype=np.float64)
        assert bounds.shape == (inst.npar, 2)
        assert np.all(inst.lims[0] < bounds[:, 0]), f"{name} lower bounds violate lims[0]"
        assert np.all(bounds[:, 0] < bounds[:, 1]), f"{name} inverted bounds"
        assert np.all(bounds[:, 1] < inst.lims[1]), f"{name} upper bounds violate lims[1]"


@pytest.mark.parametrize("k_cls", [QuasiPeriodicKernel, QuasiPeriodicKernelEP, PeriodicKernel])
@pytest.mark.parametrize("p1_val", [0.05, 0.25, 1.0, 1.45, 1.99])
def test_quasiperiodic_and_periodic_psd_across_full_p1_support(
    canonical_gp_dataset, k_cls, p1_val
):
    """Issue #30 regression: p1 >= 1 in (0, 2] must remain strictly PSD (Gamma = 1/p1 > 0)."""
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    k = k_cls()
    pv = k.pv0.copy()
    pv[1] = p1_val

    gp = SplitGP(k, splits=canonical_gp_dataset["splits"])
    K1, K2 = gp._covariance_matrix(X_tr, pv=pv, separate=True)
    eig1 = np.linalg.eigvalsh(0.5 * (K1 + K1.T))
    eig_tot = np.linalg.eigvalsh(0.5 * (K1 + K2 + (K1 + K2).T))
    assert np.min(eig1) >= -1e-13
    assert np.min(eig_tot) >= -1e-13
    assert np.isfinite(gp.lnlikelihood(pv, y_tr, X_tr))


@pytest.mark.parametrize("kname", list(kernels.keys()))
@pytest.mark.parametrize("use_splits", [False, True])
def test_all_five_kernels_matrix_parity_vs_reference_numpy(
    canonical_gp_dataset, kname, use_splits
):
    X = canonical_gp_dataset["X"]
    X_tr = canonical_gp_dataset["X_tr"]
    splits = canonical_gp_dataset["splits"] if use_splits else None

    k = kernels[kname]()
    pv = k.pv0.copy()

    # Stateless check: covariance right after __init__ matches covariance after set_pv(pv)
    gp = SplitGP(k, splits=splits)
    K1_init, K2_init = gp._covariance_matrix(X_tr, separate=True)
    K1_set, K2_set = gp._covariance_matrix(X_tr, pv=pv, separate=True)
    npt.assert_allclose(K1_init, K1_set, rtol=0.0, atol=1e-15)
    npt.assert_allclose(K2_init, K2_set, rtol=0.0, atol=1e-15)

    K1_ref, K2_ref, wn_var_ref = ref.kernel_matrices_numpy(kname, pv, X_tr, None, splits=splits)
    npt.assert_allclose(K1_set, K1_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(K2_set, K2_ref, rtol=1e-12, atol=1e-12)

    K_wn = gp._covariance_matrix(X_tr, pv=pv, separate=False, add_wn=True)
    npt.assert_allclose(
        K_wn,
        K1_ref + K2_ref + wn_var_ref * np.eye(X_tr.shape[0]),
        rtol=1e-12,
        atol=1e-12,
    )

    # Rectangular cross-covariance (N=128, N_tr=64)
    K1_rect, K2_rect = gp._covariance_matrix(X, X_tr, pv=pv, separate=True)
    K1_rect_ref, K2_rect_ref, _ = ref.kernel_matrices_numpy(kname, pv, X, X_tr, splits=splits)
    npt.assert_allclose(K1_rect, K1_rect_ref, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(K2_rect, K2_rect_ref, rtol=1e-12, atol=1e-12)

    # Direct build_tinygp_kernels parity (string & int kernel_id)
    _, _, diag_j = build_tinygp_kernels(pv, pad_splits(splits), kname)
    npt.assert_allclose(float(diag_j), wn_var_ref, rtol=1e-14, atol=1e-16)
    t_ref = X_tr[0, 0]
    X_tr_rel = jnp.asarray(X_tr - np.array([t_ref, 0.0, 0.0]))
    s_rel = pad_splits(None if splits is None else np.asarray(splits) - t_ref)
    k1_s, k2_s, _ = build_tinygp_kernels(pv, s_rel, k.kernel_id)
    for idx_j in (1, 50):
        npt.assert_allclose(
            float(k1_s.evaluate(X_tr_rel[0], X_tr_rel[idx_j])),
            K1_ref[0, idx_j],
            rtol=1e-12,
            atol=1e-12,
        )
        npt.assert_allclose(
            float(k2_s.evaluate(X_tr_rel[0], X_tr_rel[idx_j])),
            K2_ref[0, idx_j],
            rtol=1e-12,
            atol=1e-12,
        )
