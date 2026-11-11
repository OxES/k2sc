"""Unit tests for JAX/SciPy/Optimistix hyperparameter optimizers (src/optimizers.py)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from k2sc.detrender import Detrender
from k2sc.kernels import BasicKernel, QuasiPeriodicKernel, pad_splits
from k2sc.optimizers import (
    MultiStartLBFGSBOptimizer,
    OptimistixOptimizer,
    OptimizationResult,
    VectorizedDEOptimizer,
    _jit_neglnposterior,
    _jit_val_and_grad,
    _jit_vmap_neglnposterior,
    _make_scipy_host_callback,
    _pv_to_unconstrained_np,
    _unconstrained_to_pv_jax,
    compute_closed_lims,
    get_optimizer,
)
from k2sc.priors import UniformPrior, build_prior_spec


@pytest.fixture(scope="module")
def spotted_star_harmonic_dataset() -> dict[str, object]:
    """Synthetic K2 spotted star (P_true = 7.3 d) + roll-drift systematics (N=240, N_tr=64)."""
    rng = np.random.default_rng(202604)
    n_total = 240
    p_true = 7.3

    t = np.linspace(2300.0, 2380.0, n_total, dtype=np.float64)
    t += rng.uniform(-0.015, 0.015, size=n_total)
    t.sort()

    # K2 6-hour roll-drift sawtooth + slow pointing drift across one split
    splits = [2340.0]
    roll_phase = ((t - 2300.0) / 0.254) % 1.0
    x = 0.35 * (roll_phase - 0.5) + 0.05 * np.sin(2.0 * np.pi * (t - 2300.0) / 45.0)
    y_pos = -0.28 * (roll_phase - 0.5) + 0.04 * np.cos(2.0 * np.pi * (t - 2300.0) / 60.0)
    inputs = np.column_stack([t, x, y_pos]).astype(np.float64)

    # Piecewise spatial sensitivity (~3000 ppm peak-to-peak roll systematics)
    pos_sys = np.where(
        t <= splits[0],
        4.5e-3 * x - 3.2e-3 * y_pos + 1.5e-3 * (x**2),
        -4.0e-3 * x + 3.6e-3 * y_pos - 1.2e-3 * (y_pos**2),
    )

    # Quasi-periodic spotted-star modulation at P_true = 7.3 d with mild spot evolution
    envelope = 1.0 + 0.15 * np.cos(2.0 * np.pi * (t - 2300.0) / 70.0)
    spot_fundamental = 3.5e-3 * np.sin(2.0 * np.pi * (t - 2300.0) / p_true + 0.4)
    spot_asymmetry = 0.80e-3 * np.cos(4.0 * np.pi * (t - 2300.0) / p_true - 0.3)
    astro = envelope * (spot_fundamental + spot_asymmetry)

    noise = rng.normal(0.0, 1.2e-4, size=n_total)
    flux = 10000.0 * (1.0 + astro + pos_sys + noise)

    # Exact 64-point hybrid block + random training subset (2 blocks of 16 + 32 random = 64)
    dt_prep = Detrender(
        flux,
        inputs,
        splits=splits,
        kernel=QuasiPeriodicKernel(period=14.6),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=42,
    )
    X_tr = dt_prep.tr_data.masked_inputs
    y_tr = dt_prep.tr_data.masked_normalised_flux

    return {
        "p_true": p_true,
        "t": t,
        "inputs": inputs,
        "flux": flux,
        "astro_norm": astro,
        "pos_norm": pos_sys,
        "splits": splits,
        "X_tr": X_tr,
        "y_tr": y_tr,
    }


def test_get_optimizer_and_jaxopt_deprecation():
    opt_auto = get_optimizer("auto")
    assert isinstance(opt_auto, MultiStartLBFGSBOptimizer)

    opt_ms = get_optimizer("multistart", n_candidates=64, top_k=2)
    assert isinstance(opt_ms, MultiStartLBFGSBOptimizer)
    assert opt_ms.n_candidates == 64
    assert opt_ms.top_k == 2

    opt_de = get_optimizer("de", de_npop=32, de_niter=10, de_max_time=30.0)
    assert isinstance(opt_de, VectorizedDEOptimizer)
    assert opt_de.npop == 32
    assert opt_de.niter == 10
    assert opt_de.max_time == 30.0

    opt_optx = get_optimizer("optimistix", n_candidates=64, top_k=2)
    assert isinstance(opt_optx, OptimistixOptimizer)

    # Passing an already-constructed optimizer returns it unchanged
    assert get_optimizer(opt_ms) is opt_ms

    with pytest.raises(ValueError, match=r"jaxopt.*optimistix"):
        get_optimizer("jaxopt")

    with pytest.raises(ValueError, match="Unknown optimizer"):
        get_optimizer("nonexistent_backend")


def test_closed_lims_and_logit_map_invariance():
    kernel = QuasiPeriodicKernel(period=7.3)
    closed = compute_closed_lims(kernel.lims)
    assert len(closed) == kernel.npar

    for k, (lo, hi) in enumerate(closed):
        a_k = float(kernel.lims[0, k])
        b_k = float(kernel.lims[1, k])
        eps_expected = 1e-7 * min(1.0, b_k - a_k)
        assert np.isclose(lo, a_k + eps_expected, rtol=0.0, atol=1e-15)
        assert np.isclose(hi, b_k - eps_expected, rtol=0.0, atol=1e-15)
        assert a_k < lo < hi < b_k

    c_arr = np.asarray(closed, dtype=np.float64)
    pv_interior = np.clip(kernel._pv, c_arr[:, 0] + 1e-4, c_arr[:, 1] - 1e-4)
    z = _pv_to_unconstrained_np(pv_interior, c_arr[:, 0], c_arr[:, 1])
    pv_roundtrip = np.asarray(
        _unconstrained_to_pv_jax(
            jnp.asarray(z), jnp.asarray(c_arr[:, 0]), jnp.asarray(c_arr[:, 1])
        )
    )
    np.testing.assert_allclose(pv_roundtrip, pv_interior, rtol=1e-12, atol=1e-12)

    # Boundary-touching pv (kernel._pv[0] == -6.0 == lims_low[0]) maps safely inside closed_lims
    z_bound = _pv_to_unconstrained_np(kernel._pv, c_arr[:, 0], c_arr[:, 1])
    pv_bound_rt = np.asarray(
        _unconstrained_to_pv_jax(
            jnp.asarray(z_bound), jnp.asarray(c_arr[:, 0]), jnp.asarray(c_arr[:, 1])
        )
    )
    assert np.all(pv_bound_rt > kernel.lims[0])
    assert np.all(pv_bound_rt < kernel.lims[1])

    # Verify uniform-prior log-density has zero gradient wrt z (no Jacobian bias toward midpoint)
    u_spec = build_prior_spec([UniformPrior(-6.0, 1.0)])
    c_u = compute_closed_lims([[-6.0], [1.0]])
    lo_j = jnp.asarray([c_u[0][0]], dtype=jnp.float64)
    hi_j = jnp.asarray([c_u[0][1]], dtype=jnp.float64)

    def _u_loss(z_vec):
        theta = _unconstrained_to_pv_jax(z_vec, lo_j, hi_j)
        from k2sc.priors import eval_ln_prior_jax

        return -eval_ln_prior_jax(theta, u_spec)

    grad_off_center = np.asarray(jax.grad(_u_loss)(jnp.asarray([2.3], dtype=jnp.float64)))
    np.testing.assert_allclose(grad_off_center, [0.0], atol=1e-15)


def test_jit_neglnposterior_out_of_bounds_vmap_and_grad_safety(canonical_gp_dataset):
    X_tr = jnp.asarray(canonical_gp_dataset["X_tr"], dtype=jnp.float64)
    y_tr = jnp.asarray(canonical_gp_dataset["y_tr"], dtype=jnp.float64)
    splits_pad = pad_splits(canonical_gp_dataset["splits"])

    kernel = BasicKernel()
    lims_low = jnp.asarray(kernel.lims[0], dtype=jnp.float64)
    lims_high = jnp.asarray(kernel.lims[1], dtype=jnp.float64)

    pv_valid = kernel._pv.copy()
    pv_at_zero_lower = kernel._pv.copy()
    pv_at_zero_lower[1] = 0.0  # LogNormPrior lower limit (strictly open)
    pv_below = kernel._pv.copy()
    pv_below[0] = -10.0
    pv_below[1] = -2.0
    pv_above = kernel._pv.copy()
    pv_above[2] = 5.0
    pv_nan = kernel._pv.copy()
    pv_nan[3] = np.nan
    pv_inf = kernel._pv.copy()
    pv_inf[4] = np.inf

    batch = jnp.asarray(
        np.stack([pv_valid, pv_at_zero_lower, pv_below, pv_above, pv_nan, pv_inf], axis=0),
        dtype=jnp.float64,
    )

    vals = np.asarray(
        _jit_vmap_neglnposterior(
            batch, X_tr, y_tr, splits_pad, lims_low, lims_high, kernel.prior_spec, kernel.kernel_id
        )
    )
    assert np.isfinite(vals[0])
    assert np.all(np.isposinf(vals[1:]))

    # Direct vmap of grad(_jit_neglnposterior) and _jit_val_and_grad must never emit NaN
    grad_fn = jax.jit(
        jax.vmap(
            jax.grad(_jit_neglnposterior),
            in_axes=(0, None, None, None, None, None, None, None),
        ),
        static_argnames=("kernel_id",),
    )
    grads = np.asarray(
        grad_fn(
            batch, X_tr, y_tr, splits_pad, lims_low, lims_high, kernel.prior_spec, kernel.kernel_id
        )
    )
    assert np.all(np.isfinite(grads)), f"Non-finite gradients encountered: {grads}"
    assert np.linalg.norm(grads[0]) > 0.0
    np.testing.assert_allclose(grads[1:], 0.0, atol=0.0)

    for row in np.asarray(batch[1:]):
        v_j, g_j = _jit_val_and_grad(
            jnp.asarray(row),
            X_tr,
            y_tr,
            splits_pad,
            lims_low,
            lims_high,
            kernel.prior_spec,
            kernel.kernel_id,
        )
        assert np.isposinf(float(v_j))
        np.testing.assert_allclose(np.asarray(g_j), 0.0, atol=0.0)

    # SciPy host callback must sanitize non-finite evaluations to (1e30, zeros)
    host_cb, _ = _make_scipy_host_callback(
        X_tr, y_tr, splits_pad, lims_low, lims_high, kernel.prior_spec, kernel.kernel_id
    )
    val_oob, grad_oob = host_cb(pv_below)
    assert val_oob == 1e30
    np.testing.assert_allclose(grad_oob, np.zeros_like(pv_below), atol=0.0)
    val_nan_cb, grad_nan_cb = host_cb(pv_nan)
    assert val_nan_cb == 1e30
    np.testing.assert_allclose(grad_nan_cb, np.zeros_like(pv_nan), atol=0.0)


@pytest.mark.parametrize(
    ("opt_name", "opt_kwargs"),
    [
        ("multistart", {"n_candidates": 64, "top_k": 3, "maxiter": 80}),
        ("de", {"npop": 32, "niter": 12, "top_k": 2, "maxiter_polish": 80}),
        ("optimistix", {"n_candidates": 64, "top_k": 3, "max_steps": 100}),
    ],
)
def test_optimizer_seed_determinism_and_global_rng_isolation(
    canonical_gp_dataset, opt_name: str, opt_kwargs: dict
):
    X_tr = canonical_gp_dataset["X_tr"]
    y_tr = canonical_gp_dataset["y_tr"]
    splits = canonical_gp_dataset["splits"]

    np.random.seed(777)
    global_before = np.random.get_state()

    opt1 = get_optimizer(opt_name, **opt_kwargs)
    res1 = opt1.optimize(X_tr, y_tr, splits, BasicKernel(), seed=2026)

    global_mid = np.random.get_state()
    assert global_before[0] == global_mid[0]
    np.testing.assert_array_equal(global_before[1], global_mid[1])
    assert global_before[2:] == global_mid[2:]

    opt2 = get_optimizer(opt_name, **opt_kwargs)
    res2 = opt2.optimize(X_tr, y_tr, splits, BasicKernel(), seed=2026)

    assert isinstance(res1, OptimizationResult)
    assert res1.success
    assert np.isfinite(res1.fun)
    assert np.isfinite(res1.grad_norm)
    np.testing.assert_allclose(res1.pv, res2.pv, rtol=0.0, atol=0.0)
    assert res1.fun == res2.fun


@pytest.mark.parametrize("opt_name", ["multistart", "de", "optimistix"])
def test_quasiperiodic_harmonic_trap_recovery(
    spotted_star_harmonic_dataset, opt_name: str
):
    """Recover P_true = 7.3 d within 2% even when seeded near 2 * P_true = 14.6 d in < 3 s."""
    ds = spotted_star_harmonic_dataset
    p_true = float(ds["p_true"])
    X_tr = ds["X_tr"]
    y_tr = ds["y_tr"]
    splits = ds["splits"]

    # Pre-warm XLA compilation for (N_tr=64, QuasiPeriodicKernel) so runtime measures
    # actual optimizer execution rather than one-time XLA compilation overhead.
    warm_kernel = QuasiPeriodicKernel(period=14.6)
    warm_opt = get_optimizer(
        opt_name,
        **(
            {"npop": 16, "niter": 2, "top_k": 1, "maxiter_polish": 5}
            if opt_name == "de"
            else {"n_candidates": 64, "top_k": 5, "maxiter": 5, "max_steps": 5}
        ),
    )
    warm_opt.optimize(X_tr, y_tr, splits, warm_kernel, seed=0)

    # Seed right on the 2 * P_true = 14.6 d harmonic trap!
    trapped_kernel = QuasiPeriodicKernel(period=2.0 * p_true)
    pv0_trapped = trapped_kernel._pv.copy()
    pv0_trapped[2] = 2.0 * p_true

    opt = get_optimizer(opt_name)
    t0 = time.perf_counter()
    res = opt.optimize(X_tr, y_tr, splits, trapped_kernel, pv0=pv0_trapped, seed=42)
    elapsed = time.perf_counter() - t0

    assert res.success, f"{opt_name} failed: {res}"
    assert elapsed < 3.0, f"{opt_name} took {elapsed:.2f}s (> 3.0s budget)"

    rel_period_err = abs(float(res.pv[2]) - p_true) / p_true
    assert rel_period_err < 0.02, (
        f"{opt_name} recovered P={res.pv[2]:.4f} d instead of {p_true:.2f} d "
        f"(rel_err={rel_period_err:.3%})"
    )

    # Verify component separation removes the synthetic K2 roll drift accurately
    dt = Detrender(
        ds["flux"],
        ds["inputs"],
        splits=splits,
        kernel=QuasiPeriodicKernel(period=float(res.pv[2])),
        tr_nrandom=32,
        tr_nblocks=2,
        tr_bspan=16,
        seed=42,
    )
    _, mp = dt.predict(res.pv, components=True)
    fm = dt.data.flux_median
    pos_pred_norm = (mp - np.nanmedian(mp)) / fm
    pos_true_norm = ds["pos_norm"] - np.median(ds["pos_norm"])
    raw_pos_rms = float(np.std(pos_true_norm))
    resid_pos_rms = float(np.std(pos_true_norm - pos_pred_norm))
    assert resid_pos_rms < 0.25 * raw_pos_rms
