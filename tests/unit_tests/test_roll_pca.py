"""Unit and integration tests for ``k2sc.pca`` spacecraft-roll PCA normalization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import astropy.io.fits as pf
import astropy.units as u
import k2sc
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from astropy.time import Time
from k2sc import cli
from k2sc.corrector import K2SCCorrector
from k2sc.k2data import K2Data
from k2sc.k2io import FITSWriter, MASTReader
from k2sc.pca import RollPCATransform, fit_roll_pca, pca_normalize_xy
from k2sc.pipeline import K2SCResult, detrend, detrend_k2data
from lightkurve import KeplerLightCurve
from scipy.special import ndtri

_MAD_NORM = float(1.0 / ndtri(0.75))

_FAST_KW = {
    "optimizer": "multistart",
    "tr_nrandom": 48,
    "tr_nblocks": 2,
    "tr_bspan": 16,
    "n_candidates": 24,
    "top_k": 2,
    "maxiter": 35,
    "seed": 42,
    "quiet": True,
}


@pytest.fixture(scope="module", autouse=True)
def _cleanup_lazy_package_attrs():
    yield
    k2sc.__dict__.pop("pipeline", None)
    k2sc.__dict__.pop("corrector", None)
    k2sc.__dict__.pop("standalone", None)


def _sigma_mad(arr: np.ndarray) -> float:
    fin = arr[np.isfinite(arr)]
    med = float(np.median(fin))
    return _MAD_NORM * float(np.median(np.abs(fin - med)))


def _make_collinear_k2_roll(
    n_pts: int = 128,
    seed: int = 2026,
    sigma_perp: float = 0.003,
) -> dict[str, np.ndarray]:
    """Generate collinear K2 roll (Delta x ~ 0.9, Delta y = 0.25 Delta x + eps_perp)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2332.0, 2356.0, n_pts, dtype=np.float64)
    phase = ((t - 2332.0) / 0.245) % 1.0
    dx = 0.9 * (phase - 0.5)
    eps_perp = rng.normal(0.0, sigma_perp, size=n_pts)
    x = 0.40 + dx
    y = -0.30 + 0.25 * dx + eps_perp

    # True roll and curved intra-pixel cross-roll systematics
    pos_sys = 4.0e-3 * np.sin(2.2 * dx) + 3.0e-3 * (
        np.cos(1.4 * (eps_perp / sigma_perp)) - 1.0
    )
    stellar = 1.2e-3 * np.sin(2.0 * np.pi * (t - 2332.0) / 6.5)
    noise = rng.normal(0.0, 8.0e-5, size=n_pts)
    flux = 1.0e4 * (1.0 + stellar + pos_sys + noise)
    flux_err = np.full(n_pts, 0.8, dtype=np.float64)
    return {
        "time": t,
        "x": x,
        "y": y,
        "dx": dx,
        "eps_perp": eps_perp,
        "flux": flux,
        "flux_err": flux_err,
        "stellar_flux": 1.0e4 * (1.0 + stellar),
        "pos_sys_flux": 1.0e4 * pos_sys,
    }


class TestRollPCATransformGeometryAndScaling:
    def test_collinear_k2_roll_decorrelation_so2_and_unit_4mad_width(self):
        rng = np.random.default_rng(11)
        n = 600
        dx = rng.uniform(-0.45, 0.45, size=n)
        eps_perp = rng.normal(0.0, 0.003, size=n)
        x = 1.25 + dx
        y = -0.75 + 0.25 * dx + eps_perp

        u1, u2, tf = pca_normalize_xy(x, y)

        assert isinstance(tf, RollPCATransform)
        assert tf.center_xy.shape == (2,)
        assert tf.rotation.shape == (2, 2)
        assert tf.scales_u.shape == (2,)
        assert tf.center_xy.dtype == np.float64
        assert tf.rotation.dtype == np.float64
        assert tf.scales_u.dtype == np.float64

        # Proper SO(2) rotation check: det(R) == +1 and R @ R.T == I_2
        assert np.linalg.det(tf.rotation) == pytest.approx(1.0, abs=1e-14)
        npt.assert_allclose(tf.rotation @ tf.rotation.T, np.eye(2), atol=1e-14)

        # Rotated 5-MAD inlier subset check
        inlier = (np.abs(u1 - np.median(u1)) <= 5.0 * _sigma_mad(u1)) & (
            np.abs(u2 - np.median(u2)) <= 5.0 * _sigma_mad(u2)
        )
        assert int(inlier.sum()) == n

        raw_corr = float(np.corrcoef(x, y)[0, 1])
        assert raw_corr > 0.998
        pca_corr = float(np.corrcoef(u1, u2)[0, 1])
        assert abs(pca_corr) < 1e-12

        # Both axes normalized so 4 * sigma_MAD == 1.0 -> sigma_MAD == 0.25
        assert _sigma_mad(u1) == pytest.approx(0.25, rel=1e-12, abs=1e-12)
        assert _sigma_mad(u2) == pytest.approx(0.25, rel=1e-12, abs=1e-12)

        expected_r1 = np.array([1.0, 0.25]) / np.hypot(1.0, 0.25)
        npt.assert_allclose(tf.rotation[0], expected_r1, atol=5e-3)

        # Round-trip inverse_transform recovers original (x, y) to machine precision
        x_rec, y_rec = tf.inverse_transform(u1, u2)
        npt.assert_allclose(x_rec, x, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(y_rec, y, rtol=1e-13, atol=1e-13)

    def test_post_init_validation_and_readonly_immutability(self):
        tf = RollPCATransform([1.0, -2.0], np.eye(2), [0.5, 0.02])

        # Attribute reassignment blocked by frozen=True
        with pytest.raises(FrozenInstanceError):
            tf.scales_u = np.ones(2)  # type: ignore[misc]

        # In-place element mutation blocked by write=False ndarray flags
        with pytest.raises(ValueError, match="read-only"):
            tf.scales_u[0] = -1.0
        with pytest.raises(ValueError, match="read-only"):
            tf.rotation[0, 0] = 2.0
        with pytest.raises(ValueError, match="read-only"):
            tf.center_xy[0] = 99.0

        # Non-finite center_xy, rotation, or scales_u rejected
        with pytest.raises(ValueError, match="center_xy must be finite"):
            RollPCATransform([np.nan, 0.0], np.eye(2), [1.0, 1.0])
        with pytest.raises(ValueError, match="rotation must be finite"):
            RollPCATransform([0.0, 0.0], [[np.inf, 0.0], [0.0, 1.0]], [1.0, 1.0])
        with pytest.raises(ValueError, match="scales_u must be finite and strictly positive"):
            RollPCATransform([0.0, 0.0], np.eye(2), [1.0, 0.0])
        with pytest.raises(ValueError, match="scales_u must be finite and strictly positive"):
            RollPCATransform([0.0, 0.0], np.eye(2), [1.0, -0.5])

        # Improper reflection (det = -1) and non-orthogonal shear rejected
        reflection = np.array([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError, match="proper orthogonal SO\\(2\\)"):
            RollPCATransform([0.0, 0.0], reflection, [1.0, 1.0])
        shear = np.array([[1.0, 0.3], [0.0, 1.0]])
        with pytest.raises(ValueError, match="proper orthogonal SO\\(2\\)"):
            RollPCATransform([0.0, 0.0], shear, [1.0, 1.0])

    def test_deterministic_sign_orientation_and_negation_symmetry(self):
        rng = np.random.default_rng(22)
        t = np.linspace(-0.45, 0.45, 300)
        perp = rng.normal(0.0, 0.002, size=t.size)

        # 1) Horizontal-dominant roll (|r1[0]| > |r1[1]|) -> r1[0] > 0
        x_h = t
        y_h = -0.35 * t + perp
        tf_h = fit_roll_pca(x_h, y_h)
        assert tf_h.rotation[0, 0] > 0.0
        assert np.linalg.det(tf_h.rotation) == pytest.approx(1.0, abs=1e-14)

        tf_h_neg = fit_roll_pca(-x_h, -y_h)
        npt.assert_allclose(tf_h_neg.rotation, tf_h.rotation, atol=1e-14)
        npt.assert_allclose(tf_h_neg.scales_u, tf_h.scales_u, atol=1e-14)
        npt.assert_allclose(tf_h_neg.center_xy, -tf_h.center_xy, atol=1e-14)

        tf_h_pert = fit_roll_pca(
            x_h + rng.normal(0.0, 1e-5, size=t.size),
            y_h + rng.normal(0.0, 1e-5, size=t.size),
        )
        assert float(np.dot(tf_h.rotation[0], tf_h_pert.rotation[0])) > 0.999999

        # 2) Vertical-dominant roll (|r1[1]| > |r1[0]|) -> r1[1] > 0
        x_v = -0.20 * t + perp
        y_v = t
        tf_v = fit_roll_pca(x_v, y_v)
        assert tf_v.rotation[0, 1] > 0.0
        assert np.linalg.det(tf_v.rotation) == pytest.approx(1.0, abs=1e-14)

        tf_v_neg = fit_roll_pca(-x_v, -y_v)
        npt.assert_allclose(tf_v_neg.rotation, tf_v.rotation, atol=1e-14)
        npt.assert_allclose(tf_v_neg.scales_u, tf_v.scales_u, atol=1e-14)

    def test_oblique_45deg_cross_roll_outliers_inside_marginal_box(self):
        rng = np.random.default_rng(45)
        n = 400
        # 45-degree roll arc: along-roll std ~ 0.26 pix, cross-roll std = 0.003 pix
        s_along = rng.uniform(-0.45, 0.45, size=n)
        s_cross = rng.normal(0.0, 0.003, size=n)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        cx, cy = 10.0, -5.0
        x = cx + inv_sqrt2 * (s_along - s_cross)
        y = cy + inv_sqrt2 * (s_along + s_cross)

        # Inject 2% (8 / 400) cross-roll outliers at (+0.5, -1.0) pix relative to centre:
        # inside the marginal 5-MAD (x, y) box (~ +-1.6 pix), ~350 sigma off the roll line!
        outlier_idx = np.array([10, 55, 102, 160, 215, 270, 325, 380])
        clean_inlier_bool = np.ones(n, dtype=bool)
        clean_inlier_bool[outlier_idx] = False
        x_contam = x.copy()
        y_contam = y.copy()
        x_contam[outlier_idx] = cx + 0.5
        y_contam[outlier_idx] = cy - 1.0

        u1, u2, tf = pca_normalize_xy(x_contam, y_contam)

        true_r1 = np.array([inv_sqrt2, inv_sqrt2])
        cos_ang = float(np.clip(np.dot(tf.rotation[0], true_r1), -1.0, 1.0))
        angle_err_deg = float(np.degrees(np.arccos(cos_ang)))
        assert angle_err_deg < 0.2

        clean_corr = float(
            np.corrcoef(u1[clean_inlier_bool], u2[clean_inlier_bool])[0, 1]
        )
        assert abs(clean_corr) < 0.05
        assert tf.scales_u[1] == pytest.approx(4.0 * 0.003, rel=0.10)

        # 25% (100 / 400) asymmetric oblique outliers at (-1, +1) perpendicular to 45 deg roll:
        # coordinate-wise median flips z_sphere 90 deg, whereas L1 spatial median stays aligned
        heavy_idx = np.arange(0, 400, 4)
        heavy_inlier = np.ones(n, dtype=bool)
        heavy_inlier[heavy_idx] = False
        x_heavy = x.copy()
        y_heavy = y.copy()
        x_heavy[heavy_idx] = cx - 1.0
        y_heavy[heavy_idx] = cy + 1.0
        u1_h, u2_h, tf_h = pca_normalize_xy(x_heavy, y_heavy)
        cos_h = float(np.clip(np.dot(tf_h.rotation[0], true_r1), -1.0, 1.0))
        assert float(np.degrees(np.arccos(cos_h))) < 0.2
        assert tf_h.scales_u[1] == pytest.approx(4.0 * 0.003, rel=0.12)
        assert abs(float(np.median(u1_h[heavy_inlier]))) < 1e-12
        assert abs(float(np.median(u2_h[heavy_inlier]))) < 1e-12

    def test_single_extreme_1e15_glitch_and_so2_rotational_equivariance(self):
        rng = np.random.default_rng(108)
        n = 400
        dx = rng.uniform(-0.45, 0.45, size=n)
        # Mild roll-arc curvature + cross-roll scatter (marginal median != rotated median)
        eps = 0.015 * (dx**2 - float(np.mean(dx**2))) + rng.normal(0.0, 0.003, size=n)
        x_clean = 1.8 + dx
        y_clean = -0.9 + 0.30 * dx + eps

        # 1) Single-cadence 1e15 and 5e11 glitches must NOT inflate rad_tol or shift center_xy
        mask_no0 = np.ones(n, dtype=bool)
        mask_no0[0] = False
        tf_ref0 = fit_roll_pca(x_clean, y_clean, clean_mask=mask_no0)

        for glitch_val in (1.0e15, -5.0e11):
            x_gl = x_clean.copy()
            y_gl = y_clean.copy()
            x_gl[0] = glitch_val
            y_gl[0] = -0.7 * glitch_val
            tf_gl = fit_roll_pca(x_gl, y_gl)
            npt.assert_allclose(tf_gl.center_xy, tf_ref0.center_xy, rtol=1e-12, atol=1e-12)
            npt.assert_allclose(tf_gl.rotation, tf_ref0.rotation, rtol=1e-12, atol=1e-12)
            npt.assert_allclose(tf_gl.scales_u, tf_ref0.scales_u, rtol=1e-12, atol=1e-12)

        # 2) SO(2) rotational equivariance across arbitrary rotation angles (with asymmetric flyers)
        x_eq = x_clean.copy()
        y_eq = y_clean.copy()
        x_eq[[12, 95, 210, 340]] += 8.0
        y_eq[[12, 95, 210, 340]] -= 11.0
        tf0 = fit_roll_pca(x_eq, y_eq)
        u1_0, u2_0 = tf0.transform(x_eq, y_eq)

        for deg in (23.5, -61.0, 142.0):
            th = np.radians(deg)
            q_mat = np.array(
                [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]],
                dtype=np.float64,
            )
            xy_rot = q_mat @ np.vstack([x_eq, y_eq])
            tf_q = fit_roll_pca(xy_rot[0], xy_rot[1])
            u1_q, u2_q = tf_q.transform(xy_rot[0], xy_rot[1])

            npt.assert_allclose(
                tf_q.center_xy, q_mat @ tf0.center_xy, rtol=1e-11, atol=1e-11
            )
            npt.assert_allclose(tf_q.scales_u, tf0.scales_u, rtol=1e-11, atol=1e-11)
            sign = 1.0 if float(np.dot(tf_q.rotation[0], q_mat @ tf0.rotation[0])) >= 0.0 else -1.0
            npt.assert_allclose(tf_q.rotation, sign * (tf0.rotation @ q_mat.T), atol=1e-11)
            npt.assert_allclose(u1_q, sign * u1_0, rtol=1e-11, atol=1e-11)
            npt.assert_allclose(u2_q, sign * u2_0, rtol=1e-11, atol=1e-11)

    def test_outlier_trimming_clean_mask_and_nan_preservation(self):
        rng = np.random.default_rng(33)
        n = 400
        dx = np.linspace(-0.45, 0.45, n)
        eps = rng.normal(0.0, 0.003, size=n)
        x_base = 0.2 + dx
        y_base = -0.1 + 0.25 * dx + eps

        tf_clean_ref = fit_roll_pca(x_base, y_base)

        x_contaminated = x_base.copy()
        y_contaminated = y_base.copy()
        flyer_idx = [15, 80, 160, 240, 330]
        nan_idx = [5, 199, 390]
        x_contaminated[flyer_idx] += 20.0
        y_contaminated[flyer_idx] -= 20.0
        x_contaminated[nan_idx[0]] = np.nan
        y_contaminated[nan_idx[1]] = np.nan
        x_contaminated[nan_idx[2]] = np.inf

        inlier_mask = np.ones(n, dtype=bool)
        inlier_mask[flyer_idx + nan_idx] = False
        tf_masked = fit_roll_pca(x_contaminated, y_contaminated, clean_mask=inlier_mask)
        tf_unmasked = fit_roll_pca(x_contaminated, y_contaminated)

        npt.assert_allclose(tf_unmasked.center_xy, tf_masked.center_xy, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tf_unmasked.rotation, tf_masked.rotation, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tf_unmasked.scales_u, tf_masked.scales_u, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tf_unmasked.rotation, tf_clean_ref.rotation, atol=2e-3)
        npt.assert_allclose(tf_unmasked.scales_u, tf_clean_ref.scales_u, rtol=2e-2)

        u1, u2 = tf_unmasked.transform(x_contaminated, y_contaminated)
        assert u1.shape == (n,) and u2.shape == (n,)
        assert np.all(np.isnan(u1[nan_idx]))
        assert np.all(np.isnan(u2[nan_idx]))
        finite_check = np.ones(n, dtype=bool)
        finite_check[nan_idx] = False
        assert np.all(np.isfinite(u1[finite_check]))
        assert np.all(np.isfinite(u2[finite_check]))
        assert np.all(np.abs(u1[flyer_idx]) > 10.0)

        # Short segment with 4 clean points + huge flagged thruster flyers:
        # clean_mask MUST be honoured (never unmasking flagged thruster flyers!)
        x_short = np.array([0.0, 0.2, 0.4, 0.6, 50.0, -60.0, 80.0])
        y_short = np.array([0.0, 0.05, 0.10, 0.15, -90.0, 70.0, -85.0])
        short_clean_mask = np.array([True, True, True, True, False, False, False])
        tf_short = fit_roll_pca(x_short, y_short, clean_mask=short_clean_mask)
        expected_short_r1 = np.array([1.0, 0.25]) / np.hypot(1.0, 0.25)
        npt.assert_allclose(tf_short.rotation[0], expected_short_r1, atol=1e-12)
        npt.assert_allclose(tf_short.center_xy, [0.3, 0.075], atol=1e-12)

        # < 3 clean points with flagged flyers falls back to identity without unmasking flyers
        tiny_mask = np.array([True, True, False, False, False, False, False])
        tf_tiny = fit_roll_pca(x_short, y_short, clean_mask=tiny_mask)
        npt.assert_array_equal(tf_tiny.rotation, np.eye(2))
        npt.assert_array_equal(tf_tiny.scales_u, [1.0, 1.0])
        npt.assert_allclose(tf_tiny.center_xy, [0.1, 0.025], atol=1e-12)

    def test_degenerate_constant_1d_discrete_flyers_and_scale_covariance(self):
        # 1) 1D covariate case: y == 0 identically
        x_1d = np.linspace(-2.0, 2.0, 120)
        y_zero = np.zeros(120, dtype=np.float64)
        u1, u2, tf_1d = pca_normalize_xy(x_1d, y_zero)
        assert np.all(np.isfinite(u1)) and np.all(np.isfinite(u2))
        npt.assert_array_equal(u2, np.zeros(120))
        assert tf_1d.scales_u[1] == 1.0
        assert _sigma_mad(u1) == pytest.approx(0.25, rel=1e-12)

        # 2) Both x and y constant -> zero division never occurs, maps identically to (0, 0)
        x_const = np.full(80, 4.2, dtype=np.float64)
        y_const = np.full(80, -1.7, dtype=np.float64)
        u1_c, u2_c, tf_c = pca_normalize_xy(x_const, y_const)
        npt.assert_array_equal(u1_c, np.zeros(80))
        npt.assert_array_equal(u2_c, np.zeros(80))
        npt.assert_allclose(tf_c.center_xy, [4.2, -1.7])
        npt.assert_array_equal(tf_c.scales_u, [1.0, 1.0])

        # 3) Large-offset (1e5) noiseless collinear line: float64 cancellation (~2e-11)
        # must fall below scale-relative tolerance and NOT blow up scales_u[1]
        dx_line = np.linspace(-1.0, 1.0, 100)
        x_big = 1.0e5 + dx_line
        y_big = 2.0 * x_big
        u1_b, u2_b, tf_b = pca_normalize_xy(x_big, y_big)
        assert tf_b.scales_u[1] == 1.0
        npt.assert_allclose(u2_b, 0.0, atol=1e-9)
        assert _sigma_mad(u1_b) == pytest.approx(0.25, rel=1e-10)

        # Exact positive scale covariance across 8 orders of magnitude
        rng = np.random.default_rng(91)
        x_sc = rng.normal(0.0, 1.0, size=200)
        y_sc = 0.4 * x_sc + rng.normal(0.0, 0.02, size=200)
        tf_unit = fit_roll_pca(x_sc, y_sc)
        tf_scaled = fit_roll_pca(1.0e4 * x_sc, 1.0e4 * y_sc)
        npt.assert_allclose(tf_scaled.rotation, tf_unit.rotation, atol=1e-12)
        npt.assert_allclose(tf_scaled.scales_u, 1.0e4 * tf_unit.scales_u, rtol=1e-12)

        # 4) Discrete zero-MAD coordinate (>50% at median) WITH extreme flyers
        x_disc_clean = np.concatenate([np.zeros(70), np.full(15, -1.0), np.full(15, 1.0)])
        y_disc_clean = np.zeros(100)
        tf_disc_ref = fit_roll_pca(x_disc_clean, y_disc_clean)
        assert _sigma_mad(x_disc_clean) == 0.0
        assert tf_disc_ref.scales_u[0] == pytest.approx(
            4.0 * float(np.std(x_disc_clean)), rel=1e-12
        )

        x_disc_fly = np.concatenate([x_disc_clean, [50.0, -45.0]])
        y_disc_fly = np.concatenate([y_disc_clean, [-30.0, 35.0]])
        tf_disc_fly = fit_roll_pca(x_disc_fly, y_disc_fly)
        npt.assert_allclose(tf_disc_fly.rotation, np.eye(2), atol=1e-12)
        npt.assert_allclose(tf_disc_fly.scales_u, tf_disc_ref.scales_u, rtol=1e-12)

        # 5) Fewer than 3 clean finite points -> identity fallback
        tf_two = fit_roll_pca([1.0, 3.0, np.nan], [2.0, 6.0, np.nan])
        npt.assert_array_equal(tf_two.rotation, np.eye(2))
        npt.assert_array_equal(tf_two.scales_u, [1.0, 1.0])
        npt.assert_allclose(tf_two.center_xy, [2.0, 4.0])

        tf_all_nan = fit_roll_pca([np.nan, np.nan], [np.nan, np.nan])
        npt.assert_array_equal(tf_all_nan.center_xy, [0.0, 0.0])
        npt.assert_array_equal(tf_all_nan.rotation, np.eye(2))
        npt.assert_array_equal(tf_all_nan.scales_u, [1.0, 1.0])


class TestEndToEndRollPCAPipelineAndCLI:
    def test_detrend_arrays_recovers_roll_and_cross_roll_drift(self):
        sim = _make_collinear_k2_roll(n_pts=128, seed=2026, sigma_perp=0.003)

        res_pca = detrend(
            sim["time"],
            sim["flux"],
            x=sim["x"],
            y=sim["y"],
            splits=[2344.0],
            roll_pca=True,
            **_FAST_KW,
        )
        res_raw = detrend(
            sim["time"],
            sim["flux"],
            x=sim["x"],
            y=sim["y"],
            splits=[2344.0],
            roll_pca=False,
            **_FAST_KW,
        )

        assert isinstance(res_pca, K2SCResult)
        assert isinstance(res_pca.pca_transform, RollPCATransform)
        assert res_raw.pca_transform is None

        # Legacy 8-tuple unpacking remains strictly unchanged
        assert len(res_pca) == 8
        det_obj, pv, tr_t, tr_p, cr, ct, cc, warn = res_pca
        assert det_obj is res_pca.detrender
        npt.assert_array_equal(pv, res_pca.pv)
        npt.assert_array_equal(tr_t, res_pca.tr_time)
        npt.assert_array_equal(tr_p, res_pca.tr_position)
        assert (cr, ct, cc, warn) == (
            res_pca.cdpp_r,
            res_pca.cdpp_t,
            res_pca.cdpp_c,
            res_pca.warn,
        )

        assert res_pca.cdpp_t < res_pca.cdpp_r
        assert res_pca.cdpp_t < 0.5 * res_raw.cdpp_t

        pos_est = res_pca.tr_position - np.nanmedian(res_pca.tr_position)
        pos_true = sim["pos_sys_flux"] - np.median(sim["pos_sys_flux"])
        rms_err_pca = float(np.sqrt(np.mean((pos_est - pos_true) ** 2)))
        rms_raw_sys = float(np.sqrt(np.mean(pos_true**2)))
        assert rms_err_pca < 0.25 * rms_raw_sys

    def test_detrend_k2data_preserves_detector_xy_and_fits_pca_headers(self, tmp_path):
        sim = _make_collinear_k2_roll(n_pts=129, seed=77)
        x_orig = sim["x"].copy()
        y_orig = sim["y"].copy()
        x_orig[4] = np.nan

        # Shared header object (as in SPLOXReader._cache.header) must NOT be mutated in-place
        shared_hdr = pf.Header({"TELESCOP": "Kepler"})
        ds = K2Data(
            211436265,
            time=sim["time"],
            cadence=np.arange(129, dtype=np.int32),
            quality=np.zeros(129, dtype=np.int32),
            fluxes=sim["flux"],
            errors=sim["flux_err"],
            x=x_orig,
            y=y_orig,
            primary_header=shared_hdr,
            data_header=shared_hdr,
            campaign=5,
        )
        ds_x_before = ds.x.copy()
        ds_y_before = ds.y.copy()
        ds_raw_x_before = ds.raw_x.copy()
        ds_raw_y_before = ds.raw_y.copy()

        results = detrend_k2data(ds, splits=[2344.0], roll_pca=True, **_FAST_KW)
        assert "ROLL_PCA" not in shared_hdr
        assert "PCA_CX" not in shared_hdr
        assert "ROLL_PCA" not in ds.primary_header
        assert len(results) == 1
        tf = results[0].pca_transform
        assert isinstance(tf, RollPCATransform)
        assert results[0].cdpp_t < results[0].cdpp_r

        # CRITICAL: K2Data physical detector coordinates must remain completely untouched
        npt.assert_array_equal(ds.x, ds_x_before)
        npt.assert_array_equal(ds.y, ds_y_before)
        npt.assert_array_equal(ds.raw_x, ds_raw_x_before)
        npt.assert_array_equal(ds.raw_y, ds_raw_y_before)

        out_fits = tmp_path / "EPIC_211436265_mast.fits"
        FITSWriter.write(out_fits, results[0].splits, ds, results)
        with pf.open(out_fits) as hdul:
            assert "ROLL_PCA" not in hdul[0].header
            assert "PCA_CX" not in hdul[0].header
            hdr = hdul[1].header
            assert hdr["ROLL_PCA"] is True
            tf_from_hdr = RollPCATransform(
                center_xy=[hdr["PCA_CX"], hdr["PCA_CY"]],
                rotation=[
                    [hdr["PCA_R11"], hdr["PCA_R12"]],
                    [hdr["PCA_R21"], hdr["PCA_R22"]],
                ],
                scales_u=[hdr["PCA_S1"], hdr["PCA_S2"]],
            )
            npt.assert_allclose(tf_from_hdr.center_xy, tf.center_xy, rtol=1e-13, atol=1e-13)
            npt.assert_allclose(tf_from_hdr.rotation, tf.rotation, rtol=1e-13, atol=1e-13)
            npt.assert_allclose(tf_from_hdr.scales_u, tf.scales_u, rtol=1e-13, atol=1e-13)

            assert np.isnan(hdul[1].data["x"][4])
            assert np.isnan(hdul[1].data["y"][4])
            npt.assert_allclose(
                hdul[1].data["x"][ds.nanmask], ds_x_before, rtol=1e-13, atol=1e-13
            )
            npt.assert_allclose(
                hdul[1].data["y"][ds.nanmask], ds_y_before, rtol=1e-13, atol=1e-13
            )

        # Re-detrending with roll_pca=False sets ROLL_PCA=False and clears PCA_* cards
        results_no_pca = detrend_k2data(ds, splits=[2344.0], roll_pca=False, **_FAST_KW)
        assert results_no_pca[0].pca_transform is None
        out_fits_raw = tmp_path / "EPIC_211436265_raw.fits"
        FITSWriter.write(out_fits_raw, results_no_pca[0].splits, ds, results_no_pca)
        with pf.open(out_fits_raw) as hdul_raw:
            assert hdul_raw[1].header["ROLL_PCA"] is False
            assert "PCA_CX" not in hdul_raw[1].header
            assert "PCA_S1" not in hdul_raw[1].header

        # Writing the earlier roll_pca=True result AFTER ds was re-detrended with roll_pca=False
        # must serialize ROLL_PCA=True and PCA_* directly from results[0].pca_transform
        out_fits_sync = tmp_path / "EPIC_211436265_sync.fits"
        FITSWriter.write(out_fits_sync, results[0].splits, ds, results)
        with pf.open(out_fits_sync) as hdul_sync:
            assert hdul_sync[1].header["ROLL_PCA"] is True
            assert hdul_sync[1].header["PCA_CX"] == pytest.approx(tf.center_xy[0])
            assert hdul_sync[1].header["PCA_S2"] == pytest.approx(tf.scales_u[1])

    def test_lightkurve_corrector_and_cli_no_roll_pca_flag(self, tmp_path, monkeypatch):
        sim = _make_collinear_k2_roll(n_pts=128, seed=88)
        lc = KeplerLightCurve(
            time=Time(sim["time"], format="bkjd"),
            flux=sim["flux"] * (u.electron / u.s),
            flux_err=sim["flux_err"] * (u.electron / u.s),
            meta={"MISSION": "K2", "CAMPAIGN": 5, "TARGETID": 211436265},
        )
        lc["pos_corr1"] = sim["x"] * u.pix
        lc["pos_corr2"] = sim["y"] * u.pix

        corr_on = K2SCCorrector(lc)
        clc_on = corr_on.correct(splits=[2344.0], roll_pca=True, **_FAST_KW)
        assert corr_on.result is not None
        tf_on = corr_on.result.pca_transform
        assert isinstance(tf_on, RollPCATransform)
        assert clc_on.meta["K2SC_ROLL_PCA"] is True
        npt.assert_allclose(clc_on.meta["K2SC_PCA_CENTER"], tf_on.center_xy)
        npt.assert_allclose(clc_on.meta["K2SC_PCA_ROTATION"], tf_on.rotation)
        npt.assert_allclose(clc_on.meta["K2SC_PCA_SCALES"], tf_on.scales_u)

        # diagnose_xy maps training set back to physical (x, y) detector coordinates
        ax_xy = corr_on.diagnose_xy()
        line_xy = ax_xy.lines[0]
        assert np.min(line_xy.get_xdata()) >= np.min(sim["x"]) - 1e-6
        assert np.max(line_xy.get_xdata()) <= np.max(sim["x"]) + 1e-6
        assert np.min(line_xy.get_ydata()) >= np.min(sim["y"]) - 1e-6
        assert np.max(line_xy.get_ydata()) <= np.max(sim["y"]) + 1e-6
        plt.close("all")

        corr_off = K2SCCorrector(lc)
        clc_off = corr_off.correct(splits=[2344.0], roll_pca=False, **_FAST_KW)
        assert corr_off.result is not None
        assert corr_off.result.pca_transform is None
        assert clc_off.meta["K2SC_ROLL_PCA"] is False
        assert "K2SC_PCA_CENTER" not in clc_off.meta
        # Re-correcting an already PCA-corrected LightCurve with roll_pca=False must purge stale K2SC_PCA_* meta
        clc_reoff = K2SCCorrector(clc_on).correct(
            splits=[2344.0], roll_pca=False, **_FAST_KW
        )
        assert clc_reoff.meta["K2SC_ROLL_PCA"] is False
        assert "K2SC_PCA_CENTER" not in clc_reoff.meta
        assert "K2SC_PCA_ROTATION" not in clc_reoff.meta
        assert "K2SC_PCA_SCALES" not in clc_reoff.meta
        assert corr_on.cdpp_t < corr_off.cdpp_t

        # CLI argument parser default True and --no-roll-pca -> False
        parser = cli.build_k2sc_parser()
        args_default = parser.parse_args(["target.fits"])
        assert args_default.roll_pca is True
        args_disabled = parser.parse_args(["target.fits", "--no-roll-pca"])
        assert args_disabled.roll_pca is False

        # Verify cli._detrend_and_write_dataset forwards args.roll_pca into detrend_k2data
        captured_roll_pca: list[bool] = []
        orig_detrend_k2data = detrend_k2data

        def _spy_detrend_k2data(dataset, **kwargs):
            captured_roll_pca.append(bool(kwargs.get("roll_pca", True)))
            return orig_detrend_k2data(dataset, **kwargs)

        import k2sc.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "detrend_k2data", _spy_detrend_k2data)

        ds_cli = K2Data(
            211436265,
            time=sim["time"],
            cadence=np.arange(128, dtype=np.int32),
            quality=np.zeros(128, dtype=np.int32),
            fluxes=sim["flux"],
            errors=sim["flux_err"],
            x=sim["x"],
            y=sim["y"],
            campaign=5,
        )
        cli_args = parser.parse_args(
            [
                "dummy.fits",
                "--no-roll-pca",
                "--save-dir",
                str(tmp_path),
                "--optimizer",
                "multistart",
                "--tr-nrandom",
                "48",
                "--tr-nblocks",
                "2",
                "--tr-bspan",
                "16",
                "--n-candidates",
                "24",
                "--maxiter",
                "30",
                "--quiet",
            ]
        )
        out_path = cli._detrend_and_write_dataset(ds_cli, MASTReader, cli_args)
        assert captured_roll_pca == [False]
        assert (tmp_path / "EPIC_211436265_mast.fits").resolve() == (
            tmp_path / out_path
        ).resolve()
