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
    return 1.4826 * float(np.median(np.abs(fin - med)))


def _make_collinear_k2_roll(
    n_pts: int = 128,
    seed: int = 2026,
    sigma_perp: float = 0.003,
) -> dict[str, np.ndarray]:
    """Generate collinear K2 roll (Delta x ~ 0.9, Delta y = 0.25 Delta x + eps_perp)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2332.0, 2356.0, n_pts, dtype=np.float64)
    # Sawtooth-like roll excursion spanning ~0.9 pix peak-to-peak around (0.4, -0.3)
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
        # Gaussian roll + tiny cross-roll so all points lie well inside 5*sigma_MAD
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

        # Compute trimmed clean subset mask matching fit_roll_pca
        sx0 = _sigma_mad(x)
        sy0 = _sigma_mad(y)
        trim = (np.abs(x - np.median(x)) <= 5.0 * sx0) & (
            np.abs(y - np.median(y)) <= 5.0 * sy0
        )
        assert int(trim.sum()) > 590

        # Principal axes decorrelate collinear (corr ~ 0.999) coordinates to machine zero
        raw_corr = float(np.corrcoef(x[trim], y[trim])[0, 1])
        assert raw_corr > 0.998
        pca_corr = float(np.corrcoef(u1[trim], u2[trim])[0, 1])
        assert abs(pca_corr) < 1e-12

        # Both axes normalized so 4 * sigma_MAD == 1.0 -> sigma_MAD == 0.25
        assert _sigma_mad(u1[trim]) == pytest.approx(0.25, rel=1e-12, abs=1e-12)
        assert _sigma_mad(u2[trim]) == pytest.approx(0.25, rel=1e-12, abs=1e-12)

        # Principal direction matches (1, 0.25) / hypot(1, 0.25) closely
        expected_r1 = np.array([1.0, 0.25]) / np.hypot(1.0, 0.25)
        npt.assert_allclose(tf.rotation[0], expected_r1, atol=5e-3)

        # Round-trip inverse_transform recovers original (x, y) to machine precision
        x_rec, y_rec = tf.inverse_transform(u1, u2)
        npt.assert_allclose(x_rec, x, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(y_rec, y, rtol=1e-13, atol=1e-13)

        # Frozen dataclass immutability
        with pytest.raises(FrozenInstanceError):
            tf.scales_u = np.ones(2)  # type: ignore[misc]

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

        # Negation symmetry: (-x, -y) yields identical rotation and scales_u
        tf_h_neg = fit_roll_pca(-x_h, -y_h)
        npt.assert_allclose(tf_h_neg.rotation, tf_h.rotation, atol=1e-14)
        npt.assert_allclose(tf_h_neg.scales_u, tf_h.scales_u, atol=1e-14)
        npt.assert_allclose(tf_h_neg.center_xy, -tf_h.center_xy, atol=1e-14)

        # Small perturbation stability
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

    def test_outlier_trimming_clean_mask_and_nan_preservation(self):
        rng = np.random.default_rng(33)
        n = 400
        dx = np.linspace(-0.45, 0.45, n)
        eps = rng.normal(0.0, 0.003, size=n)
        x_base = 0.2 + dx
        y_base = -0.1 + 0.25 * dx + eps

        tf_clean_ref = fit_roll_pca(x_base, y_base)

        # Inject 5 extreme +20.0-pixel thruster flyers and 3 NaN cadences
        x_contaminated = x_base.copy()
        y_contaminated = y_base.copy()
        flyer_idx = [15, 80, 160, 240, 330]
        nan_idx = [5, 199, 390]
        x_contaminated[flyer_idx] += 20.0
        y_contaminated[flyer_idx] -= 20.0
        x_contaminated[nan_idx[0]] = np.nan
        y_contaminated[nan_idx[1]] = np.nan
        x_contaminated[nan_idx[2]] = np.inf

        # Reference on exact non-flyer finite cadences
        inlier_mask = np.ones(n, dtype=bool)
        inlier_mask[flyer_idx + nan_idx] = False
        tf_masked = fit_roll_pca(x_contaminated, y_contaminated, clean_mask=inlier_mask)
        tf_unmasked = fit_roll_pca(x_contaminated, y_contaminated)

        # Unflagged +20-pix flyers are rejected by 5*sigma_MAD trimming -> rotation & scales_u unperturbed!
        npt.assert_allclose(tf_unmasked.rotation, tf_masked.rotation, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tf_unmasked.scales_u, tf_masked.scales_u, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tf_unmasked.rotation, tf_clean_ref.rotation, atol=2e-3)
        npt.assert_allclose(tf_unmasked.scales_u, tf_clean_ref.scales_u, rtol=2e-2)

        # transform() projects ALL finite cadences (including flyers) and preserves NaN at non-finite cadences
        u1, u2 = tf_unmasked.transform(x_contaminated, y_contaminated)
        assert u1.shape == (n,) and u2.shape == (n,)
        assert np.all(np.isnan(u1[nan_idx]))
        assert np.all(np.isnan(u2[nan_idx]))
        finite_check = np.ones(n, dtype=bool)
        finite_check[nan_idx] = False
        assert np.all(np.isfinite(u1[finite_check]))
        assert np.all(np.isfinite(u2[finite_check]))
        # Flyer cadences project to large finite coordinates without being dropped
        assert np.all(np.abs(u1[flyer_idx]) > 10.0)

        # Fallback when clean_mask has < 10 True cadences
        sparse_clean = np.zeros(n, dtype=bool)
        sparse_clean[:5] = True
        tf_fallback = fit_roll_pca(x_contaminated, y_contaminated, clean_mask=sparse_clean)
        npt.assert_allclose(tf_fallback.rotation, tf_unmasked.rotation, atol=1e-13)

    def test_degenerate_constant_1d_discrete_and_tiny_sample_safety(self):
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

        # 3) Noiseless collinear line y = 2 * x -> cross-roll rounding noise (~1e-16) is NOT amplified
        x_line = np.linspace(-1.0, 1.0, 100)
        y_line = 2.0 * x_line
        u1_l, u2_l, tf_l = pca_normalize_xy(x_line, y_line)
        assert tf_l.scales_u[1] == 1.0
        npt.assert_allclose(u2_l, 0.0, atol=1e-14)
        assert _sigma_mad(u1_l) == pytest.approx(0.25, rel=1e-12)

        # 4) Discrete coordinate with zero MAD (>50% at median) but non-zero std -> 4*std fallback
        x_disc = np.concatenate([np.zeros(70), np.full(15, -1.0), np.full(15, 1.0)])
        y_disc = np.zeros(100)
        u1_d, u2_d, tf_d = pca_normalize_xy(x_disc, y_disc)
        assert _sigma_mad(x_disc) == 0.0
        expected_s0 = 4.0 * float(np.std(x_disc))
        assert tf_d.scales_u[0] == pytest.approx(expected_s0, rel=1e-12)
        assert float(np.std(u1_d)) == pytest.approx(0.25, rel=1e-12)
        npt.assert_array_equal(u2_d, np.zeros(100))

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

        # With sigma_perp = 0.003 pix, raw inverse-length-scale bounds [0, 70] cannot resolve
        # cross-roll drift, whereas roll_pca=True resolves both roll and cross-roll cleanly!
        assert res_pca.cdpp_t < res_pca.cdpp_r
        assert res_pca.cdpp_t < 0.5 * res_raw.cdpp_t

        pos_est = res_pca.tr_position - np.nanmedian(res_pca.tr_position)
        pos_true = sim["pos_sys_flux"] - np.median(sim["pos_sys_flux"])
        rms_err_pca = float(np.sqrt(np.mean((pos_est - pos_true) ** 2)))
        rms_raw_sys = float(np.sqrt(np.mean(pos_true**2)))
        assert rms_err_pca < 0.25 * rms_raw_sys

    def test_detrend_k2data_preserves_detector_xy_and_fits_columns(self, tmp_path):
        # Use 129 raw cadences with 1 NaN so ds.npoints == 128 reuses XLA compilation
        sim = _make_collinear_k2_roll(n_pts=129, seed=77)
        x_orig = sim["x"].copy()
        y_orig = sim["y"].copy()
        x_orig[4] = np.nan

        ds = K2Data(
            211436265,
            time=sim["time"],
            cadence=np.arange(129, dtype=np.int32),
            quality=np.zeros(129, dtype=np.int32),
            fluxes=sim["flux"],
            errors=sim["flux_err"],
            x=x_orig,
            y=y_orig,
            campaign=5,
        )
        ds_x_before = ds.x.copy()
        ds_y_before = ds.y.copy()
        ds_raw_x_before = ds.raw_x.copy()
        ds_raw_y_before = ds.raw_y.copy()

        results = detrend_k2data(ds, splits=[2344.0], roll_pca=True, **_FAST_KW)
        assert len(results) == 1
        assert isinstance(results[0].pca_transform, RollPCATransform)
        assert results[0].cdpp_t < results[0].cdpp_r

        # CRITICAL: K2Data physical detector coordinates must remain completely untouched
        npt.assert_array_equal(ds.x, ds_x_before)
        npt.assert_array_equal(ds.y, ds_y_before)
        npt.assert_array_equal(ds.raw_x, ds_raw_x_before)
        npt.assert_array_equal(ds.raw_y, ds_raw_y_before)

        out_fits = tmp_path / "EPIC_211436265_mast.fits"
        FITSWriter.write(out_fits, results[0].splits, ds, results)
        with pf.open(out_fits) as hdul:
            assert np.isnan(hdul[1].data["x"][4])
            assert np.isnan(hdul[1].data["y"][4])
            npt.assert_allclose(
                hdul[1].data["x"][ds.nanmask], ds_x_before, rtol=1e-13, atol=1e-13
            )
            npt.assert_allclose(
                hdul[1].data["y"][ds.nanmask], ds_y_before, rtol=1e-13, atol=1e-13
            )

        # Also verify roll_pca=False on K2Data leaves pca_transform as None
        results_no_pca = detrend_k2data(ds, splits=[2344.0], roll_pca=False, **_FAST_KW)
        assert results_no_pca[0].pca_transform is None

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
        assert isinstance(corr_on.result.pca_transform, RollPCATransform)
        assert clc_on.meta["K2SC_ROLL_PCA"] is True

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
