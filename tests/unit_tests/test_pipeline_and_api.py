"""Unit and integration tests for ``k2sc.pipeline``, ``k2sc.corrector``, and ``k2sc.standalone``."""

from __future__ import annotations

import astropy.io.fits as pf
import astropy.units as u
import jax.numpy as jnp
import k2sc
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from astropy.time import Time
from astropy.utils.masked import Masked
from k2sc.core import (
    M_NOTFINITE,
    M_OUTLIER_D,
    M_OUTLIER_U,
    M_PERIODIC,
    M_QUALITY,
)
from k2sc.corrector import K2SCCorrector, _to_float64_array
from k2sc.k2data import K2Data
from k2sc.k2io import FITSWriter
from k2sc.pipeline import K2SCResult, SCResult, detrend, resolve_splits
from k2sc.standalone import k2sc_lc
from lightkurve import KeplerLightCurve, LightCurve, TessLightCurve
from lightkurve.correctors import DesignMatrix
from lightkurve.correctors.corrector import Corrector

# Keep PEP 562 lazy attributes unbound on `k2sc` during pytest collection
k2sc.__dict__.pop("pipeline", None)
k2sc.__dict__.pop("corrector", None)
k2sc.__dict__.pop("standalone", None)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_lazy_package_attrs():
    yield
    k2sc.__dict__.pop("pipeline", None)
    k2sc.__dict__.pop("corrector", None)
    k2sc.__dict__.pop("standalone", None)

# Fast, deterministic optimizer settings with shared training shape (48 + 2*16 = 80)
# so XLA compiles each kernel graph once across the test module.
_FAST_KW = {
    "optimizer": "multistart",
    "tr_nrandom": 48,
    "tr_nblocks": 2,
    "tr_bspan": 16,
    "n_candidates": 20,
    "top_k": 2,
    "maxiter": 30,
    "seed": 42,
    "quiet": True,
}


def _make_synthetic_k2_series(
    n_pts: int = 128,
    seed: int = 101,
    periodic_amp: float = 1.8e-3,
    period: float = 4.2,
) -> dict[str, np.ndarray]:
    """Generate realistic K2 Campaign 5 synthetic time, drift (x, y), and flux."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2332.0, 2356.0, n_pts, dtype=np.float64)
    roll = ((t - 2332.0) / 0.25) % 1.0 - 0.5
    x = 0.35 * roll + 0.05 * np.sin(2.0 * np.pi * (t - 2332.0) / 15.0)
    y = -0.28 * roll + 0.04 * np.cos(2.0 * np.pi * (t - 2332.0) / 19.0)

    seg = (t > 2344.0).astype(int)
    pos_sys = np.where(seg == 0, 4.5e-3 * x - 3.2e-3 * y, -4.0e-3 * x + 3.5e-3 * y)
    stellar = periodic_amp * np.sin(2.0 * np.pi * (t - 2332.0) / period)
    noise = rng.normal(0.0, 1.2e-4, size=n_pts)
    base_flux = 1.0e4 * (1.0 + stellar + pos_sys + noise)
    flux_err = np.full(n_pts, 1.2, dtype=np.float64)
    return {
        "time": t,
        "x": x,
        "y": y,
        "flux": base_flux,
        "flux_err": flux_err,
        "stellar": 1.0e4 * (1.0 + stellar),
        "pos_sys": 1.0e4 * pos_sys,
    }


class TestExportsAndAstropyUnwrapping:
    def test_top_level_lazy_exports(self):
        assert k2sc.detrend is detrend
        assert k2sc.K2SCResult is K2SCResult
        assert k2sc.K2SCCorrector is K2SCCorrector
        assert SCResult is K2SCResult
        assert issubclass(K2SCCorrector, Corrector)

    def test_to_float64_array_all_astropy_types(self):
        # 1. Plain Time (bkjd) and ISO Time
        t_bkjd = Time([2300.5, 2301.5, 2302.5], format="bkjd")
        npt.assert_allclose(_to_float64_array(t_bkjd), [2300.5, 2301.5, 2302.5])

        t_iso = Time(["2020-01-01T00:00:00", "2020-01-02T00:00:00"], format="isot")
        arr_iso = _to_float64_array(t_iso)
        assert arr_iso.dtype == np.float64
        npt.assert_allclose(arr_iso[1] - arr_iso[0], 1.0, atol=1e-9)

        # 2. Masked Time
        t_masked = Time(Masked([2300.0, 2301.0, 2302.0], mask=[False, True, False]), format="bkjd")
        arr_tm = _to_float64_array(t_masked)
        assert arr_tm.dtype == np.float64
        assert np.isnan(arr_tm[1])
        npt.assert_allclose(arr_tm[[0, 2]], [2300.0, 2302.0])

        # 3. MaskedQuantity & Quantity
        mq = Masked([10.0, 20.0, 30.0] * (u.electron / u.s), mask=[True, False, False])
        arr_mq = _to_float64_array(mq)
        assert arr_mq.dtype == np.float64
        assert np.isnan(arr_mq[0])
        npt.assert_allclose(arr_mq[1:], [20.0, 30.0])

        q = np.array([1.5, -2.5], dtype=np.float32) * u.pix
        arr_q = _to_float64_array(q)
        assert arr_q.dtype == np.float64
        npt.assert_allclose(arr_q, [1.5, -2.5])

        # 4. MaskedNDArray & np.ma.MaskedArray
        mnd = Masked(np.array([4.0, 5.0, 6.0]), mask=[False, False, True])
        arr_mnd = _to_float64_array(mnd)
        assert np.isnan(arr_mnd[2])
        npt.assert_allclose(arr_mnd[:2], [4.0, 5.0])

        mma = np.ma.MaskedArray([7.0, 8.0], mask=[False, True])
        arr_mma = _to_float64_array(mma)
        assert arr_mma[0] == 7.0
        assert np.isnan(arr_mma[1])


class TestResolveSplits:
    def test_k2_campaign_splits_filtered_to_bounds(self):
        t_c5 = np.linspace(2305.0, 2380.0, 100)
        assert resolve_splits(t_c5, splits="auto", campaign=5) == [2344.0]
        assert resolve_splits(t_c5, splits=None, campaign=5) == [2344.0]

        # Campaign 8 has [2579.0, 2598.5]; restrict t_max so only the first split is inside
        t_c8_half = np.linspace(2560.0, 2590.0, 80)
        assert resolve_splits(t_c8_half, splits="auto", campaign=8) == [2579.0]

        # C0 and C19 present in DEFAULT_K2_SPLITS
        assert resolve_splits(np.linspace(1940.0, 1970.0, 50), campaign=0) == [1956.0]
        assert resolve_splits(np.linspace(3490.0, 3510.0, 50), campaign=19) == [3502.0]

    def test_explicit_splits_and_empty(self):
        t = np.linspace(100.0, 150.0, 60)
        assert resolve_splits(t, splits=[90.0, 100.0, 120.5, 150.0, 160.0]) == [120.5]
        assert resolve_splits(t, splits=[]) == []
        assert resolve_splits(t, splits="none") == []

    def test_tess_downlink_gap_and_long_span_segmentation(self):
        # TESS sector (30-min FFI cadence): two 13.0 d orbits separated by a 1.4 d downlink gap at ~13.7 d
        t_orb1 = np.arange(0.0, 13.0, 1.0 / 48.0)
        t_orb2 = np.arange(14.4, 27.4, 1.0 / 48.0)
        t_tess = np.concatenate([t_orb1, t_orb2])
        splits_tess = resolve_splits(t_tess, splits="auto", mission="TESS")
        assert isinstance(splits_tess, list)
        assert len(splits_tess) == 1
        npt.assert_allclose(splits_tess[0], 13.7, atol=0.05)

        # Unknown campaign continuous 70 d baseline (> 30 d) -> floor(70/30)+1 = 3 segments (2 splits)
        t_long = np.linspace(500.0, 570.0, 200)
        splits_long = resolve_splits(t_long, splits="auto", campaign=None)
        assert len(splits_long) == 2
        assert all(500.0 < s < 570.0 for s in splits_long)


class TestLightkurveCorrectorAndPolymorphicAPI:
    def test_kepler_lc_masked_quantity_nan_unpack_units_meta_and_diagnose(self):
        sim = _make_synthetic_k2_series(n_pts=128, seed=202)
        n = sim["time"].size

        flux_vals = sim["flux"].copy()
        flux_mask = np.zeros(n, dtype=bool)
        flux_mask[10] = True  # Masked flux cadence
        flux_vals[25] = 1.08e4  # Strong upward spike outlier

        pos1 = sim["x"].copy()
        pos2 = sim["y"].copy()
        # Inject NaNs into pos_corr1 / pos_corr2 that are NOT NaN in flux
        pos1[[4, 60]] = np.nan
        pos2[95] = np.nan

        quality = np.zeros(n, dtype=np.int32)
        quality[40] = 1 << 20  # K2 thruster firing
        quality[41] = 16  # Non-zero benign K2 flag

        lc = KeplerLightCurve(
            time=Time(sim["time"], format="bkjd"),
            flux=Masked(flux_vals * (u.electron / u.s), mask=flux_mask),
            flux_err=sim["flux_err"] * (u.electron / u.s),
            quality=quality,
            meta={"MISSION": "K2", "CAMPAIGN": 5, "TARGETID": 211436265},
        )
        lc["pos_corr1"] = pos1 * u.pix
        lc["pos_corr2"] = pos2 * u.pix

        corr = K2SCCorrector(lc)
        assert isinstance(corr, Corrector)
        with pytest.raises(RuntimeError, match="before `.diagnose"):
            corr.diagnose()

        clc = corr.correct(**_FAST_KW)
        assert len(clc) == len(lc)
        assert clc.flux.unit == (u.electron / u.s)
        assert corr.tr_position.unit == (u.electron / u.s)
        assert corr.tr_time.unit == (u.electron / u.s)

        for col_name in (
            "k2sc_tr_position",
            "k2sc_tr_time",
            "k2sc_corr_flux",
            "k2sc_detrended_flux",
        ):
            assert col_name in clc.colnames
            assert len(clc[col_name]) == len(lc)
            assert clc[col_name].unit == (u.electron / u.s)
        assert "k2sc_mflags" in clc.colnames
        assert clc["k2sc_mflags"].dtype == np.uint8

        # Invalid coordinate cadences [4, 60, 95] unpack to NaN trends and M_NOTFINITE
        bad_coord_idx = [4, 60, 95]
        assert np.all(np.isnan(_to_float64_array(clc["k2sc_tr_position"])[bad_coord_idx]))
        assert np.all(np.isnan(_to_float64_array(clc["k2sc_tr_time"])[bad_coord_idx]))
        for idx in bad_coord_idx + [10]:
            assert (clc["k2sc_mflags"][idx] & M_NOTFINITE) != 0

        # Quality flag checks: both 40 and 41 flagged M_QUALITY; thruster 40 excluded from fit
        assert (clc["k2sc_mflags"][40] & M_QUALITY) != 0
        assert (clc["k2sc_mflags"][41] & M_QUALITY) != 0
        assert not corr.cadence_mask[40]
        # Outlier spike at 25 flagged upward outlier
        assert (clc["k2sc_mflags"][25] & M_OUTLIER_U) != 0

        # Metadata checks
        assert clc.meta["K2SC_SPLITS"] == [2344.0]
        assert clc.meta["K2SC_COVARIATES"] == "pos_corr"
        assert clc.meta["K2SC_CDPP_T"] < clc.meta["K2SC_CDPP_R"]
        assert isinstance(clc.meta["K2SC_PV"], np.ndarray)

        # Diagnostic plots return expected Axes objects without error
        axes_diag = corr.diagnose()
        assert len(axes_diag) == 3
        ax_xy = corr.diagnose_xy()
        assert hasattr(ax_xy, "plot")
        plt.close("all")

        # Polymorphic k2sc.detrend(lc, return_result=True)
        res_direct = k2sc.detrend(lc, return_result=True, **_FAST_KW)
        assert isinstance(res_direct, K2SCResult)
        assert res_direct.lightcurve is not None
        assert len(res_direct.lightcurve) == len(lc)

    def test_tess_lightcurve_centroid_fallback_and_auto_downlink_split(self):
        rng = np.random.default_rng(303)
        t1 = np.linspace(1500.0, 1512.0, 64)
        t2 = np.linspace(1514.6, 1526.6, 64)
        t = np.concatenate([t1, t2])
        n = t.size

        # Absolute CCD pixel centroids around (1024.3, 512.7)
        dx = 0.25 * np.sin(2.0 * np.pi * (t - 1500.0) / 1.2)
        dy = 0.20 * np.cos(2.0 * np.pi * (t - 1500.0) / 1.7)
        cent_col = 1024.3 + dx
        cent_row = 512.7 + dy

        orb = (t > 1513.3).astype(int)
        sys_flux = np.where(orb == 0, 5e-3 * dx - 3e-3 * dy, -4e-3 * dx + 4e-3 * dy)
        flux = (1.0 + sys_flux + rng.normal(0.0, 1.5e-4, size=n)) * (u.electron / u.s)
        quality = np.zeros(n, dtype=np.int32)
        quality[15] = 8  # TESS Earth-point quality flag -> must be masked out

        lc_tess = TessLightCurve(
            time=Time(t, format="btjd"),
            flux=flux,
            flux_err=np.full(n, 1.5e-4) * (u.electron / u.s),
            quality=quality,
            meta={"MISSION": "TESS"},
        )
        # All-NaN pos_corr columns (as found in some TESS products) -> must fall back to centroids
        lc_tess["pos_corr1"] = np.full(n, np.nan) * u.pix
        lc_tess["pos_corr2"] = np.full(n, np.nan) * u.pix
        lc_tess["centroid_col"] = cent_col * u.pix
        lc_tess["centroid_row"] = cent_row * u.pix

        clc = k2sc.detrend(lc_tess, **_FAST_KW)
        res = clc.k2sc_result
        assert clc.meta["K2SC_COVARIATES"] == "centroid"
        assert len(res.splits) == 1
        npt.assert_allclose(res.splits[0], 1513.3, atol=0.05)
        assert (res.mflags[15] & M_QUALITY) != 0
        assert res.cdpp_t < res.cdpp_r

    def test_generic_lc_design_matrix_transit_masks_and_remove_time_trend(self):
        sim = _make_synthetic_k2_series(n_pts=128, seed=404, periodic_amp=2.5e-3, period=5.2)
        t = sim["time"]
        flux = sim["flux"].copy()

        # Inject periodic box transits (epoch=2335.0, period=6.0, duration=0.6 d, depth=1.5%)
        t0_tr, per_tr, dur_tr = 2335.0, 6.0, 0.6
        phase = ((t - t0_tr) / per_tr + 0.5) % 1.0
        in_transit = np.abs(phase - 0.5) * per_tr <= 0.5 * dur_tr
        flux[in_transit] -= 150.0  # 1.5% transit on 1e4 continuum

        lc_gen = LightCurve(
            time=Time(t, format="jd"),
            flux=flux * u.dimensionless_unscaled,
            flux_err=sim["flux_err"] * u.dimensionless_unscaled,
        )

        # Missing covariates error check
        with pytest.raises(ValueError, match="No usable position covariates"):
            K2SCCorrector(lc_gen)

        # Pass a Lightkurve DesignMatrix with an added constant intercept + 3 basis vectors
        dm = DesignMatrix(
            np.column_stack([np.ones_like(t), sim["x"], sim["y"], sim["x"] * sim["y"]]),
            name="motion_dm",
        )

        # 1) Tuple transit_mask=(t0, period, duration) with remove_time_trend=False
        corr_tuple = K2SCCorrector(lc_gen, covariates=dm)
        clc_tuple = corr_tuple.correct(
            transit_mask=(t0_tr, per_tr, dur_tr),
            splits=[2344.0],
            remove_time_trend=False,
            **_FAST_KW,
        )
        mflags_t = clc_tuple["k2sc_mflags"]
        assert np.all((mflags_t[in_transit] & M_PERIODIC) != 0)
        # Bugfix check: in-transit cadences must NEVER be flagged M_NOTFINITE or M_OUTLIER_D
        assert np.all((mflags_t[in_transit] & M_NOTFINITE) == 0)
        assert np.all((mflags_t[in_transit] & M_OUTLIER_D) == 0)

        # 2) Boolean transit_mask + remove_time_trend=True (removes stellar wave while keeping transit dip)
        corr_bool = K2SCCorrector(lc_gen, x=sim["x"], y=sim["y"])
        clc_whitened = corr_bool.correct(
            transit_mask=in_transit,
            splits=[2344.0],
            remove_time_trend=True,
            **_FAST_KW,
        )
        npt.assert_allclose(
            _to_float64_array(clc_whitened.flux),
            _to_float64_array(clc_whitened["k2sc_detrended_flux"]),
            rtol=1e-13,
        )
        # Transit depth (~150) remains clearly present in whitened light curve
        oot_med = np.nanmedian(_to_float64_array(clc_whitened.flux)[~in_transit])
        it_med = np.nanmedian(_to_float64_array(clc_whitened.flux)[in_transit])
        assert (oot_med - it_med) > 100.0


class TestK2DataArraysAndLegacyShims:
    def test_k2data_multi_aperture_regression_and_fits_write(self, tmp_path):
        sim1 = _make_synthetic_k2_series(n_pts=128, seed=505, periodic_amp=1.2e-3)
        sim2 = _make_synthetic_k2_series(n_pts=128, seed=606, periodic_amp=1.2e-3)

        x_raw = sim1["x"].copy()
        x_raw[3] = np.nan  # Also test K2Data nanmask unpacking across 2 apertures
        fluxes = np.vstack([sim1["flux"], 1.5 * sim2["flux"]])
        errors = np.vstack([sim1["flux_err"], 1.5 * sim2["flux_err"]])

        ds = K2Data(
            211436265,
            time=sim1["time"],
            cadence=np.arange(128, dtype=np.int32),
            quality=np.zeros(128, dtype=np.int32),
            fluxes=fluxes,
            errors=errors,
            x=x_raw,
            y=sim1["y"],
            campaign=5,
        )
        assert ds.nsets == 2
        assert ds.npoints == 127

        results, detrenders = k2sc.detrend(ds, **_FAST_KW)
        assert isinstance(results, list) and len(results) == 2
        assert isinstance(detrenders, list) and len(detrenders) == 2
        for i in range(2):
            assert isinstance(results[i], K2SCResult)
            assert isinstance(results[i].detrender.tr_pv, np.ndarray)
            assert results[i].detrender.tr_pv.shape == results[i].pv.shape
            assert results[i].cdpp_t < results[i].cdpp_r

        out_fits = tmp_path / "multi_ap.fits"
        FITSWriter.write(out_fits, results[0].splits, ds, results)
        with pf.open(out_fits) as hdul:
            assert len(hdul[1].data) == 128
            assert "flux_1" in hdul[1].data.columns.names
            assert "flux_2" in hdul[1].data.columns.names
            assert np.isnan(hdul[1].data["trposi_1"][3])
            assert np.isnan(hdul[1].data["trposi_2"][3])

    def test_raw_arrays_8tuple_unpack_and_ls_fap_threshold(self):
        sim = _make_synthetic_k2_series(n_pts=128, seed=707, periodic_amp=4.0e-3, period=5.5)
        # Pass JAX arrays directly and exercise default optimizer="auto" path end-to-end
        auto_kw = {**_FAST_KW, "optimizer": "auto"}
        res = k2sc.detrend(
            jnp.asarray(sim["time"]),
            jnp.asarray(sim["flux"]),
            x=jnp.asarray(sim["x"]),
            y=jnp.asarray(sim["y"]),
            campaign=5,
            ls_max_fap=-3.0,  # Relaxed FAP threshold -> must trigger QuasiPeriodicKernel!
            **auto_kw,
        )
        assert isinstance(res, K2SCResult)
        assert res.is_periodic
        assert res.ls_period is not None and abs(res.ls_period - 5.5) < 0.6
        assert res.detrender.kernel.name == "QuasiPeriodicKernel"
        assert res.warn == (not res.success)

        # Legacy 8-tuple unpacking & indexing
        assert len(res) == 8
        det, pv, tr_t, tr_p, cr, ct, cc, warn = res
        assert det is res.detrender
        npt.assert_array_equal(pv, res.pv)
        npt.assert_array_equal(tr_t, res.tr_time)
        npt.assert_array_equal(tr_p, res.tr_position)
        assert (cr, ct, cc, warn) == (res.cdpp_r, res.cdpp_t, res.cdpp_c, res.warn)
        assert res[0] is res.detrender
        assert res[7] == res.warn

    def test_sparse_all_nan_flux_graceful_skip(self):
        t = np.linspace(2300.0, 2310.0, 40)
        x = np.sin(t)
        y = np.cos(t)
        flux = np.full(40, np.nan)
        flux[:5] = 1000.0  # Fewer than minimum required finite points

        res = k2sc.detrend(t, flux, x=x, y=y, splits=[], **_FAST_KW)
        assert not res.success
        assert res.warn
        assert res.cdpp_r == -1.0
        assert isinstance(res.detrender.tr_pv, np.ndarray)
        assert np.all(np.isnan(res.detrender.tr_pv))

    def test_deprecated_k2sc_lc_monkeypatch_shim(self):
        sim = _make_synthetic_k2_series(n_pts=128, seed=808)
        lc = KeplerLightCurve(
            time=Time(sim["time"], format="bkjd"),
            flux=sim["flux"] * (u.electron / u.s),
            flux_err=sim["flux_err"] * (u.electron / u.s),
            meta={"CAMPAIGN": 5, "TARGETID": 211436265},
        )
        lc["pos_corr1"] = sim["x"] * u.pix
        lc["pos_corr2"] = sim["y"] * u.pix

        lc.__class__ = k2sc_lc
        with pytest.warns(DeprecationWarning, match="deprecated"):
            lc.k2sc(**_FAST_KW)

        assert lc.tr_position.unit == (u.electron / u.s)
        assert lc.tr_time.unit == (u.electron / u.s)
        assert lc.corr_flux.unit == (u.electron / u.s)
        assert len(lc.corr_flux) == 128
        assert isinstance(lc.pv, np.ndarray)
        assert lc.cdpp_t < lc.cdpp_r
        assert lc.detrender is not None
