"""Comprehensive unit tests for K2SC core constants, K2Data, DtData, I/O readers/writers,
K2SCFile, and SegmentedDetrender deprecation shim (PR 2).
"""

from __future__ import annotations

import importlib
import warnings
from collections import namedtuple
from pathlib import Path

import astropy.io.fits as pf
import numpy as np
import pytest
from k2sc.core import (
    DEFAULT_K2_SPLITS,
    M_FLARE,
    M_NOTFINITE,
    M_OUTLIER_D,
    M_OUTLIER_U,
    M_PERIODIC,
    M_QUALITY,
    M_TRANSIT,
    default_splits,
)
from k2sc.dtdata import DtData
from k2sc.k2data import K2Data
from k2sc.k2io import AMCReader, FITSWriter, MASTReader, SPLOXReader, select_reader
from k2sc.k2scfile import K2SCFile
from k2sc.kernels import BasicKernel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_OP_DIR = _REPO_ROOT / "tests" / "data_op"

_DummyDetrender = namedtuple("_DummyDetrender", ["kernel", "tr_pv"])
_DummyResult = namedtuple(
    "_DummyResult",
    ["detrender", "tr_time", "tr_position", "cdpp_r", "cdpp_t", "cdpp_c", "warn"],
)


def _make_dummy_results(nsets: int, npoints: int) -> list[_DummyResult]:
    kernel = BasicKernel()
    det = _DummyDetrender(kernel=kernel, tr_pv=kernel.pv0.copy())
    results = []
    for i in range(nsets):
        tr_t = np.full(npoints, 1000.0 + i * 10.0, dtype=np.float64) + np.linspace(
            -0.5, 0.5, npoints
        )
        tr_p = np.full(npoints, 1000.0 + i * 10.0, dtype=np.float64) + np.sin(
            np.linspace(0.0, 2.0 * np.pi, npoints)
        )
        results.append(
            _DummyResult(
                detrender=det,
                tr_time=tr_t,
                tr_position=tr_p,
                cdpp_r=120.5 + i,
                cdpp_t=45.2 + i,
                cdpp_c=32.1 + i,
                warn=False,
            )
        )
    return results


class TestCoreConstantsAndSplits:
    def test_bitmask_constants(self) -> None:
        assert M_QUALITY == 1
        assert M_FLARE == 2
        assert M_TRANSIT == 4
        assert M_OUTLIER_U == 8
        assert M_OUTLIER_D == 16
        assert M_NOTFINITE == 32
        assert M_PERIODIC == 64

    def test_default_k2_splits_campaigns_c0_to_c19(self) -> None:
        assert default_splits is DEFAULT_K2_SPLITS
        required_keys = set(range(20)) | {91, 92, 101, 102, 111, 112}
        assert required_keys.issubset(DEFAULT_K2_SPLITS.keys())
        # Issue 10: Campaign 8 splits
        assert DEFAULT_K2_SPLITS[8] == [2579.0, 2598.5]
        for camp, splits in DEFAULT_K2_SPLITS.items():
            assert isinstance(camp, int)
            assert isinstance(splits, list)
            assert len(splits) >= 1
            assert all(isinstance(s, float) for s in splits)
            assert splits == sorted(splits)


class TestAMCReaderAndPrimaryHeaderNone:
    def test_amc_reader_text_detection_and_fits_write_without_headers(
        self, tmp_path: Path
    ) -> None:
        amc_file = tmp_path / "EPIC_211436265_lc.dat"
        header = "dates cadences flux1 err1 flux2 err2 xpos ypos quality\n"
        n_rows = 24
        t = np.linspace(2300.0, 2310.0, n_rows)
        cad = np.arange(1000, 1000 + n_rows)
        f1 = np.full(n_rows, 5000.0)
        e1 = np.full(n_rows, 1.5)
        f2 = np.full(n_rows, 6200.0)
        e2 = np.full(n_rows, 1.8)
        x = np.linspace(-0.2, 0.2, n_rows)
        y = np.linspace(0.1, -0.1, n_rows)
        qual = np.zeros(n_rows, dtype=int)

        # Inject a NaN in xpos at row 5
        x[5] = np.nan
        qual[7] = 16

        rows = [header]
        for k in range(n_rows):
            rows.append(
                f"{t[k]:.6f} {cad[k]:d} {f1[k]:.4f} {e1[k]:.4f} "
                f"{f2[k]:.4f} {e2[k]:.4f} {x[k]:.6f} {y[k]:.6f} {qual[k]:d}\n"
            )
        amc_file.write_text("".join(rows), encoding="utf-8")

        # Issue 28: AMCReader.can_read in Python 3
        assert AMCReader.can_read(str(amc_file)) is True
        assert select_reader(str(amc_file)) is AMCReader

        k2data = AMCReader.read(str(amc_file), campaign=5)
        assert k2data.epic == 211436265
        assert k2data.campaign == 5
        assert k2data.nsets == 2
        assert k2data.npoints == n_rows - 1
        assert k2data.nanmask.size == n_rows
        assert not k2data.nanmask[5]
        assert k2data.primary_header is None
        assert k2data.data_header is None

        # Issue 16: FITSWriter.write when data.primary_header is None
        out_fits = tmp_path / "EPIC_211436265_amc.fits"
        dtres = _make_dummy_results(k2data.nsets, k2data.npoints)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FITSWriter.write(str(out_fits), [2305.0], k2data, dtres)
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warns == []

        assert out_fits.exists()
        with pf.open(str(out_fits)) as hdul:
            assert hdul[0].header["ORIGIN"] == "SPLOX: Stars and Planets at Oxford"
            assert hdul[1].header["EXTNAME"] == "K2SC"
            assert hdul[1].header["EPIC"] == 211436265
            tbl = hdul[1].data
            assert len(tbl) == n_rows
            assert np.isnan(tbl["x"][5])
            assert np.isnan(tbl["flux_1"][5])
            assert np.isnan(tbl["trposi_2"][5])
            assert tbl["cadence"][5] == 1005
            assert (tbl["mflags_1"][5] & M_NOTFINITE) != 0
            assert (tbl["mflags_2"][5] & M_NOTFINITE) != 0

        # Multi-aperture K2SCFile loading (aperture=1 and aperture=2) without KeyError: 'flux'
        kf1 = K2SCFile(str(out_fits), aperture=1)
        kf2 = K2SCFile(str(out_fits), aperture=2)
        assert kf1.aperture == 1
        assert kf2.aperture == 2
        assert not kf1.is_hlsp
        assert np.allclose(kf1.original_flux, 5000.0)
        assert np.allclose(kf2.original_flux, 6200.0)
        with pytest.raises(KeyError, match="Aperture 3"):
            K2SCFile(str(out_fits), aperture=3)

    def test_amc_reader_txt_extension_and_single_row(self, tmp_path: Path) -> None:
        """AMCReader must accept .txt files and handle 1-row files safely (ndmin=2)."""
        amc_txt = tmp_path / "EPIC_203794605.txt"
        amc_txt.write_text(
            "dates cadences flux1 err1 xpos ypos quality\n"
            "2301.25 1001 5100.0 1.2 0.15 -0.08 0\n",
            encoding="utf-8",
        )
        assert AMCReader.can_read(str(amc_txt)) is True
        assert select_reader(str(amc_txt)) is AMCReader
        ds = AMCReader.read(str(amc_txt), campaign=5)
        assert ds.epic == 203794605
        assert ds.nsets == 1
        assert ds.npoints == 1
        assert np.isclose(ds.time[0], 2301.25)
        assert np.isclose(ds.fluxes[0, 0], 5100.0)


class TestMASTReaderAndSPLOXReader:
    def test_mast_reader_on_real_c05_fits(self) -> None:
        fits_files = sorted(_DATA_OP_DIR.glob("ktwo*-c05_llc.fits"))
        assert len(fits_files) >= 1
        fpath = fits_files[0]

        assert MASTReader.can_read(str(fpath)) is True
        assert select_reader(str(fpath)) is MASTReader

        ds_pdc = MASTReader.read(str(fpath), sid=0, type="pdc")
        ds_sap = MASTReader.read(str(fpath), sid=0, type="sap")

        assert ds_pdc.epic == 211436265
        assert ds_pdc.campaign == 5
        assert ds_pdc.nsets == 1
        assert ds_pdc.npoints > 3000
        assert ds_pdc.raw_time.size >= ds_pdc.npoints
        assert int(np.sum(ds_pdc.nanmask)) == ds_pdc.npoints
        assert ds_sap.npoints == ds_pdc.npoints

        # Verify mask_periodic_signal flags in-transit cadences with M_PERIODIC
        t0 = float(ds_pdc.time[10])
        ds_pdc.mask_periodic_signal(center=t0, period=5.0, duration=0.2)
        assert np.any((ds_pdc.mflags[0] & M_PERIODIC) != 0)
        assert np.any((ds_pdc.mflags[0] & M_PERIODIC) == 0)

    def test_splox_reader_synthetic_fits(self, tmp_path: Path) -> None:
        splox_path = tmp_path / "splox_test.fits"
        nobj, napt, nexp = 2, 6, 12
        objno = np.array([200000001, 200000002], dtype=np.int32)
        f_rel = np.zeros((nobj, nexp, napt), dtype=np.float64)
        f_med = np.full((nobj, napt), 1000.0, dtype=np.float64)
        x = np.tile(np.linspace(-0.1, 0.1, nexp), (nobj, 1))
        y = np.tile(np.linspace(0.05, -0.05, nexp), (nobj, 1))

        col1 = [
            pf.Column(name="objno", format="J", array=objno),
            pf.Column(
                name="f",
                format=f"{nexp * napt}D",
                dim=f"({napt},{nexp})",
                array=f_rel.reshape(nobj, -1),
            ),
            pf.Column(name="f_med", format=f"{napt}D", array=f_med),
            pf.Column(name="x", format=f"{nexp}D", array=x),
            pf.Column(name="y", format=f"{nexp}D", array=y),
            pf.Column(name="f_scatter", format="D", array=np.ones(nobj)),
        ]
        hdu1 = pf.BinTableHDU.from_columns(col1)

        col2 = [
            pf.Column(
                name="mjd_obs", format="D", array=np.linspace(2300.0, 2310.0, nexp)
            ),
            pf.Column(name="cadence", format="J", array=np.arange(1, nexp + 1)),
        ]
        hdu2 = pf.BinTableHDU.from_columns(col2)
        pf.HDUList([pf.PrimaryHDU(), hdu1, hdu2]).writeto(
            str(splox_path), overwrite=True
        )

        assert SPLOXReader.can_read(str(splox_path)) is True
        assert select_reader(str(splox_path)) is SPLOXReader
        assert SPLOXReader.nstars(str(splox_path)) == nobj

        ds = SPLOXReader.read(str(splox_path), sid=1, campaign=5)
        assert ds.epic == 200000002
        assert ds.nsets == napt
        assert ds.npoints == nexp - 1
        assert ds.quality.dtype == np.int32

    def test_mast_reader_renamed_file_header_epic_and_tess_generic_fallbacks(
        self, tmp_path: Path
    ) -> None:
        """MASTReader resolves EPIC/TICID from header first and supports TESS/generic FITS."""
        n = 30
        t = np.linspace(1325.0, 1350.0, n, dtype=np.float64)
        cad = np.arange(1, n + 1, dtype=np.int32)
        flux = 10000.0 + 20.0 * np.sin(t)
        err = np.full(n, 2.0, dtype=np.float64)
        qual = np.zeros(n, dtype=np.int32)
        qual[4] = 16
        mom_x = np.linspace(100.0, 100.5, n)
        mom_y = np.linspace(200.0, 199.5, n)

        # 1. Renamed FITS file (issue 23): '203794605.fits' with KEPLERID in primary header
        renamed = tmp_path / "203794605.fits"
        phead = pf.Header()
        phead["KEPLERID"] = 203794605
        phead["CAMPAIGN"] = 2
        cols_k2 = [
            pf.Column(name="TIME", format="D", array=t),
            pf.Column(name="CADENCENO", format="J", array=cad),
            pf.Column(name="SAP_FLUX", format="D", array=flux),
            pf.Column(name="SAP_FLUX_ERR", format="D", array=err),
            pf.Column(name="PDCSAP_FLUX", format="D", array=flux),
            pf.Column(name="PDCSAP_FLUX_ERR", format="D", array=err),
            pf.Column(name="SAP_QUALITY", format="J", array=qual),
            pf.Column(name="POS_CORR1", format="D", array=np.linspace(-0.2, 0.2, n)),
            pf.Column(name="POS_CORR2", format="D", array=np.linspace(0.2, -0.2, n)),
        ]
        pf.HDUList(
            [pf.PrimaryHDU(header=phead), pf.BinTableHDU.from_columns(cols_k2)]
        ).writeto(str(renamed), overwrite=True)
        assert MASTReader.can_read(str(renamed)) is True
        ds_ren = MASTReader.read(str(renamed))
        assert ds_ren.epic == 203794605
        assert ds_ren.campaign == 2

        # 1b. Renamed FITS with NO header ID falls back to filename digits (no epic=0 collision)
        renamed_nohdr = tmp_path / "203794605_custom.fits"
        pf.HDUList([pf.PrimaryHDU(), pf.BinTableHDU.from_columns(cols_k2)]).writeto(
            str(renamed_nohdr), overwrite=True
        )
        assert MASTReader.read(str(renamed_nohdr)).epic == 203794605

        # 2. TESS SPOC FITS: TICID, QUALITY (not SAP_QUALITY), POS_CORR sparse (<15 finite) -> MOM_CENTR
        tess_path = tmp_path / "tess_target.fits"
        phead_t = pf.Header()
        phead_t["TICID"] = 261136679
        pos_sparse = np.full(n, np.nan, dtype=np.float64)
        pos_sparse[:5] = 0.01
        cols_tess = [
            pf.Column(name="TIME", format="D", array=t),
            pf.Column(name="CADENCENO", format="J", array=cad),
            pf.Column(name="PDCSAP_FLUX", format="D", array=flux),
            pf.Column(name="PDCSAP_FLUX_ERR", format="D", array=err),
            pf.Column(name="QUALITY", format="J", array=qual),
            pf.Column(name="POS_CORR1", format="D", array=pos_sparse),
            pf.Column(name="POS_CORR2", format="D", array=pos_sparse),
            pf.Column(name="MOM_CENTR1", format="D", array=mom_x),
            pf.Column(name="MOM_CENTR2", format="D", array=mom_y),
        ]
        pf.HDUList(
            [pf.PrimaryHDU(header=phead_t), pf.BinTableHDU.from_columns(cols_tess)]
        ).writeto(str(tess_path), overwrite=True)
        assert MASTReader.can_read(str(tess_path)) is True
        assert select_reader(str(tess_path)) is MASTReader
        ds_tess = MASTReader.read(str(tess_path))
        assert ds_tess.epic == 261136679
        assert ds_tess.npoints == n
        assert np.allclose(ds_tess.x, mom_x)
        assert np.allclose(ds_tess.y, mom_y)
        assert ds_tess.quality[4] == 16

        # 3. Generic light-curve FITS: single FLUX/FLUX_ERR, no quality/cadence, CENTROID_COL/ROW
        gen_path = tmp_path / "generic_star.fits"
        phead_g = pf.Header()
        phead_g["OBJECT"] = "EPIC 201234567"
        cols_gen = [
            pf.Column(name="TIME", format="D", array=t),
            pf.Column(name="FLUX", format="D", array=flux),
            pf.Column(name="FLUX_ERR", format="D", array=err),
            pf.Column(name="CENTROID_COL", format="D", array=mom_x),
            pf.Column(name="CENTROID_ROW", format="D", array=mom_y),
        ]
        pf.HDUList(
            [pf.PrimaryHDU(header=phead_g), pf.BinTableHDU.from_columns(cols_gen)]
        ).writeto(str(gen_path), overwrite=True)
        assert MASTReader.can_read(str(gen_path)) is True
        ds_gen = MASTReader.read(str(gen_path))
        assert ds_gen.epic == 201234567
        assert ds_gen.npoints == n
        assert np.all(ds_gen.quality == 0)
        assert np.allclose(ds_gen.x, mom_x)
        assert np.allclose(ds_gen.y, mom_y)

    def test_mast_header_preservation_and_zero_verify_warnings(
        self, tmp_path: Path
    ) -> None:
        """MAST primary and table header keywords survive FITSWriter.write cleanly (issue 7)."""
        from astropy.io.fits.verify import VerifyWarning
        from k2sc.kernels import PeriodicKernel, QuasiPeriodicKernel

        fits_files = sorted(_DATA_OP_DIR.glob("ktwo*-c05_llc.fits"))
        assert len(fits_files) >= 1
        src_fits = fits_files[0]

        with pf.open(str(src_fits)) as src_hdul:
            expected_primary = {
                k: src_hdul[0].header[k]
                for k in (
                    "KEPLERID",
                    "CAMPAIGN",
                    "RA_OBJ",
                    "DEC_OBJ",
                    "CHANNEL",
                    "MODULE",
                    "OUTPUT",
                    "TELESCOP",
                    "INSTRUME",
                )
                if k in src_hdul[0].header
            }
            expected_table = {
                k: src_hdul[1].header[k]
                for k in ("KEPLERID", "TELESCOP", "INSTRUME")
                if k in src_hdul[1].header
            }

        ds = MASTReader.read(str(src_fits), type="pdc")

        for kernel_cls in (BasicKernel, QuasiPeriodicKernel, PeriodicKernel):
            kern = kernel_cls()
            pv = kern.pv0.copy() + 0.123456789
            det = _DummyDetrender(kernel=kern, tr_pv=pv)
            dtres = [
                _DummyResult(
                    detrender=det,
                    tr_time=np.full(ds.npoints, 1000.0),
                    tr_position=np.full(ds.npoints, 1000.0),
                    cdpp_r=120.0,
                    cdpp_t=45.0,
                    cdpp_c=32.0,
                    warn=False,
                )
            ]
            out_fits = tmp_path / f"preserved_{kernel_cls.__name__}.fits"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                FITSWriter.write(str(out_fits), [2344.0], ds, dtres)
            verify_warns = [w for w in caught if issubclass(w.category, VerifyWarning)]
            assert verify_warns == [], (
                f"Unexpected VerifyWarning for {kernel_cls.__name__}: {verify_warns}"
            )

            with pf.open(str(out_fits)) as out_hdul:
                for k, v in expected_primary.items():
                    assert out_hdul[0].header[k] == v, f"Primary header {k} mismatch"
                for k, v in expected_table.items():
                    assert out_hdul[1].header[k] == v, f"Table header {k} mismatch"
                assert (
                    out_hdul[0].header["ORIGIN"] == "SPLOX: Stars and Planets at Oxford"
                )
                assert out_hdul[1].header["EXTNAME"] == "K2SC"


class TestFITSWriterUnpackAndK2SCFileRoundTrip:
    def test_fits_writer_nan_unpack_and_k2scfile_roundtrip(
        self, tmp_path: Path
    ) -> None:
        n_full = 60
        raw_time = np.linspace(2300.0, 2360.0, n_full, dtype=np.float64)
        raw_cadence = np.arange(5000, 5000 + n_full, dtype=np.int32)
        raw_quality = np.zeros(n_full, dtype=np.int32)
        raw_quality[12] = 128
        raw_x = 0.1 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_full))
        raw_y = 0.1 * np.cos(np.linspace(0.0, 4.0 * np.pi, n_full))
        raw_flux = 10000.0 + 50.0 * raw_x - 30.0 * raw_y
        raw_err = np.full(n_full, 2.0, dtype=np.float64)

        # Inject NaNs into time, x, y, and flux
        raw_time[3] = np.nan
        raw_x[10] = np.nan
        raw_y[25] = np.nan
        raw_flux[40] = np.nan

        phead = pf.Header()
        phead["CAMPAIGN"] = 5
        dhead = pf.Header()

        data = K2Data(
            211999888,
            time=raw_time,
            cadence=raw_cadence,
            quality=raw_quality,
            fluxes=raw_flux,
            errors=raw_err,
            x=raw_x,
            y=raw_y,
            primary_header=phead,
            data_header=dhead,
            campaign=5,
        )

        # 3 coordinate NaNs removed in data.time, while nanmask has full size 60
        assert data.npoints == n_full - 3
        assert data.nanmask.size == n_full

        # Flag one upward and one downward outlier in the valid subset
        data.mflags[0, 15] |= M_OUTLIER_U
        data.mflags[0, 20] |= M_OUTLIER_D
        data.mflags[0, 30] |= M_PERIODIC

        dtres = _make_dummy_results(1, data.npoints)
        out_fits = tmp_path / "k2sc_roundtrip.fits"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FITSWriter.write(str(out_fits), [2330.0], data, dtres)
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warns == []

        # Issue 9: Output FITS table must have full input length n_full (60), not 57
        with pf.open(str(out_fits)) as hdul:
            tbl = hdul[1].data
            assert len(tbl) == n_full
            for bad_idx in (3, 10, 25):
                assert np.isnan(tbl["time"][bad_idx])
                assert np.isnan(tbl["x"][bad_idx])
                assert np.isnan(tbl["y"][bad_idx])
                assert np.isnan(tbl["flux"][bad_idx])
                assert np.isnan(tbl["trtime"][bad_idx])
                assert np.isnan(tbl["trposi"][bad_idx])
                assert tbl["cadence"][bad_idx] == raw_cadence[bad_idx]
                assert tbl["quality"][bad_idx] == raw_quality[bad_idx]
                assert (tbl["mflags"][bad_idx] & M_NOTFINITE) != 0

            # Cadence and quality at valid indices match exact input integers
            assert np.array_equal(tbl["cadence"], raw_cadence)
            assert np.array_equal(tbl["quality"], raw_quality)

        # Issue 23: Read back via K2SCFile without np.bool errors
        kfile_raw = K2SCFile(
            str(out_fits), normalize=False, remove_nans=False, remove_outliers=False
        )
        assert kfile_raw.time.size == n_full
        assert kfile_raw.original_flux.size == n_full
        assert kfile_raw.detrended_flux.size == n_full
        assert kfile_raw.residual_flux.size == n_full
        assert kfile_raw.position_trend.size == n_full
        assert kfile_raw.time_trend.size == n_full

        kfile_clean = K2SCFile(
            str(out_fits), normalize=True, remove_nans=True, remove_outliers=True
        )
        # 60 total - 3 coord NaNs - 1 flux NaN - 2 outliers = 54 finite non-outlier points
        assert kfile_clean.time.size == n_full - 6
        assert np.all(np.isfinite(kfile_clean.detrended_flux))
        assert np.isclose(np.nanmedian(kfile_clean.original_flux), 1.0, rtol=1e-3)

        t_up, f_up = kfile_clean.outliers_up
        t_dn, f_dn = kfile_clean.outliers_down
        assert t_up.size == 1 and f_up.size == 1
        assert t_dn.size == 1 and f_dn.size == 1

        orig_out_t, orig_out_f = kfile_clean.original_outliers
        det_out_t, det_out_f = kfile_clean.detrended_outliers
        assert orig_out_t.size == 2 and orig_out_f.size == 2
        assert det_out_t.size == 2 and det_out_f.size == 2

        kfile_clean.set_normalize(False)
        kfile_clean.set_remove_outliers(False)
        kfile_clean.set_remove_nans(False)
        assert kfile_clean.time.size == n_full

    def test_k2scfile_hlsp_no_double_subtraction(self, tmp_path: Path) -> None:
        """K2SCFile must detect k2mastify HLSP extensions (PDC/SAP) and never double-subtract."""
        n = 50
        t = np.linspace(2300.0, 2320.0, n, dtype=np.float64)
        tr_pos = 1000.0 + 40.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, n))
        tr_time = 1000.0 + 25.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, n))
        resid_true = 10000.0 + np.linspace(-2.0, 2.0, n)
        # Simulate k2mastify formula: flux_hlsp = raw - (trposi - mean) - (trtime - mean)
        raw_true = (
            resid_true + (tr_pos - np.nanmean(tr_pos)) + (tr_time - np.nanmean(tr_time))
        )
        flux_hlsp = (
            raw_true - tr_pos + np.nanmean(tr_pos) - tr_time + np.nanmean(tr_time)
        )
        mflags = np.zeros(n, dtype=np.uint8)
        mflags[5] = M_OUTLIER_U

        def _make_hlsp_hdu(extname: str, offset: float) -> pf.BinTableHDU:
            cols = [
                pf.Column(name="time", format="D", array=t),
                pf.Column(name="flux", format="D", array=flux_hlsp + offset),
                pf.Column(name="trposi", format="D", array=tr_pos),
                pf.Column(name="trtime", format="D", array=tr_time),
                pf.Column(name="mflags", format="B", array=mflags),
            ]
            hdu = pf.BinTableHDU.from_columns(cols)
            hdu.header["EXTNAME"] = extname
            return hdu

        hlsp_path = tmp_path / "hlsp_k2sc_k2_llc_211436265-c05_kepler_v1_lc.fits"
        pf.HDUList(
            [pf.PrimaryHDU(), _make_hlsp_hdu("PDC", 0.0), _make_hlsp_hdu("SAP", 50.0)]
        ).writeto(str(hlsp_path), overwrite=True)

        kf_pdc = K2SCFile(str(hlsp_path), ext=1)
        assert kf_pdc.is_hlsp is True
        np.testing.assert_allclose(
            kf_pdc.residual_flux, flux_hlsp, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            kf_pdc.original_flux, raw_true, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            kf_pdc.detrended_flux,
            raw_true - (tr_pos - np.nanmean(tr_pos)),
            rtol=1e-12,
            atol=1e-12,
        )

        # Also check SAP extension (ext="SAP" / ext=2)
        kf_sap = K2SCFile(str(hlsp_path), ext="SAP")
        assert kf_sap.is_hlsp is True
        np.testing.assert_allclose(
            kf_sap.residual_flux, flux_hlsp + 50.0, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            kf_sap.original_flux, raw_true + 50.0, rtol=1e-12, atol=1e-12
        )


class TestDtDataAndTrainingSet:
    def test_dtdata_covariate_dimensions_2d_3d_5d(self) -> None:
        n = 40
        flux = np.linspace(1000.0, 1010.0, n)
        t = np.linspace(0.0, 10.0, n)
        x = np.sin(t)
        y = np.cos(t)

        # 2D inputs: (t, x)
        d2 = DtData(flux, np.column_stack([t, x]))
        assert d2.masked_inputs.shape == (n, 2)
        assert np.allclose(d2.mx, x)
        assert np.allclose(d2.ux, x)
        assert np.allclose(d2.my, np.zeros(n))
        assert np.allclose(d2.uy, np.zeros(n))

        # 3D inputs: (t, x, y)
        d3 = DtData(flux, np.column_stack([t, x, y]))
        assert d3.masked_inputs.shape == (n, 3)
        assert np.allclose(d3.mx, x)
        assert np.allclose(d3.my, y)

        # 5D inputs: (t, x, y, c3, c4)
        c3 = x**2
        c4 = y**2
        d5 = DtData(flux, np.column_stack([t, x, y, c3, c4]))
        assert d5.masked_inputs.shape == (n, 5)
        assert np.allclose(d5.mx, x)
        assert np.allclose(d5.my, y)

    def test_dtdata_dynamic_mask_updates_and_outliers_property(self) -> None:
        n = 50
        t = np.arange(n, dtype=np.float64)
        x = 0.1 * np.sin(t)
        y = 0.1 * np.cos(t)
        flux = np.full(n, 500.0, dtype=np.float64)
        flux[10] = 2000.0  # Upward spike
        flux[20] = 100.0  # Downward spike

        d = DtData(flux, np.column_stack([t, x, y]))
        assert d.nptm == n
        assert d.nptu == n
        assert d.outliers[0].size == 0

        new_mask = np.ones(n, dtype=bool)
        new_mask[[10, 20]] = False
        d.mask = new_mask

        assert d.nptm == n - 2
        assert d.mf.size == n - 2
        assert d.mt.size == n - 2
        assert d.mx.size == n - 2
        assert d.my.size == n - 2
        assert d.mi.shape == (n - 2, 3)
        assert np.isclose(d.flux_median, 500.0)
        assert np.allclose(d.masked_normalised_flux, np.zeros(n - 2))

        out_t, out_norm_f = d.outliers
        assert np.array_equal(out_t, np.array([10.0, 20.0]))
        assert np.allclose(out_norm_f, np.array([2000.0 / 500.0, 100.0 / 500.0]))

    def test_create_training_set_determinism_and_rng_isolation(self) -> None:
        n = 300
        t = np.linspace(2300.0, 2380.0, n)
        inputs = np.column_stack([t, np.sin(t), np.cos(t)])
        flux = 1000.0 + np.sin(0.5 * t)
        d = DtData(flux, inputs)

        np.random.seed(999)
        expected_global_draw = np.random.RandomState(999).rand(5)

        tr1 = d.create_training_set(nrandom=50, nblocks=4, bspan=25, seed=42)
        actual_global_draw = np.random.rand(5)
        tr2 = d.create_training_set(nrandom=50, nblocks=4, bspan=25, seed=42)
        tr3 = d.create_training_set(
            nrandom=50, nblocks=4, bspan=25, rng=np.random.default_rng(42)
        )

        # Global NumPy RNG was untouched by create_training_set
        assert np.allclose(actual_global_draw, expected_global_draw)
        assert np.array_equal(tr1.unmasked_time, tr2.unmasked_time)
        assert np.array_equal(tr1.unmasked_time, tr3.unmasked_time)
        assert tr1.nptu == 50 + 4 * 25
        assert np.isclose(tr1.flux_median, d.flux_median)

    def test_create_training_set_non_overlapping_blocks_and_max_tries_safety(
        self,
    ) -> None:
        n = 200
        t = np.arange(n, dtype=np.float64)
        inputs = np.column_stack([t, np.zeros(n), np.zeros(n)])
        d = DtData(np.ones(n), inputs)

        # Pure block sampling (nrandom=0): 4 blocks of 25 must yield 100 strictly unique points
        tr_blocks = d.create_training_set(nrandom=0, nblocks=4, bspan=25, seed=7)
        assert tr_blocks.nptu == 100
        assert np.unique(tr_blocks.unmasked_time).size == 100

        # Impossible block request (10 blocks of 30 in 100 points) with small max_tries:
        # must never leak -bspan sentinels or duplicate points
        d_small = DtData(np.ones(100), inputs[:100])
        tr_exhaust = d_small.create_training_set(
            nrandom=10, nblocks=10, bspan=30, max_tries=5, seed=11
        )
        assert tr_exhaust.nptu == 100
        assert np.all(tr_exhaust.unmasked_time >= 0.0)
        assert np.unique(tr_exhaust.unmasked_time).size == 100
        assert np.all(np.diff(tr_exhaust.unmasked_time) > 0.0)

    def test_create_training_set_short_datasets(self) -> None:
        # Dataset with ids.size <= bspan and nblocks <= 0
        n = 25
        t = np.arange(n, dtype=np.float64)
        inputs = np.column_stack([t, np.sin(t), np.cos(t)])
        d = DtData(np.full(n, 100.0), inputs)

        tr_short = d.create_training_set(nrandom=40, nblocks=5, bspan=50, seed=0)
        assert tr_short.nptu == n
        assert np.array_equal(tr_short.unmasked_time, t)

        tr_noblocks = d.create_training_set(nrandom=10, nblocks=0, bspan=50, seed=0)
        assert tr_noblocks.nptu == 10

        # Empty mask dataset
        empty_mask = np.zeros(n, dtype=bool)
        d_empty = DtData(np.full(n, 100.0), inputs, mask=empty_mask)
        tr_empty = d_empty.create_training_set(nrandom=10, nblocks=2, bspan=5, seed=0)
        assert tr_empty.nptu == 0


class TestSegmentedDetrenderDeprecation:
    def test_segdetrender_deprecation_and_methods(
        self, canonical_gp_dataset: dict[str, object]
    ) -> None:
        with pytest.deprecated_call():
            import k2sc.segdetrender as seg_mod

            importlib.reload(seg_mod)

        X = np.asarray(canonical_gp_dataset["X_tr"], dtype=np.float64)
        y = 1000.0 * (1.0 + np.asarray(canonical_gp_dataset["y_tr"], dtype=np.float64))
        splits = list(canonical_gp_dataset["splits"])

        with pytest.deprecated_call():
            sdet = seg_mod.SegmentedDetrender(
                y, X, splits=splits, tr_nrandom=20, tr_bspan=10, tr_nblocks=2
            )

        mask_mat = sdet._get_mask(X[:, 0], X[:, 0])
        assert mask_mat.dtype == bool
        assert mask_mat.shape == (X.shape[0], X.shape[0])

        K = sdet.compute_cmat(sdet.kernel.pv0, X, X, add_wn=True)
        assert K.shape == (X.shape[0], X.shape[0])
        assert np.all(np.isfinite(K))

        nll = sdet.negll(sdet.kernel.pv0)
        assert np.isfinite(nll)
