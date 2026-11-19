"""End-to-end unit tests for ``k2sc.cli`` (``k2sc``, ``k2plot``, ``k2ginfo``, ``k2mastify``)."""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import astropy.io.fits as pf
import k2sc
import numpy as np
import pytest
from astropy.io.fits.verify import VerifyWarning
from k2sc import cli
from k2sc.k2scfile import K2SCFile

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_FITS = _REPO_ROOT / "tests" / "data_op" / "ktwo211436265-c05_llc.fits"


@pytest.fixture(scope="module", autouse=True)
def _preserve_package_lazy_namespace():
    yield
    k2sc.__dict__.pop("pipeline", None)
    k2sc.__dict__.pop("corrector", None)


class TestCLIModuleHygiene:
    def test_cli_does_not_import_heavy_optional_packages(self):
        # Verify zero unconditional mpi4py, seaborn, IPython, or eager kernels imports in cli.py,
        # and verify configure_single_thread_env runs before numpy/astropy under MPI.
        with open(_REPO_ROOT / "src" / "cli.py", encoding="utf-8") as fh:
            src_text = fh.read()
        assert "seaborn" not in src_text
        assert "IPython" not in src_text
        assert "\nfrom .kernels import kernels" not in src_text
        assert src_text.index("configure_single_thread_env()") < src_text.index(
            "import numpy as np"
        )

    def test_bin_wrappers_are_python3_thin_entrypoints(self):
        for script_name, fn_name in (
            ("k2sc", "main"),
            ("k2plot", "main_plot"),
            ("k2ginfo", "main_ginfo"),
            ("k2mastify", "main_mastify"),
        ):
            script_path = _REPO_ROOT / "bin" / script_name
            content = script_path.read_text(encoding="utf-8")
            assert content.startswith("#!/usr/bin/env python3")
            assert f"from k2sc.cli import {fn_name}" in content


class TestFilelistExpansion:
    def test_at_filelist_and_txt_expansion(self, tmp_path):
        f1 = tmp_path / "ktwo211436265-c05_llc.fits"
        f2 = tmp_path / "sub" / "ktwo211495446-c05_llc.fits"
        f2.parent.mkdir()
        f1.write_bytes(b"placeholder")
        f2.write_bytes(b"placeholder")

        list_file = tmp_path / "targets.list"
        list_file.write_text(
            "# K2 C5 batch list\n"
            "\n"
            "ktwo211436265-c05_llc.fits  # relative to list dir\n"
            f"{f2}  # explicit path\n",
            encoding="utf-8",
        )

        exp_at, from_dir_at = cli.expand_input_files([f"@{list_file}"])
        assert not from_dir_at
        assert exp_at == [str(f1.resolve()), str(f2.resolve())]

        exp_txt, from_dir_txt = cli.expand_input_files([str(list_file)])
        assert not from_dir_txt
        assert exp_txt == [str(f1.resolve()), str(f2.resolve())]

    def test_amc_txt_lightcurve_not_misread_as_filelist(self, tmp_path):
        """An AMC .txt light curve must NOT be expanded as a filelist (issues 14, 27, 28)."""
        amc_txt = tmp_path / "EPIC_211436265.txt"
        amc_txt.write_text(
            "dates cadences flux1 err1 xpos ypos quality\n"
            "2301.0 1000 5000.0 1.2 0.1 -0.1 0\n"
            "2301.02 1001 5002.0 1.2 0.11 -0.09 0\n",
            encoding="utf-8",
        )
        expanded, from_dir = cli.expand_input_files([str(amc_txt)])
        assert not from_dir
        assert expanded == [str(amc_txt.resolve())]


class TestEndToEndCLIWorkflow:
    def test_main_k2sc_plot_ginfo_and_mastify_end_to_end(self, tmp_path):
        assert _SAMPLE_FITS.exists(), f"Expected sample FITS at {_SAMPLE_FITS}"

        # Create a trimmed 180-cadence copy of real MAST C05 FITS so full end-to-end
        # CLI detrending + FITS writing + PDF plotting runs in ~0.5 s on shared CI runners.
        trimmed_fits = tmp_path / _SAMPLE_FITS.name
        with pf.open(_SAMPLE_FITS) as hdul:
            hdu0 = hdul[0].copy()
            hdu1 = pf.BinTableHDU(data=hdul[1].data[:180], header=hdul[1].header.copy())
            pf.HDUList([hdu0, hdu1]).writeto(trimmed_fits, overwrite=True)

        pdc_dir = tmp_path / "pdc_out"
        sap_dir = tmp_path / "sap_out"
        pdc_dir.mkdir()
        sap_dir.mkdir()

        # Write a @filelist pointing to the real MAST FITS input
        flist = tmp_path / "inputs.txt"
        flist.write_text(f"# test target\n{trimmed_fits.name}\n", encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught_cli:
            warnings.simplefilter("always")
            rc_pdc = cli.main(
                [
                    f"@{flist}",
                    "--save-dir",
                    str(pdc_dir),
                    "--flux-type",
                    "pdc",
                    "-c",
                    "5",
                    "--optimizer",
                    "multistart",
                    "--tr-nrandom",
                    "48",
                    "--tr-nblocks",
                    "2",
                    "--tr-bspan",
                    "16",
                    "--n-candidates",
                    "20",
                    "--maxiter",
                    "30",
                    "--seed",
                    "7",
                    "--quiet",
                ]
            )
        assert rc_pdc == 0
        verify_warns_cli = [
            w for w in caught_cli if issubclass(w.category, VerifyWarning)
        ]
        assert verify_warns_cli == [], (
            f"Unexpected FITS VerifyWarning: {verify_warns_cli}"
        )

        out_pdc = pdc_dir / "EPIC_211436265_mast.fits"
        assert out_pdc.exists()

        # Verify output FITS structure via both astropy.io.fits and K2SCFile
        with pf.open(out_pdc) as hdul:
            assert hdul[1].header["EXTNAME"] == "K2SC"
            assert hdul[1].header["EPIC"] == 211436265
            assert np.isfinite(hdul[1].header["CDPPR"])
            assert np.isfinite(hdul[1].header["CDPPT"])
            assert np.isfinite(hdul[1].header["CDPPC"])
            assert len(hdul[1].data) == 180
            for col in ("time", "flux", "trtime", "trposi", "mflags", "x", "y"):
                assert col in hdul[1].data.columns.names

        kfile = K2SCFile(out_pdc)
        assert kfile.time.size > 150
        assert np.all(np.isfinite(kfile.detrended_flux))
        assert np.all(np.isfinite(kfile.residual_flux))

        # Test k2plot (main_plot) PDF generation
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()
        rc_plot = cli.main_plot([str(out_pdc), "--save-dir", str(plot_dir)])
        assert rc_plot == 0
        out_pdf = plot_dir / "EPIC_211436265.pdf"
        assert out_pdf.exists()
        assert out_pdf.read_bytes().startswith(b"%PDF-")

        # Test k2ginfo (main_ginfo) smoke check (.csv and .fits outputs work without PyTables)
        ginfo_csv = tmp_path / "summary.csv"
        assert cli.main_ginfo([str(ginfo_csv), str(pdc_dir)]) == 0
        csv_text = ginfo_csv.read_text(encoding="utf-8")
        assert "211436265" in csv_text

        ginfo_fits = tmp_path / "summary.fits"
        assert cli.main_ginfo([str(ginfo_fits), str(pdc_dir)]) == 0
        with pf.open(ginfo_fits) as gh:
            assert int(gh[1].data["epic"][0]) == 211436265

        # Test k2mastify (main_mastify) combining PDC and SAP K2SC FITS outputs
        shutil.copy2(out_pdc, sap_dir / out_pdc.name)
        hlsp_dir = tmp_path / "hlsp"
        with warnings.catch_warnings(record=True) as caught_mastify:
            warnings.simplefilter("always")
            assert cli.main_mastify([str(pdc_dir), str(sap_dir), str(hlsp_dir)]) == 0
        verify_warns_mastify = [
            w for w in caught_mastify if issubclass(w.category, VerifyWarning)
        ]
        assert verify_warns_mastify == []
        hlsp_files = list(hlsp_dir.rglob("hlsp_k2sc_*.fits"))
        assert len(hlsp_files) == 1
        with pf.open(hlsp_files[0]) as hh:
            assert hh[1].header["EXTNAME"] == "PDC"
            assert hh[2].header["EXTNAME"] == "SAP"

        # Verify K2SCFile and k2plot read the k2mastify HLSP file without double-subtracting trends
        kfile_hlsp = K2SCFile(hlsp_files[0])
        assert kfile_hlsp.is_hlsp is True
        np.testing.assert_allclose(
            kfile_hlsp.original_flux, kfile.original_flux, rtol=1e-10, atol=1e-8
        )
        hlsp_pdf = plot_dir / "hlsp_plot.pdf"
        assert cli.main_plot([str(hlsp_files[0]), "-o", str(hlsp_pdf)]) == 0
        assert hlsp_pdf.exists()
        assert hlsp_pdf.read_bytes().startswith(b"%PDF-")

    def test_cli_error_codes_and_mpi_master_worker_protocol(
        self, tmp_path, monkeypatch
    ):
        # Non-existent save directory returns ENOENT cleanly (without Logger TypeError)
        rc_bad_dir = cli.main(
            [
                str(_SAMPLE_FITS),
                "--save-dir",
                str(tmp_path / "does_not_exist"),
                "--quiet",
            ]
        )
        assert rc_bad_dir != 0

        # Periodic kernel override without --kernel-period returns 1
        rc_no_per = cli.main(
            [
                str(_SAMPLE_FITS),
                "--save-dir",
                str(tmp_path),
                "--kernel",
                "quasiperiodic",
                "--quiet",
            ]
        )
        assert rc_no_per == 1

        # Verify MPI master does NOT leak a worker when an input file in the list is missing,
        # and sends integer -1 shutdown sentinel cleanly to all workers.
        sent_messages: list[tuple[int, int, object]] = []
        valid_copy = tmp_path / _SAMPLE_FITS.name
        with pf.open(_SAMPLE_FITS) as hdul:
            pf.HDUList(
                [
                    hdul[0].copy(),
                    pf.BinTableHDU(
                        data=hdul[1].data[:40], header=hdul[1].header.copy()
                    ),
                ]
            ).writeto(valid_copy, overwrite=True)

        class _MockMasterComm:
            def Get_rank(self) -> int:
                return 0

            def Get_size(self) -> int:
                return 3  # Master + 2 workers (ranks 1 and 2)

            def send(self, obj, dest: int, tag: int = 0) -> None:
                sent_messages.append((dest, tag, obj))

            def Iprobe(self, source: int, tag: int = 0) -> bool:
                return any(
                    d == source and t == 0 and p != -1 for d, t, p in sent_messages
                )

            def recv(self, source: int, tag: int = 0):
                return 211436265

        missing_path = str(tmp_path / "ktwo999999999-c05_llc.fits")
        out_mpi_dir = tmp_path / "mpi_out"
        out_mpi_dir.mkdir()
        rc_master = cli.main(
            [missing_path, str(valid_copy), "--save-dir", str(out_mpi_dir), "--quiet"],
            comm=_MockMasterComm(),
        )
        assert rc_master == 0
        shutdowns = sorted(
            dest for dest, tag, payload in sent_messages if tag == 0 and payload == -1
        )
        # Both workers 1 and 2 received -1 shutdown sentinel (zero leaked workers!)
        assert shutdowns == [1, 2]

        # Verify MPI worker terminates cleanly on integer -1 sentinel even after receiving a K2Data object
        monkeypatch.setattr(
            cli, "_detrend_and_write_dataset", lambda ds, rdr, args, mpi_rank=0: "ok"
        )
        worker_acks: list[tuple[int, int, int]] = []

        class _DummyDataset:
            epic = 211436265

        class _MockWorkerComm:
            def __init__(self):
                self._queue = [(_DummyDataset(), None), -1]

            def Get_rank(self) -> int:
                return 1

            def Get_size(self) -> int:
                return 2

            def recv(self, source: int = 0, tag: int = 0):
                return self._queue.pop(0)

            def send(self, obj, dest: int, tag: int = 0) -> None:
                worker_acks.append((dest, tag, obj))

        rc_worker = cli.main(
            [str(valid_copy), "--save-dir", str(out_mpi_dir), "--quiet"],
            comm=_MockWorkerComm(),
        )
        assert rc_worker == 0
        assert worker_acks == [(0, 2, 211436265)]
        assert "seaborn" not in sys.modules

    def test_main_ginfo_periodic_kernel_and_iscale_labels(self, tmp_path):
        """k2ginfo must retain 9-param PeriodicKernel rows and use *_iscale column names."""
        specs = [
            (211000001, "BasicKernel", "[-5.0 0.2 -4.5 18.0 19.0 -3.2]"),
            (
                211000002,
                "QuasiPeriodicKernel",
                "[-5.1 0.3 6.5 0.02 -4.8 20.0 21.0 -3.1]",
            ),
            (
                211000003,
                "PeriodicKernel",
                "[-5.2 0.25 8.1 -5.8 0.01 -4.9 17.0 18.5 -3.0]",
            ),
        ]
        fdir = tmp_path / "ginfo_fits"
        fdir.mkdir()
        for epic, kname, hps in specs:
            cols = [pf.Column(name="time", format="D", array=np.array([1.0, 2.0]))]
            hdu = pf.BinTableHDU.from_columns(cols)
            hdu.header["EXTNAME"] = "K2SC"
            hdu.header["EPIC"] = epic
            hdu.header["KER_NAME"] = kname
            hdu.header["KER_HPS1"] = hps
            hdu.header["CDPPR"] = 100.0
            hdu.header["CDPPT"] = 50.0
            hdu.header["CDPPC"] = 40.0
            pf.HDUList([pf.PrimaryHDU(), hdu]).writeto(
                fdir / f"EPIC_{epic:09d}_mast.fits", overwrite=True
            )

        files = sorted(str(p) for p in fdir.glob("EPIC_*.fits"))
        dfb, dfq, dfp, summary = cli._build_ginfo_dataframes(files)
        assert len(dfb) == 1 and int(dfb.index[0]) == 211000001
        assert len(dfq) == 1 and int(dfq.index[0]) == 211000002
        assert len(dfp) == 1 and int(dfp.index[0]) == 211000003
        assert len(summary) == 3

        # Verify *_iscale labels in QuasiPeriodic and Periodic DataFrames
        assert "time_iscale" in dfq.columns
        assert "x_iscale" in dfq.columns
        assert "y_iscale" in dfq.columns
        assert "time_scale" not in dfq.columns
        assert "x_scale" not in dfq.columns
        assert "y_scale" not in dfq.columns
        assert "tl_iscale" in dfp.columns
        assert "x_iscale" in dfp.columns
        assert "y_iscale" in dfp.columns

        out_csv = tmp_path / "summary_all.csv"
        assert cli.main_ginfo([str(out_csv), str(fdir)]) == 0
        csv_text = out_csv.read_text(encoding="utf-8")
        assert "PeriodicKernel" in csv_text
        assert "211000003" in csv_text

    def test_cli_mpi_single_thread_env_and_mixed_reader_batch(
        self, tmp_path, monkeypatch
    ):
        """cli.main calls configure_single_thread_env under MPI and resolves readers per file."""
        # 1. Verify configure_single_thread_env is invoked when mpi_size > 1
        called: list[bool] = []
        monkeypatch.setattr(
            cli, "configure_single_thread_env", lambda: called.append(True)
        )

        # Prepare one trimmed MAST FITS and one AMC .txt light curve
        mast_file = tmp_path / _SAMPLE_FITS.name
        with pf.open(_SAMPLE_FITS) as hdul:
            pf.HDUList(
                [
                    hdul[0].copy(),
                    pf.BinTableHDU(
                        data=hdul[1].data[:20], header=hdul[1].header.copy()
                    ),
                ]
            ).writeto(mast_file, overwrite=True)

        amc_txt = tmp_path / "EPIC_203794605.txt"
        rows = ["dates cadences flux1 err1 xpos ypos quality\n"]
        for i in range(20):
            rows.append(
                f"{2300.0 + i * 0.02:.5f} {1000 + i} {5000.0 + i:.3f} 1.0 "
                f"{0.01 * i:.4f} {-0.01 * i:.4f} 0\n"
            )
        amc_txt.write_text("".join(rows), encoding="utf-8")

        # 2. Mixed-reader non-MPI batch: resolve MASTReader for .fits and AMCReader for .txt
        dispatched: list[tuple[int, str]] = []
        monkeypatch.setattr(
            cli,
            "_detrend_and_write_dataset",
            lambda ds, rdr, args, mpi_rank=0: (
                dispatched.append((int(ds.epic), rdr.__name__)) or "ok"
            ),
        )
        rc_mix = cli.main(
            [str(mast_file), str(amc_txt), "--save-dir", str(tmp_path), "--quiet"]
        )
        assert rc_mix == 0
        assert called == []  # single-process run did not force single-thread env
        assert dispatched == [(211436265, "MASTReader"), (203794605, "AMCReader")]

        # 3. Under MPI (mpi_size > 1), configure_single_thread_env MUST be called
        class _MockWorkerDoneComm:
            def Get_rank(self) -> int:
                return 1

            def Get_size(self) -> int:
                return 4

            def recv(self, source: int = 0, tag: int = 0):
                return -1

        rc_mpi = cli.main(
            [str(mast_file), "--save-dir", str(tmp_path), "--quiet"],
            comm=_MockWorkerDoneComm(),
        )
        assert rc_mpi == 0
        assert called == [True]
