"""End-to-end unit tests for ``k2sc.cli`` (``k2sc``, ``k2plot``, ``k2ginfo``, ``k2mastify``)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import astropy.io.fits as pf
import k2sc
import numpy as np
import pytest
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
        # Verify zero unconditional mpi4py, seaborn, or IPython imports in cli.py
        with open(_REPO_ROOT / "src" / "cli.py", encoding="utf-8") as fh:
            src_text = fh.read()
        assert "seaborn" not in src_text
        assert "IPython" not in src_text

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
        assert cli.main_mastify([str(pdc_dir), str(sap_dir), str(hlsp_dir)]) == 0
        hlsp_files = list(hlsp_dir.rglob("hlsp_k2sc_*.fits"))
        assert len(hlsp_files) == 1
        with pf.open(hlsp_files[0]) as hh:
            assert hh[1].header["EXTNAME"] == "PDC"
            assert hh[2].header["EXTNAME"] == "SAP"

    def test_cli_error_codes_and_mpi_master_worker_protocol(self, tmp_path, monkeypatch):
        # Non-existent save directory returns ENOENT cleanly (without Logger TypeError)
        rc_bad_dir = cli.main(
            [str(_SAMPLE_FITS), "--save-dir", str(tmp_path / "does_not_exist"), "--quiet"]
        )
        assert rc_bad_dir != 0

        # Periodic kernel override without --kernel-period returns 1
        rc_no_per = cli.main(
            [str(_SAMPLE_FITS), "--save-dir", str(tmp_path), "--kernel", "quasiperiodic", "--quiet"]
        )
        assert rc_no_per == 1

        # Verify MPI master does NOT leak a worker when an input file in the list is missing,
        # and sends integer -1 shutdown sentinel cleanly to all workers.
        sent_messages: list[tuple[int, int, object]] = []
        valid_copy = tmp_path / _SAMPLE_FITS.name
        with pf.open(_SAMPLE_FITS) as hdul:
            pf.HDUList([hdul[0].copy(), pf.BinTableHDU(data=hdul[1].data[:40], header=hdul[1].header.copy())]).writeto(
                valid_copy, overwrite=True
            )

        class _MockMasterComm:
            def Get_rank(self) -> int:
                return 0

            def Get_size(self) -> int:
                return 3  # Master + 2 workers (ranks 1 and 2)

            def send(self, obj, dest: int, tag: int = 0) -> None:
                sent_messages.append((dest, tag, obj))

            def Iprobe(self, source: int, tag: int = 0) -> bool:
                return any(d == source and t == 0 and p != -1 for d, t, p in sent_messages)

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
        shutdowns = sorted(dest for dest, tag, payload in sent_messages if tag == 0 and payload == -1)
        # Both workers 1 and 2 received -1 shutdown sentinel (zero leaked workers!)
        assert shutdowns == [1, 2]

        # Verify MPI worker terminates cleanly on integer -1 sentinel even after receiving a K2Data object
        monkeypatch.setattr(cli, "_detrend_and_write_dataset", lambda ds, rdr, args, mpi_rank=0: "ok")
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
