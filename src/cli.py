"""Command-line entry points for ``k2sc``, ``k2plot``, ``k2ginfo``, and ``k2mastify``."""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from datetime import UTC, datetime
from glob import glob
from os.path import abspath, basename, dirname, exists, isdir, join, splitext
from typing import Any

import astropy.io.fits as pf
import numpy as np

from ._jax_env import ensure_float64
from .core import M_OUTLIER_D, M_OUTLIER_U
from .k2io import FITSWriter, SPLOXReader, select_reader
from .k2scfile import K2SCFile, submed
from .kernels import kernels

ensure_float64()

_MPI_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "MPI_LOCALNRANKS",
    "SLURM_NTASKS",
)
_FILELIST_EXTENSIONS = {".txt", ".list", ".lst"}
_RE_EPIC = re.compile(r"EPIC_0*([0-9]+)")


def _detect_mpi_comm(explicit_comm: Any | None = None) -> tuple[Any | None, int, int]:
    """Return ``(comm, rank, size)``, lazily importing ``mpi4py`` only under active MPI."""
    if explicit_comm is not None:
        return explicit_comm, int(explicit_comm.Get_rank()), int(explicit_comm.Get_size())

    mpi_requested = False
    for var in _MPI_ENV_VARS:
        val = os.environ.get(var, "")
        if val.isdigit() and int(val) > 1:
            mpi_requested = True
            break

    if not mpi_requested:
        return None, 0, 1

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())
    except ImportError:
        return None, 0, 1


def _parse_splits_arg(raw: str) -> list[float] | str:
    """Parse ``--splits`` CLI argument such as ``'[2344, 2360]'``, ``'auto'``, or ``'[]'``."""
    cleaned = raw.strip()
    if cleaned.lower() == "auto":
        return "auto"
    if cleaned.lower() in ("none", "[]", ""):
        return []
    inner = cleaned.strip("[]() ")
    if not inner:
        return []
    vals = np.fromstring(inner, sep=",", dtype=np.float64)
    if vals.size == 0 or (inner.count(",") > 0 and vals.size != inner.count(",") + 1):
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return [float(p) for p in parts]
    return [float(v) for v in vals]


def _read_filelist_entries(list_path: str) -> list[str]:
    """Read file paths from a text list file, stripping comments and blank lines."""
    abs_list = abspath(list_path)
    base_dir = dirname(abs_list)
    entries: list[str] = []
    with open(abs_list, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if exists(line):
                entries.append(abspath(line))
            else:
                rel_candidate = join(base_dir, line)
                entries.append(abspath(rel_candidate) if exists(rel_candidate) else line)
    return entries


def expand_input_files(raw_files: Sequence[str]) -> tuple[list[str], bool]:
    """Expand directories, ``@filelist`` arguments, and ``.txt``/``.list`` files.

    Returns ``(files_list, from_directory)``.
    """
    if len(raw_files) == 1 and not raw_files[0].startswith("@") and isdir(raw_files[0]):
        dpath = raw_files[0]
        matched = sorted(glob(join(dpath, "ktwo*.fits")))
        if not matched:
            matched = sorted(glob(join(dpath, "*.fits")))
        return [abspath(f) for f in matched], True

    expanded: list[str] = []
    for item in raw_files:
        if item.startswith("@"):
            list_file = item[1:]
            if not exists(list_file):
                raise FileNotFoundError(f"Input file list {list_file!r} does not exist.")
            expanded.extend(_read_filelist_entries(list_file))
        elif exists(item) and not isdir(item) and splitext(item)[1].lower() in _FILELIST_EXTENSIONS:
            expanded.extend(_read_filelist_entries(item))
        elif isdir(item):
            matched = sorted(glob(join(item, "ktwo*.fits")))
            if not matched:
                matched = sorted(glob(join(item, "*.fits")))
            expanded.extend(abspath(f) for f in matched)
        else:
            expanded.append(abspath(item) if exists(item) else item)
    return expanded, False


def build_k2sc_parser() -> ArgumentParser:
    """Construct the ``k2sc`` CLI ``ArgumentParser``."""
    ap = ArgumentParser(
        prog="k2sc",
        description="K2SC: Kepler/K2 and TESS systematics correction using JAX Gaussian Processes",
    )
    gts = ap.add_argument_group("Training set options")
    gps = ap.add_argument_group(
        "Period search",
        description="Options to control the initial Lomb-Scargle period search",
    )
    god = ap.add_argument_group("Outlier detection")
    gde = ap.add_argument_group(
        "Global optimization",
        description="Options to control global and gradient hyperparameter optimization",
    )

    ap.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Input FITS/AMC file(s), directory, or @filelist.txt.",
    )
    ap.add_argument(
        "-c", "--campaign", metavar="C", type=int, default=None, help="Campaign number"
    )
    ap.add_argument(
        "--splits",
        default=None,
        type=_parse_splits_arg,
        help="Comma-separated list of time values for spatial kernel splits, or 'auto'",
    )
    ap.add_argument(
        "--quiet", action="store_true", default=False, help="Suppress info log messages"
    )
    ap.add_argument(
        "--save-dir", default=".", help="Directory in which to save output FITS files"
    )
    ap.add_argument("--start-i", default=0, type=int)
    ap.add_argument("--end-i", default=None, type=int)
    ap.add_argument("--seed", default=0, type=int, help="Deterministic RNG seed")
    ap.add_argument("--logfile", default="", type=str)
    ap.add_argument("--flux-type", default="pdc", type=str)
    ap.add_argument(
        "--default-position-kernel",
        choices=["SqrExp", "Exp"],
        default="SqrExp",
    )
    ap.add_argument(
        "--no-roll-pca",
        dest="roll_pca",
        action="store_false",
        default=True,
        help="Disable PCA spacecraft-roll rotation and robust 4-MAD axis normalization",
    )
    ap.add_argument(
        "--kernel",
        choices=list(kernels.keys()),
        default=None,
        help="Kernel to use (overrides automatic kernel selection)",
    )
    ap.add_argument(
        "--kernel-period",
        type=float,
        default=None,
        help="Period for (quasi)periodic kernels when overriding period detection",
    )
    ap.add_argument("--p-mask-center", type=float, default=None, help="Transit mask zero epoch")
    ap.add_argument("--p-mask-period", type=float, default=None, help="Transit mask period")
    ap.add_argument(
        "--p-mask-duration", type=float, default=None, help="Transit mask duration"
    )

    gts.add_argument("--tr-nrandom", default=400, type=int, help="Number of random samples")
    gts.add_argument("--tr-nblocks", default=6, type=int, help="Number of sample blocks")
    gts.add_argument("--tr-bspan", default=50, type=int, help="Span of a single block")

    gde.add_argument(
        "--optimizer",
        choices=["auto", "multistart", "de", "optimistix"],
        default="auto",
        help="Global + gradient hyperparameter optimizer backend (default: auto)",
    )
    gde.add_argument(
        "--de-npop",
        default=100,
        type=int,
        help="Population size for differential evolution (--optimizer de)",
    )
    gde.add_argument(
        "--de-niter",
        default=150,
        type=int,
        help="Maximum generations for differential evolution (--optimizer de)",
    )
    gde.add_argument(
        "--de-max-time",
        default=300.0,
        type=float,
        help="Maximum wall-time in seconds for differential evolution",
    )
    gde.add_argument(
        "--n-candidates",
        default=None,
        type=int,
        help="Optional candidate screening budget for multistart/optimistix",
    )
    gde.add_argument(
        "--maxiter",
        default=None,
        type=int,
        help="Optional local L-BFGS-B iteration limit",
    )

    gps.add_argument(
        "--ls-max-fap",
        default=-50.0,
        type=float,
        help="Maximum log10(FAP) threshold to select a quasiperiodic kernel",
    )
    gps.add_argument(
        "--ls-min-period", default=0.05, type=float, help="Minimum period to search (days)"
    )
    gps.add_argument(
        "--ls-max-period", default=25.0, type=float, help="Maximum period to search (days)"
    )

    god.add_argument("--outlier-sigma", default=5.0, type=float)
    god.add_argument("--outlier-mwidth", default=25, type=int)
    return ap


def _format_output_filename(template: str, epic: int) -> str:
    """Format reader output filename without leading spaces in EPIC identifiers."""
    return template.format(int(epic)).replace(" ", "0")


def _detrend_and_write_dataset(
    dataset,
    reader_cls,
    args: Namespace,
    mpi_rank: int = 0,
) -> str:
    """Detrend a single ``K2Data`` dataset and write its K2SC FITS output."""
    worker_logger = logging.getLogger(f"k2sc.worker.{mpi_rank}.{int(dataset.epic)}")
    worker_logger.setLevel(logging.WARNING if args.quiet else logging.INFO)

    fh: logging.Handler | None = None
    log_stream = None
    if args.logfile:
        worker_log_path = f"{args.logfile}.{mpi_rank:03d}"
        log_stream = open(worker_log_path, mode="w", encoding="utf-8")  # noqa: SIM115
        fh = logging.StreamHandler(log_stream)
        fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        fh.setLevel(logging.DEBUG)
        worker_logger.addHandler(fh)

    opt_kwargs: dict[str, Any] = {}
    if args.n_candidates is not None:
        opt_kwargs["n_candidates"] = int(args.n_candidates)
    if args.maxiter is not None:
        opt_kwargs["maxiter"] = int(args.maxiter)

    from .pipeline import detrend_k2data

    try:
        results = detrend_k2data(
            dataset,
            campaign=args.campaign,
            splits=args.splits,
            quiet=args.quiet,
            seed=args.seed,
            default_position_kernel=args.default_position_kernel,
            roll_pca=args.roll_pca,
            kernel=args.kernel,
            kernel_period=args.kernel_period,
            p_mask_center=args.p_mask_center,
            p_mask_period=args.p_mask_period,
            p_mask_duration=args.p_mask_duration,
            tr_nrandom=args.tr_nrandom,
            tr_nblocks=args.tr_nblocks,
            tr_bspan=args.tr_bspan,
            optimizer=args.optimizer,
            de_npop=args.de_npop,
            de_niter=args.de_niter,
            de_max_time=args.de_max_time,
            ls_max_fap=args.ls_max_fap,
            ls_min_period=args.ls_min_period,
            ls_max_period=args.ls_max_period,
            outlier_sigma=args.outlier_sigma,
            outlier_mwidth=args.outlier_mwidth,
            logger=worker_logger,
            **opt_kwargs,
        )
        splits_used = results[0].splits if results else []
        out_name = _format_output_filename(reader_cls.fn_out_template, int(dataset.epic))
        fpath_out = join(args.save_dir, out_name)
        FITSWriter.write(fpath_out, splits_used, dataset, results)
        if not args.quiet:
            worker_logger.info("Finished writing %s", fpath_out)
        return fpath_out
    finally:
        if fh is not None:
            fh.flush()
            worker_logger.removeHandler(fh)
            fh.close()
        if log_stream is not None:
            log_stream.close()


def main(argv: Sequence[str] | None = None, *, comm: Any | None = None) -> int:
    """Entry point for the ``k2sc`` batch detrending CLI."""
    ap = build_k2sc_parser()
    args = ap.parse_args(argv)

    mpi_comm, mpi_rank, mpi_size = _detect_mpi_comm(comm)
    with_mpi = mpi_comm is not None and mpi_size > 1
    mpi_root = 0

    master_logger = logging.getLogger("k2sc.master")
    master_logger.setLevel(logging.WARNING if args.quiet else logging.INFO)
    stream_handler: logging.Handler | None = None
    if not master_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        master_logger.addHandler(stream_handler)

    master_fh: logging.Handler | None = None
    master_log_file = None
    if mpi_rank == 0 and args.logfile:
        master_log_file = open(args.logfile, mode="w", encoding="utf-8")  # noqa: SIM115
        master_fh = logging.StreamHandler(master_log_file)
        master_fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        master_fh.setLevel(logging.DEBUG)
        master_logger.addHandler(master_fh)

    try:
        if not args.files:
            master_logger.error("No input files provided.")
            return 2

        if not exists(args.save_dir):
            master_logger.error("Error: the save directory %s does not exist.", args.save_dir)
            return errno.ENOENT

        if args.kernel and "periodic" in args.kernel and args.kernel_period is None:
            master_logger.critical(
                "Need --kernel-period when overriding automatic kernel selection with %s.",
                args.kernel,
            )
            return 1

        try:
            expanded_files, _from_dir = expand_input_files(args.files)
        except FileNotFoundError as exc:
            master_logger.error("%s", exc)
            return errno.ENOENT

        if not expanded_files:
            master_logger.error("No matching input files found.")
            return 1

        probe_file = next((f for f in expanded_files if exists(f)), expanded_files[0])
        reader = select_reader(probe_file)
        if reader is None:
            master_logger.critical("Unrecognized input file type for file %s", probe_file)
            return 1

        if len(expanded_files) == 1 and issubclass(reader, SPLOXReader):
            singlefile = True
            infile = expanded_files[0]
            end_idx = (
                int(args.end_i)
                if args.end_i is not None
                else int(reader.nstars(infile))
            )
            stop_idx = min(end_idx, int(reader.nstars(infile)))
            items: list[Any] = list(range(int(args.start_i), stop_idx))
        else:
            singlefile = False
            infile = ""
            items = list(expanded_files[args.start_i : args.end_i])

        n_items = len(items)
        if mpi_rank == 0 and not args.quiet:
            if not with_mpi:
                master_logger.info("Detrending %d light curves without MPI", n_items)
            else:
                master_logger.info(
                    "Detrending %d light curves using %d MPI worker ranks",
                    n_items,
                    mpi_size - 1,
                )
            master_logger.info("Saving results to %s (optimizer=%s)", args.save_dir, args.optimizer)

        if not with_mpi:
            for item in items:
                if singlefile:
                    sid = int(item)
                    cur_file = infile
                else:
                    sid = 0
                    cur_file = str(item)
                    if not exists(cur_file):
                        master_logger.warning(
                            "Input file %s does not exist; skipping.", cur_file
                        )
                        continue
                dataset = reader.read(
                    cur_file,
                    sid=sid,
                    type=args.flux_type,
                    campaign=args.campaign,
                )
                _detrend_and_write_dataset(dataset, reader, args, mpi_rank=0)
            return 0

        # MPI master / worker dispatch
        assert mpi_comm is not None
        if mpi_rank == 0:
            pending = list(reversed(items))
            free_workers = list(range(1, mpi_size))
            active_workers: list[int] = []
            n_finished = 0

            while pending or active_workers:
                while pending and free_workers:
                    item = pending.pop()
                    if singlefile:
                        sid = int(item)
                        cur_file = infile
                        if not args.quiet:
                            master_logger.info("Processing star %d", sid)
                    else:
                        sid = 0
                        cur_file = str(item)
                        if not exists(cur_file):
                            master_logger.warning(
                                "Input file %s does not exist; skipping.", cur_file
                            )
                            continue
                        if not args.quiet:
                            master_logger.info("Processing file %s", abspath(cur_file))

                    # Only claim a free worker AFTER verifying the file exists (preventing worker leak)
                    w = free_workers.pop()
                    dataset = reader.read(
                        cur_file,
                        sid=sid,
                        type=args.flux_type,
                        campaign=args.campaign,
                    )
                    mpi_comm.send((dataset, reader), dest=w, tag=0)
                    active_workers.append(w)

                for w in list(active_workers):
                    if mpi_comm.Iprobe(w, 2):
                        _done_epic = mpi_comm.recv(source=w, tag=2)
                        active_workers.remove(w)
                        free_workers.append(w)
                        n_finished += 1

                        if args.logfile and master_log_file is not None:
                            w_log = f"{args.logfile}.{w:03d}"
                            if exists(w_log):
                                with open(w_log, encoding="utf-8") as fh_w:
                                    master_log_file.write(fh_w.read())
                        if not args.quiet:
                            master_logger.info(
                                "Finished %3d of %3d light curves", n_finished, n_items
                            )

            for w in free_workers:
                mpi_comm.send(-1, dest=w, tag=0)
        else:
            while True:
                payload = mpi_comm.recv(source=mpi_root, tag=0)
                if isinstance(payload, int) and payload == -1:
                    break
                dataset, worker_reader = payload
                _detrend_and_write_dataset(dataset, worker_reader, args, mpi_rank=mpi_rank)
                mpi_comm.send(int(dataset.epic), dest=mpi_root, tag=2)

        return 0
    finally:
        if master_fh is not None:
            master_fh.flush()
            master_logger.removeHandler(master_fh)
            master_fh.close()
        if master_log_file is not None:
            master_log_file.close()
        if stream_handler is not None:
            master_logger.removeHandler(stream_handler)


def _extract_epic_from_fits_or_name(fname: str, header) -> int:
    """Resolve integer EPIC / target identifier from FITS header or filename."""
    for key in ("EPIC", "OBJECT", "KEPLERID"):
        if key in header:
            with contextlib.suppress(TypeError, ValueError):
                return int(header[key])
    m = _RE_EPIC.search(basename(fname))
    if m:
        return int(m.group(1))
    digits = re.findall(r"([0-9]{5,})", basename(fname))
    return int(digits[0]) if digits else 0


def main_plot(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``k2plot``: render diagnostic PDF plots for a K2SC FITS file."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    ap = ArgumentParser(
        prog="k2plot",
        description="Generate diagnostic PDF report from a K2SC FITS output file.",
    )
    ap.add_argument("file_name", type=str, help="Input K2SC FITS file")
    ap.add_argument("--save-dir", default=".", type=str, help="Directory for output PDF")
    ap.add_argument("--output", "-o", default=None, type=str, help="Explicit output PDF path")
    ap.add_argument("--show", default=False, action="store_true", help="Display interactively")
    args = ap.parse_args(argv)

    fname = args.file_name
    if not exists(fname):
        print(f"Error: file {fname!r} does not exist.", file=sys.stderr)
        return errno.ENOENT

    hdr = pf.getheader(fname, 1)
    data = pf.getdata(fname, 1)
    epic = _extract_epic_from_fits_or_name(fname, hdr)
    col_names = [c.lower() for c in data.columns.names]

    # Detect single-aperture ('flux') vs multi-aperture ('flux_1', 'flux_2', ...)
    if "flux" in col_names:
        ap_suffixes = [""]
    else:
        ap_indices = sorted(
            int(c.split("_")[1])
            for c in col_names
            if c.startswith("flux_") and c.split("_")[1].isdigit()
        )
        ap_suffixes = [f"_{idx}" for idx in ap_indices]

    if not ap_suffixes:
        print(f"Error: no K2SC flux columns found in {fname!r}.", file=sys.stderr)
        return 1

    pdf_path = (
        args.output
        if args.output is not None
        else join(args.save_dir, f"EPIC_{epic:09d}.pdf")
    )
    c_header = "#002147"

    time_arr = np.asarray(data["time"], dtype=np.float64)
    x_arr = np.asarray(data["x"], dtype=np.float64) if "x" in col_names else np.zeros_like(time_arr)
    y_arr = np.asarray(data["y"], dtype=np.float64) if "y" in col_names else np.zeros_like(time_arr)

    # Also instantiate K2SCFile when single-aperture to validate K2SCFile readability
    k2sc_file_obj = K2SCFile(fname) if ap_suffixes == [""] else None

    with PdfPages(pdf_path) as pdf:
        for ap_idx, sfx in enumerate(ap_suffixes, start=1):
            hdr_sfx = "" if sfx == "" else str(ap_idx)
            if k2sc_file_obj is not None and sfx == "":
                flux_raw = np.asarray(data["flux"], dtype=np.float64)
                tr_time = np.asarray(data["trtime"], dtype=np.float64)
                tr_pos = np.asarray(data["trposi"], dtype=np.float64)
                mflags = np.asarray(data["mflags"], dtype=np.uint8)
            else:
                flux_raw = np.asarray(data[f"flux{sfx}"], dtype=np.float64)
                tr_time = np.asarray(data[f"trtime{sfx}"], dtype=np.float64)
                tr_pos = np.asarray(data[f"trposi{sfx}"], dtype=np.float64)
                mflags = np.asarray(data[f"mflags{sfx}"], dtype=np.uint8)

            flux_corr = flux_raw - submed(tr_pos)
            flux_res = flux_corr - submed(tr_time)

            fin = np.isfinite(time_arr) & np.isfinite(flux_raw)
            is_outlier = (mflags & (M_OUTLIER_U | M_OUTLIER_D)).astype(bool)
            inlier = fin & (~is_outlier)

            cdpp_r = hdr.get(f"cdpp{hdr_sfx}r", float("nan"))
            cdpp_t = hdr.get(f"cdpp{hdr_sfx}t", float("nan"))
            cdpp_c = hdr.get(f"cdpp{hdr_sfx}c", float("nan"))
            ap_warn = hdr.get(f"ap{hdr_sfx}_warn", False)
            ker_name = hdr.get("ker_name", "Unknown")

            fig = plt.figure(figsize=(8.3, 11.7))
            banner = fig.add_axes([0, 0, 1, 1])
            banner.xaxis.set_visible(False)
            banner.yaxis.set_visible(False)
            banner.set_zorder(-1000)
            banner.add_patch(Rectangle((0, 0.93), 1, 0.07, fill=True, color=c_header))
            ap_label = "" if sfx == "" else f" (Ap {ap_idx})"
            fig.text(
                0.04,
                0.975,
                f"EPIC {epic:09d}{ap_label}",
                va="top",
                size=18,
                color="w",
                weight="bold",
            )
            fig.text(
                0.04,
                0.905,
                f"Kernel: {ker_name}   |   Fit OK: {not bool(ap_warn)}   |   "
                f"CDPP raw / pos / full: {cdpp_r:.1f} / {cdpp_t:.1f} / {cdpp_c:.1f} ppm",
                va="top",
                size=9.5,
                color="#111111",
            )

            gs = fig.add_gridspec(
                6,
                1,
                bottom=0.06,
                top=0.87,
                left=0.12,
                right=0.96,
                hspace=0.10,
            )
            axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]

            if np.any(fin):
                med_f = float(np.nanmedian(flux_raw[fin]))
                scale = med_f if (np.isfinite(med_f) and med_f != 0.0) else 1.0
                axes[0].plot(time_arr[inlier], flux_raw[inlier] / scale, ".", ms=2, color="0.25")
                if np.any(fin & is_outlier):
                    axes[0].plot(
                        time_arr[fin & is_outlier],
                        flux_raw[fin & is_outlier] / scale,
                        "rx",
                        ms=4,
                    )
                axes[1].plot(
                    time_arr[inlier],
                    flux_corr[inlier] / scale,
                    ".",
                    ms=2,
                    color="C0",
                )
                fin_tt = fin & np.isfinite(tr_time)
                if np.any(fin_tt):
                    axes[1].plot(
                        time_arr[fin_tt],
                        tr_time[fin_tt] / scale,
                        "-",
                        lw=1.0,
                        color="C1",
                    )
                fin_res = inlier & np.isfinite(flux_res)
                if np.any(fin_res):
                    axes[2].plot(
                        time_arr[fin_res],
                        flux_res[fin_res] / scale,
                        ".",
                        ms=2,
                        color="0.2",
                    )
                fin_tp = fin & np.isfinite(tr_pos)
                if np.any(fin_tp):
                    axes[3].plot(
                        time_arr[fin_tp],
                        tr_pos[fin_tp] / scale,
                        "-",
                        lw=1.0,
                        color="C2",
                    )
                axes[4].plot(time_arr[fin], x_arr[fin], "-", lw=0.8, color="0.3")
                axes[5].plot(time_arr[fin], y_arr[fin], "-", lw=0.8, color="0.3")
                t_bounds = (float(np.min(time_arr[fin])), float(np.max(time_arr[fin])))
                if t_bounds[1] > t_bounds[0]:
                    for ax_i in axes:
                        ax_i.set_xlim(t_bounds)

            labels = [
                "Raw flux",
                "Raw - XY",
                "Residual",
                "XY trend",
                "X [pix]",
                "Y [pix]",
            ]
            for idx_a, (ax_i, yl) in enumerate(zip(axes, labels, strict=True)):
                ax_i.set_ylabel(yl, fontsize=8.5)
                if idx_a < len(axes) - 1:
                    plt.setp(ax_i.get_xticklabels(), visible=False)
            axes[-1].set_xlabel("Time [d]")

            pdf.savefig(fig)
            if not args.show:
                plt.close(fig)

    if args.show:
        plt.show()
    return 0


def _read_cdpp_triplet(hdr) -> list[float]:
    """Extract ``[cdpp_r, cdpp_t, cdpp_c]`` supporting both single- and multi-aperture headers."""
    if "cdppr" in hdr:
        return [float(hdr["cdppr"]), float(hdr["cdppt"]), float(hdr["cdppc"])]
    return [float(hdr["cdpp1r"]), float(hdr["cdpp1t"]), float(hdr["cdpp1c"])]


def main_ginfo(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``k2ginfo``: gather hyperparameters and CDPP statistics across FITS files."""
    ap = ArgumentParser(
        prog="k2ginfo",
        description="Gather kernel hyperparameters and CDPP metrics from K2SC FITS files.",
    )
    ap.add_argument(
        "hdfname",
        help="Output summary file (.h5/.hdf requiring PyTables, or .csv / .fits)",
    )
    ap.add_argument("datadir", help="Directory containing K2SC EPIC_*.fits output files")
    ap.add_argument("--random-sample", default=None, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args(argv)

    files = sorted(glob(join(args.datadir, "EPIC_*.fits")))
    if not files:
        files = sorted(glob(join(args.datadir, "*.fits")))
    if not files:
        print(f"No FITS files found in {args.datadir!r}.", file=sys.stderr)
        return 1

    if args.random_sample is not None and args.random_sample < len(files):
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(files), size=int(args.random_sample), replace=False))
        files = [files[int(i)] for i in chosen]

    c_basic = [
        "time_amplitude",
        "time_iscale",
        "xy_amplitude",
        "x_iscale",
        "y_iscale",
        "white_noise",
        "cdppr",
        "cdppt",
        "cdppc",
    ]
    c_qperi = [
        "time_amplitude",
        "time_scale",
        "time_period",
        "time_evolution",
        "xy_amplitude",
        "x_scale",
        "y_scale",
        "white_noise",
        "cdppr",
        "cdppt",
        "cdppc",
    ]

    hps_basic: list[np.ndarray] = []
    hps_qperi: list[np.ndarray] = []
    summary_rows: list[dict[str, Any]] = []

    for fpath in files:
        hdr = pf.getheader(fpath, 1)
        ker_name = str(hdr.get("ker_name", "BasicKernel"))
        epic = _extract_epic_from_fits_or_name(fpath, hdr)
        raw_hps = str(hdr.get("ker_hps1", "[]")).strip("[] ")
        v = np.fromstring(raw_hps, sep=" ", dtype=np.float64)
        s = _read_cdpp_triplet(hdr)

        row_vec = np.r_[[float(epic)], v, s]
        if "basic" in ker_name.lower() and v.size == 6:
            hps_basic.append(row_vec)
        elif v.size == 8:
            hps_qperi.append(row_vec)

        summary_rows.append(
            {
                "epic": int(epic),
                "ker_name": ker_name,
                "cdppr": s[0],
                "cdppt": s[1],
                "cdppc": s[2],
                "ker_hps1": " ".join(f"{val:.8g}" for val in v),
            }
        )

    out_ext = splitext(args.hdfname)[1].lower()
    if out_ext in (".fits", ".fit"):
        from astropy.table import Table

        tbl = Table(rows=summary_rows)
        tbl.write(args.hdfname, format="fits", overwrite=True)
        return 0

    import pandas as pd

    if hps_basic:
        arr_b = np.asarray(hps_basic, dtype=np.float64)
        dfb = pd.DataFrame(arr_b[:, 1:], columns=c_basic, index=arr_b[:, 0].astype(np.uint32))
    else:
        dfb = pd.DataFrame(columns=c_basic, dtype=np.float64)

    if hps_qperi:
        arr_q = np.asarray(hps_qperi, dtype=np.float64)
        dfp = pd.DataFrame(arr_q[:, 1:], columns=c_qperi, index=arr_q[:, 0].astype(np.uint32))
    else:
        dfp = pd.DataFrame(columns=c_qperi, dtype=np.float64)

    if out_ext == ".csv":
        pd.DataFrame(summary_rows).to_csv(args.hdfname, index=False)
        return 0

    dfb.to_hdf(args.hdfname, key="basic", mode="w")
    dfp.to_hdf(args.hdfname, key="quasiperiodic", mode="a")
    return 0


def _index_epic_files(pattern: str) -> dict[int, str]:
    """Map integer EPIC identifiers to sorted matching file paths."""
    mapping: dict[int, str] = {}
    for fpath in sorted(glob(pattern)):
        m = _RE_EPIC.search(basename(fpath))
        if m:
            mapping[int(m.group(1))] = fpath
    return mapping


def main_mastify(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``k2mastify``: merge PDC and SAP K2SC FITS files into MAST HLSP layout."""
    ap = ArgumentParser(
        prog="k2mastify",
        description="Combine PDC and SAP K2SC FITS outputs into MAST HLSP FITS files.",
    )
    ap.add_argument("pdc_path", type=str)
    ap.add_argument("sap_path", type=str)
    ap.add_argument("out_path", type=str)
    ap.add_argument("--release-version", type=int, default=1)
    ap.add_argument("--logfile", type=str, default=None)
    args = ap.parse_args(argv)

    pdc_map = _index_epic_files(join(args.pdc_path, "EPIC*fits"))
    sap_map = _index_epic_files(join(args.sap_path, "EPIC*fits"))
    common_epics = sorted(set(pdc_map) & set(sap_map))

    logger = logging.getLogger("k2sc.mastify")
    logger.setLevel(logging.INFO)
    fh: logging.Handler | None = None
    log_stream = None
    if args.logfile:
        log_stream = open(args.logfile, mode="w", encoding="utf-8")  # noqa: SIM115
        fh = logging.StreamHandler(log_stream)
        fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    fntemplate = "hlsp_k2sc_k2_llc_{epic:09d}-c{c:02d}_kepler_v{v:1d}_lc.fits"
    now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        for idx, epic in enumerate(common_epics, start=1):
            with pf.open(pdc_map[epic]) as fpdc, pf.open(sap_map[epic]) as fsap:
                epri = fpdc[0].copy()
                epdc = fpdc[1].copy()
                esap = fsap[1].copy()

                epri.header["origin"] = epdc.header.get("origin", "SPLOX")
                epri.header["creator"] = epdc.header.get("program", "k2SC v2.0")
                epri.header["date"] = now_str

                for k in ("procver", "filever", "timversn", "checksum", "object"):
                    if k in epri.header:
                        epri.header.remove(k)

                for ext_hdu in (epdc, esap):
                    if "object" in ext_hdu.header:
                        ext_hdu.header.remove("object")
                    flux_arr = np.asarray(ext_hdu.data["flux"], dtype=np.float64)
                    trposi = np.asarray(ext_hdu.data["trposi"], dtype=np.float64)
                    trtime = np.asarray(ext_hdu.data["trtime"], dtype=np.float64)
                    ext_hdu.data["flux"][:] = (
                        flux_arr
                        - trposi
                        + np.nanmean(trposi)
                        - trtime
                        + np.nanmean(trtime)
                    )

                epdc.header["EXTNAME"] = "PDC"
                esap.header["EXTNAME"] = "SAP"
                hdu_list = pf.HDUList([epri, epdc, esap])

                campaign = int(epri.header.get("campaign", 0))
                epic_str = f"{epic:09d}"
                cdirname = f"c{campaign:02d}"
                fdirname = f"{epic_str[:4]}00000"
                out_dir = join(args.out_path, cdirname, fdirname)
                os.makedirs(out_dir, exist_ok=True)
                out_fname = fntemplate.format(
                    epic=epic, c=campaign, v=int(args.release_version)
                )
                hdu_list.writeto(join(out_dir, out_fname), overwrite=True)

            logger.info(
                "Finished %05i/%05i -- campaign %i EPIC %09i",
                idx,
                len(common_epics),
                campaign,
                epic,
            )
        return 0
    finally:
        if fh is not None:
            fh.flush()
            logger.removeHandler(fh)
            fh.close()
        if log_stream is not None:
            log_stream.close()


__all__ = [
    "build_k2sc_parser",
    "expand_input_files",
    "main",
    "main_ginfo",
    "main_mastify",
    "main_plot",
]
