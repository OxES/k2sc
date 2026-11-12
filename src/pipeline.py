"""Unified K2SC detrending pipeline for arrays, K2Data, Lightkurve, and CLI."""

from __future__ import annotations

import copy
import logging
import math
import time as clock
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from ._jax_env import ensure_float64
from .cdpp import cdpp
from .core import (
    DEFAULT_K2_SPLITS,
    M_NOTFINITE,
    M_OUTLIER_D,
    M_OUTLIER_U,
    M_PERIODIC,
    M_QUALITY,
)
from .detrender import Detrender
from .k2data import K2Data
from .kernels import (
    BasicKernel,
    BasicKernelEP,
    DtKernel,
    QuasiPeriodicKernel,
    QuasiPeriodicKernelEP,
    kernels,
)
from .ls import fasper
from .optimizers import OptimizationResult
from .pca import RollPCATransform, pca_normalize_xy
from .utils import fold, sigma_clip

ensure_float64()

_MIN_FINITE_POINTS = 15
_K2_THRUSTER_FLAG = 1 << 20

_KERNEL_NAME_ALIASES: dict[str, str] = {
    "basic": "basic",
    "basickernel": "basic",
    "basic_ep": "basic_ep",
    "basickernelep": "basic_ep",
    "periodic": "periodic",
    "periodickernel": "periodic",
    "quasiperiodic": "quasiperiodic",
    "quasiperiodickernel": "quasiperiodic",
    "quasiperiodic_ep": "quasiperiodic_ep",
    "quasiperiodickernelep": "quasiperiodic_ep",
}


@dataclass
class K2SCResult:
    """Unified result container for K2SC light curve detrending.

    Supports both modern attribute access and legacy 8-tuple ``SCResult``
    indexing/unpacking ``(detrender, pv, tr_time, tr_position, cdpp_r, cdpp_t, cdpp_c, warn)``
    where ``warn = not success``.
    """

    detrender: Detrender
    pv: np.ndarray
    tr_time: np.ndarray
    tr_position: np.ndarray
    cdpp_r: float
    cdpp_t: float
    cdpp_c: float
    warn: bool
    success: bool
    corr_flux: np.ndarray
    detrended_flux: np.ndarray
    mflags: np.ndarray
    outlier_mask: np.ndarray
    splits: list[float]
    is_periodic: bool
    ls_period: float | None
    ls_fap: float | None
    opt_result: OptimizationResult | None
    lightcurve: Any | None = None
    pca_transform: RollPCATransform | None = None

    def _legacy_tuple(self) -> tuple[Any, ...]:
        return (
            self.detrender,
            self.pv,
            self.tr_time,
            self.tr_position,
            self.cdpp_r,
            self.cdpp_t,
            self.cdpp_c,
            self.warn,
        )

    def __iter__(self):
        return iter(self._legacy_tuple())

    def __getitem__(self, item):
        return self._legacy_tuple()[item]

    def __len__(self) -> int:
        return 8


SCResult = K2SCResult


def resolve_splits(
    time,
    splits: Sequence[float] | np.ndarray | str | None = "auto",
    campaign: int | None = None,
    mission: str | None = None,
) -> list[float]:
    """Resolve piecewise spatial GP time-split boundaries inside ``(t_min, t_max)``.

    Parameters
    ----------
    time : array_like
        1D timestamp array (days). Non-finite values are ignored.
    splits : sequence of float, ``"auto"``, or None, optional
        Explicit split timestamps, or ``"auto"`` / ``None`` for mission-aware
        automatic resolution.
    campaign : int or None, optional
        K2 campaign identifier (e.g. 0..19, 91, 92, 101, 102, 111, 112).
    mission : str or None, optional
        Mission identifier (e.g. ``"K2"``, ``"TESS"``, ``"KEPLER"``).

    Returns
    -------
    list[float]
        Sorted list of split timestamps strictly inside ``(t_min, t_max)``.
        Never returns ``None``.
    """
    t_arr = np.asarray(time, dtype=np.float64).ravel()
    t_fin = np.sort(t_arr[np.isfinite(t_arr)])
    if t_fin.size < 2:
        return []

    t_min = float(t_fin[0])
    t_max = float(t_fin[-1])
    if t_max <= t_min:
        return []

    if splits is not None and not isinstance(splits, str):
        s_arr = np.asarray(splits, dtype=np.float64).ravel()
        valid = s_arr[np.isfinite(s_arr) & (s_arr > t_min) & (s_arr < t_max)]
        if valid.size == 0:
            return []
        return [float(v) for v in np.unique(valid)]

    if isinstance(splits, str):
        mode = splits.strip().lower()
        if mode in ("none", ""):
            return []
        if mode != "auto":
            raise ValueError(
                f"Unknown splits mode {splits!r}; expected 'auto', 'none', None, or a sequence of floats."
            )

    mission_norm = (mission or "").strip().upper()
    if campaign is not None and mission_norm != "TESS":
        try:
            camp_int = int(campaign)
        except (TypeError, ValueError):
            camp_int = None
        if camp_int is not None and camp_int in DEFAULT_K2_SPLITS:
            return [
                float(s)
                for s in sorted(DEFAULT_K2_SPLITS[camp_int])
                if t_min < float(s) < t_max
            ]

    # Gap-based breakpoint detection (e.g. TESS ~13.7 d mid-sector downlink gap)
    # combined with segmentation of continuous spans exceeding 30.0 days.
    dt = np.diff(t_fin)
    dt_pos = dt[dt > 0.0]
    med_dt = float(np.median(dt_pos)) if dt_pos.size > 0 else 0.0204
    gap_thresh = max(0.5, 10.0 * med_dt)

    gap_indices = np.flatnonzero(dt > gap_thresh)
    gap_splits = [0.5 * float(t_fin[idx] + t_fin[idx + 1]) for idx in gap_indices]

    # Segment any contiguous data span > 30.0 d into equal-duration chunks
    seg_starts = [t_min] + [float(t_fin[idx + 1]) for idx in gap_indices]
    seg_ends = [float(t_fin[idx]) for idx in gap_indices] + [t_max]

    all_splits: list[float] = list(gap_splits)
    for s_start, s_end in zip(seg_starts, seg_ends, strict=True):
        span = s_end - s_start
        if span > 30.0:
            n_chunks = math.floor(span / 30.0) + 1
            interior = np.linspace(s_start, s_end, n_chunks + 1)[1:-1]
            all_splits.extend(float(v) for v in interior)

    filtered = [s for s in sorted(set(all_splits)) if t_min < s < t_max]
    return filtered


def _coerce_covariates(
    n_pts: int,
    x=None,
    y=None,
    covariates=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and map 2D/D_c-dimensional covariates onto (valid_coords_xy, x_2d, y_2d)."""
    if covariates is not None and (x is not None or y is not None):
        raise ValueError("Pass either (x, y) or covariates, not both.")

    if covariates is not None:
        cov_raw = getattr(covariates, "values", covariates)
        cov_arr = np.asarray(cov_raw, dtype=np.float64)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr[:, np.newaxis]
        if cov_arr.ndim != 2 or cov_arr.shape[0] != n_pts or cov_arr.shape[1] < 1:
            raise ValueError(
                f"Expected covariates of shape ({n_pts}, D_c >= 1), got {cov_arr.shape}."
            )
        valid_xy = np.all(np.isfinite(cov_arr), axis=1)
        d_c = cov_arr.shape[1]
        if d_c == 1:
            x_out = cov_arr[:, 0].copy()
            y_out = np.where(valid_xy, 0.0, np.nan)
            return valid_xy, x_out, y_out
        if d_c == 2:
            return valid_xy, cov_arr[:, 0].copy(), cov_arr[:, 1].copy()

        # D_c > 2: drop constant intercept columns, then project >2 non-constant
        # columns onto their dominant 2 orthogonal singular axes so the 2D spatial
        # kernel incorporates variance across all supplied covariates.
        x_out = np.full(n_pts, np.nan, dtype=np.float64)
        y_out = np.full(n_pts, np.nan, dtype=np.float64)
        if int(valid_xy.sum()) == 0:
            return valid_xy, x_out, y_out

        sub = cov_arr[valid_xy]
        stds = np.std(sub, axis=0)
        non_const = stds > 0.0
        sub_nc = sub[:, non_const] if np.any(non_const) else sub
        if sub_nc.shape[1] == 1:
            x_out[valid_xy] = sub_nc[:, 0]
            y_out[valid_xy] = 0.0
        elif sub_nc.shape[1] == 2:
            x_out[valid_xy] = sub_nc[:, 0]
            y_out[valid_xy] = sub_nc[:, 1]
        else:
            centred = sub_nc - np.median(sub_nc, axis=0, keepdims=True)
            _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            proj = centred @ vt[:2].T
            x_out[valid_xy] = proj[:, 0]
            y_out[valid_xy] = proj[:, 1]
        return valid_xy, x_out, y_out

    if x is None or y is None:
        raise ValueError("Must provide both `x` and `y` arrays (or `covariates`).")

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size != n_pts or y_arr.size != n_pts:
        raise ValueError(
            f"Length mismatch: time has {n_pts} elements, x has {x_arr.size}, y has {y_arr.size}."
        )
    valid_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
    return valid_xy, x_arr.copy(), y_arr.copy()


def _detrend_for_clipping(
    cflux: np.ndarray,
    valid_mask: np.ndarray,
    mwidth: int | None,
) -> np.ndarray:
    """Apply optional running-median baseline flattening of width ``mwidth`` before sigma-clipping."""
    if mwidth is None or int(mwidth) <= 1:
        return cflux
    w = int(mwidth)
    idx_valid = np.flatnonzero(valid_mask & np.isfinite(cflux))
    if idx_valid.size < max(5, w):
        return cflux
    vals_valid = cflux[idx_valid]
    med_base = float(np.nanmedian(vals_valid))
    baseline_valid = median_filter(vals_valid, size=w, mode="nearest")
    idx_all = np.arange(cflux.size, dtype=np.float64)
    baseline_all = np.interp(idx_all, idx_valid.astype(np.float64), baseline_valid)
    return cflux - baseline_all + med_base


def _run_ls_period_search(
    time_masked: np.ndarray,
    resid_masked: np.ndarray,
    ls_min_period: float,
    ls_max_period: float,
) -> tuple[float | None, float | None, float | None]:
    """Run JAX-accelerated Lomb-Scargle period search returning (period, power, fap)."""
    if time_masked.size < 8:
        return None, None, None
    t_span = float(np.max(time_masked) - np.min(time_masked))
    if not np.isfinite(t_span) or t_span <= float(ls_min_period):
        return None, None, None

    freq, power, nout, _jmax, _prob = fasper(time_masked, resid_masked, 6.0, 0.5)
    valid_f = freq > 0.0
    if not np.any(valid_f):
        return None, None, None

    periods = np.where(valid_f, 1.0 / np.where(valid_f, freq, 1.0), np.inf)
    in_window = valid_f & (periods > float(ls_min_period)) & (periods < float(ls_max_period))
    if not np.any(in_window):
        return None, None, None

    periods_w = periods[in_window]
    power_w = power[in_window]
    j = int(np.argmax(power_w))
    best_period = float(periods_w[j])
    best_power = float(power_w[j])

    expy = math.exp(-min(max(0.0, best_power), 700.0))
    effm = 2.0 * float(nout) / 6.0
    fap = expy * effm
    if fap > 0.01:
        fap = 1.0 - (1.0 - expy) ** effm
    return best_period, best_power, float(fap)


def _instantiate_kernel(
    kernel: str | DtKernel | None,
    kernel_period: float | None,
    default_position_kernel: str,
    is_periodic: bool,
    ls_period: float | None,
) -> DtKernel:
    """Instantiate a fresh DtKernel instance from explicit override or automatic selection."""
    if isinstance(kernel, DtKernel):
        k_inst = copy.deepcopy(kernel)
        if kernel_period is not None and hasattr(k_inst, "period"):
            k_inst._pv[2] = float(kernel_period)
            k_inst.period = float(kernel_period)
            k_inst.set_pv(k_inst._pv)
        return k_inst

    if isinstance(kernel, str):
        key_clean = kernel.strip().lower()
        canonical = _KERNEL_NAME_ALIASES.get(key_clean)
        if canonical is None or canonical not in kernels:
            raise ValueError(
                f"Unknown kernel {kernel!r}. Choose from {sorted(kernels.keys())}."
            )
        if "periodic" in canonical:
            eff_period = kernel_period if kernel_period is not None else ls_period
            if eff_period is None:
                raise ValueError(
                    f"Kernel {canonical!r} requires `kernel_period` when no Lomb-Scargle period was detected."
                )
            return kernels[canonical](period=float(eff_period))
        return kernels[canonical]()

    if kernel is not None:
        raise TypeError(f"Expected kernel to be str, DtKernel, or None; got {type(kernel)!r}.")

    pos_key = default_position_kernel.strip().lower()
    if pos_key == "sqrexp":
        return QuasiPeriodicKernel(period=ls_period) if is_periodic else BasicKernel()
    if pos_key == "exp":
        return QuasiPeriodicKernelEP(period=ls_period) if is_periodic else BasicKernelEP()
    raise ValueError(
        f"Unknown default_position_kernel {default_position_kernel!r}; expected 'SqrExp' or 'Exp'."
    )


def _compute_periodic_transit_mask(
    time_valid: np.ndarray,
    center: float,
    period: float,
    duration: float,
) -> np.ndarray:
    """Return boolean array on ``time_valid`` where ``True`` indicates in-transit cadences."""
    per = float(period)
    dur = float(duration)
    if per <= 0.0 or dur < 0.0:
        raise ValueError(f"Transit period must be > 0 and duration >= 0 (got {per}, {dur}).")
    folded = np.asarray(fold(time_valid, per, float(center), shift=0.5), dtype=np.float64)
    return np.abs(folded - 0.5) * per <= 0.5 * dur


def _initial_outlier_and_period_pass(
    time_v: np.ndarray,
    flux_v: np.ndarray,
    inputs_v: np.ndarray,
    base_fit_mask_v: np.ndarray,
    splits_list: list[float],
    tr_nrandom: int,
    tr_nblocks: int,
    tr_bspan: int,
    outlier_sigma: float,
    outlier_mwidth: int | None,
    outlier_max_iter: int,
    ls_min_period: float,
    ls_max_period: float,
    seed: int | np.random.Generator | None,
    log: logging.Logger,
    quiet: bool,
) -> tuple[np.ndarray, float | None, float | None, float | None]:
    """Run initial BasicKernelEP outlier screen and Legendre-detrended LS period search."""
    mask_v = base_fit_mask_v.copy()
    if int(mask_v.sum()) < _MIN_FINITE_POINTS:
        return mask_v, None, None, None

    det_init = Detrender(
        flux_v,
        inputs_v,
        mask=mask_v,
        splits=splits_list,
        kernel=BasicKernelEP(),
        tr_nrandom=tr_nrandom,
        tr_nblocks=tr_nblocks,
        tr_bspan=tr_bspan,
        seed=seed,
    )
    ttrend, ptrend = det_init.predict(det_init.kernel.pv0, components=True)
    cflux = (
        flux_v
        - ptrend
        + np.nanmedian(ptrend)
        - ttrend
        + np.nanmedian(ttrend)
    )
    cflux_med = float(np.nanmedian(cflux))
    if np.isfinite(cflux_med) and cflux_med != 0.0:
        cflux = cflux / cflux_med

    if not quiet:
        log.info("Starting initial outlier detection")
    clip_input = _detrend_for_clipping(cflux, mask_v, outlier_mwidth)
    omask = mask_v & sigma_clip(
        clip_input,
        max_iter=outlier_max_iter,
        max_sigma=outlier_sigma,
        mexc=mask_v,
    )
    ofrac = float((~omask).sum()) / float(max(1, omask.size))
    if ofrac < 0.25:
        mask_v &= omask
        if not quiet:
            log.info("  Flagged %i (%4.1f%%) outliers.", int((~omask).sum()), 100.0 * ofrac)
    elif not quiet:
        log.info(
            "  Found %i (%4.1f%%) outliers. Not flagging.",
            int((~omask).sum()),
            100.0 * ofrac,
        )

    if ofrac >= 0.9 or int(mask_v.sum()) < _MIN_FINITE_POINTS:
        if not quiet:
            log.info("Too many outliers or too few points, skipping Lomb-Scargle period search")
        return mask_v, None, None, None

    if not quiet:
        log.info("Starting Lomb-Scargle period search")
    nflux = flux_v - ptrend + np.nanmedian(ptrend)
    ntime = time_v - float(np.mean(time_v[mask_v]))
    deg = min(9, max(1, int(mask_v.sum()) // 10))
    pflux = np.polynomial.Legendre.fit(ntime[mask_v], nflux[mask_v], deg=deg)(ntime)
    resid = nflux - pflux

    period, power, fap = _run_ls_period_search(
        time_v[mask_v],
        resid[mask_v],
        ls_min_period=ls_min_period,
        ls_max_period=ls_max_period,
    )
    return mask_v, period, power, fap


def _fit_single_aperture_valid(
    time_v: np.ndarray,
    flux_v: np.ndarray,
    inputs_v: np.ndarray,
    fit_mask_v: np.ndarray,
    mexc_final_v: np.ndarray,
    mflags_v: np.ndarray,
    splits_list: list[float],
    kernel_inst: DtKernel,
    is_periodic: bool,
    ls_period: float | None,
    ls_fap: float | None,
    tr_nrandom: int,
    tr_nblocks: int,
    tr_bspan: int,
    optimizer: str,
    de_npop: int,
    de_niter: int,
    de_max_time: float,
    ls_min_period: float,
    ls_max_period: float,
    outlier_sigma: float,
    outlier_mwidth: int | None,
    outlier_max_iter: int,
    seed: int | np.random.Generator | None,
    log: logging.Logger,
    quiet: bool,
    iset: int = 0,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[
    Detrender,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    bool,
    OptimizationResult | None,
]:
    """Optimize GP hyperparameters and compute trends/outliers/CDPP on valid-coord arrays."""
    tstart = clock.perf_counter()
    extra_opt = dict(optimizer_kwargs) if optimizer_kwargs is not None else {}

    n_valid = flux_v.size
    if n_valid == 0:
        raise ValueError("No finite (time, x, y) coordinates available for GP detrending.")

    safe_mask = fit_mask_v if np.any(fit_mask_v) else np.isfinite(flux_v)
    if not np.any(safe_mask):
        safe_mask = np.ones(n_valid, dtype=bool)

    detrender = Detrender(
        flux_v,
        inputs_v,
        mask=safe_mask,
        splits=splits_list,
        kernel=kernel_inst,
        tr_nrandom=tr_nrandom,
        tr_nblocks=tr_nblocks,
        tr_bspan=tr_bspan,
        seed=seed,
    )

    n_fit = int(np.count_nonzero(fit_mask_v & np.isfinite(flux_v)))
    if n_fit < _MIN_FINITE_POINTS:
        if not quiet:
            log.info("Skipping dataset %i, not enough finite datapoints (%i)", iset, n_fit)
        pv = np.full(kernel_inst.npar, np.nan, dtype=np.float64)
        detrender.tr_pv = pv.copy()
        nan_arr = np.full(n_valid, np.nan, dtype=np.float64)
        mflags_out = mflags_v.copy()
        mflags_out[~np.isfinite(flux_v)] |= M_NOTFINITE
        inlier_v = np.zeros(n_valid, dtype=bool)
        return (
            detrender,
            pv,
            nan_arr.copy(),
            nan_arr.copy(),
            nan_arr.copy(),
            nan_arr.copy(),
            mflags_out,
            inlier_v,
            -1.0,
            -1.0,
            -1.0,
            False,
            None,
        )

    if not quiet:
        log.info("Starting hyperparameter optimization (%s)", optimizer)
    opt_res = detrender.optimize(
        optimizer=optimizer,
        seed=seed,
        ls_min_period=ls_min_period,
        ls_max_period=ls_max_period,
        de_npop=de_npop,
        de_niter=de_niter,
        de_max_time=de_max_time,
        **extra_opt,
    )
    pv = np.asarray(opt_res.pv, dtype=np.float64).copy()
    detrender.tr_pv = pv.copy()
    success = bool(opt_res.success)

    t_time_v, t_pos_v = detrender.predict(pv, components=True)
    mt = float(np.nanmedian(t_time_v))
    mp = float(np.nanmedian(t_pos_v))
    tt_v = t_time_v - mt
    tp_v = t_pos_v - mp

    corr_v = flux_v - tp_v
    detr_v = flux_v - tp_v - tt_v

    if not quiet:
        log.info("Starting final outlier detection")
    cflux = detr_v.copy()
    cflux_med = float(np.nanmedian(cflux))
    if np.isfinite(cflux_med) and cflux_med != 0.0:
        cflux = cflux / cflux_med

    minf_v = np.isfinite(cflux)
    clip_arr = _detrend_for_clipping(cflux, mexc_final_v, outlier_mwidth)
    _mexc_ret, mlow_v, mhigh_v = sigma_clip(
        clip_arr,
        max_iter=outlier_max_iter,
        max_sigma=outlier_sigma,
        separate_masks=True,
        mexc=mexc_final_v,
    )

    mflags_out = mflags_v.copy()
    mflags_out[~minf_v] |= M_NOTFINITE
    mflags_out[~mhigh_v] |= M_OUTLIER_U
    mflags_out[~mlow_v] |= M_OUTLIER_D
    inlier_v = minf_v & mlow_v & mhigh_v

    if not quiet:
        log.info("  %5i too high", int((~mhigh_v).sum()))
        log.info("  %5i too low", int((~mlow_v).sum()))
        log.info("  %5i not finite", int((~minf_v).sum()))

    dd = detrender.data
    if dd.masked_time.size >= 3:
        cdpp_r = float(cdpp(dd.masked_time, dd.masked_flux))
        cdpp_t = float(cdpp(dd.unmasked_time, corr_v, exclude=~dd.mask))
        cdpp_c = float(cdpp(dd.unmasked_time, detr_v, exclude=~dd.mask))
    else:
        cdpp_r, cdpp_t, cdpp_c = float("nan"), float("nan"), float("nan")

    if not quiet:
        log.info("  CDPP - raw - %6.3f", cdpp_r)
        log.info("  CDPP - position component removed - %6.3f", cdpp_t)
        log.info("  CDPP - full reduction - %6.3f", cdpp_c)
        log.info("Detrending time %6.3f s", clock.perf_counter() - tstart)

    return (
        detrender,
        pv,
        t_time_v,
        t_pos_v,
        corr_v,
        detr_v,
        mflags_out,
        inlier_v,
        cdpp_r,
        cdpp_t,
        cdpp_c,
        success,
        opt_res,
    )


def detrend_array(
    time,
    flux,
    *,
    x=None,
    y=None,
    covariates=None,
    flux_err=None,
    quality=None,
    cadence_mask=None,
    transit_mask=None,
    p_mask_center: float | None = None,
    p_mask_period: float | None = None,
    p_mask_duration: float | None = None,
    initial_mflags=None,
    mission: str | None = None,
    campaign: int | None = None,
    splits: Sequence[float] | np.ndarray | str | None = "auto",
    kernel: str | DtKernel | None = None,
    kernel_period: float | None = None,
    default_position_kernel: str = "SqrExp",
    roll_pca: bool = True,
    tr_nrandom: int = 400,
    tr_nblocks: int = 6,
    tr_bspan: int = 50,
    optimizer: str = "auto",
    de_npop: int = 100,
    de_niter: int = 150,
    de_max_time: float = 300.0,
    ls_max_fap: float = -50.0,
    ls_min_period: float = 0.05,
    ls_max_period: float = 25.0,
    outlier_sigma: float = 5.0,
    outlier_mwidth: int | None = 25,
    outlier_max_iter: int = 10,
    seed: int | np.random.Generator | None = 0,
    quiet: bool = True,
    logger: logging.Logger | None = None,
    **optimizer_kwargs,
) -> K2SCResult:
    """Detrend a 1D light curve array, preserving full input length ``N`` under NaN coordinates."""
    _ = flux_err  # accepted for API symmetry across array / LightCurve callers
    log = logger if logger is not None else logging.getLogger("k2sc.pipeline")

    time_full = np.asarray(time, dtype=np.float64).ravel()
    flux_full = np.asarray(flux, dtype=np.float64).ravel()
    n_full = time_full.size
    if flux_full.size != n_full:
        raise ValueError(
            f"Length mismatch: time has {n_full} elements, flux has {flux_full.size}."
        )
    if n_full == 0:
        raise ValueError("Input time and flux arrays must be non-empty.")

    valid_xy, x_full, y_full = _coerce_covariates(
        n_full, x=x, y=y, covariates=covariates
    )
    valid_coords = np.isfinite(time_full) & valid_xy

    if initial_mflags is not None:
        mflags_full = np.asarray(initial_mflags, dtype=np.uint8).ravel().copy()
        if mflags_full.size != n_full:
            raise ValueError(
                f"initial_mflags length ({mflags_full.size}) != input length ({n_full})."
            )
    else:
        mflags_full = np.zeros(n_full, dtype=np.uint8)

    mission_norm = (mission or "").strip().upper()
    q_fit_full = np.ones(n_full, dtype=bool)
    if quality is not None:
        q_raw = np.asarray(quality).ravel()
        if q_raw.size != n_full:
            raise ValueError(
                f"quality length ({q_raw.size}) != input length ({n_full})."
            )
        if np.issubdtype(q_raw.dtype, np.floating):
            q_int = np.where(np.isfinite(q_raw), q_raw, 0).astype(np.int64)
        else:
            q_int = q_raw.astype(np.int64)
        mflags_full[q_int != 0] |= M_QUALITY
        if mission_norm == "TESS":
            q_fit_full &= q_int == 0
        else:
            q_fit_full &= (q_int & _K2_THRUSTER_FLAG) == 0

    mflags_full[~np.isfinite(flux_full)] |= M_NOTFINITE
    mflags_full[~valid_coords] |= M_NOTFINITE

    # Periodic / transit masks
    if (
        p_mask_center is not None
        and p_mask_period is not None
        and p_mask_duration is not None
        and np.any(valid_coords)
    ):
        in_tr_p = _compute_periodic_transit_mask(
            time_full[valid_coords],
            p_mask_center,
            p_mask_period,
            p_mask_duration,
        )
        valid_idx = np.flatnonzero(valid_coords)
        mflags_full[valid_idx[in_tr_p]] |= M_PERIODIC

    if transit_mask is not None:
        is_tuple_spec = isinstance(transit_mask, (tuple, list)) and (
            len(transit_mask) == 3
            and not isinstance(transit_mask[0], (bool, np.bool_))
        )
        if is_tuple_spec:
            if np.any(valid_coords):
                t0_m, per_m, dur_m = transit_mask
                in_tr = _compute_periodic_transit_mask(
                    time_full[valid_coords],
                    float(t0_m),
                    float(per_m),
                    float(dur_m),
                )
                valid_idx = np.flatnonzero(valid_coords)
                mflags_full[valid_idx[in_tr]] |= M_PERIODIC
        else:
            tmask_bool = np.asarray(transit_mask, dtype=bool).ravel()
            if tmask_bool.size != n_full:
                raise ValueError(
                    f"Boolean transit_mask length ({tmask_bool.size}) != input length ({n_full})."
                )
            mflags_full[tmask_bool] |= M_PERIODIC

    cmask_full = np.ones(n_full, dtype=bool)
    if cadence_mask is not None:
        cmask_arr = np.asarray(cadence_mask, dtype=bool).ravel()
        if cmask_arr.size != n_full:
            raise ValueError(
                f"cadence_mask length ({cmask_arr.size}) != input length ({n_full})."
            )
        cmask_full &= cmask_arr

    if not np.any(valid_coords):
        raise ValueError("No cadences with simultaneously finite time and spatial covariates.")

    pca_tf: RollPCATransform | None = None
    if roll_pca:
        clean_pca = valid_coords & np.isfinite(flux_full) & q_fit_full & cmask_full
        u1_full, u2_full, pca_tf = pca_normalize_xy(
            x_full, y_full, clean_mask=clean_pca
        )
        x_v = u1_full[valid_coords]
        y_v = u2_full[valid_coords]
    else:
        x_v = x_full[valid_coords]
        y_v = y_full[valid_coords]

    time_v = time_full[valid_coords]
    flux_v = flux_full[valid_coords]
    inputs_v = np.column_stack([time_v, x_v, y_v])
    mflags_v = mflags_full[valid_coords].copy()

    splits_list = resolve_splits(
        time_v, splits=splits, campaign=campaign, mission=mission
    )

    not_periodic_v = ~(mflags_v & M_PERIODIC).astype(bool)
    mexc_final_v = not_periodic_v & q_fit_full[valid_coords] & cmask_full[valid_coords]
    base_fit_mask_v = np.isfinite(flux_v) & mexc_final_v

    fit_mask_v, det_period, _det_power, det_fap = _initial_outlier_and_period_pass(
        time_v=time_v,
        flux_v=flux_v,
        inputs_v=inputs_v,
        base_fit_mask_v=base_fit_mask_v,
        splits_list=splits_list,
        tr_nrandom=tr_nrandom,
        tr_nblocks=tr_nblocks,
        tr_bspan=tr_bspan,
        outlier_sigma=outlier_sigma,
        outlier_mwidth=outlier_mwidth,
        outlier_max_iter=outlier_max_iter,
        ls_min_period=ls_min_period,
        ls_max_period=ls_max_period,
        seed=seed,
        log=log,
        quiet=quiet,
    )

    fap_thresh = 10.0 ** float(ls_max_fap)
    is_periodic = bool(det_fap is not None and det_fap < fap_thresh)
    ls_period = det_period if is_periodic else None
    ls_fap = det_fap

    kernel_inst = _instantiate_kernel(
        kernel=kernel,
        kernel_period=kernel_period,
        default_position_kernel=default_position_kernel,
        is_periodic=is_periodic,
        ls_period=ls_period if ls_period is not None else det_period,
    )

    (
        detrender,
        pv,
        t_time_v,
        t_pos_v,
        corr_v,
        detr_v,
        mflags_v_out,
        inlier_v,
        cdpp_r,
        cdpp_t,
        cdpp_c,
        success,
        opt_res,
    ) = _fit_single_aperture_valid(
        time_v=time_v,
        flux_v=flux_v,
        inputs_v=inputs_v,
        fit_mask_v=fit_mask_v,
        mexc_final_v=mexc_final_v,
        mflags_v=mflags_v,
        splits_list=splits_list,
        kernel_inst=kernel_inst,
        is_periodic=is_periodic,
        ls_period=ls_period,
        ls_fap=ls_fap,
        tr_nrandom=tr_nrandom,
        tr_nblocks=tr_nblocks,
        tr_bspan=tr_bspan,
        optimizer=optimizer,
        de_npop=de_npop,
        de_niter=de_niter,
        de_max_time=de_max_time,
        ls_min_period=ls_min_period,
        ls_max_period=ls_max_period,
        outlier_sigma=outlier_sigma,
        outlier_mwidth=outlier_mwidth,
        outlier_max_iter=outlier_max_iter,
        seed=seed,
        log=log,
        quiet=quiet,
        iset=0,
        optimizer_kwargs=optimizer_kwargs,
    )

    tr_time_full = np.full(n_full, np.nan, dtype=np.float64)
    tr_pos_full = np.full(n_full, np.nan, dtype=np.float64)
    corr_full = np.full(n_full, np.nan, dtype=np.float64)
    detr_full = np.full(n_full, np.nan, dtype=np.float64)
    outlier_mask_full = np.zeros(n_full, dtype=bool)

    tr_time_full[valid_coords] = t_time_v
    tr_pos_full[valid_coords] = t_pos_v
    corr_full[valid_coords] = corr_v
    detr_full[valid_coords] = detr_v
    outlier_mask_full[valid_coords] = inlier_v

    mflags_full[valid_coords] = mflags_v_out
    mflags_full[~valid_coords] |= M_NOTFINITE
    mflags_full[~np.isfinite(flux_full)] |= M_NOTFINITE

    warn = not success
    return K2SCResult(
        detrender=detrender,
        pv=pv,
        tr_time=tr_time_full,
        tr_position=tr_pos_full,
        cdpp_r=cdpp_r,
        cdpp_t=cdpp_t,
        cdpp_c=cdpp_c,
        warn=warn,
        success=success,
        corr_flux=corr_full,
        detrended_flux=detr_full,
        mflags=mflags_full,
        outlier_mask=outlier_mask_full,
        splits=splits_list,
        is_periodic=is_periodic,
        ls_period=ls_period,
        ls_fap=ls_fap,
        opt_result=opt_res,
        lightcurve=None,
        pca_transform=pca_tf,
    )


def detrend_k2data(
    dataset: K2Data,
    *,
    campaign: int | None = None,
    splits: Sequence[float] | np.ndarray | str | None = "auto",
    quiet: bool = False,
    seed: int | np.random.Generator | None = 0,
    default_position_kernel: str = "SqrExp",
    roll_pca: bool = True,
    kernel: str | DtKernel | None = None,
    kernel_period: float | None = None,
    transit_mask=None,
    p_mask_center: float | None = None,
    p_mask_period: float | None = None,
    p_mask_duration: float | None = None,
    tr_nrandom: int = 400,
    tr_nblocks: int = 6,
    tr_bspan: int = 50,
    optimizer: str = "auto",
    de_npop: int = 100,
    de_niter: int = 150,
    de_max_time: float = 300.0,
    ls_max_fap: float = -50.0,
    ls_min_period: float = 0.05,
    ls_max_period: float = 25.0,
    outlier_sigma: float = 5.0,
    outlier_mwidth: int | None = 25,
    outlier_max_iter: int = 10,
    logger: logging.Logger | None = None,
    **optimizer_kwargs,
) -> list[K2SCResult]:
    """Detrend all ``dataset.nsets`` apertures on a ``K2Data`` instance in-place.

    Updates ``dataset.mflags[iset]``, ``dataset.is_periodic``, ``dataset.ls_period``,
    ``dataset.ls_power``, and ``dataset.ls_fap`` for downstream ``FITSWriter.write``.
    """
    ds = dataset
    log = logger if logger is not None else logging.getLogger("k2sc.pipeline")

    eff_campaign = ds.campaign if ds.campaign is not None else campaign
    splits_list = resolve_splits(ds.time, splits=splits, campaign=eff_campaign, mission="K2")

    if not quiet:
        log.info("Resolved splits %s (campaign=%s)", splits_list, eff_campaign)

    if (
        p_mask_center is not None
        and p_mask_period is not None
        and p_mask_duration is not None
    ):
        ds.mask_periodic_signal(p_mask_center, p_mask_period, p_mask_duration)

    if transit_mask is not None:
        is_tuple_spec = isinstance(transit_mask, (tuple, list)) and (
            len(transit_mask) == 3
            and not isinstance(transit_mask[0], (bool, np.bool_))
        )
        if is_tuple_spec:
            t0_m, per_m, dur_m = transit_mask
            ds.mask_periodic_signal(float(t0_m), float(per_m), float(dur_m))
        else:
            tmask_arr = np.asarray(transit_mask, dtype=bool).ravel()
            if tmask_arr.size == ds.npoints:
                ds.mflags[:, tmask_arr] |= M_PERIODIC
            elif tmask_arr.size == ds.nanmask.size:
                ds.mflags[:, tmask_arr[ds.nanmask]] |= M_PERIODIC
            else:
                raise ValueError(
                    f"Boolean transit_mask length ({tmask_arr.size}) matches neither "
                    f"ds.npoints ({ds.npoints}) nor raw length ({ds.nanmask.size})."
                )

    mthf = (ds.quality & _K2_THRUSTER_FLAG) == 0
    pca_tf: RollPCATransform | None = None
    if roll_pca:
        has_finite_flux = (
            np.any(np.isfinite(ds.fluxes), axis=0)
            if ds.fluxes.size > 0
            else np.zeros(ds.time.size, dtype=bool)
        )
        clean_pca = (
            np.isfinite(ds.time)
            & np.isfinite(ds.x)
            & np.isfinite(ds.y)
            & mthf
            & has_finite_flux
        )
        u1, u2, pca_tf = pca_normalize_xy(ds.x, ds.y, clean_mask=clean_pca)
        inputs = np.column_stack([ds.time, u1, u2])
    else:
        inputs = np.column_stack([ds.time, ds.x, ds.y])
    fap_thresh = 10.0 ** float(ls_max_fap)

    fit_masks: list[np.ndarray] = []
    mexc_masks: list[np.ndarray] = []
    fallback_period: float | None = None

    for iset in range(ds.nsets):
        flux = ds.fluxes[iset]
        mper = ~(ds.mflags[iset] & M_PERIODIC).astype(bool)
        mexc_i = mper & mthf
        base_mask_i = np.isfinite(flux) & mexc_i

        mask_i, period_i, power_i, fap_i = _initial_outlier_and_period_pass(
            time_v=ds.time,
            flux_v=flux,
            inputs_v=inputs,
            base_fit_mask_v=base_mask_i,
            splits_list=splits_list,
            tr_nrandom=tr_nrandom,
            tr_nblocks=tr_nblocks,
            tr_bspan=tr_bspan,
            outlier_sigma=outlier_sigma,
            outlier_mwidth=outlier_mwidth,
            outlier_max_iter=outlier_max_iter,
            ls_min_period=ls_min_period,
            ls_max_period=ls_max_period,
            seed=seed,
            log=log,
            quiet=quiet,
        )
        fit_masks.append(mask_i)
        mexc_masks.append(mexc_i)

        if period_i is not None and fallback_period is None:
            fallback_period = period_i
        if (
            fap_i is not None
            and fap_i < fap_thresh
            and (not ds.is_periodic or ds.ls_fap is None or fap_i < ds.ls_fap)
        ):
            ds.is_periodic = True
            ds.ls_period = period_i
            ds.ls_power = power_i
            ds.ls_fap = fap_i

    results: list[K2SCResult] = []
    for iset in range(ds.nsets):
        kernel_i = _instantiate_kernel(
            kernel=kernel,
            kernel_period=kernel_period,
            default_position_kernel=default_position_kernel,
            is_periodic=bool(ds.is_periodic),
            ls_period=ds.ls_period if ds.ls_period is not None else fallback_period,
        )

        (
            detrender_i,
            pv_i,
            t_time_i,
            t_pos_i,
            corr_i,
            detr_i,
            mflags_i,
            inlier_i,
            cdpp_r,
            cdpp_t,
            cdpp_c,
            success_i,
            opt_res_i,
        ) = _fit_single_aperture_valid(
            time_v=ds.time,
            flux_v=ds.fluxes[iset],
            inputs_v=inputs,
            fit_mask_v=fit_masks[iset],
            mexc_final_v=mexc_masks[iset],
            mflags_v=ds.mflags[iset],
            splits_list=splits_list,
            kernel_inst=kernel_i,
            is_periodic=bool(ds.is_periodic),
            ls_period=ds.ls_period,
            ls_fap=ds.ls_fap,
            tr_nrandom=tr_nrandom,
            tr_nblocks=tr_nblocks,
            tr_bspan=tr_bspan,
            optimizer=optimizer,
            de_npop=de_npop,
            de_niter=de_niter,
            de_max_time=de_max_time,
            ls_min_period=ls_min_period,
            ls_max_period=ls_max_period,
            outlier_sigma=outlier_sigma,
            outlier_mwidth=outlier_mwidth,
            outlier_max_iter=outlier_max_iter,
            seed=seed,
            log=log,
            quiet=quiet,
            iset=iset,
            optimizer_kwargs=optimizer_kwargs,
        )

        ds.mflags[iset][:] = mflags_i
        warn_i = not success_i
        results.append(
            K2SCResult(
                detrender=detrender_i,
                pv=pv_i,
                tr_time=t_time_i,
                tr_position=t_pos_i,
                cdpp_r=cdpp_r,
                cdpp_t=cdpp_t,
                cdpp_c=cdpp_c,
                warn=warn_i,
                success=success_i,
                corr_flux=corr_i,
                detrended_flux=detr_i,
                mflags=mflags_i.copy(),
                outlier_mask=inlier_i,
                splits=list(splits_list),
                is_periodic=bool(ds.is_periodic),
                ls_period=ds.ls_period,
                ls_fap=ds.ls_fap,
                opt_result=opt_res_i,
                lightcurve=None,
                pca_transform=pca_tf,
            )
        )

    return results


def detrend(
    data,
    flux=None,
    *,
    x=None,
    y=None,
    covariates=None,
    flux_err=None,
    quality=None,
    cadence_mask=None,
    transit_mask=None,
    mission: str | None = None,
    campaign: int | None = None,
    splits: Sequence[float] | np.ndarray | str | None = "auto",
    quiet: bool = False,
    save_dir: str = ".",
    seed: int | np.random.Generator | None = 0,
    flux_type: str = "pdc",
    default_position_kernel: str = "SqrExp",
    roll_pca: bool = True,
    kernel: str | DtKernel | None = None,
    kernel_period: float | None = None,
    p_mask_center: float | None = None,
    p_mask_period: float | None = None,
    p_mask_duration: float | None = None,
    tr_nrandom: int = 400,
    tr_nblocks: int = 6,
    tr_bspan: int = 50,
    optimizer: str = "auto",
    de_npop: int = 100,
    de_niter: int = 150,
    de_max_time: float = 300.0,
    ls_max_fap: float = -50.0,
    ls_min_period: float = 0.05,
    ls_max_period: float = 25.0,
    outlier_sigma: float = 5.0,
    outlier_mwidth: int | None = 25,
    outlier_max_iter: int = 10,
    remove_time_trend: bool = False,
    return_result: bool = False,
    **optimizer_kwargs,
):
    """Polymorphic K2SC detrending entry point.

    Accepts:
    1. A ``lightkurve.LightCurve`` (``KeplerLightCurve``, ``TessLightCurve``, or
       generic ``LightCurve``): delegates to ``K2SCCorrector(data, ...).correct(...)``
       and returns the corrected ``LightCurve`` (or ``K2SCResult`` if ``return_result=True``).
    2. A ``K2Data`` instance: detrends all ``ds.nsets`` apertures and returns
       ``(results[0], results[0].detrender)`` for ``nsets == 1`` or
       ``(results, [r.detrender for r in results])`` for ``nsets > 1``.
    3. 1D NumPy/JAX arrays ``detrend(time, flux, x=..., y=..., covariates=...)``:
       returns a ``K2SCResult``.
    """
    _ = (save_dir, flux_type)  # accepted for legacy standalone.detrend signature compatibility

    from .corrector import HAS_LIGHTKURVE, K2SCCorrector, LightCurve

    if HAS_LIGHTKURVE and LightCurve is not None and isinstance(data, LightCurve):
        corr = K2SCCorrector(data, x=x, y=y, covariates=covariates)
        clc = corr.correct(
            cadence_mask=cadence_mask,
            splits=splits,
            campaign=campaign,
            kernel=kernel,
            kernel_period=kernel_period,
            default_position_kernel=default_position_kernel,
            roll_pca=roll_pca,
            transit_mask=transit_mask,
            p_mask_center=p_mask_center,
            p_mask_period=p_mask_period,
            p_mask_duration=p_mask_duration,
            tr_nrandom=tr_nrandom,
            tr_nblocks=tr_nblocks,
            tr_bspan=tr_bspan,
            optimizer=optimizer,
            de_npop=de_npop,
            de_niter=de_niter,
            de_max_time=de_max_time,
            ls_max_fap=ls_max_fap,
            ls_min_period=ls_min_period,
            ls_max_period=ls_max_period,
            outlier_sigma=outlier_sigma,
            outlier_mwidth=outlier_mwidth,
            outlier_max_iter=outlier_max_iter,
            seed=seed,
            remove_time_trend=remove_time_trend,
            quiet=quiet,
            **optimizer_kwargs,
        )
        if return_result:
            assert corr.result is not None
            return corr.result
        return clc

    if isinstance(data, K2Data):
        results = detrend_k2data(
            data,
            campaign=campaign,
            splits=splits,
            quiet=quiet,
            seed=seed,
            default_position_kernel=default_position_kernel,
            roll_pca=roll_pca,
            kernel=kernel,
            kernel_period=kernel_period,
            transit_mask=transit_mask,
            p_mask_center=p_mask_center,
            p_mask_period=p_mask_period,
            p_mask_duration=p_mask_duration,
            tr_nrandom=tr_nrandom,
            tr_nblocks=tr_nblocks,
            tr_bspan=tr_bspan,
            optimizer=optimizer,
            de_npop=de_npop,
            de_niter=de_niter,
            de_max_time=de_max_time,
            ls_max_fap=ls_max_fap,
            ls_min_period=ls_min_period,
            ls_max_period=ls_max_period,
            outlier_sigma=outlier_sigma,
            outlier_mwidth=outlier_mwidth,
            outlier_max_iter=outlier_max_iter,
            **optimizer_kwargs,
        )
        if data.nsets == 1:
            return results[0], results[0].detrender
        return results, [r.detrender for r in results]

    if flux is None:
        raise ValueError(
            "When calling `detrend(time, flux, ...)` on 1D arrays, `flux` must be provided."
        )

    return detrend_array(
        data,
        flux,
        x=x,
        y=y,
        covariates=covariates,
        flux_err=flux_err,
        quality=quality,
        cadence_mask=cadence_mask,
        transit_mask=transit_mask,
        p_mask_center=p_mask_center,
        p_mask_period=p_mask_period,
        p_mask_duration=p_mask_duration,
        mission=mission,
        campaign=campaign,
        splits=splits,
        kernel=kernel,
        kernel_period=kernel_period,
        default_position_kernel=default_position_kernel,
        roll_pca=roll_pca,
        tr_nrandom=tr_nrandom,
        tr_nblocks=tr_nblocks,
        tr_bspan=tr_bspan,
        optimizer=optimizer,
        de_npop=de_npop,
        de_niter=de_niter,
        de_max_time=de_max_time,
        ls_max_fap=ls_max_fap,
        ls_min_period=ls_min_period,
        ls_max_period=ls_max_period,
        outlier_sigma=outlier_sigma,
        outlier_mwidth=outlier_mwidth,
        outlier_max_iter=outlier_max_iter,
        seed=seed,
        quiet=quiet,
        **optimizer_kwargs,
    )


__all__ = [
    "K2SCResult",
    "SCResult",
    "detrend",
    "detrend_array",
    "detrend_k2data",
    "resolve_splits",
]
