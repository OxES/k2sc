"""Idiomatic Lightkurve v2+ Gaussian Process systematics corrector for K2SC."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._jax_env import ensure_float64
from .core import M_PERIODIC
from .kernels import DtKernel
from .pipeline import K2SCResult, _coerce_covariates, detrend_array

ensure_float64()

try:
    from lightkurve import KeplerLightCurve, LightCurve, TessLightCurve
    from lightkurve.correctors import DesignMatrix
    from lightkurve.correctors.corrector import Corrector as _BaseCorrector

    HAS_LIGHTKURVE = True
except ImportError:
    _BaseCorrector = object
    LightCurve = None  # type: ignore[assignment,misc]
    KeplerLightCurve = None  # type: ignore[assignment,misc]
    TessLightCurve = None  # type: ignore[assignment,misc]
    DesignMatrix = None  # type: ignore[assignment,misc]
    HAS_LIGHTKURVE = False

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _to_float64_array(col: Any) -> np.ndarray:
    """Unwrap Astropy Time, MaskedQuantity, MaskedNDArray, MaskedArray, or Quantity to float64.

    Masked entries are deterministically mapped to ``np.nan`` without unit-stripping warnings.
    """
    mask: np.ndarray | None = None

    if hasattr(col, "jd") and hasattr(col, "value"):
        if getattr(col, "masked", False) and hasattr(col, "mask"):
            mask = np.asarray(col.mask, dtype=bool)
        val = col.value
        val_unmasked = getattr(val, "unmasked", val)
        val_probe = np.asarray(getattr(val_unmasked, "value", val_unmasked))
        if np.issubdtype(val_probe.dtype, np.number) or np.issubdtype(
            val_probe.dtype, np.bool_
        ):
            col = val
        else:
            col = col.jd

    if hasattr(col, "unmasked"):
        inner = col.unmasked
        raw = getattr(inner, "value", inner)
        arr = np.asarray(raw, dtype=np.float64).copy()
        col_mask = np.asarray(col.mask, dtype=bool)
        if mask is not None:
            col_mask = col_mask | mask
        arr[col_mask] = np.nan
        return arr

    if isinstance(col, np.ma.MaskedArray):
        arr = np.asarray(col.filled(np.nan), dtype=np.float64).copy()
        if mask is not None:
            arr[mask] = np.nan
        return arr

    if hasattr(col, "value"):
        col = col.value

    arr = np.asarray(col, dtype=np.float64).copy()
    if mask is not None:
        arr[mask] = np.nan
    return arr


def _extract_quality_array(col: Any, n_pts: int) -> np.ndarray:
    """Unwrap quality bitmask column (including MaskedNDArray / Quantity) to int64."""
    if col is None:
        return np.zeros(n_pts, dtype=np.int64)
    if hasattr(col, "unmasked"):
        inner = getattr(col.unmasked, "value", col.unmasked)
        raw = np.asarray(inner)
        mask = np.asarray(col.mask, dtype=bool)
        if np.issubdtype(raw.dtype, np.floating):
            arr = np.where(np.isfinite(raw) & ~mask, raw, 0).astype(np.int64)
        else:
            arr = raw.astype(np.int64, copy=True)
            arr[mask] = 0
        return arr.ravel()
    if isinstance(col, np.ma.MaskedArray):
        raw = np.asarray(col.filled(0))
        if np.issubdtype(raw.dtype, np.floating):
            return np.where(np.isfinite(raw), raw, 0).astype(np.int64).ravel()
        return raw.astype(np.int64).ravel()
    raw_val = getattr(col, "value", col)
    raw_arr = np.asarray(raw_val)
    if np.issubdtype(raw_arr.dtype, np.floating):
        return np.where(np.isfinite(raw_arr), raw_arr, 0).astype(np.int64).ravel()
    return raw_arr.astype(np.int64).ravel()


def _lookup_lc_column(lc: Any, name: str) -> Any | None:
    """Return column from ``lc`` (case-insensitive) or direct attribute if present."""
    colnames = list(getattr(lc, "colnames", ()))
    if name in colnames:
        return lc[name]
    lower_map = {c.lower(): c for c in colnames}
    if name.lower() in lower_map:
        return lc[lower_map[name.lower()]]
    if hasattr(lc, name):
        return getattr(lc, name)
    return None


def _has_usable_pair(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check >50% finite values and positive variance on both spatial columns."""
    if c1.size == 0 or c2.size != c1.size:
        return False
    fin1 = np.isfinite(c1)
    fin2 = np.isfinite(c2)
    if float(fin1.mean()) <= 0.5 or float(fin2.mean()) <= 0.5:
        return False
    return bool(float(np.nanstd(c1)) > 0.0 and float(np.nanstd(c2)) > 0.0)


def _infer_mission_and_campaign(
    lc: Any,
    campaign_arg: int | None,
) -> tuple[str | None, int | None]:
    """Infer mission ('K2', 'TESS', 'KEPLER', or None) and K2 campaign from LightCurve."""
    meta = getattr(lc, "meta", {}) or {}
    mission_meta = str(meta.get("MISSION", meta.get("TELESCOP", ""))).strip().upper()

    if (TessLightCurve is not None and isinstance(lc, TessLightCurve)) or (
        mission_meta == "TESS"
    ):
        return "TESS", campaign_arg

    camp = campaign_arg
    if camp is None:
        if "CAMPAIGN" in meta and meta["CAMPAIGN"] is not None:
            camp = meta["CAMPAIGN"]
        elif getattr(lc, "campaign", None) is not None:
            camp = lc.campaign

    if KeplerLightCurve is not None and isinstance(lc, KeplerLightCurve):
        mission = "K2" if (camp is not None or mission_meta == "K2") else "KEPLER"
        return mission, camp

    if mission_meta in ("K2", "KEPLER"):
        return mission_meta, camp
    if camp is not None:
        return "K2", camp
    return None, camp


class K2SCCorrector(_BaseCorrector):
    """Gaussian Process time + position systematics corrector for Lightkurve v2+.

    Supports ``KeplerLightCurve``, ``TessLightCurve``, and generic ``LightCurve``
    instances, unwrapping Astropy ``Time``, ``Quantity``, and ``MaskedNDArray``
    columns while preserving full light curve length and flux units.
    """

    def __init__(
        self,
        lc: LightCurve,
        x: np.ndarray | str | None = None,
        y: np.ndarray | str | None = None,
        covariates: np.ndarray | DesignMatrix | None = None,
    ):
        if not HAS_LIGHTKURVE:
            raise ImportError(
                "K2SCCorrector requires 'lightkurve>=2.4'. "
                "Install it with `pip install 'k2sc[lightkurve]'`."
            )
        if LightCurve is not None and not isinstance(lc, LightCurve):
            raise TypeError(
                f"Expected a lightkurve.LightCurve instance, got {type(lc)!r}."
            )

        if _BaseCorrector is not object:
            with contextlib.suppress(Exception):
                super().__init__(lc)
        # Always retain the full-length unmodified LightCurve (never dropping NaN rows)
        self.original_lc = lc
        self.corrected_lc: Any | None = None
        self.cadence_mask = np.ones(len(lc), dtype=bool)

        self._time_arr = _to_float64_array(lc.time).ravel()
        self._flux_arr = _to_float64_array(lc.flux).ravel()
        n_pts = self._time_arr.size

        flux_err_col = _lookup_lc_column(lc, "flux_err")
        self._flux_err_arr = (
            _to_float64_array(flux_err_col).ravel()
            if flux_err_col is not None
            else np.full(n_pts, np.nan, dtype=np.float64)
        )
        qual_col = _lookup_lc_column(lc, "quality")
        self._quality_arr = _extract_quality_array(qual_col, n_pts)

        self._x_arr, self._y_arr, self._cov_arr, self.covariate_source = (
            self._resolve_lc_covariates(lc, x=x, y=y, covariates=covariates)
        )

        self.result: K2SCResult | None = None
        self.detrender = None
        self.tr_position = None
        self.tr_time = None
        self.pv: np.ndarray | None = None
        self.cdpp_r: float | None = None
        self.cdpp_t: float | None = None
        self.cdpp_c: float | None = None

    @staticmethod
    def _resolve_lc_covariates(
        lc: Any,
        x: np.ndarray | str | None = None,
        y: np.ndarray | str | None = None,
        covariates: Any | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
        """Resolve spatial covariates via the 4-stage Lightkurve fallback chain."""
        n_pts = len(lc)
        if covariates is not None and (x is not None or y is not None):
            raise ValueError("Pass either (`x`, `y`) or `covariates`, not both.")

        # 1. Explicit covariates or explicit (x, y)
        if covariates is not None:
            if (
                DesignMatrix is not None
                and isinstance(covariates, DesignMatrix)
                or hasattr(covariates, "values")
            ):
                cov_mat = np.asarray(covariates.values, dtype=np.float64)
            else:
                cov_mat = _to_float64_array(covariates)
            if cov_mat.ndim == 1:
                cov_mat = cov_mat[:, np.newaxis]
            if cov_mat.ndim != 2 or cov_mat.shape[0] != n_pts:
                raise ValueError(
                    f"Explicit covariates shape {cov_mat.shape} does not match LightCurve length {n_pts}."
                )
            return None, None, cov_mat, "covariates"

        if (x is None) != (y is None):
            raise ValueError("Both `x` and `y` must be provided together.")

        if x is not None and y is not None:
            if isinstance(x, str):
                col_x = _lookup_lc_column(lc, x)
                if col_x is None:
                    raise ValueError(f"Column {x!r} not found in LightCurve.")
                x_arr = _to_float64_array(col_x).ravel()
            else:
                x_arr = _to_float64_array(x).ravel()

            if isinstance(y, str):
                col_y = _lookup_lc_column(lc, y)
                if col_y is None:
                    raise ValueError(f"Column {y!r} not found in LightCurve.")
                y_arr = _to_float64_array(col_y).ravel()
            else:
                y_arr = _to_float64_array(y).ravel()

            if x_arr.size != n_pts or y_arr.size != n_pts:
                raise ValueError(
                    f"Explicit x/y lengths ({x_arr.size}, {y_arr.size}) != LightCurve length ({n_pts})."
                )
            return x_arr, y_arr, None, "explicit_xy"

        # 2. pos_corr1 / pos_corr2 (>50% finite and positive variance)
        pc1_col = _lookup_lc_column(lc, "pos_corr1")
        pc2_col = _lookup_lc_column(lc, "pos_corr2")
        if pc1_col is not None and pc2_col is not None:
            pc1 = _to_float64_array(pc1_col).ravel()
            pc2 = _to_float64_array(pc2_col).ravel()
            if _has_usable_pair(pc1, pc2):
                return pc1, pc2, None, "pos_corr"

        # 3. Median-centred centroid_col / centroid_row or mom_centr1 / mom_centr2
        for col1_name, col2_name, label in (
            ("centroid_col", "centroid_row", "centroid"),
            ("mom_centr1", "mom_centr2", "mom_centr"),
        ):
            c1_col = _lookup_lc_column(lc, col1_name)
            c2_col = _lookup_lc_column(lc, col2_name)
            if c1_col is not None and c2_col is not None:
                c1 = _to_float64_array(c1_col).ravel()
                c2 = _to_float64_array(c2_col).ravel()
                if _has_usable_pair(c1, c2):
                    c1_centred = c1 - float(np.nanmedian(c1))
                    c2_centred = c2 - float(np.nanmedian(c2))
                    return c1_centred, c2_centred, None, label

        # 4. Clear error guiding the caller
        raise ValueError(
            "No usable position covariates found in LightCurve (checked 'pos_corr1'/'pos_corr2', "
            "'centroid_col'/'centroid_row', and 'mom_centr1'/'mom_centr2' with >50% finite values "
            "and non-zero variance). Pass explicit `x=..., y=...` or `covariates=...`."
        )

    def correct(
        self,
        cadence_mask: np.ndarray | None = None,
        splits: Sequence[float] | np.ndarray | str | None = "auto",
        campaign: int | None = None,
        kernel: str | DtKernel | None = None,
        kernel_period: float | None = None,
        default_position_kernel: str = "SqrExp",
        roll_pca: bool = True,
        transit_mask: tuple[float, float, float] | np.ndarray | None = None,
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
        seed: int | np.random.Generator | None = 0,
        remove_time_trend: bool = False,
        quiet: bool = True,
        **optimizer_kwargs,
    ) -> LightCurve:
        """Fit the K2SC additive GP model and return a systematics-corrected LightCurve.

        Parameters
        ----------
        cadence_mask : np.ndarray of bool, optional
            Lightkurve convention mask where ``True`` marks valid cadences to include
            in GP training/fitting and ``False`` excludes cadences from fitting.
        transit_mask : tuple ``(t0, period, duration)`` or np.ndarray of bool, optional
            Either periodic transit ephemeris ``(t0, period, duration)`` in days, or
            a boolean array where ``True`` marks in-transit cadences to exclude from
            GP training (flagged with ``M_PERIODIC``).
        remove_time_trend : bool, optional
            If ``False`` (default), ``corrected_lc.flux`` removes only the position
            systematics ``tr_position - nanmedian(tr_position)`` (preserving stellar
            variability and transits). If ``True``, also removes ``tr_time - nanmedian(tr_time)``.
        """
        mission, eff_campaign = _infer_mission_and_campaign(
            self.original_lc, campaign
        )

        res = detrend_array(
            self._time_arr,
            self._flux_arr,
            x=self._x_arr,
            y=self._y_arr,
            covariates=self._cov_arr,
            flux_err=self._flux_err_arr,
            quality=self._quality_arr,
            cadence_mask=cadence_mask,
            transit_mask=transit_mask,
            p_mask_center=p_mask_center,
            p_mask_period=p_mask_period,
            p_mask_duration=p_mask_duration,
            mission=mission,
            campaign=eff_campaign,
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

        flux_unit = getattr(self.original_lc.flux, "unit", None)

        def _with_unit(arr: np.ndarray):
            return arr * flux_unit if flux_unit is not None else arr.copy()

        corrected_lc = self.original_lc.copy()
        target_flux = res.detrended_flux if remove_time_trend else res.corr_flux
        corrected_lc["flux"] = _with_unit(target_flux)
        corrected_lc["k2sc_tr_position"] = _with_unit(res.tr_position)
        corrected_lc["k2sc_tr_time"] = _with_unit(res.tr_time)
        corrected_lc["k2sc_corr_flux"] = _with_unit(res.corr_flux)
        corrected_lc["k2sc_detrended_flux"] = _with_unit(res.detrended_flux)
        corrected_lc["k2sc_mflags"] = np.asarray(res.mflags, dtype=np.uint8)

        meta_updates: dict[str, Any] = {
            "K2SC_KERNEL": str(res.detrender.kernel.name),
            "K2SC_PV": np.asarray(res.pv, dtype=np.float64).copy(),
            "K2SC_CDPP_R": float(res.cdpp_r),
            "K2SC_CDPP_T": float(res.cdpp_t),
            "K2SC_CDPP_C": float(res.cdpp_c),
            "K2SC_SPLITS": [float(s) for s in res.splits],
            "K2SC_SUCCESS": bool(res.success),
            "K2SC_WARN": bool(res.warn),
            "K2SC_PERIODIC": bool(res.is_periodic),
            "K2SC_COVARIATES": str(self.covariate_source),
            "K2SC_ROLL_PCA": bool(roll_pca),
        }
        if res.ls_period is not None:
            meta_updates["K2SC_LS_PERIOD"] = float(res.ls_period)
        if res.ls_fap is not None:
            meta_updates["K2SC_LS_FAP"] = float(res.ls_fap)
        corrected_lc.meta.update(meta_updates)

        res.lightcurve = corrected_lc
        with contextlib.suppress(Exception):
            object.__setattr__(corrected_lc, "k2sc_result", res)

        # Reconstruct full-length fit cadence mask
        valid_xy, _x_tmp, _y_tmp = _coerce_covariates(
            self._time_arr.size,
            x=self._x_arr,
            y=self._y_arr,
            covariates=self._cov_arr,
        )
        valid_coords = np.isfinite(self._time_arr) & valid_xy
        full_fit_mask = np.zeros(self._time_arr.size, dtype=bool)
        if res.detrender.data.mask.size == int(valid_coords.sum()):
            full_fit_mask[valid_coords] = res.detrender.data.mask
        full_fit_mask &= ~(res.mflags & M_PERIODIC).astype(bool)

        self.corrected_lc = corrected_lc
        self.result = res
        self.detrender = res.detrender
        self.tr_position = _with_unit(res.tr_position)
        self.tr_time = _with_unit(res.tr_time)
        self.pv = res.pv
        self.cdpp_r = res.cdpp_r
        self.cdpp_t = res.cdpp_t
        self.cdpp_c = res.cdpp_c
        self.cadence_mask = full_fit_mask
        return corrected_lc

    def diagnose(self, ax=None) -> np.ndarray:
        """Plot a 3-panel diagnostic summary of raw, position-corrected, and residual flux."""
        if self.corrected_lc is None or self.result is None:
            raise RuntimeError(
                "No corrected light curve available; call `.correct()` before `.diagnose()`."
            )

        import matplotlib.pyplot as plt

        if ax is None:
            _fig, axes_raw = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes = np.asarray(axes_raw, dtype=object).ravel()
        else:
            axes = np.asarray(ax, dtype=object).ravel()
            if axes.size != 3:
                raise ValueError(f"Expected 3 matplotlib Axes, got {axes.size}.")

        res = self.result
        t = self._time_arr
        f_raw = self._flux_arr
        fin_t = np.isfinite(t)
        out_idx = fin_t & np.isfinite(f_raw) & (~res.outlier_mask)

        pos_zero = res.tr_position - float(np.nanmedian(res.tr_position))
        total_model = res.tr_time + pos_zero

        # Panel 0: Raw flux + training set + outliers + full GP model
        axes[0].plot(t[fin_t], f_raw[fin_t], ".", color="0.65", ms=3, label="Raw flux")
        if res.detrender is not None:
            axes[0].plot(
                res.detrender.tr_data.unmasked_time,
                res.detrender.tr_data.unmasked_flux,
                "k.",
                ms=3.5,
                alpha=0.7,
                label="Training set",
            )
        if np.any(out_idx):
            axes[0].plot(
                t[out_idx],
                f_raw[out_idx],
                "rx",
                ms=5,
                label="Flagged outliers",
            )
        fin_mod = fin_t & np.isfinite(total_model)
        if np.any(fin_mod):
            order = np.argsort(t[fin_mod])
            axes[0].plot(
                t[fin_mod][order],
                total_model[fin_mod][order],
                color="C0",
                lw=1.2,
                label="GP (time + position)",
            )
        axes[0].set_ylabel("Raw flux")
        axes[0].legend(loc="best", fontsize=8)

        # Panel 1: Position-corrected flux + stellar time trend + splits
        fin_corr = fin_t & np.isfinite(res.corr_flux)
        axes[1].plot(
            t[fin_corr],
            res.corr_flux[fin_corr],
            ".",
            color="0.35",
            ms=3,
            label="Position-corrected",
        )
        fin_tt = fin_t & np.isfinite(res.tr_time)
        if np.any(fin_tt):
            order_t = np.argsort(t[fin_tt])
            axes[1].plot(
                t[fin_tt][order_t],
                res.tr_time[fin_tt][order_t],
                color="C1",
                lw=1.3,
                label="GP time trend",
            )
        for idx_s, s_val in enumerate(res.splits):
            axes[1].axvline(
                s_val,
                color="C3",
                ls="--",
                lw=0.9,
                alpha=0.7,
                label="Split" if idx_s == 0 else None,
            )
        axes[1].set_ylabel("Corrected flux")
        axes[1].legend(loc="best", fontsize=8)

        # Panel 2: Fully whitened residuals
        fin_det = fin_t & np.isfinite(res.detrended_flux)
        axes[2].plot(
            t[fin_det],
            res.detrended_flux[fin_det],
            ".",
            color="0.25",
            ms=3,
            label=f"Residual (CDPP={res.cdpp_c:.1f} ppm)",
        )
        axes[2].set_ylabel("Residual flux")
        axes[2].set_xlabel("Time [d]")
        axes[2].legend(loc="best", fontsize=8)

        return axes

    def diagnose_xy(self, ax=None) -> Axes:
        """Plot the 2D spatial systematics model ``tr_position(x, y)`` over centroid space."""
        if self.corrected_lc is None or self.result is None:
            raise RuntimeError(
                "No corrected light curve available; call `.correct()` before `.diagnose_xy()`."
            )

        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

        res = self.result
        _valid_xy, x_full, y_full = _coerce_covariates(
            self._time_arr.size,
            x=self._x_arr,
            y=self._y_arr,
            covariates=self._cov_arr,
        )
        pos_rel = res.tr_position - float(np.nanmedian(res.tr_position))
        fin = np.isfinite(x_full) & np.isfinite(y_full) & np.isfinite(pos_rel)

        mappable = None
        if int(fin.sum()) >= 4 and float(np.std(y_full[fin])) > 0.0:
            try:
                mappable = ax.tripcolor(x_full[fin], y_full[fin], pos_rel[fin], shading="flat")
            except (RuntimeError, ValueError):
                mappable = None
        if mappable is None and np.any(fin):
            mappable = ax.scatter(x_full[fin], y_full[fin], c=pos_rel[fin], s=16)

        if res.detrender is not None:
            tr_x = res.detrender.tr_data.mx
            tr_y = res.detrender.tr_data.my
            if res.pca_transform is not None:
                tr_x, tr_y = res.pca_transform.inverse_transform(tr_x, tr_y)
            ax.plot(
                tr_x,
                tr_y,
                "o",
                ms=3,
                color="k",
                mec="w",
                mew=0.4,
                label="Training set",
            )
        if mappable is not None and ax.figure is not None:
            ax.figure.colorbar(mappable, ax=ax, label="Position trend")

        ax.set_xlabel("x covariate")
        ax.set_ylabel("y covariate")
        ax.set_title("K2SC 2D Spatial Systematics")
        return ax


__all__ = ["HAS_LIGHTKURVE", "K2SCCorrector", "_to_float64_array"]
