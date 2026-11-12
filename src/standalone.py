"""Backward-compatibility standalone entry points and deprecated ``k2sc_lc`` shim."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .corrector import (
    HAS_LIGHTKURVE,
    K2SCCorrector,
    KeplerLightCurve,
    _extract_quality_array,
    _lookup_lc_column,
    _to_float64_array,
)
from .k2data import K2Data
from .ls import psearch
from .pipeline import K2SCResult, SCResult, detrend

_BaseLC = KeplerLightCurve if (HAS_LIGHTKURVE and KeplerLightCurve is not None) else object


class k2sc_lc(_BaseLC):  # type: ignore[misc,valid-type]
    """Deprecated Lightkurve subclass shim retained for legacy notebook compatibility.

    Use ``K2SCCorrector(lc).correct()`` or ``k2sc.detrend(lc)`` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "`k2sc_lc` is deprecated and will be removed in a future release; "
            "use `k2sc.K2SCCorrector(lc).correct()` or `k2sc.detrend(lc)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not HAS_LIGHTKURVE:
            raise ImportError(
                "`k2sc_lc` requires 'lightkurve>=2.4'. Install it via `pip install 'k2sc[lightkurve]'`."
            )
        super().__init__(*args, **kwargs)

    def get_k2data(self) -> K2Data:
        """Construct a ``K2Data`` instance from this LightCurve (deprecated)."""
        warnings.warn(
            "`k2sc_lc.get_k2data()` is deprecated; use `k2sc.K2SCCorrector(lc)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        time_arr = _to_float64_array(self.time).ravel()
        n_pts = time_arr.size

        cad_col = _lookup_lc_column(self, "cadenceno")
        cadence = (
            _extract_quality_array(cad_col, n_pts)
            if cad_col is not None
            else np.arange(n_pts, dtype=np.int32)
        )
        qual_col = _lookup_lc_column(self, "quality")
        quality = _extract_quality_array(qual_col, n_pts)

        flux_arr = _to_float64_array(self.flux).ravel()
        err_col = _lookup_lc_column(self, "flux_err")
        err_arr = (
            _to_float64_array(err_col).ravel()
            if err_col is not None
            else np.zeros_like(flux_arr)
        )

        x_arr, y_arr, cov_arr, _src = K2SCCorrector._resolve_lc_covariates(self)
        if cov_arr is not None:
            x_arr = cov_arr[:, 0]
            y_arr = cov_arr[:, 1] if cov_arr.shape[1] > 1 else np.zeros_like(x_arr)
        assert x_arr is not None and y_arr is not None

        target_id = getattr(self, "targetid", None)
        if target_id is None:
            target_id = (getattr(self, "meta", {}) or {}).get("TARGETID", 0)
        try:
            epic = int(target_id)
        except (TypeError, ValueError):
            epic = 0

        campaign = getattr(self, "campaign", None)
        if campaign is None:
            campaign = (getattr(self, "meta", {}) or {}).get("CAMPAIGN", None)

        primary_header = getattr(self, "primary_header", None)
        data_header = getattr(self, "data_header", None)

        return K2Data(
            epic,
            time=time_arr,
            cadence=cadence,
            quality=quality,
            fluxes=flux_arr,
            errors=err_arr,
            x=x_arr,
            y=y_arr,
            primary_header=primary_header,
            data_header=data_header,
            campaign=campaign,
        )

    def k2sc(self, **kwargs: Any):
        """Detrend in-place by delegating to ``K2SCCorrector`` (deprecated)."""
        warnings.warn(
            "`lc.k2sc()` / `k2sc_lc` monkeypatching is deprecated; "
            "use `k2sc.K2SCCorrector(lc).correct()` or `k2sc.detrend(lc)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not HAS_LIGHTKURVE:
            raise ImportError(
                "`k2sc_lc` requires 'lightkurve>=2.4'. Install it via `pip install 'k2sc[lightkurve]'`."
            )

        kw = dict(kwargs)
        kw.pop("save_dir", None)
        kw.pop("flux_type", None)
        x = kw.pop("x", None)
        y = kw.pop("y", None)
        covariates = kw.pop("covariates", None)

        corr = K2SCCorrector(self, x=x, y=y, covariates=covariates)
        clc = corr.correct(**kw)
        assert corr.result is not None

        for attr_name, attr_val in (
            ("detrender", corr.detrender),
            ("tr_position", corr.tr_position),
            ("tr_time", corr.tr_time),
            ("pv", corr.pv),
            ("corr_flux", clc.flux),
            ("cdpp_r", corr.cdpp_r),
            ("cdpp_t", corr.cdpp_t),
            ("cdpp_c", corr.cdpp_c),
            ("k2sc_result", corr.result),
        ):
            object.__setattr__(self, attr_name, attr_val)
        return clc


__all__ = ["K2SCResult", "SCResult", "detrend", "k2sc_lc", "psearch"]
