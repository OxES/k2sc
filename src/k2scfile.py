"""Utility class to work with K2SC output FITS files."""

from __future__ import annotations

import astropy.io.fits as pf
import numpy as np

from .core import M_OUTLIER_D, M_OUTLIER_U, M_PERIODIC

_HLSP_EXTNAMES = frozenset({"PDC", "SAP"})


def submed(a):
    """Subtract the finite median of ``a`` without warnings on all-NaN arrays."""
    arr = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(arr)
    if not np.any(fin):
        return np.full_like(arr, np.nan, dtype=np.float64)
    return arr - float(np.median(arr[fin]))


def _submean(a) -> np.ndarray:
    """Subtract the finite mean of ``a`` without warnings on all-NaN arrays."""
    arr = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(arr)
    if not np.any(fin):
        return np.zeros_like(arr, dtype=np.float64)
    return arr - float(np.mean(arr[fin]))


class K2SCFile:
    """Utility class to work with K2SC output and MAST HLSP FITS files."""

    def __init__(
        self,
        filename,
        normalize: bool = False,
        remove_nans: bool = True,
        remove_outliers: bool = False,
        aperture: int = 1,
        ext: int | str = 1,
    ):
        """Initialize K2SCFile.

        Parameters
        ----------
        filename : str or Path
            K2SC output FITS file or ``k2mastify`` HLSP FITS file path.
        normalize : bool, optional
            Normalize the light curves by their finite median.
        remove_nans : bool, optional
            Remove points with either non-finite time or flux.
        remove_outliers : bool, optional
            Remove points marked as outliers by K2SC.
        aperture : int, optional
            1-based aperture index for multi-aperture (``flux_1..N``) files
            (default 1).
        ext : int or str, optional
            FITS table extension index or name (default 1, e.g. ``1``/``"PDC"``
            or ``2``/``"SAP"`` for ``k2mastify`` HLSP files).
        """
        fname_str = str(filename)
        self._fname = fname_str
        self._aperture = int(aperture)
        if self._aperture < 1:
            raise ValueError(f"aperture must be a positive integer (got {aperture})")

        with pf.open(fname_str) as hdul:
            self._phead = hdul[0].header.copy()
            hdu = hdul[ext]
            self._dhead = hdu.header.copy()
            self._data = hdu.data.copy()

        col_map = {c.lower(): c for c in self._data.columns.names}
        if "flux" in col_map:
            if self._aperture != 1:
                raise ValueError(
                    f"Single-aperture file {fname_str!r} only supports aperture=1 "
                    f"(got {self._aperture})"
                )
            sfx = ""
        else:
            sfx = f"_{self._aperture}"
            if f"flux{sfx}" not in col_map:
                avail = sorted(
                    int(c.split("_")[1])
                    for c in col_map
                    if c.startswith("flux_") and c.split("_")[1].isdigit()
                )
                raise KeyError(
                    f"Aperture {self._aperture} (column 'flux{sfx}') not found in "
                    f"{fname_str!r}; available apertures: {avail}"
                )

        self._time = np.asarray(self._data[col_map["time"]], dtype=np.float64)
        flux_col = np.asarray(self._data[col_map[f"flux{sfx}"]], dtype=np.float64)
        self._tposi = np.asarray(self._data[col_map[f"trposi{sfx}"]], dtype=np.float64)
        self._ttime = np.asarray(self._data[col_map[f"trtime{sfx}"]], dtype=np.float64)
        mflags = np.asarray(self._data[col_map[f"mflags{sfx}"]], dtype=np.uint8)

        extname = str(self._dhead.get("EXTNAME", "")).strip().upper()
        self._is_hlsp = extname in _HLSP_EXTNAMES

        if self._is_hlsp:
            # k2mastify HLSP files store already-detrended flux:
            # flux_hlsp = raw - (trposi - nanmean(trposi)) - (trtime - nanmean(trtime))
            self._flux_residual = flux_col
            self._flux_detrended = flux_col + _submean(self._ttime)
            self._flux = self._flux_detrended + _submean(self._tposi)
            self._minf = np.isfinite(flux_col) & np.isfinite(self._time)
        else:
            # Raw k2sc CLI output (EXTNAME='K2SC' or 'RESULTS') stores raw input flux
            self._flux = flux_col
            self._flux_detrended = self._flux - submed(self._tposi)
            self._flux_residual = self._flux_detrended - submed(self._ttime)
            self._minf = np.isfinite(self._flux) & np.isfinite(self._time)

        self._mper = ~(mflags & M_PERIODIC).astype(bool)
        self._ou = ~(mflags & M_OUTLIER_U).astype(bool)
        self._od = ~(mflags & M_OUTLIER_D).astype(bool)
        self._om = self._ou & self._od

        self._normalize = bool(normalize)
        self._remove_nans = bool(remove_nans)
        self._remove_outliers = bool(remove_outliers)

    def _n(self, a, remove_outliers=None):
        if self._normalize:
            fin = np.isfinite(a)
            med = float(np.median(a[fin])) if np.any(fin) else np.nan
            return self._clean(a / med, remove_outliers=remove_outliers)
        return self._clean(a, remove_outliers=remove_outliers)

    def _clean(self, a, remove_nans=None, remove_outliers=None):
        mask = np.ones(a.size, dtype=bool)
        remove_nans = remove_nans if remove_nans is not None else self._remove_nans
        remove_outliers = (
            remove_outliers if remove_outliers is not None else self._remove_outliers
        )
        if remove_nans:
            mask &= self._minf
        if remove_outliers:
            mask &= self._om
        return a[mask]

    def set_normalize(self, v):
        self._normalize = bool(v)

    def set_remove_nans(self, v):
        self._remove_nans = bool(v)

    def set_remove_outliers(self, v):
        self._remove_outliers = bool(v)

    @property
    def is_hlsp(self) -> bool:
        """True when the loaded extension is a ``k2mastify`` HLSP table (``PDC``/``SAP``)."""
        return self._is_hlsp

    @property
    def aperture(self) -> int:
        """1-based aperture index loaded from the FITS table."""
        return self._aperture

    @property
    def time(self):
        """Time array"""
        return self._clean(self._time)

    @property
    def original_flux(self):
        """Original flux (reconstructed from trends on ``k2mastify`` HLSP files)"""
        return self._n(self._flux)

    @property
    def detrended_flux(self):
        """Flux with the position trend removed"""
        return self._n(self._flux_detrended)

    @property
    def residual_flux(self):
        """Flux with both the position and time trends removed"""
        return self._n(self._flux_residual)

    @property
    def position_trend(self):
        """Position trend from K2SC"""
        return self._clean(self._tposi)

    @property
    def time_trend(self):
        """Time trend from K2SC"""
        return self._clean(self._ttime)

    @property
    def outlier_mask(self):
        """Mask that excludes points marked as outliers by K2SC"""
        return self._clean(self._om, remove_outliers=False)

    @property
    def outlier_mask_up(self):
        """Mask that excludes points marked as upwards outliers by K2SC"""
        return self._clean(self._ou, remove_outliers=False)

    @property
    def outlier_mask_down(self):
        """Mask that excludes points marked as downwards outliers by K2SC"""
        return self._clean(self._od, remove_outliers=False)

    @property
    def original_outliers(self):
        """Time and original flux datapoints marked as outliers by K2SC"""
        return (
            self._clean(self._time, remove_outliers=False)[~self.outlier_mask],
            self._n(self._flux, remove_outliers=False)[~self.outlier_mask],
        )

    @property
    def detrended_outliers(self):
        """Time and detrended flux datapoints marked as outliers by K2SC"""
        return (
            self._clean(self._time, remove_outliers=False)[~self.outlier_mask],
            self._n(self._flux_detrended, remove_outliers=False)[~self.outlier_mask],
        )

    @property
    def outliers_up(self):
        """Points marked as upwards outliers by K2SC"""
        return self._time[~self._ou], self._flux[~self._ou]

    @property
    def outliers_down(self):
        """Points marked as downwards outliers by K2SC"""
        return self._time[~self._od], self._flux[~self._od]
