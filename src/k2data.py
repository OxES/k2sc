"""Contains the K2Data class for working with K2 datasets with 1..n flux and error
arrays (using different aperture size, etc.).
"""

from __future__ import annotations

import numpy as np

from .core import M_PERIODIC, M_QUALITY
from .utils import fold


def _safe_int32_array(arr, fill_value: int = 0) -> np.ndarray:
    """Convert an array to int32, replacing any non-finite floating values with fill_value."""
    raw = np.asarray(arr)
    if np.issubdtype(raw.dtype, np.floating):
        return np.where(np.isfinite(raw), raw, fill_value).astype(np.int32)
    return raw.astype(np.int32, copy=True)


class K2Data:
    """Encapsulates the K2 data.

    Encapsulates the K2 data with 1..n flux and flux uncertainty arrays. Separate
    flux and uncertainty arrays can correspond to different aperture sizes, etc.

    Notes
    -----
    We remove the cadences where either time, x, or y is nan since input nans screw
    up the GP fitting and reduction. This must be accounted for when saving the data.

    Parameters
    ----------
    epic    : int
              EPIC number of the star
    time    : array_like
              ndarray or list containing the time values
    cadence : array_like
              ndarray or list containing the cadence values
    quality : array_like
              ndarray or list containing the quality values
    fluxes  : array_like
              a list, multiple lists, or [nsets,npoints] ndarray of flux values
    errors  : array_like
              a list, multiple lists, or [nsets,npoints] ndarray of flux values
    x       : array_like
              ndarray or list of x values
    y       : array_like
              ndarray or list of y values

    Attributes
    ----------
    nsets       : int
                  Number of datasets
    npoints     : int
                  Number of datapoints
    is_periodic : bool
                  Does the flux show clear periodic variability
    ls_period   : float
                  Period of the strongest periodic variability detected
    ls_power    : float
                  Lomb-Scargle power of the strongest periodic variability detected
    """

    def __init__(
        self,
        epic,
        time,
        cadence,
        quality,
        fluxes,
        errors,
        x,
        y,
        primary_header=None,
        data_header=None,
        campaign=None,
    ):
        self.epic = epic
        self.campaign = campaign

        self.raw_time = np.asarray(time, dtype=np.float64).copy()
        self.raw_cadence = _safe_int32_array(cadence, fill_value=-1)
        self.raw_quality = _safe_int32_array(quality, fill_value=0)
        self.raw_fluxes = np.atleast_2d(np.asarray(fluxes, dtype=np.float64)).copy()
        self.raw_errors = np.atleast_2d(np.asarray(errors, dtype=np.float64)).copy()
        self.raw_x = np.asarray(x, dtype=np.float64).copy()
        self.raw_y = np.asarray(y, dtype=np.float64).copy()

        self.nanmask = nm = (
            np.isfinite(self.raw_time) & np.isfinite(self.raw_x) & np.isfinite(self.raw_y)
        )
        self.time = np.extract(nm, self.raw_time)
        self.cadence = np.extract(nm, self.raw_cadence).astype(np.int32)
        self.quality = np.extract(nm, self.raw_quality).astype(np.int32)
        self.fluxes = self.raw_fluxes[:, nm]
        self.errors = self.raw_errors[:, nm]
        self.x = np.extract(nm, self.raw_x)
        self.y = np.extract(nm, self.raw_y)
        self.primary_header = primary_header
        self.data_header = data_header

        self.nsets = self.fluxes.shape[0]
        self.npoints = self.fluxes.shape[1]

        self.is_periodic = False
        self.ls_period = None
        self.ls_power = None
        self.ls_fap = None

        qmask = np.all(np.isfinite(self.fluxes), axis=0) & (self.quality == 0)
        self.mflags = np.zeros((self.nsets, self.npoints), dtype=np.uint8)
        self.mflags[:, ~qmask] |= M_QUALITY

    def mask_periodic_signal(self, center, period, duration):
        folded = np.asarray(fold(self.time, period, center, shift=0.5), dtype=np.float64)
        self.pmask = np.abs(folded - 0.5) * period > 0.5 * duration
        self.mflags[:, ~self.pmask] |= M_PERIODIC
