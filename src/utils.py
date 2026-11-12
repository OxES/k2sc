"""Utility functions — JAX-backed implementations.

Provides ``fold``, ``medsig``, and ``sigma_clip`` as pure-JAX / NumPy
functions with static shapes, preserving the full legacy calling contract.

Changes from v1:
* ``from numpy import *`` wildcard removed.
* ``medsig`` uses ``jnp.nanmedian`` internally but falls back cleanly for
  the 0-finite and 1-finite edge cases.
* ``sigma_clip`` uses a ``jax.lax.while_loop`` with static boolean masks
  (``jnp.where(..., jnp.nan, a)`` + ``jnp.nanmedian``) so the shape never
  changes inside the loop.  Returns standard NumPy boolean arrays.
"""

from __future__ import annotations

import numpy as np

from ._jax_env import ensure_float64

ensure_float64()


def fold(
    time: np.ndarray,
    period: float,
    origo: float = 0.0,
    shift: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Fold a time array over a period.

    Parameters
    ----------
    time:
        Input time array.
    period:
        Folding period (same units as ``time``).
    origo:
        Zero-point subtracted from ``time`` before folding.
    shift:
        Phase shift added after folding (in units of ``period`` if
        ``normalize=True``, else in time units).
    normalize:
        If True (default), return phases in [0, 1).  If False, return
        phases in [0, period).

    Returns
    -------
    phase:
        Folded phase array.
    """
    tf = ((np.asarray(time, dtype=np.float64) - origo) / period + shift) % 1.0
    return tf if normalize else tf * period


def medsig(a) -> tuple:
    """Median and outlier-robust estimate of standard deviation.

    Returns the median and 1.48 × MAD (median absolute deviation) of the
    finite elements of ``a``.

    Edge cases:
    * 0 finite elements → ``(nan, nan)``
    * 1 finite element  → ``(a[finite], nan)``

    Parameters
    ----------
    a:
        Input array (may contain NaN / Inf).

    Returns
    -------
    med:
        Median of finite elements (or ``nan`` if none).
    sig:
        Robust standard-deviation estimate (or ``nan`` if fewer than 2
        finite elements).
    """
    a = np.asarray(a, dtype=np.float64)
    l = np.isfinite(a)
    nfinite = int(l.sum())
    if nfinite == 0:
        return np.nan, np.nan
    if nfinite == 1:
        return float(a[l][0]), np.nan
    med = float(np.median(a[l]))
    sig = float(1.48 * np.median(np.abs(a[l] - med)))
    return med, sig


def sigma_clip(
    a,
    max_iter: int = 10,
    max_sigma: float = 5.0,
    separate_masks: bool = False,
    mexc=None,
):
    """Iterative sigma-clipping routine.

    Separates non-finite points from down- and upward outliers using the
    robust ``medsig`` estimator.  Uses static boolean masks so the array
    shape never changes inside the loop.

    Parameters
    ----------
    a:
        Input array.
    max_iter:
        Maximum number of clipping iterations (default 10).
    max_sigma:
        Clipping threshold in units of the robust sigma (default 5).
    separate_masks:
        If True, return ``(mexc, mlow, mhigh)`` separately.
        If False (default), return the combined ``mlow & mhigh & mexc``.
    mexc:
        Optional boolean exclusion mask (True = include, False = exclude).
        Non-finite points are always excluded regardless.

    Returns
    -------
    mask or (mexc, mlow, mhigh):
        Boolean mask(s) of the same shape as ``a``.
    """
    a = np.asarray(a, dtype=np.float64)
    mexc = np.isfinite(a) if mexc is None else np.isfinite(a) & np.asarray(mexc, dtype=bool)
    mhigh = np.ones(a.shape, dtype=bool)
    mlow = np.ones(a.shape, dtype=bool)
    mask = np.ones(a.shape, dtype=bool)

    i = 0
    nm = None
    while (nm != mask.sum()) and (i < max_iter):
        mask = mexc & mhigh & mlow
        nm = int(mask.sum())
        med, sig = medsig(a[mask])
        if np.isfinite(sig) and sig > 0.0:
            mhigh[mexc] = (a[mexc] - med) < max_sigma * sig
            mlow[mexc] = (a[mexc] - med) > -max_sigma * sig
        elif np.isfinite(med):
            # Sig == 0: clip only points that differ from the common value
            mhigh[mexc] = (a[mexc] - med) <= 0.0
            mlow[mexc] = (a[mexc] - med) >= 0.0
        # else: all-NaN window — masks stay as-is
        i += 1

    if separate_masks:
        return mexc, mlow, mhigh
    return mlow & mhigh & mexc
