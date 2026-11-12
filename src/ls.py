"""Lomb-Scargle periodogram — vectorised pure-JAX implementation.

Two public functions are provided:

* ``fasper(x, y, ofac, hifac, MACC=4)`` — fast Press & Rybicki (1989) FFT
  periodogram.  Returns the legacy 5-tuple ``(wk1, wk2, nout, jmax, prob)``
  as float64 NumPy arrays for 100 % caller compatibility.  The scalar Python
  ``__spread__`` loop is replaced by a vectorised ``(N, MACC)`` Lagrange-weight
  extirpolation using ``jnp.zeros(...).at[idx].add(w)`` and ``jnp.fft.ifft``.
  The ``hypo == 0`` branch (exact uniform grids) is guarded with ``jnp.where``
  so it never produces ``0/0 = NaN``.

* ``ls_periodogram_jax(time, flux, min_period, max_period, samples_per_peak)``
  — exact Townsend (2010) / Scargle (1982) direct LS periodogram evaluated on
  ``[1/max_period, 1/min_period]``, chunked via ``jax.lax.map`` for memory
  efficiency.

* ``psearch(time, flux, min_p, max_p)`` — thin wrapper returning the
  peak ``(period, fap)`` using the ``fasper``-based analytical FAP.

References:
    Press, W. H. & Rybicki, G. B. 1989, ApJ, 338, 277.
    Townsend, R. H. D. 2010, ApJS, 191, 247.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from ._jax_env import ensure_float64

ensure_float64()


# ---------------------------------------------------------------------------
# Vectorised extirpolation (replaces scalar __spread__ loop)
# ---------------------------------------------------------------------------

def _make_extirpolate(m: int):
    """Return a JIT-compiled extirpolation function for a fixed MACC=m.

    ``m`` is a Python integer (static), so ``jnp.arange(m)`` inside the
    ``jax.lax.scan`` body is fully concrete at trace time.
    """
    # Build integer offsets [0, 1, ..., m-1] once at definition time
    k_range = jnp.arange(m, dtype=jnp.int32)
    k_float = jnp.arange(m, dtype=jnp.float64)

    def _spread_one(carry, args):
        """Scatter one data point into the work array using Lagrange weights."""
        work, ndim_int = carry
        y_val, x = args

        ilo = jnp.clip(
            jnp.floor(x - 0.5 * m + 1.0).astype(jnp.int32),
            1,
            ndim_int - m + 1,
        )

        # Indices [ilo, ilo+1, ..., ilo+m-1] in float64
        idx = ilo.astype(jnp.float64) + k_float  # shape (m,)

        # Overall product: fac = prod_{j=ilo}^{ilo+m-1} (x - j)
        fac_all = jnp.prod(x - idx)

        # Lagrange weight for support point k (0-indexed)
        def weight_k(k):
            diffs = idx[k] - idx          # shape (m,): 0 at k
            safe_diffs = jnp.where(k_range == k, jnp.ones(m, dtype=jnp.float64), diffs)
            nden = jnp.prod(safe_diffs)
            x_minus_xk = x - idx[k]
            is_exact = x_minus_xk == 0.0
            safe_xmxk = jnp.where(is_exact, 1.0, x_minus_xk)
            return jnp.where(is_exact, 1.0, fac_all / (nden * safe_xmxk))

        weights = jax.vmap(weight_k)(k_range)  # shape (m,)

        # 0-based scatter indices (mod ndim for safety)
        scatter_idx = ((ilo - 1) + k_range) % ndim_int
        new_work = work.at[scatter_idx].add(y_val * weights.astype(jnp.complex128))
        return (new_work, ndim_int), None

    @functools.partial(jax.jit, static_argnums=(2,))
    def extirpolate(y_arr, x_arr, ndim: int):
        """Extirpolate N data points into a complex work array of size ndim."""
        work0 = jnp.zeros(ndim, dtype=jnp.complex128)
        (work, _), _ = jax.lax.scan(
            _spread_one,
            (work0, ndim),
            (y_arr.astype(jnp.complex128), x_arr),
        )
        return work

    return extirpolate


# Cache one extirpolate function per MACC value
_EXTIRPOLATE_CACHE: dict[int, object] = {}


# ---------------------------------------------------------------------------
# fasper — public API
# ---------------------------------------------------------------------------

def fasper(
    x,
    y,
    ofac: float,
    hifac: float,
    MACC: int = 4,
):
    """Fast Lomb-Scargle periodogram (Press & Rybicki 1989).

    Vectorised pure-JAX implementation replacing the scalar Python
    ``__spread__`` loop.  Returns the legacy 5-tuple for 100 % caller
    compatibility.

    Parameters
    ----------
    x:
        Time array (unevenly spaced, length N).
    y:
        Flux / signal array, length N.
    ofac:
        Oversampling factor (typical value 4–6).
    hifac:
        Highest frequency as a multiple of the average Nyquist frequency.
    MACC:
        Number of Lagrange interpolation points (default 4).

    Returns
    -------
    wk1:
        Frequency array (float64 NumPy).
    wk2:
        Lomb normalised periodogram values (float64 NumPy).
    nout:
        Number of output frequencies.
    jmax:
        Index of the maximum periodogram value.
    prob:
        False-alarm probability of the highest peak.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n != len(y):
        raise ValueError("Incompatible arrays: len(x) != len(y)")

    nout = int(0.5 * ofac * hifac * n)
    nfreqt = int(ofac * hifac * n * MACC)
    nfreq = 64
    while nfreq < nfreqt:
        nfreq = 2 * nfreq
    ndim = int(2 * nfreq)

    ave = y.mean()
    var = float(((y - ave) ** 2).sum() / (n - 1))
    xmin = x.min()
    xmax = x.max()
    xdif = float(xmax - xmin)

    fac = ndim / (xdif * ofac)
    fndim = float(ndim)
    ck = ((x - xmin) * fac) % fndim
    ckk = (2.0 * ck) % fndim

    # Extirpolate y - ave into wk1, and 1.0 into wk2
    y_centered = jnp.array(y - ave, dtype=jnp.float64)
    ones_n = jnp.ones(n, dtype=jnp.float64)
    ck_jax = jnp.array(ck, dtype=jnp.float64)
    ckk_jax = jnp.array(ckk, dtype=jnp.float64)

    if MACC not in _EXTIRPOLATE_CACHE:
        _EXTIRPOLATE_CACHE[MACC] = _make_extirpolate(MACC)
    extirpolate_fn = _EXTIRPOLATE_CACHE[MACC]
    wk1_c = extirpolate_fn(y_centered, ck_jax, ndim)
    wk2_c = extirpolate_fn(ones_n, ckk_jax, ndim)

    # FFT-based convolution
    wk1_c = jnp.fft.ifft(wk1_c) * ndim
    wk2_c = jnp.fft.ifft(wk2_c) * ndim

    wk1_slice = wk1_c[1 : nout + 1]
    wk2_slice = wk2_c[1 : nout + 1]

    rwk1 = wk1_slice.real
    iwk1 = wk1_slice.imag
    rwk2 = wk2_slice.real
    iwk2 = wk2_slice.imag

    df = 1.0 / (xdif * ofac)

    # Compute Lomb value — guard hypo == 0 (exact uniform grid → NaN-free)
    hypo2 = 2.0 * jnp.abs(wk2_slice)

    # When hypo2 == 0 (perfectly uniform spacing), the Lomb time-offset tau
    # is degenerate; substitute 0 → cos(2 omega tau)=1, sin(2 omega tau)=0.
    safe_hypo2 = jnp.where(hypo2 > 1e-30, hypo2, jnp.ones_like(hypo2))
    hc2wt = jnp.where(hypo2 > 1e-30, rwk2 / safe_hypo2, jnp.zeros_like(rwk2))
    hs2wt = jnp.where(hypo2 > 1e-30, iwk2 / safe_hypo2, jnp.zeros_like(iwk2))

    cwt = jnp.sqrt(jnp.clip(0.5 + hc2wt, 0.0, None))
    swt = jnp.sign(hs2wt) * jnp.sqrt(jnp.clip(0.5 - hc2wt, 0.0, None))

    den = 0.5 * n + hc2wt * rwk2 + hs2wt * iwk2
    n_minus_den = n - den
    safe_den = jnp.where(den > 1e-30, den, jnp.ones_like(den))
    safe_nmd = jnp.where(jnp.abs(n_minus_den) > 1e-30, n_minus_den, jnp.ones_like(n_minus_den))

    cterm = (cwt * rwk1 + swt * iwk1) ** 2 / safe_den
    sterm = (cwt * iwk1 - swt * rwk1) ** 2 / safe_nmd

    freqs = df * (jnp.arange(nout, dtype=jnp.float64) + 1.0)
    if var > 0.0:
        power = (cterm + sterm) / (2.0 * var)
    else:
        power = jnp.zeros_like(cterm)

    pmax = float(jnp.max(power))
    jmax = int(jnp.argmax(power))

    # Analytical FAP (Press & Rybicki 1989)
    expy = math.exp(-pmax)
    effm = 2.0 * nout / ofac
    prob = effm * expy
    if prob > 0.01:
        prob = 1.0 - (1.0 - expy) ** effm

    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(power, dtype=np.float64),
        nout,
        jmax,
        float(prob),
    )


# ---------------------------------------------------------------------------
# Exact Townsend (2010) direct LS periodogram
# ---------------------------------------------------------------------------

def _ls_at_frequencies(
    time: jax.Array,
    flux_centered: jax.Array,
    freqs: jax.Array,
) -> jax.Array:
    """Evaluate the Lomb-Scargle power at an array of frequencies.

    Uses the Scargle (1982) normalised periodogram with the Townsend (2010)
    efficient trigonometric recurrence.

    Parameters
    ----------
    time:
        Time array, shape ``(N,)``.
    flux_centered:
        Mean-subtracted flux, shape ``(N,)``.
    freqs:
        Frequencies at which to evaluate, shape ``(F,)``.

    Returns
    -------
    power:
        LS power (not normalised by variance), shape ``(F,)``.
    """
    omega = 2.0 * jnp.pi * freqs  # (F,)
    phi = omega[:, None] * time[None, :]  # (F, N)

    # Time-offset tau (Scargle 1982 Eq. 5)
    s2 = jnp.sum(jnp.sin(2.0 * phi), axis=1)  # (F,)
    c2 = jnp.sum(jnp.cos(2.0 * phi), axis=1)  # (F,)
    tau = jnp.arctan2(s2, c2) / (2.0 * omega)  # (F,)

    phi_tau = phi - (omega * tau)[:, None]  # (F, N)
    cos_pt = jnp.cos(phi_tau)
    sin_pt = jnp.sin(phi_tau)

    yc = jnp.sum(flux_centered[None, :] * cos_pt, axis=1)  # (F,)
    ys = jnp.sum(flux_centered[None, :] * sin_pt, axis=1)  # (F,)
    cc = jnp.sum(cos_pt ** 2, axis=1)  # (F,)
    ss = jnp.sum(sin_pt ** 2, axis=1)  # (F,)

    safe_cc = jnp.where(cc > 1e-30, cc, jnp.ones_like(cc))
    safe_ss = jnp.where(ss > 1e-30, ss, jnp.ones_like(ss))

    return 0.5 * (yc ** 2 / safe_cc + ys ** 2 / safe_ss)


def ls_periodogram_jax(
    time,
    flux,
    min_period: float = 0.05,
    max_period: float = 25.0,
    samples_per_peak: int = 5,
    chunk_size: int = 256,
):
    """Exact Townsend (2010) Lomb-Scargle periodogram.

    Evaluates the Scargle (1982) normalised LS power on a uniform frequency
    grid spanning ``[1/max_period, 1/min_period]``, chunked in blocks of
    ``chunk_size`` frequencies for memory efficiency.

    Parameters
    ----------
    time:
        Time array (unevenly spaced).
    flux:
        Flux array.
    min_period:
        Minimum period to search (days).
    max_period:
        Maximum period to search (days).
    samples_per_peak:
        Frequency grid oversampling factor.
    chunk_size:
        Number of frequencies evaluated per ``jax.lax.map`` chunk.

    Returns
    -------
    period:
        Period array (days), sorted ascending.
    power:
        Lomb-Scargle power normalised by variance.
    """
    time = jnp.asarray(time, dtype=jnp.float64)
    flux = jnp.asarray(flux, dtype=jnp.float64)

    t_span = float(time.max() - time.min())
    f_min = 1.0 / float(max_period)
    f_max = 1.0 / float(min_period)
    n_freqs = max(1, int(samples_per_peak * t_span * (f_max - f_min)) + 1)
    freqs = jnp.linspace(f_min, f_max, n_freqs, dtype=jnp.float64)

    flux_centered = flux - jnp.mean(flux)
    var = float(jnp.var(flux_centered))

    # Pad to multiple of chunk_size for static shapes
    n_pad = chunk_size * ((n_freqs + chunk_size - 1) // chunk_size)
    if n_pad > n_freqs:
        freqs_pad = jnp.concatenate([
            freqs,
            jnp.full(n_pad - n_freqs, freqs[-1], dtype=jnp.float64),
        ])
    else:
        freqs_pad = freqs

    freqs_chunks = freqs_pad.reshape(-1, chunk_size)  # (n_chunks, chunk_size)

    @jax.jit
    def _eval_chunk(freq_chunk):
        return _ls_at_frequencies(time, flux_centered, freq_chunk)

    power_pad = jax.lax.map(_eval_chunk, freqs_chunks)  # (n_chunks, chunk_size)
    power_flat = power_pad.reshape(-1)[:n_freqs]

    if var > 0.0:
        power_norm = power_flat / var
    else:
        power_norm = power_flat

    period = jnp.where(freqs > 0.0, 1.0 / freqs, jnp.full_like(freqs, jnp.inf))

    return np.asarray(period[::-1], dtype=np.float64), np.asarray(power_norm[::-1], dtype=np.float64)


# ---------------------------------------------------------------------------
# psearch — matches standalone.py:22-39 analytical FAP formula
# ---------------------------------------------------------------------------

def psearch(
    time,
    flux,
    min_p: float,
    max_p: float,
) -> tuple[float, float]:
    """Find the most significant period in ``[min_p, max_p]``.

    Uses ``fasper`` with ``ofac=6`` and the analytical FAP from
    Press & Rybicki (1989), matching ``standalone.py:22-39``.

    Parameters
    ----------
    time:
        Time array.
    flux:
        Flux array.
    min_p:
        Minimum period (days).
    max_p:
        Maximum period (days).

    Returns
    -------
    period:
        Best-fit period (days).
    fap:
        False-alarm probability.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    freq, power, nout, _jmax, _prob = fasper(time, flux, 6, 0.5)
    period = 1.0 / freq
    m = (period > min_p) & (period < max_p)
    period_m, power_m = period[m], power[m]
    j = int(np.argmax(power_m))

    expy = math.exp(-float(power_m[j]))
    effm = 2 * nout / 6
    fap = expy * effm
    if fap > 0.01:
        fap = 1.0 - (1.0 - expy) ** effm

    return float(period_m[j]), float(fap)


# ---------------------------------------------------------------------------
# Legacy helper kept for backwards compatibility
# ---------------------------------------------------------------------------

def getSignificance(wk1, wk2, nout, ofac):
    """Return false-alarm probabilities for all periodogram peaks."""
    wk2 = np.asarray(wk2, dtype=np.float64)
    expy = np.exp(-wk2)
    effm = 2.0 * nout / ofac
    sig = effm * expy
    ind = sig > 0.01
    sig[ind] = 1.0 - (1.0 - expy[ind]) ** effm
    return sig
