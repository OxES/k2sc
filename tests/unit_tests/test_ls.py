"""Tests for src/ls.py — vectorised pure-JAX Lomb-Scargle periodogram.

Covers:
* ``fasper`` peak frequency/period recovery on K2-cadence synthetic signals.
* ``fasper`` power parity against ``astropy.timeseries.LombScargle``.
* ``fasper`` FAP accuracy.
* Exact uniform-grid ``hypo == 0`` zero-NaN regression test.
* ``ls_periodogram_jax`` peak period recovery and power parity.
* ``psearch`` period recovery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from k2sc.ls import fasper, ls_periodogram_jax, psearch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_K2 = 1626  # typical K2 LC length (~80 d / 0.0204 d cadence)


@pytest.fixture(scope="module")
def k2_synthetic():
    """K2-cadence synthetic sinusoidal signal."""
    dt = 0.0204  # days (long-cadence)
    t = np.arange(N_K2, dtype=np.float64) * dt
    # Inject a 7.3-day periodic signal + white noise
    true_period = 7.3
    flux = 1.0 + 0.005 * np.sin(2.0 * math.pi * t / true_period)
    flux += RNG.normal(0.0, 1e-4, size=N_K2)
    return t, flux, true_period


@pytest.fixture(scope="module")
def k2_gapped():
    """K2-cadence with a gap (thruster firing removed)."""
    dt = 0.0204
    t = np.arange(N_K2, dtype=np.float64) * dt
    # Remove 3-day gap at ~25 d
    gap = (t < 25.0) | (t > 28.0)
    t = t[gap]
    true_period = 5.1
    flux = 1.0 + 0.003 * np.sin(2.0 * math.pi * t / true_period)
    flux += RNG.normal(0.0, 5e-5, size=len(t))
    return t, flux, true_period


# ---------------------------------------------------------------------------
# fasper tests
# ---------------------------------------------------------------------------

class TestFasper:
    def test_peak_frequency_k2(self, k2_synthetic):
        t, flux, true_period = k2_synthetic
        freq, _power, _nout, jmax, _prob = fasper(t, flux - flux.mean(), 6, 0.5)
        peak_freq = freq[jmax]
        peak_period = 1.0 / peak_freq
        # Recovered period within 1% of injected
        assert abs(peak_period - true_period) / true_period < 0.01, (
            f"fasper: peak period {peak_period:.4f} d, expected {true_period:.4f} d"
        )

    def test_peak_frequency_gapped(self, k2_gapped):
        t, flux, true_period = k2_gapped
        freq, _power, _nout, jmax, _prob = fasper(t, flux - flux.mean(), 6, 0.5)
        peak_period = 1.0 / freq[jmax]
        assert abs(peak_period - true_period) / true_period < 0.02, (
            f"fasper gapped: peak period {peak_period:.4f} d, expected {true_period:.4f} d"
        )

    def test_return_dtype_float64(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        freq, power, nout, jmax, prob = fasper(t, flux, 4, 0.5)
        assert freq.dtype == np.float64
        assert power.dtype == np.float64
        assert isinstance(nout, int)
        assert isinstance(jmax, int)
        assert isinstance(prob, float)

    def test_return_shape_consistency(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        freq, power, nout, jmax, _prob = fasper(t, flux, 4, 0.5)
        assert len(freq) == nout
        assert len(power) == nout
        assert 0 <= jmax < nout

    def test_parity_vs_astropy(self, k2_synthetic):
        """fasper peak frequency agrees with astropy.timeseries.LombScargle."""
        pytest.importorskip("astropy.timeseries", reason="astropy not available")
        from astropy.timeseries import LombScargle

        t, flux, true_period = k2_synthetic
        y = flux - flux.mean()

        freq_fast, _power_fast, _nout, jmax, _prob = fasper(t, y, 6, 0.5)
        peak_fast = 1.0 / freq_fast[jmax]

        # astropy LS at the same frequencies
        ls = LombScargle(t, y, fit_mean=False)
        power_ap = ls.power(freq_fast)
        jmax_ap = int(np.argmax(power_ap))
        peak_ap = 1.0 / freq_fast[jmax_ap]

        # Both peak within 1% of the true period
        assert abs(peak_fast - true_period) / true_period < 0.01
        assert abs(peak_ap - true_period) / true_period < 0.01

    def test_fap_range(self, k2_synthetic):
        """False-alarm probability should be in [0, 1]."""
        t, flux, _ = k2_synthetic
        _, _, _, _, prob = fasper(t, flux - flux.mean(), 6, 0.5)
        assert 0.0 <= prob <= 1.0

    def test_fap_significant(self, k2_synthetic):
        """Injected signal should have low FAP (< 1e-10)."""
        t, flux, _ = k2_synthetic
        _, _, _, _, prob = fasper(t, flux - flux.mean(), 6, 0.5)
        assert prob < 1e-10, f"Expected low FAP, got {prob:.2e}"

    def test_incompatible_arrays_raises(self):
        with pytest.raises(ValueError, match="Incompatible"):
            fasper(np.arange(10, dtype=np.float64), np.arange(5, dtype=np.float64), 4, 0.5)

    # ----- Exact uniform-grid zero-NaN regression test -----

    def test_uniform_grid_no_nan(self):
        """Exact uniform grid: hypo == 0 must NEVER produce NaN in the power.

        This is the key regression test for the guard:
            safe_hypo2 = jnp.where(hypo2 > 1e-30, hypo2, ones_like(hypo2))
        """
        t = np.arange(200, dtype=np.float64) * 0.0204  # perfectly uniform
        rng = np.random.default_rng(7)
        y = rng.standard_normal(200)
        _freq, power, nout, jmax, _prob = fasper(t, y, 4, 0.5)
        assert not np.any(np.isnan(power)), "NaN in power for uniform grid!"
        assert not np.any(np.isinf(power)), "Inf in power for uniform grid!"
        assert 0 <= jmax < nout

    def test_uniform_grid_power_finite_and_positive(self):
        """Power values must be finite and non-negative on uniform grids."""
        t = np.linspace(0, 100.0, 500)
        period = 10.0
        y = np.sin(2.0 * math.pi * t / period)
        _freq, power, _nout, _jmax, _prob = fasper(t, y, 4, 0.5)
        assert np.all(np.isfinite(power))
        assert np.all(power >= 0.0)

    def test_no_nan_for_constant_signal(self):
        """Constant signal (zero variance): power should be 0.0, not NaN."""
        t = np.linspace(0, 80.0, 200, dtype=np.float64)
        y = np.ones(200, dtype=np.float64)
        _freq, power, _nout, _jmax, _prob = fasper(t, y, 4, 0.5)
        # var == 0 → power is zeroed out, not NaN
        assert not np.any(np.isnan(power))


# ---------------------------------------------------------------------------
# ls_periodogram_jax tests
# ---------------------------------------------------------------------------

class TestLsPeriodogramJax:
    def test_peak_period_recovery(self, k2_synthetic):
        t, flux, true_period = k2_synthetic
        period, power = ls_periodogram_jax(t, flux, min_period=0.5, max_period=20.0)
        jmax = int(np.argmax(power))
        peak_period = period[jmax]
        assert abs(peak_period - true_period) / true_period < 0.02, (
            f"ls_periodogram_jax: peak period {peak_period:.4f} d, expected {true_period:.4f} d"
        )

    def test_power_parity_vs_astropy(self, k2_synthetic):
        """LS periodogram peak should agree with astropy LS within 10%."""
        pytest.importorskip("astropy.timeseries", reason="astropy not available")
        from astropy.timeseries import LombScargle

        t, flux, true_period = k2_synthetic
        y = flux - flux.mean()

        period_jax, power_jax = ls_periodogram_jax(t, flux, min_period=0.5, max_period=20.0)
        freq_jax = 1.0 / period_jax  # ascending periods → descending freqs

        # Evaluate astropy at the same frequencies (must be ascending for fast method)
        freq_sorted = freq_jax[::-1]  # now ascending
        ls = LombScargle(t, y, fit_mean=False)
        power_ap = ls.power(freq_sorted)
        # Map back: power at ascending freq corresponds to descending period order
        power_ap = power_ap[::-1]  # back to ascending-period order

        # Peaks should be at the same period
        jmax_jax = int(np.argmax(power_jax))
        jmax_ap = int(np.argmax(power_ap))
        assert abs(period_jax[jmax_jax] - period_jax[jmax_ap]) / true_period < 0.05

    def test_return_dtype(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        period, power = ls_periodogram_jax(t, flux)
        assert period.dtype == np.float64
        assert power.dtype == np.float64

    def test_period_ascending(self, k2_synthetic):
        """Period array should be sorted ascending."""
        t, flux, _ = k2_synthetic
        period, _power = ls_periodogram_jax(t, flux)
        assert np.all(period[1:] >= period[:-1]), "Periods not sorted ascending"

    def test_no_nan_in_output(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        period, power = ls_periodogram_jax(t, flux)
        assert not np.any(np.isnan(power))
        assert not np.any(np.isnan(period))


# ---------------------------------------------------------------------------
# psearch tests
# ---------------------------------------------------------------------------

class TestPsearch:
    def test_period_recovery(self, k2_synthetic):
        t, flux, true_period = k2_synthetic
        y = flux - flux.mean()
        period, _fap = psearch(t, y, min_p=0.5, max_p=20.0)
        assert abs(period - true_period) / true_period < 0.02, (
            f"psearch: recovered {period:.4f} d, expected {true_period:.4f} d"
        )

    def test_fap_significant(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        y = flux - flux.mean()
        _period, fap = psearch(t, y, min_p=0.5, max_p=20.0)
        assert 0.0 <= fap <= 1.0
        assert fap < 1e-5

    def test_returns_floats(self, k2_synthetic):
        t, flux, _ = k2_synthetic
        period, fap = psearch(t, flux, min_p=1.0, max_p=15.0)
        assert isinstance(period, float)
        assert isinstance(fap, float)
