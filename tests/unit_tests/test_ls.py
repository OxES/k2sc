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
            fasper(
                np.arange(10, dtype=np.float64), np.arange(5, dtype=np.float64), 4, 0.5
            )

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

    def test_uniform_grid_hypo2_zero_power_parity_vs_astropy(self):
        """Uniform grid frequencies where hypo2 == 0 or hs2wt == 0 match Astropy LS.

        Regression test for M1: previously hc2wt fell back to 0.0 (not +0.5) when
        hypo2 <= 1e-30 and three-valued jnp.sign(0.0) == 0.0 broke c^2 + s^2 = 1,
        halving LS power at exact uniform-grid frequencies.
        """
        pytest.importorskip("astropy.timeseries", reason="astropy not available")
        from astropy.timeseries import LombScargle

        n = 100
        dt = 0.0204
        t = np.arange(n, dtype=np.float64) * dt
        rng = np.random.default_rng(314)
        y = 0.8 * np.sin(2.0 * math.pi * t / 0.25) + 0.5 * rng.standard_normal(n)

        freqs, power_fast, nout, _jmax, _prob = fasper(t, y, ofac=4, hifac=0.5)
        assert nout == 100

        # Verify at least one evaluated frequency actually hits hypo2 <= 1e-12
        # (at k = n - 1 = 99 -> index 98, 2 * f * n * dt = 50 is an integer not divisible by n)
        c2 = np.sum(np.cos(4.0 * np.pi * freqs[:, None] * t[None, :]), axis=1)
        s2 = np.sum(np.sin(4.0 * np.pi * freqs[:, None] * t[None, :]), axis=1)
        hypo2_exact = 2.0 * np.hypot(c2, s2)
        zero_mask = hypo2_exact < 1e-10
        assert np.any(zero_mask), (
            "Expected at least one exact hypo2 == 0 grid frequency"
        )

        ls = LombScargle(t, y, center_data=True, fit_mean=False)
        power_ap = ls.power(freqs, normalization="standard") * ((n - 1) / 2.0)

        # At hypo2 == 0 points, fasper must match Astropy closely (never 0.5x!)
        np.testing.assert_allclose(
            power_fast[zero_mask], power_ap[zero_mask], rtol=0.05, atol=1e-4
        )
        # And across all frequencies with meaningful power
        sig_mask = power_ap > 0.1
        np.testing.assert_allclose(
            power_fast[sig_mask], power_ap[sig_mask], rtol=0.05, atol=1e-3
        )

    def test_fasper_edge_case_guards(self):
        """fasper must raise clear ValueError on degenerate inputs."""
        with pytest.raises(ValueError, match="at least 2"):
            fasper(np.array([1.0]), np.array([2.0]), 4, 0.5)
        with pytest.raises(ValueError, match="at least 2"):
            fasper(np.array([]), np.array([]), 4, 0.5)
        with pytest.raises(ValueError, match="positive baseline"):
            fasper(np.ones(10), np.arange(10, dtype=np.float64), 4, 0.5)
        with pytest.raises(ValueError, match="nout < 1"):
            fasper(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0.2, 0.2)
        with pytest.raises(ValueError, match="positive"):
            fasper(
                np.arange(10, dtype=np.float64),
                np.arange(10, dtype=np.float64),
                0.0,
                0.5,
            )


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

        period_jax, power_jax = ls_periodogram_jax(
            t, flux, min_period=0.5, max_period=20.0
        )
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

    def test_high_bjd_shift_invariance(self, k2_synthetic):
        """High-BJD timestamps must be shift-invariant via t - t[0] recentring."""
        t, flux, _ = k2_synthetic
        t_bjd = 2458000.0 + t
        p_bjd, pow_bjd = ls_periodogram_jax(
            t_bjd, flux, min_period=0.5, max_period=20.0
        )
        p_shift, pow_shift = ls_periodogram_jax(
            t_bjd - t_bjd[0], flux, min_period=0.5, max_period=20.0
        )
        _p_rel, pow_rel = ls_periodogram_jax(t, flux, min_period=0.5, max_period=20.0)
        # Exact shift invariance wrt t_bjd - t_bjd[0]
        np.testing.assert_allclose(p_bjd, p_shift, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(pow_bjd, pow_shift, rtol=1e-13, atol=1e-13)
        # Matches unquantized zero-origin timestamps within float64 BJD input quantization
        np.testing.assert_allclose(pow_bjd, pow_rel, rtol=1e-6, atol=1e-8)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="at least 2"):
            ls_periodogram_jax(np.array([1.0]), np.array([1.0]))
        with pytest.raises(ValueError, match="min_period"):
            ls_periodogram_jax(
                np.arange(10.0), np.arange(10.0), min_period=5.0, max_period=2.0
            )
        with pytest.raises(ValueError, match="positive baseline"):
            ls_periodogram_jax(np.ones(10, dtype=np.float64), np.arange(10.0))
        t_nan = np.arange(10.0)
        t_nan[3] = np.nan
        with pytest.raises(ValueError, match="positive baseline"):
            ls_periodogram_jax(t_nan, np.arange(10.0))


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

    def test_empty_period_window_returns_nan_and_unit_fap(self, k2_synthetic):
        """When no evaluated frequencies fall in (min_p, max_p), return (nan, 1.0)."""
        t, flux, _ = k2_synthetic
        period, fap = psearch(t[:100], flux[:100], min_p=500.0, max_p=1000.0)
        assert np.isnan(period)
        assert fap == 1.0

        period_inv, fap_inv = psearch(t[:100], flux[:100], min_p=10.0, max_p=5.0)
        assert np.isnan(period_inv)
        assert fap_inv == 1.0
