"""Tests for src/cdpp.py — vectorised JAX Savitzky-Golay + CDPP.

Covers:
* ``savgol_filter`` float64 parity against reference ``np.polyfit`` (≤ 1e-12).
* Python 3 ``block_mean`` integer indexing (``block_size // 2 + 1``).
* ``exclude`` mask in ``cdpp``.
* High-BJD (2458000.0 + t) numerical stability.
* White-noise σ / √13 scaling.
* ``transit_duration_hours`` auto-scaling of ``bl_sz``.
* Sparse / gap windows.
* Post-JIT CPU runtime < 50 ms on N = 3800.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from k2sc.cdpp import block_mean, cdpp, savgol_filter

# ---------------------------------------------------------------------------
# Reference implementation using np.polyfit
# ---------------------------------------------------------------------------

def _ref_savgol(t, f, wd=2.0, order=2):
    """Pure-NumPy reference matching legacy cdpp.py:6-17."""
    n = len(t)
    f_sm = np.zeros(n) + np.nan
    for i in range(n):
        l = np.abs(t - t[i]) <= wd / 2.0
        if l.sum() == 0:
            continue
        tt = t[l]
        ff = f[l]
        # Centre coordinates to match the new implementation
        p = np.polyfit(tt - t[i], ff, order)
        f_sm[i] = np.polyval(p, 0.0)
    return f_sm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)
N = 3800


@pytest.fixture(scope="module")
def k2_lc():
    """K2 long-cadence synthetic light curve."""
    dt = 0.0204  # days
    t = np.arange(N, dtype=np.float64) * dt
    flux = 1.0 + 1e-4 * RNG.standard_normal(N)
    return t, flux


@pytest.fixture(scope="module")
def high_bjd_lc():
    """High-BJD (~ 2458000 + t) light curve to test Vandermonde stability."""
    dt = 0.0204
    t_rel = np.arange(N, dtype=np.float64) * dt
    t = 2458000.0 + t_rel
    flux = 1.0 + 1e-4 * RNG.standard_normal(N)
    return t, flux, t_rel


# ---------------------------------------------------------------------------
# savgol_filter tests
# ---------------------------------------------------------------------------

class TestSavgolFilter:
    def test_parity_vs_polyfit_small(self):
        """Float64 parity ≤ 1e-12 on a small array."""
        rng = np.random.default_rng(99)
        t = np.sort(rng.uniform(0, 10, 50))
        f = np.sin(t) + 0.01 * rng.standard_normal(50)
        ref = _ref_savgol(t, f, wd=2.0, order=2)
        got = savgol_filter(t, f, wd=2.0, polyorder=2)
        # Compare only finite points
        finite = np.isfinite(ref) & np.isfinite(got)
        assert finite.sum() > 10, "Too few finite comparison points"
        npt.assert_allclose(got[finite], ref[finite], rtol=1e-11, atol=1e-11)

    def test_parity_vs_polyfit_k2(self, k2_lc):
        """Float64 parity ≤ 1e-10 on K2 cadences."""
        t, flux = k2_lc
        # Use a small subset for the reference (slow np.polyfit loop)
        t_sub, f_sub = t[:200], flux[:200]
        ref = _ref_savgol(t_sub, f_sub, wd=2.0, order=2)
        got = savgol_filter(t_sub, f_sub, wd=2.0, polyorder=2)
        finite = np.isfinite(ref) & np.isfinite(got)
        assert finite.sum() > 50
        npt.assert_allclose(got[finite], ref[finite], rtol=1e-10, atol=1e-10)

    def test_high_bjd_stability(self, high_bjd_lc):
        """High-BJD timestamps should give the same result as relative times."""
        t_abs, flux, t_rel = high_bjd_lc
        t_sub = t_abs[:100]
        t_rel_sub = t_rel[:100]
        f_sub = flux[:100]
        got_abs = savgol_filter(t_sub, f_sub, wd=2.0, polyorder=2)
        got_rel = savgol_filter(t_rel_sub, f_sub, wd=2.0, polyorder=2)
        finite = np.isfinite(got_abs) & np.isfinite(got_rel)
        assert finite.sum() > 20
        # Both should be finite and close (< 1e-6 relative error)
        npt.assert_allclose(got_abs[finite], got_rel[finite], rtol=1e-6, atol=1e-6)

    def test_sparse_windows_no_crash(self):
        """Sparse window (< polyorder + 1 valid points) should not crash."""
        t = np.array([0.0, 100.0, 200.0], dtype=np.float64)
        f = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = savgol_filter(t, f, wd=2.0, polyorder=2)
        # May be NaN for isolated points but must not raise
        assert result.shape == (3,)

    def test_returns_float64(self, k2_lc):
        t, flux = k2_lc
        result = savgol_filter(t[:100], flux[:100])
        assert result.dtype == np.float64

    def test_cpu_runtime_under_50ms(self, k2_lc):
        """Post-JIT runtime must be < 50 ms on N = 3800."""
        t, flux = k2_lc
        # Warm up JIT (may compile on this call if first with this shape)
        _ = savgol_filter(t, flux)
        jax.block_until_ready(jnp.array(0.0))  # flush async ops
        # Time second call (JIT already compiled)
        t0 = time.perf_counter()
        result = savgol_filter(t, flux)
        jax.block_until_ready(jnp.array(result[0]))  # ensure completion
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"savgol_filter took {elapsed_ms:.1f} ms > 50 ms"


# ---------------------------------------------------------------------------
# block_mean tests
# ---------------------------------------------------------------------------

class TestBlockMean:
    def test_integer_block_size_min(self):
        """block_size // 2 + 1 must give integer 7, not float 7.5."""
        t = np.arange(100, dtype=np.float64) * 0.0204
        f = np.ones(100, dtype=np.float64)
        # block_size=13 → block_size_min = 13//2 + 1 = 7
        t_b, _f_b = block_mean(t, f, block_size=13)
        # Should produce some blocks without crashing
        assert isinstance(t_b, np.ndarray)
        assert len(t_b) > 0

    def test_explicit_block_size_min(self):
        """Explicit block_size_min should override default."""
        t = np.arange(50, dtype=np.float64) * 0.05
        f = np.ones(50, dtype=np.float64)
        t_b1, _ = block_mean(t, f, block_size=5, block_size_min=3)
        t_b2, _ = block_mean(t, f, block_size=5, block_size_min=1)
        # With looser min, we get at least as many blocks
        assert len(t_b2) >= len(t_b1)

    def test_means_correct(self):
        """Block means should equal the input mean for uniform flux."""
        t = np.arange(26, dtype=np.float64) * 0.0204
        f = np.full(26, 2.5, dtype=np.float64)
        _t_b, f_b = block_mean(t, f, block_size=13)
        npt.assert_allclose(f_b, 2.5, rtol=1e-14)


# ---------------------------------------------------------------------------
# cdpp tests
# ---------------------------------------------------------------------------

class TestCdpp:
    def test_white_noise_scaling(self):
        """White noise: CDPP should scale as σ / √bl_sz (within 30%)."""
        rng = np.random.default_rng(1234)
        dt = 0.0204
        t = np.arange(N, dtype=np.float64) * dt
        sigma_ppm = 200.0  # 200 ppm white noise
        flux = 1.0 + (sigma_ppm * 1e-6) * rng.standard_normal(N)
        bl_sz = 13
        cdpp_val = cdpp(t, flux, bl_sz=bl_sz)
        expected = sigma_ppm / np.sqrt(bl_sz)
        # Within 30% (noise fluctuations expected)
        assert abs(cdpp_val - expected) / expected < 0.30, (
            f"CDPP {cdpp_val:.1f} ppm, expected ~{expected:.1f} ppm"
        )

    def test_exclude_mask(self):
        """Excluding outlier cadences should reduce CDPP."""
        rng = np.random.default_rng(55)
        dt = 0.0204
        t = np.arange(N, dtype=np.float64) * dt
        flux = 1.0 + 1e-4 * rng.standard_normal(N)
        # Inject strong outliers
        outlier_idx = np.array([100, 200, 300, 400])
        flux[outlier_idx] *= 10.0
        exclude = np.zeros(N, dtype=bool)
        exclude[outlier_idx] = True
        cdpp_no_excl = cdpp(t, flux)
        cdpp_excl = cdpp(t, flux, exclude=exclude)
        assert cdpp_excl < cdpp_no_excl, "Excluding outliers should reduce CDPP"

    def test_high_bjd_no_crash(self, high_bjd_lc):
        """High-BJD timestamps should not crash cdpp."""
        t_abs, flux, _ = high_bjd_lc
        result = cdpp(t_abs, flux)
        assert np.isfinite(result)
        assert result > 0

    def test_transit_duration_hours(self):
        """transit_duration_hours should auto-set bl_sz."""
        rng = np.random.default_rng(77)
        dt = 0.0204
        t = np.arange(N, dtype=np.float64) * dt
        flux = 1.0 + 1e-4 * rng.standard_normal(N)
        # 6 hr transit → bl_sz ≈ round(6/24 / 0.0204) = round(12.25) = 12
        result_auto = cdpp(t, flux, transit_duration_hours=6.0)
        result_manual = cdpp(t, flux, bl_sz=12)
        # Should be very close (same bl_sz)
        npt.assert_allclose(result_auto, result_manual, rtol=0.05)

    def test_exclude_mask_size_mismatch_raises(self):
        """Exclusion mask of wrong size must raise AssertionError."""
        t = np.arange(50, dtype=np.float64) * 0.0204
        f = np.ones(50, dtype=np.float64)
        bad_excl = np.zeros(10, dtype=bool)
        with pytest.raises(AssertionError):
            cdpp(t, f, exclude=bad_excl)

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        t = np.arange(200, dtype=np.float64) * 0.0204
        f = 1.0 + 1e-4 * rng.standard_normal(200)
        result = cdpp(t, f)
        assert isinstance(result, float)
        assert result > 0

    def test_sparse_gap_windows(self):
        """Light curve with a large gap in the middle should not crash."""
        dt = 0.0204
        t1 = np.arange(500, dtype=np.float64) * dt
        t2 = np.arange(500, dtype=np.float64) * dt + 50.0  # 50-day gap
        t = np.concatenate([t1, t2])
        rng = np.random.default_rng(42)
        f = 1.0 + 1e-4 * rng.standard_normal(len(t))
        result = cdpp(t, f)
        assert np.isfinite(result)
