"""Tests for src/utils.py — JAX-backed fold, medsig, sigma_clip.

Covers:
* ``fold`` with normalize=True/False, origo, shift.
* ``medsig`` with 0, 1, N finite elements, NaNs/Infs.
* ``sigma_clip`` with separate_masks=True/False, mexc exclusion mask,
  all-NaN array, and convergence.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
from k2sc.utils import fold, medsig, sigma_clip

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# fold tests
# ---------------------------------------------------------------------------

class TestFold:
    def test_basic_fold_normalize(self):
        """fold with normalize=True should give phases in [0, 1)."""
        t = np.linspace(0.0, 10.0, 100)
        phase = fold(t, period=2.0)
        assert phase.min() >= 0.0
        assert phase.max() < 1.0

    def test_basic_fold_no_normalize(self):
        """fold with normalize=False should give values in [0, period)."""
        t = np.linspace(0.0, 10.0, 100)
        period = 3.5
        phase = fold(t, period=period, normalize=False)
        assert phase.min() >= 0.0
        assert phase.max() < period

    def test_fold_normalize_vs_no_normalize(self):
        """fold(normalize=True) * period should equal fold(normalize=False)."""
        t = np.linspace(0.0, 20.0, 50)
        period = 4.1
        p_norm = fold(t, period=period, normalize=True)
        p_unnorm = fold(t, period=period, normalize=False)
        npt.assert_allclose(p_norm * period, p_unnorm, rtol=1e-14)

    def test_fold_origo(self):
        """fold with origo should subtract the reference time."""
        t = np.array([2.0, 4.0, 6.0])
        fold(t, period=2.0, origo=0.0)
        phase_origo = fold(t, period=2.0, origo=2.0)
        # phase_origo[i] = phase_no_origo[i] - 1.0 (mod 1) = same values
        npt.assert_allclose(phase_origo, fold(t - 2.0, period=2.0), rtol=1e-14)

    def test_fold_shift(self):
        """Shift by 0.5 should shift all phases by 0.5 (mod 1)."""
        t = np.linspace(0.0, 10.0, 11)
        period = 2.0
        p0 = fold(t, period=period, shift=0.0)
        p1 = fold(t, period=period, shift=0.5)
        expected = (p0 + 0.5) % 1.0
        npt.assert_allclose(p1, expected, rtol=1e-14)

    def test_fold_returns_float64(self):
        t = np.arange(10, dtype=np.float32)
        phase = fold(t, period=3.0)
        assert phase.dtype == np.float64

    def test_fold_period_exactly_divides(self):
        """When time is an exact multiple of period, phase should be 0."""
        t = np.array([0.0, 2.0, 4.0, 6.0])
        phase = fold(t, period=2.0)
        npt.assert_allclose(phase, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# medsig tests
# ---------------------------------------------------------------------------

class TestMedSig:
    def test_zero_finite_returns_nan_nan(self):
        """Array with no finite elements → (nan, nan)."""
        a = np.array([np.nan, np.inf, -np.inf])
        med, sig = medsig(a)
        assert np.isnan(med)
        assert np.isnan(sig)

    def test_one_finite_returns_nan_sig(self):
        """Array with exactly one finite element → (value, nan)."""
        a = np.array([np.nan, 3.14, np.inf])
        med, sig = medsig(a)
        npt.assert_almost_equal(med, 3.14)
        assert np.isnan(sig)

    def test_normal_distribution(self):
        """medsig on a large Gaussian: median ≈ μ, sig ≈ σ."""
        rng = np.random.default_rng(42)
        a = rng.normal(14.0, 5.0, 100000)
        med, sig = medsig(a)
        npt.assert_almost_equal(med, 14.0, decimal=1)
        npt.assert_almost_equal(sig, 5.0, decimal=1)

    def test_with_nans_and_infs(self):
        """NaN/Inf values should be excluded from the computation."""
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 1.0, 1000)
        # Inject NaN/Inf
        a[::20] = np.nan
        a[1::30] = np.inf
        med, sig = medsig(a)
        assert np.isfinite(med)
        assert np.isfinite(sig)
        assert abs(med) < 0.2  # Should be close to 0

    def test_all_same_value(self):
        """Constant array: median = value, sig = 0."""
        a = np.full(100, 5.0)
        med, sig = medsig(a)
        npt.assert_almost_equal(med, 5.0)
        npt.assert_almost_equal(sig, 0.0)

    def test_two_elements(self):
        """Two finite elements: median = midpoint, sig = 1.48 × half-range."""
        a = np.array([1.0, 3.0])
        med, sig = medsig(a)
        npt.assert_almost_equal(med, 2.0)
        npt.assert_almost_equal(sig, 1.48 * 1.0)


# ---------------------------------------------------------------------------
# sigma_clip tests
# ---------------------------------------------------------------------------

class TestSigmaClip:
    def _make_array(self, seed=0):
        """100-element array with 5 outliers."""
        npt_count = 100
        badi = [6, 23, 34, 55, 76]
        rng = np.random.default_rng(seed)
        arr = rng.normal(1.0, 0.1, npt_count)
        arr[badi] = [-2.0, 1.8, 31.0, np.nan, -5.0]
        return arr, badi

    def test_combined_mask(self):
        """Combined mask should exclude all outliers and NaN."""
        arr, badi = self._make_array()
        tmask = np.ones(100, dtype=bool)
        tmask[badi] = False
        m = sigma_clip(arr)
        npt.assert_array_equal(m, tmask)

    def test_separated_masks(self):
        """Separate masks: mexc, mlow, mhigh correctly identify outlier types."""
        arr, _badi = self._make_array()
        tinf = np.ones(100, dtype=bool)
        thig = np.ones(100, dtype=bool)
        tlow = np.ones(100, dtype=bool)
        tinf[55] = False   # NaN → not finite
        thig[23] = False   # 1.8 is high outlier (>5σ)
        thig[34] = False   # 31.0 is very high
        tlow[6] = False    # -2.0 is low outlier
        tlow[76] = False   # -5.0 is low outlier
        minf, mlow, mhig = sigma_clip(arr, separate_masks=True)
        npt.assert_array_equal(minf, tinf)
        npt.assert_array_equal(mlow, tlow)
        npt.assert_array_equal(mhig, thig)

    def test_mexc_exclusion_mask(self):
        """Points excluded by mexc should always be in mexc=False."""
        rng = np.random.default_rng(99)
        arr = rng.normal(0.0, 1.0, 50)
        mexc_in = np.ones(50, dtype=bool)
        mexc_in[0:5] = False  # exclude first 5
        m = sigma_clip(arr, mexc=mexc_in)
        # First 5 must be False in result
        npt.assert_array_equal(m[0:5], False)

    def test_all_nan_array(self):
        """All-NaN array: combined mask should be all False."""
        a = np.full(10, np.nan)
        m = sigma_clip(a)
        assert not np.any(m)

    def test_convergence(self):
        """Sigma-clip should converge: re-applying the result changes nothing."""
        rng = np.random.default_rng(13)
        arr = rng.normal(0.0, 1.0, 500)
        arr[0] = 100.0  # strong outlier
        m = sigma_clip(arr, max_sigma=3.0)
        m2 = sigma_clip(arr[m], max_sigma=3.0)
        # After convergence, further clipping should not change the fraction
        assert m2.mean() > 0.95  # essentially all clean points retained

    def test_returns_numpy_bool(self):
        """sigma_clip should return a NumPy boolean array."""
        arr = np.arange(10, dtype=np.float64)
        m = sigma_clip(arr)
        assert isinstance(m, np.ndarray)
        assert m.dtype == bool

    def test_separate_masks_returns_numpy_bool(self):
        arr = np.arange(20, dtype=np.float64)
        mexc, _mlow, _mhig = sigma_clip(arr, separate_masks=True)
        assert isinstance(mexc, np.ndarray)
        assert mexc.dtype == bool

    def test_max_sigma(self):
        """Tight max_sigma should clip more points."""
        rng = np.random.default_rng(66)
        arr = rng.normal(0.0, 1.0, 1000)
        m5 = sigma_clip(arr, max_sigma=5.0)
        m2 = sigma_clip(arr, max_sigma=2.0)
        # Tighter sigma clips more → fewer True values
        assert m2.sum() <= m5.sum()

    def test_max_iter_stops_loop(self):
        """max_iter=1 should still return a valid mask without infinite loop."""
        arr = np.concatenate([np.ones(50), [100.0, -100.0]])
        m = sigma_clip(arr, max_iter=1, max_sigma=3.0)
        assert isinstance(m, np.ndarray)
        assert m.sum() > 0
