# K2SC: *Kepler*/K2 and TESS Systematics Correction

`k2sc` (**v2.0.0**) is a Python package for *Kepler*/K2 and TESS space-photometry systematics correction using additive Gaussian Processes (GPs) powered by [`jax`](https://github.com/jax-ml/jax), [`tinygp`](https://github.com/dfm/tinygp), and [`equinox`](https://github.com/patrick-kidger/equinox).

## Contributors

[Suzanne Aigrain](https://github.com/saigrain), [Hannu Parviainen](https://github.com/hpparvi), [Benjamin Pope](https://github.com/benjaminpope)

---

## Installation

`k2sc` requires **Python >= 3.11** and installs directly from pure-Python wheels—no C/C++ compiler or `george` installation is required. Importing `k2sc` automatically activates JAX 64-bit floating-point precision (`jax_enable_x64=True`).

```bash
# Core GP engine, functional API, and CLI utilities
pip install k2sc

# With Lightkurve v2+ integration (K2SCCorrector)
pip install "k2sc[lightkurve]"

# Full installation (lightkurve, mpi, optimistix, aux)
pip install "k2sc[all]"
```

---

## Quickstart 1 — Idiomatic Lightkurve v2+ API (`K2SCCorrector`)

The recommended interactive workflow uses `K2SCCorrector`, a native `lightkurve.correctors.Corrector` subclass. See the interactive tutorial notebook (`notebooks/lightkurve.ipynb`) for a full walkthrough.

```python
import lightkurve as lk
from k2sc import K2SCCorrector

lc = lk.search_lightcurve("EPIC 211436265", mission="K2", campaign=5).download()

corr = K2SCCorrector(lc)
clc = corr.correct(
    transit_mask=(2310.45, 3.30, 0.12),  # optional (t0, period, duration) in days
    remove_time_trend=False,             # preserve stellar variability and transits
    optimizer="auto",
    seed=0,
)

print(f"CDPP: raw={corr.cdpp_r:.1f} ppm, pos-corrected={corr.cdpp_t:.1f} ppm, whitened={corr.cdpp_c:.1f} ppm")
pos_trend = clc["k2sc_tr_position"]
time_trend = clc["k2sc_tr_time"]

corr.diagnose()
corr.diagnose_xy()
```

---

## Quickstart 2 — TESS Light Curve Detrending

For `TessLightCurve` inputs, `K2SCCorrector` automatically falls back from `pos_corr1`/`pos_corr2` to median-centred `centroid_col`/`centroid_row` (or `mom_centr1`/`mom_centr2`), masks non-zero TESS quality flags, and detects mid-sector perigee downlink gaps via `splits="auto"`:

```python
import lightkurve as lk
from k2sc import K2SCCorrector

tlc = lk.search_lightcurve("TIC 25155310", mission="TESS", sector=1, author="SPOC").download()
corr = K2SCCorrector(tlc)
ctlc = corr.correct(splits="auto", optimizer="auto", seed=0)
```

---

## Quickstart 3 — Functional & Standalone Array API (`k2sc.detrend`)

`k2sc.detrend` accepts 1D NumPy/JAX arrays, `LightCurve` objects, or `K2Data` instances:

```python
import numpy as np
import k2sc

rng = np.random.default_rng(0)
time = np.linspace(0.0, 40.0, 600)
x = 0.2 * ((time % 0.25) / 0.25 - 0.5)
y = 0.15 * np.sin(2.0 * np.pi * time / 0.25)
flux = 1000.0 * (1.0 + 0.003 * np.sin(2.0 * np.pi * time / 6.5) + 0.008 * x - 0.005 * y + rng.normal(0.0, 2e-4, time.size))

res = k2sc.detrend(time, flux, x=x, y=y, splits=[20.0], optimizer="auto", seed=0)
# Also dispatches directly on LightCurve instances:
# clc = k2sc.detrend(lc, seed=0)
```

---

## Optimizer Guide

Pass `optimizer` in Python or `--optimizer` on the command line alongside a deterministic integer `seed`:

- **`"auto"` / `"multistart"`** (`MultiStartLBFGSBOptimizer`, default): Deterministic $M = 256$ candidate screening (`pv0` + scrambled Sobol quasi-random sequences + prior/harmonic-stratified draws across $\{P_{\mathrm{LS}}, \tfrac{1}{2}P_{\mathrm{LS}}, 2P_{\mathrm{LS}}\}$) evaluated in static chunks via `jax.lax.map` on CPU (`jax.vmap` on GPU/TPU), followed by exact-gradient SciPy `L-BFGS-B` polishing from the top $K = 5$ diverse basins. For multi-process / MPI batch runs, call `k2sc.configure_single_thread_env()` before importing `numpy`/`jax` to pin BLAS, OpenMP, and XLA CPU Eigen threads to `1` per worker.
- **`"de"`** (`VectorizedDEOptimizer`): Vectorized SciPy Differential Evolution (`--de-npop`, `--de-niter`, `--de-max-time`) evaluated per generation in static chunks (`jax.lax.map` on CPU, `jax.vmap` on GPU/TPU), polished from the top 3 diverse basins with exact-gradient `L-BFGS-B`.
- **`"optimistix"`** (`OptimistixOptimizer`): Pure-JAX `optimistix.BFGS` over unconstrained logit/sigmoid coordinates (`pip install "k2sc[optimistix]"`).

---

## Command-Line Interface

```bash
# Single file, directory, or @filelist batch input
k2sc ktwo211436265-c05_llc.fits --save-dir ./out
k2sc @filelist.txt -c 5 --flux-type pdc --optimizer auto --seed 0 --save-dir ./out
k2sc ktwo211436265-c05_llc.fits --no-roll-pca --save-dir ./out_raw_xy

# MPI parallel execution (requires k2sc[mpi])
mpirun -n 8 k2sc ./campaign05_fits/ -c 5 --save-dir ./out --logfile k2sc.log

# Companion diagnostic, catalogue-summary, and MAST HLSP utilities
k2plot ./out/EPIC_211436265_mast.fits --save-dir ./plots
k2ginfo summary.csv ./out
k2mastify ./out_pdc ./out_sap ./hlsp_out --release-version 2
```

---

## Mathematical Summary

`k2sc` decomposes normalized relative flux along $\mathbf{x} = (t, \tilde{u}_1, \tilde{u}_2)$ into an astrophysical time GP and a piecewise-segmented spatial GP:

$$f(t, x, y) = f_{\mathrm{time}}(t) + f_{\mathrm{pos}}(\tilde{u}_1, \tilde{u}_2; t) + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 10^{2 p_{-1}}).$$

- **Spacecraft-Roll PCA Normalization (`k2sc.pca`, `roll_pca=True` default):** Because K2 `(POS_CORR1, POS_CORR2)` centroids are collinear along the spacecraft roll arc while orthogonal thermal/focus drift spans only millipixels, `k2sc` rotates `(x, y)` into a right-handed $\mathrm{SO}(2)$ roll frame (seeded from the 2D $L_1$ spatial median and spherical PCA, refined with iterative $5\sigma$ principal-axis trimming, and recentred on the trimmed principal-axis medians $(\mu_x, \mu_y)$) and normalizes each axis by $s_k = 4\sigma_{\mathrm{MAD}}(u_k)$ so $[-2\sigma_k, +2\sigma_k] \mapsto [-0.5, +0.5]$. Pass `roll_pca=False` (or `--no-roll-pca` on the CLI) to fit in raw `(x, y)` coordinates. The fitted `RollPCATransform` is stored on `K2SCResult.pca_transform`, recorded in `LightCurve.meta` (`K2SC_ROLL_PCA`, `K2SC_PCA_CENTER`, `K2SC_PCA_ROTATION`, `K2SC_PCA_SCALES`), and written to the `K2SC` FITS extension header (`ROLL_PCA`, `PCA_CX`, `PCA_CY`, `PCA_R11`..`PCA_R22`, `PCA_S1`, `PCA_S2`) while keeping physical detector `(x, y)` columns unchanged.
- **`SplitGP` & `PiecewiseSpatialKernel`:** Spatial covariances are multiplied by $M_{\mathrm{seg}}(t, t'; \mathbf{s}) = \mathbb{I}(c(t) = c(t'))$ where $c(t) = \sum_k \mathbb{I}(t > s_k)$, guaranteeing positive semidefiniteness across roll-angle breakpoint splits $\mathbf{s}$ and masked cadences.
- **Kernels:** Provides `BasicKernel` (`basic`), `BasicKernelEP` (`basic_ep`), `QuasiPeriodicKernel` (`quasiperiodic`), `QuasiPeriodicKernelEP` (`quasiperiodic_ep`), and `PeriodicKernel` (`periodic`), with spatial hyperparameters (`x_iscale`, `y_iscale` in `ExpSquared(scale=1/sqrt(p))` and `Exp(scale=1/sqrt(p))`) parameterizing inverse-squared length-scales in normalized $(\tilde{u}_1, \tilde{u}_2)$ coordinates, strictly positive-semidefinite harmonic complexity $\Gamma(p_1) = 1/p_1 > 0$, and zero-lag autodiff-safe `SafeL1Distance` exponential metrics.

---

## Citing

If you use `k2sc` in your research, please cite **Aigrain, Parviainen & Pope (2016, MNRAS, 459, 2408; DOI [`10.1093/mnras/stw706`](https://doi.org/10.1093/mnras/stw706))**:

```bibtex
@article{Aigrain2016,
    author  = {Aigrain, Suzanne and Parviainen, Hannu and Pope, Benjamin},
    title   = {{K2SC: flexible systematics correction and detrending of K2 light curves using Gaussian process regression}},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume  = {459},
    number  = {3},
    pages   = {2408--2419},
    year    = {2016},
    month   = {07},
    doi     = {10.1093/mnras/stw706},
    eprint  = {1603.09167},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.IM},
    url     = {https://doi.org/10.1093/mnras/stw706}
}
```
