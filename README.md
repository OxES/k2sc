# K2SC: *Kepler*/K2 and TESS Systematics Correction

[![Tests](https://github.com/OxES/k2sc/actions/workflows/tests.yml/badge.svg)](https://github.com/OxES/k2sc/actions/workflows/tests.yml)
[![Licence](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](https://www.gnu.org/licenses/gpl-3.0.html)
[![MNRAS](https://img.shields.io/badge/MNRAS-10.1093%2Fmnras%2Fstw706-blue.svg)](https://doi.org/10.1093/mnras/stw706)
[![arXiv](https://img.shields.io/badge/arXiv-1603.09167-blue.svg?style=flat)](https://arxiv.org/abs/1603.09167)

`k2sc` (**v2.0.0**) is a Python package for space-photometry systematics correction and astrophysical detrending of *Kepler*/K2 and TESS light curves using additive Gaussian Processes (GPs). Originally developed by Aigrain, Parviainen & Pope (2016), `k2sc` v2.0 replaces the legacy C++ `george` backend and serial Differential Evolution solver with a pure-Python [`jax`](https://github.com/jax-ml/jax) + [`tinygp`](https://github.com/dfm/tinygp) + [`equinox`](https://github.com/patrick-kidger/equinox) engine, exact reverse-mode automatic differentiation, and native [`lightkurve`](https://github.com/lightkurve/lightkurve) v2+ integration—achieving **~1–3 s per star on CPU (~100× speedup)**.

---

## Installation

`k2sc` requires **Python >= 3.11** and installs cleanly from pure-Python wheels with no C/C++ compiler or `george` dependency:

```bash
# Core GP engine, standalone API, and CLI tools
pip install k2sc

# Recommended for interactive analysis with Lightkurve v2+
pip install "k2sc[lightkurve]"

# Full installation (Lightkurve, MPI batch support, Optimistix, HDF5/CSV tools)
pip install "k2sc[all]"
```

For development or running the test suite from a local clone:

```bash
git clone https://github.com/OxES/k2sc.git
cd k2sc
pip install -e ".[test]"
pytest
```

### Optional Dependency Extras

| Extra | Packages Installed | Purpose |
|---|---|---|
| `k2sc[lightkurve]` | `lightkurve>=2.4` | Idiomatic `K2SCCorrector` and `LightCurve` dispatch in `k2sc.detrend` |
| `k2sc[mpi]` | `mpi4py>=3.1` | Multi-rank MPI batch processing (`mpirun -n N k2sc`) |
| `k2sc[optimistix]` | `optimistix>=0.0.7` | Pure-JAX `OptimistixOptimizer` (`BFGS`) backend |
| `k2sc[aux]` | `pandas>=2.0`, `tables>=3.9` | HDF5 and CSV catalogue aggregation in `k2ginfo` |
| `k2sc[all]` | All extras above | Complete feature set |

> **Automatic JAX `float64` Precision:** Importing `k2sc` automatically enables 64-bit floating-point arithmetic in JAX (`jax_enable_x64=True`) and shifts internal GP coordinates relative to a reference epoch $t_{\mathrm{ref}}$, preserving full 53-bit IEEE-754 mantissa precision on BJD/BKJD/BTJD timestamps and white-noise variances down to $\sigma_n^2 = 10^{-12}$.

---

## Quickstart 1 — Idiomatic Lightkurve v2+ API (`K2SCCorrector`)

`K2SCCorrector` subclasses `lightkurve.correctors.Corrector`, unwrapping Astropy `Time`, `Quantity`, and `MaskedNDArray` columns automatically while preserving full light curve length and physical flux units:

```python
import lightkurve as lk
from k2sc import K2SCCorrector

# Download a K2 light curve from MAST (or load a local FITS file via lk.read)
lc = lk.search_lightcurve("EPIC 211436265", mission="K2", campaign=5).download()

# Fit the simultaneous time + position GP model
corr = K2SCCorrector(lc)
clc = corr.correct(optimizer="auto", seed=0)

# Mask known planetary transits (t0, period, duration in days) during GP training
clc_masked = corr.correct(
    transit_mask=(2310.45, 3.30, 0.12),
    remove_time_trend=False,  # True also subtracts the smooth stellar GP time component
    seed=0,
)

# Inspect 6.5-hour CDPP metrics (ppm) before and after correction
print(f"CDPP raw={corr.cdpp_r:.1f} ppm -> pos-corrected={corr.cdpp_t:.1f} ppm (whitened={corr.cdpp_c:.1f} ppm)")

# Access separated GP trend components and quality/outlier bitmask columns
pos_trend = clc["k2sc_tr_position"]
time_trend = clc["k2sc_tr_time"]
whitened_flux = clc["k2sc_detrended_flux"]
flags = clc["k2sc_mflags"]

# Render 3-panel time-series diagnostics and 2D centroid spatial systematics map
corr.diagnose()
corr.diagnose_xy()
```

### Columns and Metadata Attached to `corrected_lc`

- `clc.flux`: Position-detrended flux (`remove_time_trend=False`, default, preserving transits and stellar variability) or fully whitened flux (`remove_time_trend=True`).
- `clc["k2sc_tr_position"]`, `clc["k2sc_tr_time"]`: Posterior mean GP position and time trends in original flux units.
- `clc["k2sc_corr_flux"]`, `clc["k2sc_detrended_flux"]`: Position-corrected and fully whitened flux series.
- `clc["k2sc_mflags"]`: Bitmask (`uint8`) recording non-finite cadences, quality flags, sigma-clipped upward/downward outliers, and periodic transit masks.
- `clc.meta`: Populated with `K2SC_KERNEL`, `K2SC_PV`, `K2SC_CDPP_R`, `K2SC_CDPP_T`, `K2SC_CDPP_C`, `K2SC_SPLITS`, `K2SC_PERIODIC`, `K2SC_LS_PERIOD`, `K2SC_LS_FAP`, `K2SC_COVARIATES`, and `K2SC_SUCCESS`.

---

## Quickstart 2 — TESS Light Curve Detrending

`K2SCCorrector` and `k2sc.detrend` support `TessLightCurve` objects out of the box, with automatic centroid covariate fallback and perigee downlink-gap split detection:

```python
import lightkurve as lk
from k2sc import K2SCCorrector

tlc = lk.search_lightcurve("TIC 25155310", mission="TESS", sector=1, author="SPOC").download()

# Automatically falls back to median-centred centroid_col / centroid_row when pos_corr is absent
# and places a spatial-kernel split across the ~13.7 d mid-sector downlink gap (splits="auto")
corr = K2SCCorrector(tlc)
ctlc = corr.correct(splits="auto", transit_mask=(1325.504, 3.2888, 0.11), seed=0)
```

**Covariate Resolution Order (`K2SCCorrector`):**
1. Explicit `x=..., y=...` (1D arrays or column names) or `covariates=...` (`(N, D_c)` array or `lightkurve.DesignMatrix`; when $D_c > 2$, non-constant columns are projected onto their two dominant orthogonal singular axes).
2. `pos_corr1` and `pos_corr2` (when $>50\%$ finite with positive variance).
3. Median-centred `centroid_col` / `centroid_row` or `mom_centr1` / `mom_centr2`.

---

## Quickstart 3 — Functional & Standalone Array API (`k2sc.detrend`)

`k2sc.detrend` provides a polymorphic functional interface accepting 1D NumPy/JAX arrays, `LightCurve` instances, or `K2Data` objects:

```python
import numpy as np
import k2sc

# 1. Standalone 1D array detrending (returns a K2SCResult dataclass; NaN-coordinate safe)
rng = np.random.default_rng(0)
time = np.linspace(0.0, 40.0, 600)
x = 0.2 * ((time % 0.25) / 0.25 - 0.5)
y = 0.15 * np.sin(2.0 * np.pi * time / 0.25)
flux = 1000.0 * (1.0 + 0.003 * np.sin(2.0 * np.pi * time / 6.5) + 0.008 * x - 0.005 * y + rng.normal(0.0, 2e-4, time.size))

res = k2sc.detrend(
    time,
    flux,
    x=x,
    y=y,
    splits=[20.0],
    optimizer="auto",
    seed=0,
)
print(res.corr_flux.shape, res.cdpp_r, res.cdpp_t, res.cdpp_c, res.is_periodic)

# 2. Direct Lightkurve dispatch (returns corrected LightCurve, or K2SCResult if return_result=True)
# clc = k2sc.detrend(lc, transit_mask=(t0, period, duration), seed=0)
```

`K2SCResult` exposes rich structured attributes (`detrender`, `pv`, `tr_time`, `tr_position`, `corr_flux`, `detrended_flux`, `cdpp_r`, `cdpp_t`, `cdpp_c`, `mflags`, `outlier_mask`, `splits`, `is_periodic`, `ls_period`, `ls_fap`, `opt_result`, `success`, `warn`) while supporting legacy 8-tuple unpacking `(detrender, pv, tr_time, tr_position, cdpp_r, cdpp_t, cdpp_c, warn)`.

---

## Hyperparameter Optimizer Guide

All entry points (`K2SCCorrector.correct`, `k2sc.detrend`, `Detrender.optimize`, and the `k2sc` CLI) accept `optimizer` / `--optimizer` and a deterministic `seed` (`np.random.default_rng(seed)`):

| Backend | Option | Algorithm & Characteristics |
|---|---|---|
| **`MultiStartLBFGSBOptimizer`** *(default)* | `"auto"` or `"multistart"` | Deterministic $M = 256$ design (`pv0` + scrambled Sobol quasi-random points + prior/harmonic-stratified draws at $\{P_{\mathrm{LS}}, \tfrac{1}{2}P_{\mathrm{LS}}, 2P_{\mathrm{LS}}\}$), screened in batched `jax.vmap` chunks and polished from the top $K = 5$ diverse basins via bounded SciPy `L-BFGS-B` using exact `jax.value_and_grad` derivatives (~0.8–2.5 s/star on CPU). |
| **`VectorizedDEOptimizer`** | `"de"` | Vectorized SciPy Differential Evolution (`vectorized=True`, `updating="deferred"`) evaluating entire generations in a single `jax.vmap` call (`--de-npop`, `--de-niter`, `--de-max-time`, early stopping when $\mathrm{ptp}(\text{energies}) < 3.0$), followed by top-3 exact-gradient `L-BFGS-B` polish. |
| **`OptimistixOptimizer`** | `"optimistix"` | Pure-JAX `optimistix.BFGS` (`jax.vmap`-batched across top-$K$ diverse seeds) over smooth logit/sigmoid unconstrained coordinates without Jacobian mode-shift bias (requires `pip install "k2sc[optimistix]"`). |

---

## Command-Line Interface Reference

### `k2sc` — Batch FITS / AMC Detrending

```bash
# Detrend a single MAST FITS file, multiple files, or an entire directory
k2sc ktwo211436265-c05_llc.fits --save-dir ./out
k2sc ./campaign05_fits/ -c 5 --optimizer auto --seed 0 --save-dir ./out

# Detrend from a text file list (@filelist or .txt/.list/.lst)
k2sc @targets.txt --flux-type pdc --save-dir ./out

# Distributed multi-rank MPI execution (requires k2sc[mpi])
mpirun -n 8 k2sc @targets.txt -c 5 --optimizer auto --save-dir ./out --logfile k2sc_run.log
```

**Key `k2sc` Flags:**
- `-c C`, `--campaign C`: K2 campaign number (`0`–`19`, `91`, `92`, `101`, `102`, `111`, `112`).
- `--splits SPLITS`: Comma-separated breakpoint timestamps (e.g. `"[2344, 2360]"`), `"auto"`, or `"[]"`.
- `--flux-type {pdc,sap}`: Input MAST flux column (`pdc` default).
- `--optimizer {auto,multistart,de,optimistix}`: Global + gradient hyperparameter optimizer (`auto` default).
- `--n-candidates N`, `--maxiter M`: Candidate screening budget and local `L-BFGS-B` iteration limit.
- `--de-npop N`, `--de-niter G`, `--de-max-time SEC`: Differential Evolution population, generation, and wall-time budgets (`--optimizer de`).
- `--kernel {basic,basic_ep,periodic,quasiperiodic,quasiperiodic_ep}` and `--kernel-period P`: Explicit GP kernel override.
- `--default-position-kernel {SqrExp,Exp}`: Default 2D spatial kernel family (`SqrExp` default).
- `--p-mask-center T0`, `--p-mask-period P`, `--p-mask-duration DUR`: Periodic transit ephemeris mask (days).
- `--ls-max-fap LOG10_FAP`, `--ls-min-period P_MIN`, `--ls-max-period P_MAX`: Lomb–Scargle periodicity selection thresholds (default `--ls-max-fap -50.0`, i.e. $\mathrm{FAP} < 10^{-50}$).
- `--outlier-sigma SIGMA`, `--outlier-mwidth W`: Iterative running-median sigma-clipping parameters (`5.0`, `25`).
- `--tr-nrandom N`, `--tr-nblocks B`, `--tr-bspan S`: GP training-set subsample parameters (`400`, `6`, `50`).
- `--start-i I`, `--end-i J`, `--seed S`, `--save-dir DIR`, `--logfile FILE`, `--quiet`.

### Companion Utilities

- **`k2plot`**: Render multi-panel diagnostic PDF reports from detrended `K2SC` FITS files:
  ```bash
  k2plot out/EPIC_211436265_mast.fits --save-dir ./plots
  ```
- **`k2ginfo`**: Aggregate fitted GP hyperparameters and raw/detrended CDPP triplets across a directory of `K2SC` FITS files into `.fits`, `.csv`, or `.h5` summary tables:
  ```bash
  k2ginfo summary.csv ./out
  ```
- **`k2mastify`**: Merge `PDC` and `SAP` detrended `K2SC` FITS products into standard MAST High-Level Science Product (`hlsp_k2sc_k2_llc_*`) multi-extension FITS files:
  ```bash
  k2mastify ./out_pdc ./out_sap ./hlsp_out --release-version 2
  ```

---

## Mathematical Summary of the GP Model

`k2sc` models normalized relative flux $y_i = F_i / \mathrm{median}(F) - 1$ at coordinates $\mathbf{x}_i = (t_i, x_i, y_i) \in \mathbb{R}^3$ as the sum of an astrophysical time kernel $k_{\mathrm{time}}(t, t')$, a piecewise-segmented 2D pointing-drift kernel $k_{\mathrm{pos}}(x, y, t; x', y', t')$, and white noise $\sigma_n^2 = 10^{2 p_{-1}}$:

$$K_{ij} = k_{\mathrm{time}}(t_i, t_j) + M_{\mathrm{seg}}(t_i, t_j; \mathbf{s})\,k_{xy}(x_i, y_i; x_j, y_j) + \sigma_n^2\,\delta_{ij}.$$

- **Piecewise Spatial Segmentation (`SplitGP` & `PiecewiseSpatialKernel`):** Roll-angle drift reversals at timestamps $\mathbf{s} = (s_1, \dots, s_K)$ partition the light curve into segments $c(t) = \sum_{k=1}^K \mathbb{I}(t > s_k)$ with $M_{\mathrm{seg}}(t, t'; \mathbf{s}) = \mathbb{I}(c(t) = c(t'))$. Because $M_{\mathrm{seg}}$ is a sum of outer-product indicator matrices, `PiecewiseSpatialKernel` is guaranteed positive semidefinite by the Schur product theorem across arbitrary training/prediction masks ($N \neq M$).
- **Temporal Kernels:** Smooth variability uses `ExpSquared` ($10^{p_0}\exp[-\tfrac{1}{2}p_1 \Delta t^2]$), while rotational spot modulation (`QuasiPeriodicKernel`, `QuasiPeriodicKernelEP`, `PeriodicKernel`) uses the strictly positive-semidefinite harmonic complexity parameterization $\Gamma(p_1) = 1/p_1 > 0$ on $p_1 \in (0, 2]$:
  $$k_{\mathrm{QP}}(t, t') = 10^{p_0}\exp\!\left(-\frac{1}{p_1}\sin^2\!\left(\frac{\pi |t - t'|}{p_2}\right) - \frac{p_3}{2}(t - t')^2\right).$$
- **Autodiff-Safe $L_1$ Kernels (`SafeL1Distance`):** Exponential spatial kernels (`BasicKernelEP`, `QuasiPeriodicKernelEP`: $10^{p_{\mathrm{xy}}}\exp(-\sqrt{p_x}|\Delta x| - \sqrt{p_y}|\Delta y|)$) and `ExpSineSquared` evaluate $|u - u'|$ via a double-`jnp.where` metric (`SafeL1Distance`), yielding exact zero distance and finite zero reverse-mode derivatives (never `0/0 = NaN`) at zero lag and duplicate centroids.
- **Posterior Component Separation:** Conditioning the factored Cholesky matrix $L$ ($K = L L^\top$, $\boldsymbol{\alpha} = K^{-1}\mathbf{y}$) separately on $k_{\mathrm{time}}$ and $k_{\mathrm{pos}}$ yields physical-flux trends $T_{\mathrm{time}}$ and $T_{\mathrm{pos}}$, from which `k2sc` forms position-corrected flux $F_{\mathrm{corr}} = F - (T_{\mathrm{pos}} - \mathrm{nanmedian}(T_{\mathrm{pos}}))$ and fully whitened flux $F_{\mathrm{detrended}} = F_{\mathrm{corr}} - (T_{\mathrm{time}} - \mathrm{nanmedian}(T_{\mathrm{time}}))$.

---

## Citation

If you use `k2sc` in your research, please cite **Aigrain, Parviainen & Pope (2016, MNRAS, 459, 2408)**:

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

## Authors

- **Suzanne Aigrain** (University of Oxford)
- **Hannu Parviainen** (Instituto de Astrofísica de Canarias / University of Oxford)
- **Benjamin Pope** (Macquarie University / University of Queensland)
