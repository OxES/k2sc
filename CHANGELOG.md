# Changelog

All notable changes to `k2sc` are documented in this file.

## [Unreleased]

### Performance & Numerical Improvements
- **Vectorized `tinygp` Kernel Hierarchy (`k2sc.kernels`, `k2sc.gp`):** Added vectorized `tinygp.kernels.Kernel` subclasses (`K2SCTimeKernel`, `K2SCSpatialKernel`, `K2SCKernel`, `_VectorizedSum`, and vectorized `PiecewiseSpatialKernel` / `_BlockMaskedKernel`) that evaluate 1-D `(N,)` diagonals in $\mathcal{O}(N)$ and 2-D `(N, M)` covariance blocks via direct broadcasting and $\mathcal{O}((N + M) S)$ piecewise split segmentation without nested `jax.vmap`.
- **Precomputed Pairwise Tensors & Analytic GP Sensitivity Gradients (`k2sc.gp`, `k2sc.detrender`, `k2sc.optimizers`):** Added `PrecomputedPairwise` (`precompute_pairwise_jax`), fast forward marginal log-likelihood (`eval_gp_lnlike_precomputed_jax`), and closed-form Cholesky sensitivity gradients (`eval_gp_lnlike_and_grad_precomputed_jax` via $W = \tfrac{1}{2}(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - K^{-1})$) across all five K2SC kernel families, cached on `Detrender` with value-based split-mutation invalidation and non-finite gradient rejection.
- **L2-Cache-Resident Static-Chunk CPU Screening & Host Deduplication (`k2sc.optimizers`):** Updated `_eval_candidates_chunked` to dispatch single-point evaluations directly to `_jit_neglnposterior_precomputed` and evaluate multi-candidate batches in static-shape `jax.lax.map` chunks on CPU (`jax.vmap` on GPU/TPU), alongside consecutive-point query deduplication in `_make_scipy_host_callback`.
- **Single-Thread Worker Environment Helper (`k2sc.configure_single_thread_env`):** Added a stdlib-only `configure_single_thread_env()` helper exported in `k2sc.__all__` that configures BLAS/OpenMP thread environment variables prior to `numpy`/`jax` initialization and appends `--xla_cpu_multi_thread_eigen=false` idempotently to `XLA_FLAGS`.

## [2.0.0] — 2026-11-11

`k2sc` v2.0.0 is a complete scientific, numerical, and architectural modernization of *K2 Systematics Correction* (Aigrain, Parviainen & Pope 2016, MNRAS, 459, 2408), migrating the Gaussian Process engine from C++ `george` to pure-Python `jax` + `tinygp` + `equinox`, introducing exact-gradient multi-start and vectorized evolutionary optimizers (~100× faster per star on CPU), and adding native Lightkurve v2+ and TESS support.

### Breaking & Packaging Changes
- **Python & PEP 621 Packaging (`pyproject.toml`):** Dropped legacy Python 2 / early Python 3 and `setup.py` packaging in favour of PEP 621 metadata requiring **Python >= 3.11** (tested continuously on Python 3.11, 3.12, and 3.13 across Linux and macOS).
- **Pure-Python Wheel Dependencies:** Removed the unmaintained C++ `george` requirement (resolving upstream issue 22) and made `mpi4py` optional (`k2sc[mpi]`). Core dependencies are now pure-Python wheels: `numpy>=1.26`, `scipy>=1.11`, `astropy>=5.3`, `jax>=0.4.35`, `jaxlib>=0.4.35`, `tinygp>=0.3.0`, `equinox>=0.11.5`, `matplotlib>=3.8`, and `tqdm>=4.65`.
- **Optional Extras:** Added modular extras `k2sc[lightkurve]`, `k2sc[mpi]`, `k2sc[optimistix]`, `k2sc[aux]`, `k2sc[test]`, and `k2sc[all]`.
- **Lazy Top-Level Exports & `float64` Initialization:** `import k2sc` automatically enforces 64-bit JAX precision (`jax_enable_x64=True`) prior to any array allocation and lazily exposes `K2SCCorrector`, `K2SCResult`, and `detrend` via PEP 562 `__getattr__`.
- **Deprecations:** Legacy `k2sc.de.DiffEvol`, `k2sc.standalone.k2sc_lc` class-clobbering, and `k2sc.segdetrender.SegmentedDetrender` are retained only as deprecated compatibility shims emitting `DeprecationWarning` (resolving upstream issue 13). Passing `optimizer="jaxopt"` raises a clear `ValueError` directing callers to `"optimistix"`, `"multistart"`, or `"de"`.

### Gaussian Process Engine (`tinygp` + JAX `float64`)
- **Unified `SplitGP` / `GeorgeGP` Solver (`k2sc.gp`):** Replaced `george.GP` and legacy index-sliced `SplitGP` with a unified `tinygp.GaussianProcess` solver (`GeorgeGP = SplitGP`) featuring internal time-origin shifting ($t - t_{\mathrm{ref}}$), exact normalized Gaussian marginal log-likelihoods (including $-\tfrac{M}{2}\ln(2\pi)$), static-shape block-identity masked conditioning (zero XLA JIT recompilation across varying cadence masks), and adaptive diagonal jitter fallback on extreme boundary predictions.
- **Guaranteed PSD Spatial Segmentation (`PiecewiseSpatialKernel`):** Replaced fragile nearest-index matrix slicing (`_get_mask`) with an `equinox.Module` kernel evaluating $M_{\mathrm{seg}}(t, t'; \mathbf{s}) = \mathbb{I}(c(t) = c(t'))$ over dynamically traced padded split arrays (`MAX_SPLITS = 16`), guaranteeing positive semidefiniteness for all $N \neq M$, masked boundary cadences, and unsorted query timestamps.
- **Autodiff-Safe $L_1$ Metric (`SafeL1Distance`):** Implemented a metrically consistent double-`jnp.where` $L_1$ distance subclass for `BasicKernelEP`, `QuasiPeriodicKernelEP`, and `ExpSineSquared`, ensuring exact zero lag distance and finite zero 1st/2nd-order reverse-mode derivatives (`0.0` rather than `NaN`) on diagonal entries and duplicate centroids.
- **Vectorized Supporting Modules (`priors.py`, `ls.py`, `cdpp.py`, `utils.py`):**
  - Rewrote `UniformPrior`, `NormalPrior`, and `LogNormPrior` as branchless JAX PyTrees (`PriorSpec`).
  - Vectorized Press & Rybicki (1989) `fasper` extirpolation in JAX (`jax.lax.scan` + `jnp.fft.ifft` with uniform-grid `hypo2 == 0` guard) and added exact chunked Townsend (2010) `ls_periodogram_jax`.
  - Replaced the scalar Python `np.polyfit` Savitzky–Golay loop in `cdpp.py` with centred local $3 \times 3$ normal equations solved via chunked `jax.vmap` (~1000× faster on CPU without high-BJD Vandermonde ill-conditioning).

### Hyperparameter Optimizers (`k2sc.optimizers`)
- **`MultiStartLBFGSBOptimizer` (`optimizer="auto"` / `"multistart"`, default):** Deterministic $M = 256$ candidate design combining `kernel.pv0`, scrambled Sobol quasi-random sequences, and prior/harmonic-stratified draws (`{P_LS, 0.5 P_LS, 2 P_LS}` with coherent multi-cycle spot evolution scales), evaluated in batched `jax.vmap` chunks and polished from the top $K = 5$ diverse basins using bounded SciPy `L-BFGS-B` with exact `jax.jit(jax.value_and_grad(...))` derivatives (~0.8–2.5 s/star on CPU).
- **`VectorizedDEOptimizer` (`optimizer="de"`):** Wraps `scipy.optimize.differential_evolution(..., vectorized=True, updating="deferred")` backed by `jax.vmap` batch evaluation, early convergence stopping ($\mathrm{ptp}(\text{energies}) < 3.0$), and top-3 exact-gradient `L-BFGS-B` polishing.
- **`OptimistixOptimizer` (`optimizer="optimistix"`):** Pure-JAX `optimistix.BFGS` batched across top-$K$ seeds in smooth logit/sigmoid unconstrained coordinates without Jacobian mode-shift bias.
- **Strict RNG Isolation:** Threaded `np.random.default_rng(seed)` through `DtData.create_training_set`, `Detrender`, and every optimizer without mutating global `np.random` state.

### Lightkurve v2+ `K2SCCorrector` & Unified Pipeline (`k2sc.corrector`, `k2sc.pipeline`)
- **Idiomatic `K2SCCorrector`:** Native `lightkurve.correctors.Corrector` subclass supporting `KeplerLightCurve`, `TessLightCurve`, and generic `LightCurve` objects, safely unwrapping Astropy `Time`, `Quantity`, `MaskedQuantity`, and `MaskedNDArray` columns while preserving units and full input array length $N$.
- **4-Stage Spatial Covariate Fallback & TESS Support:** Automatically resolves explicit `(x, y)` or `covariates` / `DesignMatrix` (with SVD projection when $D_c > 2$), `pos_corr1`/`pos_corr2`, or median-centred `centroid_col`/`centroid_row` and `mom_centr1`/`mom_centr2`.
- **Mission-Aware Split Resolution (`resolve_splits`):** Expanded `DEFAULT_K2_SPLITS` to cover all K2 Campaigns C0–C19 (including sub-campaigns C91/92, C101/102, C111/112; resolving upstream issue 10) and added automatic gap/span breakpoint detection ($\Delta t > \max(0.5\,\mathrm{d}, 10\,\mathrm{median}(\Delta t))$ and spans $> 30\,\mathrm{d}$) for TESS perigee downlink gaps and custom light curves.
- **Unified `K2SCResult` & Polymorphic `k2sc.detrend`:** Consolidated array, `LightCurve`, `K2Data` (all `nsets` apertures), and CLI detrending into `k2sc.pipeline`, returning `K2SCResult` with full-length NaN-coordinate unpacking, transit masking (`transit_mask=(t0, period, duration)` or boolean mask), and backward-compatible 8-tuple iteration/indexing.
- **Diagnostics:** Added `K2SCCorrector.diagnose()` and `K2SCCorrector.diagnose_xy()`, and removed module-scope `matplotlib.use("Agg")` calls so interactive notebook backends are never hijacked.

### CLI Consolidation (`k2sc.cli`)
- Unified `k2sc`, `k2plot`, `k2ginfo`, and `k2mastify` into `pyproject.toml` `[project.scripts]` entry points backed by `k2sc.cli` (keeping `bin/*` as thin executable wrappers).
- Added `--optimizer {auto,multistart,de,optimistix}`, `--n-candidates`, `--maxiter`, `--seed`, and `@filelist` / `.txt` / `.list` batch input file expansion (resolving upstream issue 8).
- Rewrote `k2plot` without unconditional `mpi4py`, `seaborn`, or `IPython` imports, supporting both single- and multi-aperture `K2SC` FITS outputs.

### Bug Fixes
- **Kernel PSD & Initialization Fixes (`k2sc.kernels`):**
  - Restored strictly positive-semidefinite harmonic complexity $\Gamma(p_1) = 1/p_1 > 0$ in `QuasiPeriodicKernel` and `QuasiPeriodicKernelEP`, fixing the `george >= 0.3` unlogged `gamma = -ln(p_1) < 0` Cholesky `LinAlgError` crash on $p_1 \in (1, 2]$ (resolving upstream issue 30).
  - Eliminated double-inverted initial kernel lengthscales between `__init__` and `set_pv` (removing the legacy `pv0 + 1e-5` workaround).
  - Fixed in-place class-attribute mutation of `DtKernel.pv0`, out-of-bounds `evolution_scale=100` initialization, and out-of-prior `bounds[2]` limits on quasiperiodic kernels.
  - Repaired `PeriodicKernel` ($\mathbf{p} \in \mathbb{R}^9$) priors, bounds, `map_pv`, and temporal/spatial kernel grouping.
- **Prior Algebra (`k2sc.priors`):** Fixed missing factor of $2$ in `LogNormPrior` (`-0.5 * ((lnx - mu) / sigma)**2` instead of `lnx**2 - mu*lnx + mu**2`) and `NormalPrior(lims=None)` unpacking crash.
- **Training Set & `Detrender` Fixes (`k2sc.dtdata`, `k2sc.detrender`):**
  - Fixed swapped `(tr_nrandom, tr_bspan, tr_nblocks)` positional arguments in `Detrender.__init__` that created 50 blocks of 6 cadences instead of 6 blocks of 50 (resolving upstream issue 19).
  - Fixed `DtData.create_training_set` crashes on short light curves (`ids.size <= bspan`), overlapping block checks, negative `-bspan` index injection, and stale `DtData.mask` / missing `DtData.outliers` properties.
  - Implemented missing `Detrender.compute_components`, replaced Python branching in `Detrender.neglnposterior` with pure-JAX `_jit_neglnposterior`, and separated `success` from `warn = not success` (resolving upstream issue 26).
- **Pipeline, NumPy 2.0, Astropy, & I/O Fixes (`k2sc.k2io`, `k2sc.pipeline`, `k2sc.cdpp`):**
  - Replaced removed NumPy 2.0 APIs (`np.int`, `np.bool`, `ndarray.ptp()`, `numpy.linalg.linalg`, wildcard `from numpy import *`) across the entire codebase (resolving upstream issue 23).
  - Fixed `FITSWriter.write` `unpack` returning compressed arrays instead of full-length unpacked arrays and prevented `NaN -> int` cast failures on integer/flag FITS columns (resolving upstream issue 9).
  - Fixed `AMCReader.can_read` Python 3 bytes-vs-str check (resolving upstream issue 28) and `FITSWriter` crash when `primary_header is None` (resolving upstream issue 16).
  - Fixed Python 3 `map`/`range` subscripting/popping crashes in CLI tools (resolving upstream issues 24 and 26), Python 3 float division `block_size // 2 + 1` in `cdpp.py`, `standalone.detrend` early `return` after aperture `0` (resolving upstream issue 20), hardcoded `1e-50` FAP and `5`-sigma thresholds ignoring `ls_max_fap` and `outlier_sigma`, ill-conditioned raw `np.polyfit(..., 9)` pre-periodogram detrending (replaced with domain-scaled `Legendre.fit`), false `M_NOTFINITE` flagging of masked transits/thruster firings, missing `campaign=args.campaign` in non-MPI CLI mode, and MPI worker leaks when an input file path is missing.
