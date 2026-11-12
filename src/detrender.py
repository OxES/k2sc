"""Gaussian Process light curve detrender backed by JAX and tinygp."""

from __future__ import annotations

import time

import jax
import numpy as np
from scipy.optimize import minimize

from ._jax_env import as_f64, ensure_float64
from .dtdata import DtData
from .gp import GeorgeGP, PrecomputedPairwise, SplitGP, precompute_pairwise_jax
from .kernels import BasicKernel, DtKernel, pad_splits
from .optimizers import (
    OptimizationResult,
    _jit_neglnposterior,
    _jit_neglnposterior_precomputed,
    _jit_val_and_grad_precomputed,
    compute_closed_lims,
    get_optimizer,
    polish_lbfgsb,
)

ensure_float64()


class Detrender:
    """Simultaneous astrophysical + pointing-drift GP detrender."""

    def __init__(
        self,
        flux,
        inputs,
        mask=None,
        p0=None,
        kernel: DtKernel | None = None,
        splits=(),
        tr_nrandom: int = 200,
        tr_bspan: int = 50,
        tr_nblocks: int = 6,
        seed=None,
        rng=None,
    ):
        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(seed)
        self._seed = seed

        self.data = DtData(flux, inputs, mask)
        if kernel is None:
            self.kernel: DtKernel = BasicKernel(p0=p0)
        else:
            self.kernel = kernel
            if p0 is not None:
                self.kernel.set_pv(p0)

        if splits is None:
            self.gp = GeorgeGP(self.kernel)
        else:
            splits_val = list(splits) if isinstance(splits, tuple) else splits
            self.gp = SplitGP(self.kernel, splits_val)

        self.tr_data = self.data.create_training_set(
            nrandom=tr_nrandom,
            nblocks=tr_nblocks,
            bspan=tr_bspan,
            rng=self._rng,
        )
        self.gp.set_inputs(self.tr_data.masked_inputs)

        self._X_tr_jax = as_f64(self.tr_data.masked_inputs)
        self._y_tr_jax = as_f64(self.tr_data.masked_normalised_flux)
        self._lims_low_jax = as_f64(self.kernel.lims[0])
        self._lims_high_jax = as_f64(self.kernel.lims[1])
        self._splits_pad_tr = pad_splits(self.gp.splits)
        self._precomp_tr: PrecomputedPairwise | None = precompute_pairwise_jax(
            self._X_tr_jax, self._splits_pad_tr
        )
        self._precomp_splits_ref = self.gp.splits

        self.tr_pv: np.ndarray | None = None
        self.opt_result: OptimizationResult | None = None

    @property
    def flux(self) -> np.ndarray:
        return self.data.masked_flux

    @property
    def time(self) -> np.ndarray:
        return self.data.masked_time

    def _training_precomp(self) -> PrecomputedPairwise:
        splits_now = self.gp.splits
        if self._precomp_tr is None or splits_now is not self._precomp_splits_ref:
            self._splits_pad_tr = pad_splits(splits_now)
            self._precomp_tr = precompute_pairwise_jax(
                self._X_tr_jax, self._splits_pad_tr
            )
            self._precomp_splits_ref = splits_now
        return self._precomp_tr

    def _objective_tensors(self, training: bool = True):
        if training:
            self._training_precomp()
            return (
                self._X_tr_jax,
                self._y_tr_jax,
                self._splits_pad_tr,
                self._lims_low_jax,
                self._lims_high_jax,
            )
        X_jax = as_f64(self.data.masked_inputs)
        y_jax = as_f64(self.data.masked_normalised_flux)
        splits_pad = pad_splits(self.gp.splits)
        return X_jax, y_jax, splits_pad, self._lims_low_jax, self._lims_high_jax

    def covariance_matrix(self, pv=None, inputs=None, separate: bool = False):
        inputs = inputs if inputs is not None else self.tr_data.masked_inputs
        self.gp.compute(inputs, pv)
        return self.gp._covariance_matrix(inputs, separate=separate)

    def neglnposterior_jax(self, pv, training: bool = True) -> jax.Array:
        """Pure-JAX negative log-posterior (autodiff, jit, and vmap safe)."""
        if training:
            precomp = self._training_precomp()
            return _jit_neglnposterior_precomputed(
                as_f64(pv),
                precomp,
                self._y_tr_jax,
                self._lims_low_jax,
                self._lims_high_jax,
                self.kernel.prior_spec,
                self.kernel.kernel_id,
            )
        X_jax, y_jax, splits_pad, lims_low, lims_high = self._objective_tensors(
            training=False
        )
        return _jit_neglnposterior(
            as_f64(pv),
            X_jax,
            y_jax,
            splits_pad,
            lims_low,
            lims_high,
            self.kernel.prior_spec,
            self.kernel.kernel_id,
        )

    def neglnposterior(self, pv, training: bool = True) -> float | jax.Array:
        """Evaluate negative log-posterior; returns float eagerly or JAX scalar when traced."""
        val = self.neglnposterior_jax(pv, training=training)
        if isinstance(pv, jax.core.Tracer) or isinstance(val, jax.core.Tracer):
            return val
        return float(val)

    def train(
        self,
        pv0=None,
        disp: bool = False,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> tuple[np.ndarray, bool]:
        """Local hyperparameter optimization on the training set (L-BFGS-B default)."""
        x0 = (
            np.asarray(pv0, dtype=np.float64).copy()
            if pv0 is not None
            else np.asarray(self.kernel._pv, dtype=np.float64).copy()
        )
        X_jax, y_jax, splits_pad, lims_low, lims_high = self._objective_tensors(
            training=True
        )
        precomp_tr = self._training_precomp()
        closed_lims = compute_closed_lims(self.kernel.lims)
        c_arr = np.asarray(closed_lims, dtype=np.float64)
        x0_clipped = np.clip(x0, c_arr[:, 0], c_arr[:, 1])

        method_clean = method.strip()
        if method_clean.upper() == "L-BFGS-B":
            ftol = float(kwargs.get("ftol", 1e-9))
            gtol = float(kwargs.get("gtol", 1e-6))
            maxiter = int(kwargs.get("maxiter", 200))
            opt_res = polish_lbfgsb(
                x0_clipped,
                X_jax,
                y_jax,
                splits_pad,
                lims_low,
                lims_high,
                self.kernel.prior_spec,
                self.kernel.kernel_id,
                closed_lims,
                ftol=ftol,
                gtol=gtol,
                maxiter=maxiter,
                precomp=precomp_tr,
            )
        else:
            t0 = time.perf_counter()
            nfev_counter = {"n": 0}

            def _scalar_obj(x: np.ndarray) -> float:
                nfev_counter["n"] += 1
                v = float(
                    _jit_neglnposterior_precomputed(
                        as_f64(x),
                        precomp_tr,
                        y_jax,
                        lims_low,
                        lims_high,
                        self.kernel.prior_spec,
                        self.kernel.kernel_id,
                    )
                )
                return v if np.isfinite(v) else 1e30

            options = dict(kwargs.get("options", {}))
            if "maxiter" in kwargs and "maxiter" not in options:
                options["maxiter"] = int(kwargs["maxiter"])
            min_kwargs = {"method": method_clean, "options": options}
            if method_clean.lower() in ("powell", "nelder-mead", "tnc", "slsqp"):
                min_kwargs["bounds"] = closed_lims

            mres = minimize(_scalar_obj, x0_clipped, **min_kwargs)
            cand_x = np.clip(
                np.asarray(mres.x, dtype=np.float64), c_arr[:, 0], c_arr[:, 1]
            )
            fin_val_j, fin_grad_j = _jit_val_and_grad_precomputed(
                as_f64(cand_x),
                precomp_tr,
                y_jax,
                lims_low,
                lims_high,
                self.kernel.prior_spec,
                self.kernel.kernel_id,
            )
            fin_val = float(fin_val_j)
            fin_grad = np.asarray(fin_grad_j, dtype=np.float64)
            grad_norm = (
                float(np.linalg.norm(fin_grad))
                if np.all(np.isfinite(fin_grad))
                else float("inf")
            )
            msg = mres.message if isinstance(mres.message, str) else str(mres.message)
            opt_res = OptimizationResult(
                pv=cand_x.copy(),
                fun=fin_val,
                grad_norm=grad_norm,
                success=bool(mres.success and np.isfinite(fin_val)),
                nfev=int(getattr(mres, "nfev", nfev_counter["n"])) + 1,
                njev=1,
                runtime_sec=time.perf_counter() - t0,
                method=method_clean,
                message=msg,
            )

        self.opt_result = opt_res
        self.tr_pv = opt_res.pv.copy()
        self.kernel.set_pv(self.tr_pv)
        if disp:
            print(
                f"[{opt_res.method}] fun={opt_res.fun:.6f} "
                f"|grad|={opt_res.grad_norm:.3e} success={opt_res.success}"
            )
        return self.tr_pv, self.opt_result.success

    def optimize(
        self,
        optimizer: str = "auto",
        seed: int | np.random.Generator | None = 0,
        pv0=None,
        training: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """Global + exact-gradient local hyperparameter optimization."""
        opt = get_optimizer(optimizer, **kwargs)
        ds = self.tr_data if training else self.data
        res = opt.optimize(
            ds.masked_inputs,
            ds.masked_normalised_flux,
            self.gp.splits,
            self.kernel,
            pv0=pv0,
            seed=seed,
        )
        self.opt_result = res
        self.tr_pv = res.pv.copy()
        self.kernel.set_pv(self.tr_pv)
        return res

    def predict(
        self,
        pv,
        inputs=None,
        components: bool = False,
        mean_only: bool = True,
    ):
        inputs = inputs if inputs is not None else self.data.unmasked_inputs
        self.gp.compute(self.data.masked_inputs, pv)
        self.gp._compute_alpha(
            self.data.masked_normalised_flux, allow_jitter_fallback=True
        )

        if components:
            mu_time, mu_pos = self.gp.predict_components(inputs)
            return (
                (1.0 + mu_time) * self.data._fm,
                (1.0 + mu_pos) * self.data._fm,
            )
        return self.gp.predict(inputs, mean_only=mean_only)

    def compute_components(self, pv, inputs=None) -> tuple[np.ndarray, np.ndarray]:
        """Compute physical-flux (time_trend, pos_trend) on masked inputs by default."""
        q_inputs = self.data.masked_inputs if inputs is None else inputs
        return self.predict(pv, inputs=q_inputs, components=True)

    def detrend_spatial(self, pv) -> np.ndarray:
        """Remove spatial systematics from masked cadences while setting excluded cadences to NaN."""
        _, mp = self.compute_components(pv)
        flux = self.data.unmasked_flux.copy()
        flux[self.data.mask] += -mp + np.nanmedian(mp)
        flux[~self.data.mask] = np.nan
        return flux

    def plot_xy(self, pv=None, ax=None, plot_wireframe: bool = False):
        """Plot centroid (x, y) trajectory for the full dataset and training set."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        if pv is None:
            ax.tripcolor(
                self.data.mx,
                self.data.my,
                np.ones(self.data.nptm, dtype=np.float64),
                vmin=0,
                vmax=1,
            )
            if plot_wireframe:
                ax.triplot(self.data.mx, self.data.my, color="w")
        else:
            _, mp = self.compute_components(pv)
            ax.tripcolor(self.data.mx, self.data.my, mp)

        ax.plot(self.tr_data.mx, self.tr_data.my, "o", ms=3, c="k", mec="w")
        return ax

    def plot_t(self, pv=None, ax=None):
        """Plot raw and spatially detrended flux vs. time."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)

        fm = self.data.flux_median
        fmin = float(np.min(self.data.masked_flux))
        fmax = float(np.max(self.data.masked_flux))
        fptp = float(np.ptp(self.data.masked_flux))

        ax.plot(self.data.mt, self.data.mf, c="0.75", lw=1)
        ax.plot(self.tr_data.ut, self.tr_data.uf, ".k", ms=6)
        plt.setp(ax, ylim=(0.999 * fmin, 1.001 * fmax))

        if pv is not None:
            fd = self.detrend_spatial(pv)
            fd += fm - np.nanmedian(fd)
            mm = np.isfinite(fd)
            ax.plot(
                self.data.unmasked_time[mm],
                fd[mm] - 0.7 * fptp,
                alpha=0.75,
                lw=1,
            )
            plt.setp(ax, ylim=(0.999 * (fmin - 0.7 * fptp), 1.001 * fmax))

        plt.setp(ax, xlim=self.data.mt[[0, -1]], xlabel="Time", ylabel="Flux")
        return ax

    def plot_report(self, pv, tid, fname=None, maxpt: int = 350):
        """Generate diagnostic multi-panel K2SC summary report figure."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        lmargin, rmargin = 0.12, 0.03

        fig = plt.figure(figsize=(8.3, 11.7))
        title_str = (
            f"EPIC {int(tid):9d}"
            if isinstance(tid, (int, np.integer))
            else f"{tid}"
        )
        fig.text(
            0.04,
            0.965,
            title_str,
            va="top",
            size=24,
            color="w",
            weight="bold",
        )

        ax = fig.add_axes([0, 0, 1, 1])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_zorder(-1000)
        ax.add_patch(Rectangle((0, 0.92), 1, 0.08, fill=True))

        ax_a = fig.add_axes([lmargin, 0.25, 1 - lmargin - rmargin, 0.3])
        ax_x = fig.add_axes([lmargin, 0.05, 1 - lmargin - rmargin, 0.1])
        ax_y = fig.add_axes([lmargin, 0.15, 1 - lmargin - rmargin, 0.1])
        ax_c = fig.add_axes([0.55, 0.6, 0.45 - rmargin, 0.3])

        ax_x.plot(self.data.masked_time, self.data.mx, lw=1)
        ax_y.plot(self.data.masked_time, self.data.my, lw=1)

        fm = float(np.median(self.data.masked_flux))
        fmin = float(np.min(self.data.masked_flux))
        fmax = float(np.max(self.data.masked_flux))
        fptp = float(np.ptp(self.data.masked_flux))

        _, mp = self.compute_components(pv)
        ms = self.data.mask
        fd = self.data.unmasked_flux.copy()
        fd[ms] += -mp + np.nanmedian(mp)
        fd[~ms] = np.nan
        fd += fm - np.nanmedian(fd)

        ax_a.plot(self.data.masked_time, self.data.masked_flux / fm, c="0.75", lw=1)
        ax_a.plot(
            self.tr_data.unmasked_time,
            self.tr_data.unmasked_flux / fm,
            ".k",
            ms=6,
        )
        ax_a.plot(*self.data.outliers, ls="", marker="o", ms=6)
        ax_a.plot(self.data.unmasked_time[ms], (fd[ms] - 0.7 * fptp) / fm, lw=1)
        ax_a.plot(self.time, (mp - 1.4 * fptp) / fm, lw=1)

        samples = self._rng.permutation(self.time.size)[:maxpt]
        ax_c.tripcolor(self.data.mx[samples], self.data.my[samples], mp[samples])
        ax_c.plot(self.tr_data.mx, self.tr_data.my, ".", ms=3, c="w", alpha=0.8)
        ax_c.plot(self.tr_data.mx, self.tr_data.my, ".", ms=1.5, c="k")

        plt.setp(ax_a, ylim=(0.999 * (fmin - 1.4 * fptp) / fm, 1.001 * fmax / fm))
        plt.setp(ax_a.get_xticklabels() + ax_y.get_xticklabels(), visible=False)
        plt.setp(ax_x, xlabel="Time", ylabel="X")
        plt.setp(ax_c, xlabel="X", ylabel="Y")
        plt.setp([ax_a, ax_x, ax_y], xlim=self.time[[0, -1]])
        plt.setp(ax_a, ylabel="Normalised flux")
        plt.setp(ax_y, ylabel="Y")

        if fname:
            fig.savefig(fname)
        return fig
