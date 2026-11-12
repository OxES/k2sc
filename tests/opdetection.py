"""Diagnostic script for initial GP outlier clipping and Lomb-Scargle periodicity detection."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from k2sc.detrender import Detrender
from k2sc.k2io import MASTReader
from k2sc.kernels import BasicKernelEP
from k2sc.ls import fasper, getSignificance
from k2sc.pipeline import resolve_splits
from k2sc.utils import medsig, sigma_clip


def normalize_flux(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Normalize a flux array by the finite median of ``reference`` (or ``values``)."""
    ref = values if reference is None else reference
    return values / float(np.nanmedian(ref))


def run_opdetection(
    fname: str,
    *,
    splits: Sequence[float] | None = None,
    max_sigma: float = 5.0,
    min_period: float = 0.25,
    max_period: float = 25.0,
    output: str | None = None,
    show: bool = True,
) -> tuple[float, float, float]:
    """Run initial GP detrending, iterative sigma-clipping, and LS period search on a MAST FITS file."""
    dataset = MASTReader.read(fname, sid=0)
    time = np.asarray(dataset.time, dtype=np.float64)
    flux = np.asarray(dataset.fluxes[0], dtype=np.float64)
    inputs = np.column_stack([time, dataset.x, dataset.y])

    valid_mask = np.isfinite(flux) & (dataset.mflags[0] == 0)

    splits_used = (
        resolve_splits(time, splits="auto", campaign=dataset.campaign, mission="K2")
        if splits is None
        else list(splits)
    )

    dt = Detrender(
        flux,
        inputs,
        mask=valid_mask,
        splits=splits_used,
        kernel=BasicKernelEP(),
        seed=0,
    )
    tt, tp = dt.predict(dt.kernel.pv0, components=True)

    fig, ax = plt.subplots(6, 1, figsize=(14, 18))
    ax[0].plot(time, normalize_flux(flux), lw=0.9, label="Raw flux")
    ax[0].set_ylabel("Normalized flux")

    pos_corr = flux - tp + float(np.nanmedian(tp))
    ax[1].plot(time, normalize_flux(pos_corr, flux), lw=0.9, label="Position-corrected")
    ax[1].plot(time, normalize_flux(tt, flux), lw=1.2, label="Initial GP time trend")
    ax[1].set_ylabel("Normalized flux")
    ax[1].legend(loc="best", fontsize=9)

    # Iterative outlier clipping via k2sc.utils.sigma_clip
    cflux = pos_corr - tt + float(np.nanmedian(tt))
    cflux = cflux / float(np.nanmedian(cflux))
    mask = sigma_clip(cflux, max_iter=10, max_sigma=max_sigma, mexc=valid_mask)
    med_c, sigma_c = medsig(cflux[mask])
    scale_sigma = sigma_c if (np.isfinite(sigma_c) and sigma_c > 0.0) else 1.0

    ax[2].plot(time, (cflux - med_c) / scale_sigma, lw=0.8)
    ax[2].axhline(max_sigma, color="C3", ls="--", lw=0.8)
    ax[2].axhline(-max_sigma, color="C3", ls="--", lw=0.8)
    ax[2].grid(axis="y", alpha=0.4)
    ax[2].set_ylabel(r"Normalized flux [$\sigma$]")

    ax[3].plot(time[mask], normalize_flux(pos_corr[mask], flux), lw=0.9, label="Inliers")
    if np.any(~mask):
        ax[3].plot(
            time[~mask],
            normalize_flux(pos_corr[~mask], flux),
            ".",
            color="C3",
            label="Clipped outliers",
        )
    ax[3].set_ylabel("Normalized flux")
    ax[3].legend(loc="best", fontsize=9)

    # Remove residual long-timescale trend with a domain-scaled Legendre polynomial
    nflux = pos_corr.copy()
    ntime = time - float(np.mean(time[mask]))
    poly_deg = min(9, max(1, int(mask.sum()) // 10))
    baseline = np.polynomial.Legendre.fit(ntime[mask], nflux[mask], deg=poly_deg)(ntime)
    resid = nflux - baseline
    ax[4].plot(time[mask], normalize_flux(nflux[mask]), lw=0.8, label="Position-corrected")
    ax[4].plot(time, normalize_flux(baseline, nflux[mask]), "k-", lw=1.3, label="Legendre baseline")
    ax[4].set_ylabel("Normalized flux")
    ax[4].set_xlabel("Time [BKJD]")
    ax[4].legend(loc="best", fontsize=9)

    freq, power, nout, _jmax, _prob = fasper(time[mask], resid[mask], 6.0, 0.5)
    valid_f = freq > 0.0
    period = np.where(valid_f, 1.0 / np.where(valid_f, freq, 1.0), np.inf)
    in_range = valid_f & (period > min_period) & (period < max_period)
    period = period[in_range]
    power = power[in_range]
    fap = getSignificance(period, power, nout, 6.0)
    j = int(np.argmax(power))

    ax[5].semilogy(period, np.maximum(fap, 1e-300), lw=0.9)
    ax[5].set_ylabel("FAP")
    ax[5].set_xlabel("Period [d]")
    ax[5].set_xlim(0.0, max_period)

    plt.setp(ax[:4], xticks=[])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.12)

    best_period = float(period[j])
    best_power = float(power[j])
    best_fap = float(fap[j])
    print(f"Best period: {best_period:.5f} d | LS power: {best_power:.2f} | FAP: {best_fap:.3e}")

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return best_period, best_power, best_fap


def main(argv: Sequence[str] | None = None) -> int:
    ap = ArgumentParser(
        description="Inspect initial K2SC outlier clipping and Lomb-Scargle periodicity detection."
    )
    ap.add_argument("fname", type=str, help="Path to a MAST K2 FITS light curve")
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        type=str,
        help="Optional path to save diagnostic figure",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Close figure instead of opening an interactive window",
    )
    args = ap.parse_args(argv)
    run_opdetection(args.fname, output=args.output, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
