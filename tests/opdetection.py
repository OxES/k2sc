# coding: utf-8
from numpy import *
from matplotlib.pyplot import *
from argparse import ArgumentParser

import math as mt
import astropy.io.fits as pf
rc('figure', figsize=(14,6))

from k2sc.k2io import MASTReader
from k2sc.ls import fasper
from k2sc.kernels import BasicKernel
from k2sc.detrender import Detrender
from k2sc.ls import fasper

def N(a,b=None):
    return a / nanmedian(b if b is not None else a)

splits = [2344]
if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('fname', type=str)
    args = ap.parse_args()

    d  = MASTReader.read(args.fname, 0)
    dt = Detrender(d.fluxes[0], transpose([d.time,d.x,d.y]), mask=d.masks[0], splits=splits, kernel=BasicKernel())
    tt,tp = dt.predict(dt.kernel.pv0+1e-5, components=True)
    time, flux = d.time, d.fluxes[0]

    fig,ax = subplots(6,1, figsize=(14,18))
    ax[0].plot(time, N(flux))
    ax[1].plot(time, N(flux-tp+tp.mean(), flux))
    ax[1].plot(time, N(tt, flux))

    ## Outlier detection
    ## -----------------
    cflux = flux-tp+median(tp)-tt+median(tt)
    cflux /= nanmedian(cflux)
    minf  = isfinite(cflux)
    mask  = minf.copy()
    nm    = None
    i     = 0

    while nm != mask.sum() and i<10:
        nm = mask.sum()
        mad = nanmedian(abs(cflux[mask]-nanmean(cflux[mask])))
        sigma = 1.4826*mad
        mhigh = zeros_like(minf)
        mlow  = zeros_like(minf)
        mhigh[minf] = cflux[minf] > 1+5*sigma
        mlow[minf]  = cflux[minf] < 1-5*sigma
        mask &= minf & (~mlow) & (~mhigh)
        i += 1


    ax[2].plot(time, (cflux-1)/sigma)
    ax[2].grid(axis='y')
    ax[3].plot(time[mask],  N(flux[mask]  - tp[mask],  flux))
    ax[3].plot(time[~mask], N(flux[~mask] - tp[~mask], flux), '.')
    setp(ax[2], ylabel='Normalised flux [$\sigma$]')
    setp(ax[3], ylabel='Normalised flux')

    nflux = flux - tp + nanmedian(tp)
    for split in dt.gp.splits:
        i = argmin(abs(time - split))
        nflux[i:] += nanmedian(nflux[i-10:i]) - nanmedian(nflux[i:i+10])
    ax[4].plot(time[mask], N(nflux[mask]))

    freq,power,nout,jmax,prob = fasper(time[mask], nflux[mask], 6, 0.5)
    period = 1./freq

    m = (period > 0.25) & (period < 25) 
    period, freq, power = period[m], freq[m], power[m]
    j = argmax(power)
    expy = exp(-power)
    effm = 2.*nout/6
    fap  = expy*effm
    mfap = fap > 0.01
    fap[mfap] = 1.0-(1.0-expy[mfap])**effm

    ax[5].semilogy(period, fap)
    setp(ax[5], ylabel='FAP', xlabel='Period [d]', xlim=(0,25))

    print period[j], power[j], fap[j]
    setp(ax[:4], xticks=[])
    setp(ax[4], xlabel='Time [BJD]')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    show()
