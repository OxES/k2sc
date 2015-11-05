import numpy as np
import pylab as pl

from numpy import isfinite

def savgol_filter(t, f, window = 2.0, order = 2):
    n = len(t)
    f_sm = np.zeros(n) + np.nan
    for i in np.arange(n):
        l = (abs(t-t[i]) <= window/2.)
        if l.sum() == 0: continue
        tt = t[l]
        ff = f[l]
        p = np.polyval(np.polyfit(tt, ff, order), t[l])
        j = np.where(tt == t[i])[0]
        f_sm[i] = p[j]
    return f_sm


def block_mean(t, f, block_size = 13, block_size_min = None):
    block_size = int(block_size)
    if block_size_min == None:
        block_size_min = block_size / 2 + 1
    n = len(t)
    dt = np.median(t[1:]-t[:-1])
    t_blocks = []
    f_blocks = []
    i = 0
    while t[i] < t[-1]:
        j = np.copy(i)
        while (t[j] - t[i]) < (block_size * dt):
            j+=1
            if j >= n: break
        if j >= (i + block_size_min):
            t_blocks.append(t[i:j].mean())
            f_blocks.append(f[i:j].mean())
        i = np.copy(j)
        if i >= n: break
    t_blocks = np.array(t_blocks)
    f_blocks = np.array(f_blocks)
    return t_blocks, f_blocks
    

def cdpp(time, flux, dfilt = 2.0, bl_sz = 13, exclude=None, plot = False):
    m = isfinite(time) & isfinite(flux)
    if exclude is not None:
        assert exclude.size == m.size, 'Exclusion mask size != time and flux array size'
        m &= ~exclude
    t,f = time[m], flux[m] / flux[m].mean()

    # filter out long-term variations
    f_sm = savgol_filter(t, f, dfilt)
    f_res = f - f_sm

    # exclude obvious outliers
    m2, s2 = f_res.mean(), f_res.std()
    l2 = abs(f_res-m2) < (5 * s2)

    # compute bin-averaged fluxes
    t_b, f_b = block_mean(t[l2], f_res[l2], block_size = bl_sz)
    cdpp = f_b.std() * 1e6

    if plot:
        pl.clf()
        pl.subplot(211)
        pl.plot(time, flux/flux[m].mean(), '.', c = 'grey', mec = 'grey')
        pl.plot(t, f, 'k.')
        pl.plot(t, f_sm, 'r-')
        pl.xlim(t.min(), t.max())
        pl.subplot(212)
        pl.plot(t, f_res * 1e6, '.', c = 'grey', mec = 'grey')
        pl.plot(t[l2], f_res[l2] * 1e6, 'k.')
        pl.plot(t_b, f_b * 1e6, 'b.')
        pl.xlim(t.min(), t.max())

    return cdpp
