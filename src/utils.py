from numpy import isfinite, nan, median, abs, ones_like

def fold(time, period, origo=0.0, shift=0.0, normalize=True):
    """Folds the given data over a given period.
    """
    tf = ((time - origo)/period + shift) % 1.
    return tf if normalize else tf*period


def medsig(a):
    """Return median and outlier-robust estimate of standard deviation
       (1.48 x median of absolute deviations).
    """
    l = isfinite(a)
    nfinite = l.sum()
    if nfinite == 0:
        return nan, nan
    if nfinite == 1:
        return a[l], nan
    med = median(a[l])
    sig = 1.48 * median(abs(a[l] - med))
    return med, sig

def sigma_clip(a, max_iter=10, max_sigma=5, separate_masks=False, mexc=None):
    """Iterative sigma-clipping routine that separates not finite points, and down- and upwards outliers.
    """
    mexc  = isfinite(a) if mexc is None else isfinite(a) & mexc
    mhigh = ones_like(mexc)
    mlow  = ones_like(mexc)
    mask  = ones_like(mexc)

    i, nm = 0, None
    while (nm != mask.sum()) and (i < max_iter):
        mask = mexc & mhigh & mlow
        nm = mask.sum()
        med, sig = medsig(a[mask])
        mhigh[mexc] = a[mexc] - med <  max_sigma*sig
        mlow[mexc]  = a[mexc] - med > -max_sigma*sig
        i += 1

    if separate_masks:
        return mexc, mlow, mhigh
    else:
        return mlow & mhigh & mexc


