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


def sigma_clip(a, max_iter=10, max_sigma=5, separate_masks=False):
    """Iterative sigma-clipping routine that separates not finite points, and down- and upwards outliers.
    """
    minf  = isfinite(a)
    mhigh = ones_like(minf)
    mlow  = ones_like(minf)
    mask  = ones_like(minf)

    i, nm = 0, None
    while (nm != mask.sum()) and (i < max_iter):
        mask = minf & mhigh & mlow
        nm = mask.sum()
        med, sig = medsig(a[mask])
        mhigh[minf] = a[minf] - med <  max_sigma*sig
        mlow[minf]  = a[minf] - med > -max_sigma*sig
        i += 1

    if separate_masks:
        return minf, mlow, mhigh
    else:
        return minf & mlow & mhigh

