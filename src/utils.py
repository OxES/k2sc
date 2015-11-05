from numpy import isfinite, nan, median, abs

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
