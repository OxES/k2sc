from numpy import isfinite, nan, median, abs

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
