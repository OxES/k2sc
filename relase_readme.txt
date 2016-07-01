K2SC-detrended K2 light curves
------------------------------

Overview 
--------

K2SC (K2 Systematics Correction, Aigrain et al. 2016) is a K2 light
curve (LC) detrending tool that uses Gaussian processes (GPs) to
robustly model the systematics due to the Kepler telescope pointing
jitter together with the astrophysical variability. The K2SC-detrended
light curves are especially suited for studying variable stars in K2
photometry (by allowing us to remove the position-dependent
systematics while keeping time-dependent variability intact), and
searches for transiting planets (by allowing us to remove both the
systematics and astrophysical variability).

The detrended light curves available here use either SAP or PDC-MAP
photometry as a starting point (stored in separate binary table
extensions in the same fits file), and consist of three components:
position-dependent trend, time-dependent trend, and residual
(detrended) flux.

Description of Data Products 
----------------------------

The detrended light curves are stored in a subdirectory structure with
the format

  c<##>/<####00000>

where

  - c<##> is the zero-padded two-digit K2 Campaign number
  - <####00000> are the first four digits of the target's EPIC ID

The light curves are stored as fits-files with the names

  hslp_k2sc_k2_llc_<#########>-c<##>_v<#>.fits

where

 - <#########> is the nine-digit EPIC ID
 - c<##> is the zero-padded two-digit K2 Campaign number
 - v<#> is the file release version

The fits files have two binary table extensions: the first contains a
k2sc-detrended PDC-MAP light curve, and the second a k2sc-detrended
SAP light curve. The binary table columns are

 - quality  : Original K2 photometry pipeline quality flags
 - cadence  : observation cadences
 - time     : observation times
 - flux     : k2sc-detrended flux
 - trend_p  : k2sc-estimated position-dependent trend
 - trend_t  : k2sc-estimated time-dependent trend
 - mflags   : k2sc outlier flags

The PDC-MAP and SAP extensions are identical, the only difference being
the flux for which the detrending was applied to. The detrended flux
has both the position- and time-dependent trends removed, and can be
directly used for transit and flare searches. The `trend_t` column
contains the time-dependent trend, and can be combined with the
detrended flux in the `flux` column for variability studies as

 flux_v = flux + trend_t - median(trend_t)

Finally, the original flux can be recovered by combining the three
components as

 flux_r = flux + trend_t - median(trend_t) + trend_p - median(trend_p)

Finally, the outliers are tagged in a bit field, where zero value
corresponds to a good-quality point, and bits

 - 2**0 : one of the K2 quality flags on
 - 2**1 : flare (reserved but not currently used) 
 - 2**2 : transit (reserved but not currently used)
 - 2**3 : upwards outlier 
 - 2**4 : downwards outlier
 - 2**5 : nonfinite flux
 - 2**6 : a periodic mask applied manually by k2sc (not used)

are used to flag different reasons for masking.
