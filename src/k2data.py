"""Contains the K2Data class for working with K2 datasets with 1..n flux and error
arrays (using different aperture size, etc.)

"""
from __future__ import division
import numpy as np

from numpy import array, atleast_2d, ones, ones_like, argmax, all, isfinite, tile, extract
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import medfilt

from ls import fasper
from utils import medsig

class K2MultiStarData(K2Data):
    """Encapsulates the K2 data.

    Encapsulates the K2 data with 1..n flux and flux uncertainty arrays for a number of stars.
    Separate flux and uncertainty arrays can correspond to different aperture sizes, etc.

    Notes
    -----
    We remove the cadences where either time, x, or y is nan since input nans screw 
    up the GP fitting and reduction. This must be accounted for when saving the data.

    Parameters
    ----------
    epic    : array_like [nexp]
              EPIC numbers of the stars
    time    : array_like [nexp]
              ndarray or list containing the time values
    cadence : array_like [nexp]
              ndarray or list containing the cadence values
    quality : array_like [nstars,nexp]
              ndarray or list containing quality value lists
    fluxes  : array_like [nstars,nsets,npoints] 
              ndarray of flux values
    errors  : array_like [nstars,nsets,npoints] 
              ndarray of flux values
    x       : array_like [nstars,nexp] 
              ndarray or list of x value lists
    y       : array_like [nstars,nexp] 
              ndarray or list of y value lists

    Attributes
    ----------
    nstars      : int
                  Number of stars
    nsets       : int
                  Number of apertures per star
    npoints     : int
                  Number of datapoints
    is_periodic : bool
                  Does the flux show clear periodic variability
    ls_period   : float
                  Period of the strongest periodic variability detected
    ls_power    : float
                  Lomb-Scargle power of the strongest periodic variability detected
    """
    def __init__(self, epic, time, cadence, quality, fluxes, errors, x, y, sap_header=None):
        self.epic = epic
        self.nanmask = nm = isfinite(time) & all(isfinite(x), 0) & all(isfinite(y), 0)
        self.time = extract(nm, time)
        self.cadence = extract(nm, cadence)
        self.quality = extract(nm, quality).astype(np.int)
        self.fluxes = atleast_3d(fluxes)[:,:,nm]
        self.errors = atleast_3d(errors)[:,:,nm]
        self.x = atleast_2d(x)[:,nm]
        self.y = atleast_2d(x)[:,nm]
        self.sap_header = sap_header

        self.nstars  = self.fluxes.shape[0]
        self.nsets   = self.fluxes.shape[1]
        self.npoints = self.fluxes.shape[2]
        self.istar   = 0

        self.kmask    = all(isfinite(self.fluxes),1) & (self.quality==0)
        self.fmasks   = ones([self.nsets, self.npoints], np.bool)
        self.omasks_u = ones_like(self.fmasks)
        self.omasks_d = ones_like(self.fmasks)
        self.masks    = tile(self.kmask, [self.nsets,1])
        
        self.is_periodic = False
        self.ls_period = None
        self.ls_power = None


class K2Data(object):
    """Encapsulates the K2 data.

    Encapsulates the K2 data with 1..n flux and flux uncertainty arrays. Separate
    flux and uncertainty arrays can correspond to different aperture sizes, etc.

    Notes
    -----
    We remove the cadences where either time, x, or y is nan since input nans screw 
    up the GP fitting and reduction. This must be accounted for when saving the data.

    Parameters
    ----------
    epic    : int
              EPIC number of the star
    time    : array_like
              ndarray or list containing the time values
    cadence : array_like
              ndarray or list containing the cadence values
    quality : array_like
              ndarray or list containing the quality values
    fluxes  : array_like
              a list, multiple lists, or [nsets,npoints] ndarray of flux values
    errors  : array_like
              a list, multiple lists, or [nsets,npoints] ndarray of flux values
    x       : array_like
              ndarray or list of x values
    y       : array_like
              ndarray or list of y values

    Attributes
    ----------
    nsets       : int
                  Number of datasets
    npoints     : int
                  Number of datapoints
    is_periodic : bool
                  Does the flux show clear periodic variability
    ls_period   : float
                  Period of the strongest periodic variability detected
    ls_power    : float
                  Lomb-Scargle power of the strongest periodic variability detected
    """
    def __init__(self, epic, time, cadence, quality, fluxes, errors, x, y, sap_header=None):
        self.epic = epic
        self.nanmask = nm = isfinite(time) & isfinite(x) & isfinite(y)
        self.time = extract(nm, time)
        self.cadence =  extract(nm, cadence)
        self.quality =  extract(nm, quality).astype(np.int)
        self.fluxes = atleast_2d(fluxes)[:,nm]
        self.errors = atleast_2d(errors)[:,nm]
        self.x = extract(nm,x)
        self.y = extract(nm,y)
        self.sap_header = sap_header

        self.nsets   = self.fluxes.shape[0]
        self.npoints = self.fluxes.shape[1]

        self.kmask    = all(isfinite(self.fluxes),0) & (self.quality==0)
        self.fmasks   = ones([self.nsets, self.npoints], np.bool)
        self.omasks_u = ones_like(self.fmasks)
        self.omasks_d = ones_like(self.fmasks)
        self.masks    = tile(self.kmask, [self.nsets,1])
        
        self.is_periodic = False
        self.ls_period = None
        self.ls_power = None


    def mask_flares(self, flare_sigma=5, flare_erosion=5):
        """Identify and mask flares.
        
        Strong flares will screw up the period search and are easy to identify.

        Parameters
        ----------
        flare_sigma : float, optional
        flare_erosion : float, optional
        """
        for iset, (flux,mask) in enumerate(zip(self.fluxes, self.masks)):
            fmed,fstd = medsig(flux[mask])
            self.fmasks[iset,mask] = (flux[mask] - fmed) <  flare_sigma *  fstd
            self.fmasks[iset,:] = binary_erosion(self.fmasks[iset,:], border_value=1, iterations=flare_erosion)
        self._update_mask()


    def mask_outliers(self, outlier_sigma=10, outlier_mwidth=5):
        """Identify short-duration outliers.

        Parameters
        ----------
        outlier_sigma : float, optional
        outlier_mwidth : float, optional
        """
        for iset, (flux,mask) in enumerate(zip(self.fluxes, self.masks)):
            r = flux - medfilt(flux, outlier_mwidth)
            fmed,fstd = medsig(flux[mask])
            rmed,rstd = medsig(r[mask])
            self.omasks_u[iset,:] = r < outlier_sigma*rstd
            self.omasks_u[iset,:] = self.omasks_u[iset,:] | ~self.fmasks[iset,:]
            self.omasks_d[iset,:] = (1. + fmed) > 0.75
            self.omasks_d[iset,:] = self.omasks_d[iset,:] | ~self.fmasks[iset,:]
        self._update_mask()


    def search_for_periodicity(self, min_period=0.25, max_period=20, ls_min_power=100):
        """Search for periodicity using a Lomb-Scargle periodogram.

        Parameters
        ----------
        min_period : float, optional
        max_period : float, optional
        ls_min_power : float, optional
        """
        for iset, (flux,mask) in enumerate(zip(self.fluxes, self.masks)):
            freq,power,_,_,_ = fasper(self.time[mask], flux[mask], 6, 0.5)
            msize = min(freq.size,power.size)
            freq,power = freq[:msize], power[:msize]
            period = 1/freq
            pmask = (period > min_period) & (period < max_period)
            period, power = period[pmask], power[pmask]
            i = argmax(power)

            if (power[i] >ls_min_power) & (power[i] > self.ls_power):
                self.is_periodic = True
                self.ls_period = period[i]
                self.ls_power = power[i]

        return self.is_periodic


    def _update_mask(self):
        self.masks = self.kmask & self.fmasks & self.omasks_u & self.omasks_d
