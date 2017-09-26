"""Contains the K2Data class for working with K2 datasets with 1..n flux and error
arrays (using different aperture size, etc.)

"""
from __future__ import division
import numpy as np

from numpy import array, atleast_2d, zeros, ones, ones_like, argmax, all, isfinite, tile, extract, zeros_like
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import medfilt

from .ls import fasper
from .utils import medsig, fold
from .core import *

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
    
    def __init__(self, epic, time, cadence, quality, fluxes, errors, x, y, primary_header=None, data_header=None, campaign=None):
        self.epic = epic
        self.campaign = campaign
        self.nanmask = nm = isfinite(time) & isfinite(x) & isfinite(y)
        self.time = extract(nm, time)
        self.cadence =  extract(nm, cadence)
        self.quality =  extract(nm, quality).astype(np.int32)
        self.fluxes = atleast_2d(fluxes)[:,nm]
        self.errors = atleast_2d(errors)[:,nm]
        self.x = extract(nm,x)
        self.y = extract(nm,y)
        self.primary_header = primary_header
        self.data_header = data_header

        self.nsets   = self.fluxes.shape[0]
        self.npoints = self.fluxes.shape[1]

        self.is_periodic = False
        self.ls_period = None
        self.ls_power = None

        qmask = all(isfinite(self.fluxes),0) & (self.quality==0)
        self.mflags   = zeros([self.nsets, self.npoints], np.uint8)
        self.mflags[:,~qmask] |= M_QUALITY


    def mask_periodic_signal(self, center, period, duration):
        self.pmask = np.abs(fold(self.time, period, center, shift=0.5) - 0.5)*period > 0.5*duration
        self.mflags[:,~self.pmask] |= M_PERIODIC

    #def __str__(self):
    #    return str(id(self))
