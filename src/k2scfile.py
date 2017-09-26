import numpy as np
import astropy.io.fits as pf

from .core import *

def submed(a):
    return a - np.nanmedian(a)

class K2SCFile(object):
    """
    Utility class to work with K2SC output files.
        
    """
    def __init__(self, filename, normalize=False, remove_nans=True, remove_outliers=False):
        """
        Initialize K2SCFile
        
        Parameters
        ----------
        
        filename : string
            K2SC file path
                
        normalize : bool, optional
            Normalize the light curves
                
        remove_nans : bool, optional
            Remove points with either not-finite time or flux

        remove_outliers: bool, optional
            Remove points marked as outliers by K2SC
        """
        self._fname  = filename
        self._phead  = pf.getheader(filename, 0)
        self._dhead  = pf.getheader(filename, 1)
        self._data   = pf.getdata(filename, 1)
        self._time   = self._data.time
        self._flux   = self._data.flux
        self._tposi  = self._data.trposi
        self._ttime  = self._data.trtime
        
        self._minf = np.isfinite(self._flux) & np.isfinite(self._time)
        self._mper = ~(self._data.mflags & M_PERIODIC).astype(np.bool)
        self._ou = ~(self._data.mflags & M_OUTLIER_U).astype(np.bool)
        self._od = ~(self._data.mflags & M_OUTLIER_D).astype(np.bool)
        self._om = self._ou & self._od
        
        self._normalize  = normalize
        self._remove_nans = remove_nans
        self._remove_outliers = remove_outliers
        

    def _n(self, a, remove_outliers=None):
        if self._normalize:
            return self._clean(a / np.nanmedian(a), remove_outliers=remove_outliers)
        else:
            return self._clean(a, remove_outliers=remove_outliers)
                
    def _clean(self, a, remove_nans=None, remove_outliers=None):
        mask = np.ones(a.size, np.bool)
        remove_nans = remove_nans if remove_nans is not None else self._remove_nans
        remove_outliers = remove_outliers if remove_outliers is not None else self._remove_outliers
        if remove_nans:
            mask &= self._minf
        if remove_outliers:
            mask &= self._om    
        return a[mask]
        
    def set_normalize(self, v):
        self._normalize = bool(v)
        
    def set_remove_nans(self, v):
        self._remove_nans = bool(v)
        
    def set_remove_outliers(self, v):
        self._remove_outliers = bool(v)
        
    @property
    def time(self):
        """Time array"""
        return self._clean(self._time)
                
    @property
    def original_flux(self):
        """Original flux"""
        return self._n(self._flux)
        
    @property
    def detrended_flux(self):
        """Flux with the position trend removed"""
        return self._n(self._flux - submed(self._tposi))
    
    @property
    def residual_flux(self):
        """Flux with both the position and time trends removed"""
        return self._n(self._flux - submed(self._tposi) - submed(self._ttime))
        
    @property
    def position_trend(self):
        """Position trend from K2SC"""
        return self._clean(self._tposi)
    
    @property
    def time_trend(self):
        """Time trend from K2SC"""
        return self._clean(self._ttime)
    
    @property
    def outlier_mask(self):
        """Mask that excludes points marked as outliers by K2SC"""
        return self._clean(self._om, remove_outliers=False)
    
    @property
    def outlier_mask_up(self):
        """Mask that excludes points marked as upwards outliers by K2SC"""
        return self._clean(self._ou, remove_outliers=False)

    @property
    def outlier_mask_down(self):
        """Mask that excludes points marked as downwards outliers by K2SC"""
        return self._clean(self._od, remove_outliers=False)

    @property
    def original_outliers(self):
        """Time and original flux datapoints marked as outliers by K2SC"""
        return (self._clean(self._time, remove_outliers=False)[~self.outlier_mask], 
                self._n(self._flux, remove_outliers=False)[~self.outlier_mask])
    
    @property
    def detrended_outliers(self):
        """Time and detrended flux datapoints marked as outliers by K2SC"""
        return (self._clean(self._time, remove_outliers=False)[~self.outlier_mask], 
                self._n(self._flux - submed(self._tposi), remove_outliers=False)[~self.outlier_mask])
    
    @property
    def outliers_up(self):
        """Points marked as upwards outliers by K2SC"""
        return self._time[~self._ou], self._flux[~self._ou]
    
    @property
    def outliers_down(self):
        """Points marked as downwards outliers by K2SC"""
        return self._time[~self._od], self._flux[~self._od]
