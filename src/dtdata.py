from __future__ import division

import math as m
import numpy as np

from numpy import nan, any, array, asarray, ones, arange, delete, where
from numpy.random import permutation
from .utils import medsig
from .core import *

class DtData(object):
    """Utility class that encapsulates the fluxes and inputs for a detrender.
    """
    def __init__(self, flux, inputs, mask=None):
        self._flux   = array(flux)
        self._inputs = array(inputs)
        self._mask   = array(mask) if mask is not None else ones(self._flux.size, np.bool) 
        self._fm, self._fs = medsig(self.masked_flux)
        self.nptm     = self.masked_flux.size
        self.nptu     = self.unmasked_flux.size

        assert self._flux.ndim == 1, 'The flux array for DtData should be 1D [npt]'
        assert self._inputs.ndim == 2, 'The input array for DtData should be 2D [npt,3]'
        assert self._inputs.shape[1] == 3, 'The input array for DtData should be 2D with the shape [npt,3]'

        self.mf, self.uf = self.masked_flux,   self.unmasked_flux
        self.mi, self.ui = self.masked_inputs, self.unmasked_inputs
        self.mt, self.ut = self.masked_time,   self.unmasked_time
        self.mx, self.ux = self.masked_x,      self.unmasked_x
        self.my, self.uy = self.masked_y,      self.unmasked_y

        
    def create_training_set(self, nrandom=100, nblocks=5, bspan=50, max_tries=100):
        """Creates a new DtData instance containing a subset of the original data.

        Creates a new DtData instance made of nblocks non-overlapping chunks of bspan
        consecutive datapoints (excluding the masked points) and nrandom datapoints
        drawn randomly from the leftover points. This is done in order to try to include
        both short- and long-time-scale trends to the training set.

        Parameters
        ----------
        nrandom : int, optional
                  Number of randomly drawn points

        nblocks : int, optional
                  Number of blocks to draw

        bspan   : int, optional
                  Size of a block

        max_tries : int, optional
                    Maximum number of tries to create a consecutive block that doesn't
                    overlap with any of the previously created blocks

        Returns
        -------
        DtData
            The training dataset as a new DtData instance.
        
        """
        ids = where(self.mask)[0]                    ## Get the masked indices
        bstarts = np.full(nblocks, -bspan, np.int)   ## Starting indices for blocks

        ## Chunks
        ## ------
        ## Attempt to draw nblock non-overlapping chunks of bspan consecutive points
        ## from the data excluding the masked points. If this fails, the leftover
        ## points will be added to the random sample set.
        i, j = 0, 0
        while True:
            if j == nblocks or i > max_tries:
                break

            idx = np.random.randint(0, ids.size-bspan)
            if not any((idx >= bstarts) & (idx < bstarts+bspan)):
                bstarts[j] = idx
                j += 1
            i += 1
    
        block_ids = (np.sort(bstarts)[:,np.newaxis] + arange(bspan)[np.newaxis,:]).ravel()
        npt_blocks = block_ids.size
        npt_random = min(ids.size, nrandom+bspan*nblocks-npt_blocks)

        idb = ids[block_ids]
        idr = np.sort((permutation(delete(ids, block_ids))[:npt_random]))

        tr_ids = np.sort(np.r_[idr, idb])
        trd = DtData(self.unmasked_flux[tr_ids], self.unmasked_inputs[tr_ids])
        trd._fm, trd._fs = self._fm, self._fs
        return trd


    @property
    def flux_median(self):
        return self._fm
        
    @property
    def flux_std(self):
        return self._fs

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        mask = array(mask)
        assert mask.ndim == 1, 'The mask array for DtData should be 1D'
        assert mask.size == self.flux.size,  'The mask array for DtData should have the same size as the flux array'
        assert mask.dtype == np.bool,  'The mask array should be boolean'
        self._mask = mask

    @property
    def masked_time(self):
        return self._inputs[self.mask,0]

    @property
    def unmasked_time(self):
        return self._inputs[:,0]

    @property
    def masked_flux(self):
        return self._flux[self.mask]

    @property
    def unmasked_flux(self):
        return self._flux

    @property
    def masked_normalised_flux(self):
        return self._flux[self.mask] / self._fm - 1.

    @property
    def unmasked_normalised_flux(self):
        return self._flux / self._fm - 1.

    @property
    def masked_inputs(self):
        return self._inputs[self.mask, :]

    @property
    def unmasked_inputs(self):
        return self._inputs

    @property
    def masked_x(self):
        return self._inputs[self.mask, 1]

    @property
    def unmasked_x(self):
        return self._inputs[:, 1]

    @property
    def masked_y(self):
        return self._inputs[self.mask, 2]

    @property
    def unmasked_y(self):
        return self._inputs[:, 2]
