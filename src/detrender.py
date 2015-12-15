from __future__ import division

from scipy.optimize import fmin, fmin_powell, minimize
from george import GP
from george.kernels import ExpSquaredKernel as ESK
from george.kernels import ExpSine2Kernel as ESn2K

import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl

from numpy import (any, array, asarray, ones, ones_like, zeros, isfinite, inf, concatenate, arange, unique, delete,
                   dot, median, abs, std, nan, diag, log, where, identity, s_, sqrt)
from numpy.random import permutation
from matplotlib.pyplot import subplots, setp

from gp import GeorgeGP, SplitGP
from kernels import BasicKernel
from utils import medsig

c_ob = '#002147'
c_bo = '#BF5700'

class Detrender(object):
    def __init__(self, flux, inputs, mask=None, p0=None, kernel=None, splits=[], tr_nrandom=100, tr_bspan=50, tr_nblocks=5):
        self.data   = DtData(flux, inputs, mask)
        self.kernel = kernel or BasicKernel()
        self.gp     = SplitGP(self.kernel, splits) if splits is not None else GeorgeGP(self.kernel)
        #self.mask_outliers(max_sigma=5, pv=self.kernel._pv)
        self.tr_data  = self.data.create_training_set(tr_nrandom, tr_bspan, tr_nblocks)
        self.gp.set_inputs(self.tr_data.masked_inputs)

    @property
    def flux(self):
        return self.data.masked_flux

    @property
    def time(self):
        return self.data.masked_time

    @property
    def outliers(self):
        return self.data.outliers


    def mask_outliers(self, max_sigma=5, pv=None):
        pv = self.kernel.pv0 if pv is None else pv
        qm = self.data.quality_mask
        self.gp.compute(self.data.masked_inputs, pv)
        self.gp._compute_alpha(self.data.masked_normalised_flux)
        predicted_flux = self.gp.predict(self.data.unmasked_inputs, mean_only=True)
        residuals = self.data.unmasked_normalised_flux - predicted_flux
        flux_median, flux_std = medsig(residuals[qm])
        outlier_mask = ones_like(self.data.mask)
        outlier_mask[qm] = abs(residuals[qm]-flux_median) < (max_sigma * flux_std)
        self.data.set_mask(outlier_mask)
        return outlier_mask.sum()


    def covariance_matrix(self, pv=None, inputs=None, separate=False):
        inputs = inputs if inputs is not None else self.tr_data.masked_inputs
        self.gp.compute(inputs, pv)
        return self.gp._covariance_matrix(inputs, separate=separate)
    
        
    def neglnposterior(self, pv, training=True):
        if any(pv < self.kernel.lims[0]) or any(self.kernel.lims[1] < pv):
            return inf
        ds = self.tr_data if training else self.data
        return -(self.kernel.ln_prior(pv) + 
                 self.gp.lnlikelihood(pv, ds.masked_normalised_flux, ds.masked_inputs))
    

    def train(self, pv0=None, disp=False):
        pv0 = pv0 if pv0 is not None else self.kernel.pv0
        #mres = minimize(self.neglnposterior, pv0, method='SLSQP')
        mres = minimize(self.neglnposterior, pv0, method='Powell')
        self.tr_pv = mres.x.copy()
        return self.tr_pv, mres.success

    
    def predict(self, pv, inputs=None, components=False, mean_only=True):
        inputs  = inputs if inputs is not None else self.data.unmasked_inputs
        self.gp.compute(self.data.masked_inputs, pv)
        self.gp._compute_alpha(self.data.masked_normalised_flux)

        if components:
            mu_time, mu_pos = self.gp.predict_components(inputs)
            return ((1. + mu_time) * self.data._fm,
                    (1. + mu_pos)  * self.data._fm)
        else:
            return self.gp.predict(inputs, mean_only=mean_only)
    

    def detrend_spatial(self, pv):
        mt, mp = self.compute_components(pv)
        flux = self.data.unmasked_flux.copy()
        flux[self.data.mask] += -mp + median(mp)
        flux[~self.data.mask] = nan
        return flux
    
    def plot_xy(self, pv=None, ax=None):
        if ax is None:
            fig,ax = subplots(1,1, figsize=(10,10))
        
        if pv is None:
            ax.tripcolor(self.data.masked_inputs[:,1], self.masked_data.inputs[:,2], ones(self.npt), vmin=0, vmax=1)
            ax.triplot(self.data.masked_inputs[:,1], self.data.masked_inputs[:,2], color='w')
        else:
            mt, mp = self.compute_components(pv)
            ax.tripcolor(self.data.masked_inputs[:,1], self.data.masked_inputs[:,2], mp)
            
        ax.plot(self.tr_data.masked_inputs[:,1], self.tr_data.masked_inputs[:,2],'.', ms=8, c='w', alpha=0.8)
        ax.plot(self.tr_data.masked_inputs[:,1], self.tr_data.masked_inputs[:,2],'.', ms=5, c='k')
        return ax
    
    
    def plot_t(self, pv=None, ax=None):
        if ax is None:        
            fig,ax = subplots(1,1)
            
        fm = median(self.data.masked_flux)
        fmin = self.data.masked_flux.min()
        fmax = self.data.masked_flux.max()
        fptp = self.data.masked_flux.ptp()

        ax.plot(self.data.masked_time, self.data.masked_flux, c='0.75', lw=1)
        ax.plot(self.tr_data.unmasked_time, self.tr_data.unmasked_flux, '.k', ms=6)
        ax.plot(*self.data.outliers, c='#BF5700', ls='', marker='o', ms=6)

        setp(ax, ylim=(0.999*fmin,1.001*fmax))

        if pv is not None:
            fd  = self.detrend_spatial(pv)
            fd +=  fm - np.nanmedian(fd)
            mm = isfinite(fd)
            ax.plot(self.data.unmasked_time[mm], fd[mm] - 0.7*fptp, alpha=0.75, lw=1, c=c_ob)
            setp(ax, ylim=(0.999*(fmin-0.7*fptp), 1.001*fmax))

        setp(ax, xlim=self.data.inputs[[0,-1],0], xlabel='Time', ylabel='Flux')
        return ax

    
    def plot_report(self, pv, tid, fname=None, maxpt=350):
        lmargin, rmargin = 0.12, 0.03

        fig = pl.figure(figsize=(8.3,11.7))
        fig.text(0.04, 0.965, 'EPIC {:9d}'.format(tid), va='top', size=24, color='w', weight='bold')

        ax = fig.add_axes([0,0,1,1])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_zorder(-1000)
        ax.add_patch(pl.Rectangle((0,0.92), 1, 0.08, fill=True, color=c_ob))

        ax_a = fig.add_axes([lmargin,0.25,1-lmargin-rmargin,0.3])
        ax_x = fig.add_axes([lmargin,0.05,1-lmargin-rmargin,0.1])
        ax_y = fig.add_axes([lmargin,0.15,1-lmargin-rmargin,0.1])
        ax_c = fig.add_axes([0.55,0.6,0.45-rmargin, 0.3])

 
        ax_x.plot(self.data.masked_time, self.data.masked_inputs[:,1], lw=1, c=c_ob)
        ax_y.plot(self.data.masked_time, self.data.masked_inputs[:,2], lw=1, c=c_ob)

  
        ## Compute stuff
        ## -------------
        fm = median(self.data.masked_flux)
        fmin = self.data.masked_flux.min()
        fmax = self.data.masked_flux.max()
        fptp = self.data.masked_flux.ptp()

        mt, mp = self.compute_components(pv)
        ms = self.data.mask
        fd = self.data.unmasked_flux.copy()
        fd[ms] += -mp + median(mp)
        fd[~ms] = nan
        fd +=  fm - np.nanmedian(fd)

        ## Plot A
        ## ------
        ax_a.plot(self.data.masked_time, self.data.masked_flux/fm, c='0.75', lw=1)
        ax_a.plot(self.tr_data.unmasked_time, self.tr_data.unmasked_flux/fm, '.k', ms=6)
        ax_a.plot(*self.data.outliers, c=c_bo, ls='', marker='o', ms=6)
        ax_a.plot(self.data.unmasked_time[ms], (fd[ms] - 0.7*fptp)/fm, lw=1, c=c_ob)

        ax_a.plot(self.time, (mp-1.4*fptp)/fm, c=c_bo, lw=1)

        samples = permutation(self.time.size)[:maxpt]

        ax_c.tripcolor(self.data.masked_inputs[samples,1], self.data.masked_inputs[samples,2], mp[samples])
        ax_c.plot(self.tr_data.masked_inputs[:,1], self.tr_data.masked_inputs[:,2],'.', ms=3, c='w', alpha=0.8)
        ax_c.plot(self.tr_data.masked_inputs[:,1], self.tr_data.masked_inputs[:,2],'.', ms=1.5, c='k')


        setp(ax_a, ylim=(0.999*(fmin-1.4*fptp)/fm, 1.001*fmax/fm))
        setp(ax_a.get_xticklabels()+ax_y.get_xticklabels(), visible=False)
        setp(ax_x, xlabel='Time', ylabel='X')
        setp(ax_c, xlabel='X', ylabel='Y')
        setp([ax_a,ax_x,ax_y], xlim=self.time[[0,-1]])
        setp(ax_a, ylabel='Normalised flux')
        setp(ax_y, ylabel='Y')

        if fname:
            fig.savefig(fname)



class DtData(object):
    """Utility class that encapsulates the fluxes and inputs for a detrender.
    """
    def __init__(self, flux, inputs, mask=None):
        self.flux  = array(flux)
        self.inputs = array(inputs)
        self.quality_mask = array(mask) if mask is not None else ones(self.flux.size, np.bool) 
        self.outlier_mask = ones(self.flux.size, np.bool)
        self.mask  =  self.quality_mask & self.outlier_mask
        self.npt = self.flux.size
        self._fm, self._fs =  medsig(self.flux[self.mask])

        assert self.flux.ndim == 1, 'The flux array for DtData should be 1D [npt]'
        assert self.inputs.ndim == 2, 'The input array for DtData should be 2D [npt,3]'
        assert self.inputs.shape[1] == 3, 'The input array for DtData should be 2D with the shape [npt,3]'
        assert self.quality_mask.ndim == 1, 'The mask array for DtData should be 1D [npt]'
        ##TODO: Change the input array axes from [npt,3] to [3,npt]

        
    def create_training_set(self, nrandom=100, nblocks=5, bspan=50):
        """Creates a new DtData instance containing a subset of the original data.
        """
        ids = (unique((permutation(self.npt-bspan)[:nblocks].reshape([nblocks,-1]) + arange(bspan)).ravel()))
        ids = np.sort(concatenate([ids, permutation(delete(arange(self.npt),ids))[:nrandom]]))
        tr_mask = zeros(self.npt, np.bool)
        tr_mask[ids] = 1
        tr_mask = (tr_mask & self.mask)
        trd = DtData(self.unmasked_flux[tr_mask], self.unmasked_inputs[tr_mask])
        trd._fm, trd._fs = self._fm, self._fs
        return trd


    def set_mask(self, outlier_mask):
        self.outlier_mask = outlier_mask
        self.mask = self.quality_mask & self.outlier_mask

    @property
    def masked_time(self):
        return self.time[self.mask]

    @property
    def unmasked_time(self):
        return self.time

    @property
    def masked_flux(self):
        return self.flux[self.mask]

    @property
    def unmasked_flux(self):
        return self.flux

    @property
    def masked_normalised_flux(self):
        return self.flux[self.mask] / self._fm - 1.

    @property
    def unmasked_normalised_flux(self):
        return self.flux / self._fm - 1.

    @property
    def masked_inputs(self):
        return self.inputs[self.mask,:]

    @property
    def unmasked_inputs(self):
        return self.inputs

    @property
    def outliers(self):
        return self.inputs[~self.outlier_mask,0], self.flux[~self.outlier_mask]

    @property
    def time(self):
        return self.inputs[:,0]

