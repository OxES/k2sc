from __future__ import division

import math as m
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl

from scipy.optimize import fmin, fmin_powell, minimize
from numpy import (any, array, asarray, ones, ones_like, zeros, isfinite, inf, concatenate, arange, unique, delete,
                   dot, median, abs, std, nan, diag, log, where, identity, s_, sqrt)
from numpy.random import permutation
from matplotlib.pyplot import subplots, setp

from numpy.linalg.linalg import LinAlgError
from .gp import GeorgeGP, SplitGP
from .kernels import BasicKernel
from .utils import medsig
from .dtdata import DtData

class Detrender(object):
    def __init__(self, flux, inputs, mask=None, p0=None, kernel=None, splits=[], tr_nrandom=200, tr_bspan=50, tr_nblocks=6):
        self.data   = DtData(flux, inputs, mask)
        self.kernel = kernel or BasicKernel()
        self.gp     = SplitGP(self.kernel, splits) if splits is not None else GeorgeGP(self.kernel)
        self.tr_data  = self.data.create_training_set(tr_nrandom, tr_bspan, tr_nblocks)
        self.gp.set_inputs(self.tr_data.masked_inputs)

    ## ======================
    ##  Convenience routines
    ## ======================

    @property
    def flux(self):
        return self.data.masked_flux

    @property
    def time(self):
        return self.data.masked_time


    ## =====================
    ##  Detrending routines
    ## =====================

    def covariance_matrix(self, pv=None, inputs=None, separate=False):
        inputs = inputs if inputs is not None else self.tr_data.masked_inputs
        self.gp.compute(inputs, pv)
        return self.gp._covariance_matrix(inputs, separate=separate)
    
        
    def neglnposterior(self, pv, training=True):
        if any(pv < self.kernel.lims[0]) or any(self.kernel.lims[1] < pv):
            return inf
        ds = self.tr_data if training else self.data
        try:
            lnlike = self.gp.lnlikelihood(pv, ds.masked_normalised_flux, ds.masked_inputs)
            return -(self.kernel.ln_prior(pv) + lnlike)
        except LinAlgError:
            return inf


    def train(self, pv0=None, disp=False):
        pv0 = pv0 if pv0 is not None else self.kernel.pv0
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
    

    ## ===================
    ##  Plotting routines
    ## ===================

    def plot_xy(self, pv=None, ax=None, plot_wireframe=False):
        """Plot the x and y points for the whole dataset and the training set.
        """
        if ax is None:
            fig,ax = subplots(1,1, figsize=(10,10))
        
        if pv is None:
            ax.tripcolor(self.data.mx, self.data.my, ones(self.data.nptm), vmin=0, vmax=1)
            if plot_wireframe:
                ax.triplot(self.data.mx, self.data.my, color='w')
        else:
            mt, mp = self.compute_components(pv)
            ax.tripcolor(self.data.mx, self.data.my, mp)
            
        ax.plot(self.tr_data.mx, self.tr_data.my, 'o', ms=3, c='k', mec='w')
        return ax
    
    
    def plot_t(self, pv=None, ax=None):
        """ Plot the flux as a function of time for the whole dataset and the training set.
        """
        if ax is None:        
            fig, ax = subplots(1,1)
            
        fm = self.data.flux_median
        fmin = self.data.masked_flux.min()
        fmax = self.data.masked_flux.max()
        fptp = self.data.masked_flux.ptp()

        ax.plot(self.data.mt, self.data.mf, c='0.75', lw=1)
        ax.plot(self.tr_data.ut, self.tr_data.uf, '.k', ms=6)
        setp(ax, ylim=(0.999*fmin,1.001*fmax))

        if pv is not None:
            fd  = self.detrend_spatial(pv)
            fd +=  fm - np.nanmedian(fd)
            mm = isfinite(fd)
            ax.plot(self.data.unmasked_time[mm], fd[mm] - 0.7*fptp, alpha=0.75, lw=1)
            setp(ax, ylim=(0.999*(fmin-0.7*fptp), 1.001*fmax))

        setp(ax, xlim=self.data.mt[[0,-1]], xlabel='Time', ylabel='Flux')
        return ax

    
    def plot_report(self, pv, tid, fname=None, maxpt=350):
        lmargin, rmargin = 0.12, 0.03

        fig = pl.figure(figsize=(8.3,11.7))
        fig.text(0.04, 0.965, 'EPIC {:9d}'.format(tid), va='top', size=24, color='w', weight='bold')

        ax = fig.add_axes([0,0,1,1])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_zorder(-1000)
        ax.add_patch(pl.Rectangle((0,0.92), 1, 0.08, fill=True))

        ax_a = fig.add_axes([lmargin,0.25,1-lmargin-rmargin,0.3])
        ax_x = fig.add_axes([lmargin,0.05,1-lmargin-rmargin,0.1])
        ax_y = fig.add_axes([lmargin,0.15,1-lmargin-rmargin,0.1])
        ax_c = fig.add_axes([0.55,0.6,0.45-rmargin, 0.3])
 
        ax_x.plot(self.data.masked_time, self.data.mx, lw=1)
        ax_y.plot(self.data.masked_time, self.data.my, lw=1)

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
        ax_a.plot(*self.data.outliers, ls='', marker='o', ms=6)
        ax_a.plot(self.data.unmasked_time[ms], (fd[ms] - 0.7*fptp)/fm, lw=1)

        ax_a.plot(self.time, (mp-1.4*fptp)/fm, lw=1)

        samples = permutation(self.time.size)[:maxpt]

        ax_c.tripcolor(self.data.mx[samples], self.data.my[samples], mp[samples])
        ax_c.plot(self.tr_data.mx, self.tr_data.my, '.', ms=3, c='w', alpha=0.8)
        ax_c.plot(self.tr_data.mx, self.tr_data.my, '.', ms=1.5, c='k')

        setp(ax_a, ylim=(0.999*(fmin-1.4*fptp)/fm, 1.001*fmax/fm))
        setp(ax_a.get_xticklabels()+ax_y.get_xticklabels(), visible=False)
        setp(ax_x, xlabel='Time', ylabel='X')
        setp(ax_c, xlabel='X', ylabel='Y')
        setp([ax_a,ax_x,ax_y], xlim=self.time[[0,-1]])
        setp(ax_a, ylabel='Normalised flux')
        setp(ax_y, ylabel='Y')

        if fname:
            fig.savefig(fname)
