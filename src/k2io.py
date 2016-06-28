"""Module for reading K2 data from different sources and writing k2sc FITS files.

    This module contains severals "readers" that can read a file containing a K2 light
    curve, and return a properly initialised K2Data instance. The module also contains
    a writer class for writing FITS files containing the K2 data and the detrending
    time series.

    Copyright (C) 2016  Suzanne Aigrain

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings
import numpy as np
import re
import astropy.io.fits as pf
from os.path import basename, splitext
from datetime import datetime
from collections import namedtuple

from k2data import K2Data

warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)

## ===  READERS  ===
## =================

class DataReader(object):
    extensions = []
    ndatasets = 0
    fn_out_template = None
    
    def __init__(self, fname):
        raise NotImplementedError

    @classmethod
    def read(cls, fname, **kwargs):
        raise NotImplementedError

    @classmethod
    def can_read(cls, fname):
        raise NotImplementedError
                
    @classmethod
    def is_extension_valid(cls, fname):
        return splitext(basename(fname))[1].lower() in cls.extensions
        
        
class AMCReader(DataReader):
    extensions = ['.bin', '.dat']
    ndatasets = None  
    fn_out_template = 'EPIC_{:9d}_amc.fits'

    @classmethod
    def read(cls, fname, **kwargs):
        epic = int(re.findall('EPIC_([0-9]+)_', basename(fname))[0])
        data = np.loadtxt(fname, skiprows=1)
        return K2Data(epic,
                      time    = data[:,0],
                      cadence = data[:,1],
                      quality = data[:,-1],
                      fluxes  = data[:,2:-3:2].T,
                      errors  = data[:,3:-3:2].T,
                      x       = data[:,-3],
                      y       = data[:,-2])

    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        with open(fname) as f:
            header = f.readline().lower().split()
            head_ok = all([cn in header for cn in 'dates cadences xpos ypos quality'.split()])
        return ext_ok and head_ok
    
    
class MASTReader(DataReader):
    extensions = ['.fits', '.fit']
    ndatasets = 1
    fn_out_template = 'EPIC_{:9d}_mast.fits'
    allowed_types = ['sap', 'pdc']
    fkeys = dict(sap = 'sap_flux', pdc = 'pdcsap_flux')

    @classmethod
    def read(cls, fname, sid, **kwargs):
        ftype = kwargs.get('type', 'pdc').lower()
        assert ftype in cls.allowed_types, 'Flux type must be either `sap` or `pdc`'
        fkey = cls.fkeys[ftype]

        try:
            epic = int(re.findall('ktwo([0-9]+)-c', basename(fname))[0])
        except:
            epic = int(re.findall('C([0-9]+)_smear', basename(fname))[0][2:]) # for smear
        data = pf.getdata(fname, 1)
        head = pf.getheader(fname, 0)
        return K2Data(epic,
                      time    = data['time'],
                      cadence = data['cadenceno'],
                      quality = data['sap_quality'],
                      fluxes  = data[fkey],
                      errors  = data[fkey+'_err'],
                      x       = data['pos_corr1'],
                      y       = data['pos_corr2'],
                      sap_header = head)    
    
    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        else:
            h = pf.getheader(fname, 1)
            fmt_ok = 'SAP_FLUX' in h.values()
            return fmt_ok
        

class SPLOXReader(DataReader):
    extensions = ['.fits', '.fit']
    ndatasets = 6
    fn_out_template = 'STAR_{:09d}.fits'
    _cache = None
    _nstars = None
    
    @classmethod
    def read(cls, fname, sid, **kwargs):
        cache = namedtuple('K2Cache', 'fname objno nobj nexp time cadence quality fluxes errors x y header')
        if not cls._cache or cls._cache.fname != fname:
            with pf.open(fname) as fin:
                data = fin[1].data
                nobj = fin[1].header['naxis2']
                nexp = fin[2].header['naxis2']
                fluxes = (1. + data['f'].reshape([nobj,nexp,-1])) * data['f_med'][:,np.newaxis,:]
                fluxes = np.swapaxes(fluxes, 1, 2) # Fluxes as [nobj,napt,nexp] ndarray
                cls._nstars = nobj         
                cls._cache = cache(fname, fin[1].data['objno'], nobj, nexp, fin[2].data['mjd_obs'],
                                   fin[2].data['cadence'], np.zeros(nexp, np.int), fluxes,
                                   np.zeros_like(fluxes), data['x'], data['y'], fin[0].header)

        return K2Data(cls._cache.objno[sid],
                      time=cls._cache.time[:-1],
                      cadence=cls._cache.cadence[:-1],
                      quality=cls._cache.quality[:-1],
                      fluxes=cls._cache.fluxes[sid,:,:-1],
                      errors=cls._cache.errors[sid,:,:-1],
                      x=cls._cache.x[sid,:-1],
                      y=cls._cache.y[sid,:-1],
                      sap_header=cls._cache.header)    

    
    @classmethod
    def nstars(cls, fname):
        return pf.getval(fname, 'naxis2', 1)

    
    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        else:
            h = pf.getheader(fname, 1)
            fmt_ok = 'F_SCATTER' in h.values()
            return fmt_ok



## ===  WRITERS  ===
## =================

class FITSWriter(object):
    @classmethod
    def write(cls, fname, splits, data, dtres):

        def unpack(arr):
            aup = np.full(data.nanmask.size, np.nan)
            aup[data.nanmask] = arr
            return arr

        C = pf.Column
        
        cols = [C(name='time',     format='D', array=unpack(data.time)),
                C(name='cadence',  format='J', array=unpack(data.cadence)),
                C(name='quality',  format='J', array=unpack(data.quality)),
                C(name='x',        format='D', array=unpack(data.x)),
                C(name='y',        format='D', array=unpack(data.y))]

        for i in range(data.nsets):
            cols.extend([C(name='flux_%d'    %(i+1), format='D', array=unpack(data.fluxes[i])),
                         C(name='error_%d'   %(i+1), format='D', array=unpack(data.errors[i])),
                         C(name='mflags_%d'  %(i+1), format='B', array=unpack(data.mflags[i])),
                         C(name='trend_t_%d' %(i+1), format='D', array=unpack(dtres[i].tr_time)),
                         C(name='trend_p_%d' %(i+1), format='D', array=unpack(dtres[i].tr_position))])

        hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
        hdu.header['extname'] = 'k2_detrend'
        hdu.header['object'] = data.epic
        hdu.header['epic']   = data.epic
        hdu.header['splits'] = str(splits)
        for i in range(data.nsets):
            hdu.header['cdpp%dr'%(i+1)] = dtres[i].cdpp_r
            hdu.header['cdpp%dt'%(i+1)] = dtres[i].cdpp_t
            hdu.header['cdpp%dc'%(i+1)] = dtres[i].cdpp_c
            hdu.header['ap%d_warn'%(i+1)] = dtres[i].warn
        hdu.header['ker_name'] = dtres[0].detrender.kernel.name
        hdu.header['ker_pars'] = ' '.join(dtres[0].detrender.kernel.names)
        hdu.header['ker_eqn']  = dtres[0].detrender.kernel.eq
        for i in range(data.nsets):
            hdu.header['ker_hps%d'%(i+1)] = str(dtres[i].detrender.tr_pv).replace('\n', '')
        hdu.header['origin'] = 'SPLOX: Stars and Planets at Oxford'
        hdu.header['program'] = 'k2_syscor v0.8'
        hdu.header['date']   = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
        
        primary_hdu = pf.PrimaryHDU(header=data.sap_header)
        hdu_list = pf.HDUList([primary_hdu, hdu])
        hdu_list.writeto(fname, clobber=True)


readers = [AMCReader,MASTReader,SPLOXReader]

def select_reader(fname):
    for R in readers:
        if R.can_read(fname):
            return R
    return None
