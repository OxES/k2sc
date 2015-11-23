"""Module for reading K2 data from different sources and writing k2sc FITS files.
The module contains severals "readers" that can read a file containing a K2 light
curve, and return a properly initialised K2Data instance. The module also contains
a writer class for writing FITS files containing the K2 data and the detrending
time series.

"""

import warnings
import numpy as np
import re
import pyfits as pf
from os.path import basename, splitext
from datetime import datetime

from k2data import K2Data

warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)

pf_version = float(re.findall('^([0-9]\.[0-9])\.*', pf.__version__)[0])

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
    ndatasets = 2  
    fn_out_template = 'EPIC_{:9d}_amc.fits'

    @classmethod
    def read(cls, fname, **kwargs):
        epic = int(re.findall('EPIC_([0-9]+)_', basename(fname))[0])
        data = np.loadtxt(fname, skiprows=1)
        return K2Data(epic,
                      time=data[:,0],
                      cadence=data[:,1],
                      quality=data[:,8],
                      fluxes=data[:,[2,4]].T,
                      errors=data[:,[3,5]].T,
                      x=data[:,6],
                      y=data[:,7])

    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        with open(fname) as f:
            fmt_ok = len(f.readline().split()) == 9
        return ext_ok and fmt_ok
    
    
class MASTReader(DataReader):
    extensions = ['.fits', '.fit']
    ndatasets = 1
    fn_out_template = 'EPIC_{:9d}_sap.fits'
    allowed_types = ['sap', 'pdc']

    @classmethod
    def read(cls, fname, **kwargs):
        ftype = 'sap_flux' if kwargs.get('type','sap').lower() == 'sap' else 'pdcsap_flux'
        epic = int(re.findall('ktwo([0-9]+)-c', basename(fname))[0])
        data = pf.getdata(fname, 1)
        head = pf.getheader(fname, 0)
        return K2Data(epic,
                      time=data['time'],
                      cadence=data['cadenceno'],
                      quality=data['sap_quality'],
                      fluxes=data[ftype],
                      errors=data[ftype+'_err'],
                      x=data['pos_corr1'],
                      y=data['pos_corr2'],
                      sap_header=head)    
    
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
    fn_out_template = 'EPIC_{:9d}_reb.fits'

    @classmethod
    def read(cls, fname, **kwargs):
        ftype = 'sap_flux' if kwargs.get('type','sap').lower() == 'sap' else 'pdcsap_flux'
        epic = int(re.findall('ktwo([0-9]+)-c', basename(fname))[0])
        with pf.open(fname) as fin:
            data = fin[1].data
            nobj = fin[1].header['naxis2']
            nexp = fin[2].header['naxis2']
            fluxes = (1. + data['f'].reshape([nobj,nexp,-1])) * data['f_med'][:,np.newaxis,:]

            return K2Data(fin[1].data['objno'],
                          time=fin[2].data['mjd_obs'],
                          cadence=fin[2].data['cadence'],
                          quality=np.zeros(nexp),
                          fluxes=fluxes,
                          errors=np.zeros_like(fluxes),
                          x=data['x'],
                          y=data['y'],
                          sap_header=fin[0].header)    
    
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

        columns = [pf.Column(name='time',     format='D', array=unpack(data.time)),
                   pf.Column(name='cadence',  format='I', array=unpack(data.cadence)),
                   pf.Column(name='quality',  format='I', array=unpack(data.quality)),
                   pf.Column(name='x',        format='D', array=unpack(data.x)),
                   pf.Column(name='y',        format='D', array=unpack(data.y))]

        for i in range(data.nsets):
            columns.extend([pf.Column(name='flux_%d'    %(i+1), format='D', array=unpack(data.fluxes[i])),
                            pf.Column(name='error_%d'   %(i+1), format='D', array=unpack(data.errors[i])),
                            pf.Column(name='mask_ol_%d' %(i+1), format='D', array=unpack(dtres[i].detrender.data.outlier_mask)),
                            pf.Column(name='trend_t_%d' %(i+1), format='D', array=unpack(dtres[i].tr_time)),
                            pf.Column(name='trend_p_%d' %(i+1), format='D', array=unpack(dtres[i].tr_position))])

        if pf_version >= 3.3:
            hdu = pf.BinTableHDU.from_columns(pf.ColDefs(columns))
        else:
            hdu = pf.new_table(columns)

        hdu.header['extname'] = 'k2_detrend'
        hdu.header['object'] = data.epic
        hdu.header['epic']   = data.epic
        hdu.header['splits'] = str(splits)
        for i in range(data.nsets):
            hdu.header['cdpp%dr'%(i+1)] = dtres[i].cdpp_r
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


readers = [AMCReader,MASTReader]

def select_reader(fname):
    for R in readers:
        if R.can_read(fname):
            return R
    return None
