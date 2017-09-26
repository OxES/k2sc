# *K2* *S*ystematics *C*orrection

[![Build Status](https://travis-ci.org/OxES/k2sc.svg?branch=master)](https://travis-ci.org/OxES/k2sc)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![MNRAS](https://img.shields.io/badge/MNRAS-10.1093%2Fmnras%2Fstv706-blue.svg)](https://doi.org/10.1093/mnras/stw706)
[![arXiv](http://img.shields.io/badge/arXiv-1603.09167-blue.svg?style=flat)](http://arxiv.org/abs/1603.09167)

Python package for K2 systematics correction using Gaussian processes.

## Installation

    git clone https://github.com/OxES/k2sc.git
    cd k2sc
    python setup.py install --user

## Basic usage

A MAST K2 light curve can be detrended by calling

    k2sc <filename>
    
where `<filename>` is either a MAST light curve filename, list of files, or a directory.

### Useful flags

- `--flux-type` can be either `pdc` or `sap`
- `--de-max-time <ss>`  maximum time (in seconds) to run global GP hyperparameter optimization (differential evolution) before switching to local optimization.
- `--de-npop <nn>` size of the de population, can be set to 50 to speed up the optimization.
- `--save-dir <path>` defines where to save the detrended files
- `--logfile <filename>` 

### MPI

K2SC supports MPI automatically (requires MPI4Py.) Call k2sc as

    mpirun -n N k2sc <files>

where `<files>` is a list of files or a directory to be detrended (for example, `path/to/ktwo*.fits`).

## Requires

 - NumPy, SciPy, astropy, George, MPI4Py

Citing
------

If you use K2SC in your reserach, please cite

    Aigrain, S., Parviainen, H. & Pope, B. (2016, accepted to MNRAS), arXiv:1603.09167

or use this ready-made BibTeX entry

    @article{Aigrain2016,
        arxivId = {1603.09167},
        author = {Aigrain, Suzanne and Parviainen, Hannu and Pope, Benjamin},
        keywords = {data analysis,methods,photometry,planetary systems,techniques},
        title = {{K2SC: Flexible systematics correction and detrending of K2 light curves using Gaussian Process regression}},
        url = {http://arxiv.org/abs/1603.09167},
        year = {2016}
    }


## Authors

 - Hannu Parviainen
 - Suzanne Aigrain
 - Benjamin Pope
