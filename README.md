# *K2* *S*ystematics *C*orrection

[![Build Status](https://travis-ci.org/OxES/k2sc.svg?branch=master)](https://travis-ci.org/OxES/k2sc)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![arXiv](http://img.shields.io/badge/arXiv-1603.09167-blue.svg?style=flat)](http://arxiv.org/abs/1603.09167)

Python package for K2 systematics correction using Gaussian processes.

## Installation

    git clone https://github.com/OxES/k2sc.git
    cd k2sc
    python setup.py install --user

## Basic usage

A MAST K2 light curve can be detrended by calling

    k2sc <filename> -c <c> --splits xxxx,yyyy --flux-type pdc
    
where `<filename>` is the MAST light curve filename, `<c>` is the campaign number, and `xxxx,yyyy` define the break points (in time) where the position-dependent systematics change (corresponds to the points of reversal of the roll-angle variation).

### Useful flags

- `--de-max-time <ss>`  maximum time (in seconds) to run global GP hyperparameter optimization (differential evolution) before switching to local optimization.
- `--de-npop <nn>` size of the de population, can be set to 50 to speed up the optimization.
- `--save-dir <path>` defines where to save the detrended files
- `--logfile <filename>` 

### MPI

K2SC supports MPI automatically (requires MPI4Py.) Call k2sc as

    mpirun -n N k2sc <files> -c <c> --splits xxxx,yyyy --flux-type pdc

where `<files>` is now a list of files to be detrended (for example, `path/to/ktwo*.fits`).

## Requires

 - NumPy
 - SciPy
 - astropy
 - George
 - MPI4Py

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
