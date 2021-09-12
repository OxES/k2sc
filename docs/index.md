# *K2* *S*ystematics *C*orrection

Python package for K2 systematics correction using Gaussian processes.

## Contributors

[Hannu Parviainen](https://github.com/hpparvi), [Suzanne Aigrain](https://github.com/saigrain), [Benjamin Pope](https://github.com/benjaminpope)

## Installation

    git clone https://github.com/OxES/k2sc.git
    cd k2sc
    python setup.py install --user

## Basic usage - `lightkurve` interface

The easiest way to use `k2sc` is together with the standard Kepler/K2/TESS data analysis package [lightkurve](https://github.com/lightkurve/lightkurve). See tutorial to the left!


## Basic usage - Command Line

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

If you use K2SC in your research, please cite

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


