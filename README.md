# K2SC

Python package for the K2 systematics correction using Gaussian processes.

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

## Authors

 - Hannu Parviainen
 - Suzanne Aigrain
 - Benjamin Pope
