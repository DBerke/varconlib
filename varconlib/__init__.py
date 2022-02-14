#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:16:18 2019

@author: dberke

VarConLib -- the Varying Constants Library

"""

import configparser
from pathlib import Path

from bidict import bidict
from tqdm import tqdm

__all__ = ['base_dir', 'data_dir', 'masks_dir', 'pickle_dir',
           'pixel_geom_files_dir', 'final_selection_file',
           'final_pair_selection_file', 'databases_dir',
           'verbose_print']

# Define some important paths to be available globally relative to the
# absolute path of the parent directory.

base_dir = Path(__file__).parent

data_dir = base_dir / 'data'

masks_dir = data_dir / 'masks'

pickle_dir = data_dir / 'pickles'

spectra_dir = data_dir / 'spectra'

pixel_geom_files_dir = data_dir / 'pixel_geom_files'

# The pickle file containing the final selection of transitions to use.
final_selection_file = pickle_dir / 'final_transitions_selection.pkl'

# The pickle file containing the final selection of pairs to use.
final_pair_selection_file = pickle_dir / 'final_pairs_selection.pkl'

# Read the config file to get values from it.
config_file = base_dir / 'config/variables.cfg'
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# Directory where data from HARPS is located.
harps_dir = Path(config['PATHS']['harps_dir'])

# Directory where most data output from analysis goes.
output_dir = Path(config['PATHS']['output_dir'])

# Directory for HARPS wavelength calibration files.
harps_wavelength_cals_dir = Path(config['PATHS']['harps_wavelength_cal_dir'])

# Directory containing blaze files for HARPS.
harps_blaze_files_dir = Path(config['PATHS']['harps_blaze_files_dir'])

# The directory to store generated stellar databases in.
databases_dir = output_dir / 'stellar_databases'

# Define a bidict of atomic number: atomic symbol for use in a few places.
elements = bidict({1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
                   8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg",  13: "Al",
                   14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
                   20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
                   26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga",
                   32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
                   38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
                   44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
                   50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
                   56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm",
                   62: "Sm", 63:  "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho",
                   68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
                   74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
                   80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
                   86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa",
                   92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk",
                   98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No",
                   103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh",
                   108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn",
                   113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts",
                   118: "Og"})


def verbose_print(verbosity):
    """Return a function depending on the value of `verbosity`.

    Function returns a different function depending on whether a script is
    called with a 'verbose' option or not. It is intended that this function be
    used to define another function whose behavior fundamentally changes
    depending on whether a 'verbose' flag is set or not. If it is, the defined
    function should print its output (though using tqdm.write() in order to
    not mess up scripts making use of tqdm.tqdm). If not, the function will
    simply do nothing at all.

    Sample usage:
        verbose = True
        vprint = varconlib.verbose_print(verbose)
    Anything passed to vprint will then be printed to stdout as long as verbose
    is *True*.

    Parameters
    ----------
    verbosity : bool
        If *True*, returns a lamdba function which passes its input through
        str() and on to tqdm.write().
        If *False*, returns a lambda function which simply returns *None*.

    Returns
    -------
    function
        A function which does different things depending on the value of
        `verbosity`; if *True* it will cast its input to a string, then write
        it using tqdm.write(), otherwise it will simply return *None* always.

    """

    if verbosity:
        return lambda x: tqdm.write(str(x))
    else:
        return lambda x: None
