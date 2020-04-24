#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:16:18 2019

@author: dberke

VarConLib -- the Varying Constants Library

"""

import configparser
from pathlib import Path

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

output_dir = Path(config['PATHS']['output_dir'])

# The directory to store generated stellar databases in.
databases_dir = output_dir / 'stellar_databases'


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
