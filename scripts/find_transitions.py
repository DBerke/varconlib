#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:41:58 2018

@author: dberke

This script takes a dictionary containing lists of `Transition` objects from
the script select_line_pairs, and an input list of spectral observations and
attempts to fit each of the transitions listed in the dictionary.
"""

import argparse
import configparser
import pickle
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import unyt as u
import obs2d
from fitting import GaussianFit

desc = 'Fit absorption features in spectra from a given list of transitions.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('object_dir', action='store',
                    help='Directory in which to find e2ds sub-folders.')
parser.add_argument('object_name', action='store',
                    help='Name of object to use for storing output.')
#parser.add_argument('--transition_dict', action='store',
#                    help='Collection of transitions to use.')

args = parser.parse_args()

observations_dir = Path(args.object_dir)
# Check that the path given exists:
if not observations_dir.exists():
    tqdm.write(observations_dir)
    raise RuntimeError('The given directory does not exist.')

# Check if the given path ends in data/reduced:
if observations_dir.match('*/data/reduced'):
    pass
else:
    observations_dir = observations_dir / 'data/reduced'
    if not observations_dir.exists():
        raise RuntimeError('The directory does not contain "data/reduced".')

# Currenttly using /Users/dberke/HD146233/data/reduced/ for a specific file.
glob_search_string = str(observations_dir) + '/2016-03-29/*e2ds_A.fits'
# Get a list of all the data files in the data directory:
data_files = [Path(string) for string in sorted(glob(glob_search_string))]

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

pickle_dir = Path(config['PATHS']['pickle_dir'])
# All unique transitions found within pairs found:
pickle_pairs_transitions_file = pickle_dir / 'pair_transitions.pkl'

# output_dir = /Users/dberke/data_output
output_dir = Path(config['PATHS']['output_dir'])
object_dir = output_dir / args.object_name
if not object_dir.exists():
    os.mkdir(object_dir)
# Define directory for output pickle files:
output_pickle_dir = object_dir / 'pickles'
if not output_pickle_dir.exists():
    os.mkdir(output_pickle_dir)
# Define paths for plots to go in:
output_plots_dir = object_dir / 'plots'
if not output_plots_dir.exists():
    os.mkdir(output_plots_dir)
closeup_dir = output_plots_dir / 'close_up'
if not closeup_dir.exists():
    os.mkdir(closeup_dir)
context_dir = output_plots_dir / 'context'
if not context_dir.exists():
    os.mkdir(context_dir)

# Read the dictionary of transitions matched with NIST.
with open(pickle_pairs_transitions_file, 'r+b') as f:
    tqdm.write('Unpickling NIST-matched transitions...')
    transitions_list = pickle.load(f)

tqdm.write('Found {} pickled transitions in {}.'.format(len(transitions_list),
           pickle_pairs_transitions_file.name))

for obs_path in tqdm(data_files) if len(data_files) > 1 else data_files:
    tqdm.write('Fitting {}...'.format(obs_path.name))
    obs = obs2d.HARPSFile2DScience(obs_path)

    fits_list = []
    for transition in tqdm(transitions_list):
        plot_closeup = closeup_dir / 'Transition_{:.4f}_{}{}.png'.format(
                transition.wavelength.to(u.angstrom).value,
                transition.atomicSymbol, transition.ionizationState)
        plot_context = context_dir / 'Transition_{:.4f}_{}{}.png'.format(
                transition.wavelength.to(u.angstrom).value,
                transition.atomicSymbol, transition.ionizationState)
        try:
            fit = GaussianFit(transition, obs, verbose=False)
            fit.plotFit(plot_closeup, plot_context)
            fits_list.append(fit)
        except RuntimeError:
            pass
    fits_list.append(fit)

    tqdm.write('Found {} transitions in {}.'.format(len(fits_list),
               obs_path.name))
    outfile = output_pickle_dir / '{}_gaussian_fits.pkl'.format(
            obs._filename.stem)
    if not outfile.parent.exists():
        os.mkdir(outfile.parent)
    with open(outfile, 'w+b') as f:
        tqdm.write(f'Pickling list of fits at {outfile}')
        pickle.dump(fits_list, f)
