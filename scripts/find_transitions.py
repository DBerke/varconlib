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
#parser.add_argument('star_dir', action='store')
#parser.add_argument('plot_dir', action='store')
#parser.add_argument('transition_list', action='store')
#parser.add_argument('results_out', action='store')

args = parser.parse_args()

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

pickle_dir = Path(config['PATHS']['pickle_dir'])
nist_matched_pickle_file = pickle_dir / 'transitions_NIST_matched.pkl'

# Define paths for plots to go in:
pictures_dir = Path(config['PATHS']['pictures_dir'])

# Read the dictionary of transitions matched with NIST.
with open(nist_matched_pickle_file, 'r+b') as f:
    tqdm.write('Unpickling NIST-matched transitions...')
    nist_dict = pickle.load(f)

observations_dir = Path('/Users/dberke/HD78660/data/reduced/')
glob_search_string = str(observations_dir) + '/*/*e2ds_A.fits'

# Get a list of all the data files in the data directory:
data_files = [Path(string) for string in sorted(glob(glob_search_string))]

for obs_path in tqdm(data_files[0:1]):
    tqdm.write('Fitting {}...'.format(obs_path.stem))
    obs = obs2d.HARPSFile2DScience(obs_path)
    fits_dict = {}
    plots_dir = pictures_dir / 'Stars' / obs_path.stem
    for species in tqdm(nist_dict.keys()):
        tqdm.write('Fitting {}...'.format(species))
        fits_list = []
        for transition in tqdm(nist_dict[species]):
            plot_file = str(plots_dir / 'Transition_{:.4f}.png'.format(
                    transition.wavelength.to(u.angstrom).value))
            fit = GaussianFit(transition, obs, verbose=False)
            fit.plotFit(outfile=plot_file)
            fits_list.append(fit)
        fits_dict[species] = tuple(fits_list)

    total = 0
    for value in fits_dict.values():
        total += len(value)
    tqdm.write('Found {} total transitions in {}.'.format(total,
               obs_path.stem))
    outfile = pickle_dir / obs_path.stem / 'gaussian_fits.pkl'
    if not outfile.parent.exists():
        os.mkdir(outfile.parent)
    with open(outfile, 'w+b') as f:
        pickle.dump(fits_dict, f)
