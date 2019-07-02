#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:33:52 2019

@author: dberke

A script to create transition pairs from lists of transition fits and return
information about them (via plots or otherwise).
"""

import argparse
import configparser
from glob import glob
from pathlib import Path
import pickle
from pprint import pprint
from math import isclose

import numpy as np
from tqdm import tqdm
import unyt as u

from transition_pair import TransitionPair


# Read the config file and set up some paths:
config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# The list of pairs of transitions chosen:
pickle_dir = Path(config['PATHS']['pickle_dir'])
pickle_pairs_file = pickle_dir / 'transition_pairs.pkl'

# Where the analysis results live:
output_dir = Path(config['PATHS']['output_dir'])

desc = 'Analyze fitted absorption features.'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('object_dir', action='store', type=str,
                    help='Object directory to search in')
parser.add_argument('suffix', action='store', type=str,
                    help='Suffix to add to directory names to search for.')

args = parser.parse_args()

# Define a list of good "blend numbers" for chooosing which blends to look at.
blends_of_interest = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# Read the list of chosen pairs.
with open(pickle_pairs_file, 'r+b') as f:
    pairs_list = pickle.load(f)

tqdm.write(f'Found {len(pairs_list)} transition pairs.')

# Find the data in the given directory.
data_dir = Path(args.object_dir)
if not data_dir.exists():
    print(data_dir)
    raise RuntimeError('The given directory does not exist.')

# Search for pickle files in the given directory.
search_str = str(data_dir) + '/*/pickles_{}/*fits.pkl'.format(args.suffix)
print(search_str)
pickle_files = [Path(path) for path in glob(search_str)]

measured_pairs_list = []

for pickle_file in tqdm(pickle_files[0:2]):
    with open(pickle_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    for pair in pairs_list:
        if pair.blendTuple in blends_of_interest:
#            tqdm.write('Found pair of interest: {}'.format(pair))
            transition_lower, transition_higher = None, None
            # Search through the transition list and find the matching
            # transitions.
            for fit in transitions_list:
                if isclose(fit.transition.lowerEnergy,
                           pair._lowerEnergyTransition.lowerEnergy,
                           rel_tol=1e-3) and\
                           isclose(fit.transition.higherEnergy,
                                   pair._lowerEnergyTransition.higherEnergy,
                                   rel_tol=1e-3):
#                    tqdm.write('Found lower energy transition.')
                    transition_lower = fit.transition
                elif isclose(fit.transition.lowerEnergy,
                             pair._higherEnergyTransition.lowerEnergy,
                             rel_tol=1e-3) and\
                             isclose(fit.transition.higherEnergy,
                                     pair._higherEnergyTransition.higherEnergy,
                                     rel_tol=1e-3):
#                    tqdm.write('Found higher energy transition.')
                    transition_higher = fit.transition
            if (transition_lower and transition_higher):
                # Create a new pair from the measured values.
                new_pair = TransitionPair(transition_lower, transition_higher)
                tqdm.write('Found pair: {}'.format(new_pair))
                measured_pairs_list.append(new_pair)

tqdm.write(f'Found {len(measured_pairs_list)} pairs.')
#for pair in measured_pairs_list:
#    print(pair)
