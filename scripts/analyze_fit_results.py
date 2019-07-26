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
import datetime as dt
from glob import glob
from os import mkdir
from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
import unyt as u

from varconlib import wavelength2velocity as wave2vel
from varconlib import date2index


plt.rc('text', usetex=True)

# Read the config file and set up some paths:
config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# The list of pairs of transitions chosen:
pickle_dir = Path(config['PATHS']['pickle_dir'])
pickle_pairs_file = pickle_dir / 'transition_pairs.pkl'

# List of individual transitions chosen:
final_selection_file = pickle_dir / 'final_transitions_selection.pkl'

# Where the analysis results live:
output_dir = Path(config['PATHS']['output_dir'])


# Set up CL arguments.
desc = 'Analyze fitted absorption features.'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('object_dir', action='store', type=str,
                    help='Object directory to search in')
parser.add_argument('suffix', action='store', type=str,
                    help='Suffix to add to directory names to search for.')
parser.add_argument('--create-plots', action='store_true', default=False,
                    help='Create plots of the offsets for each pair.')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Print more information about the process.')

args = parser.parse_args()

if args.create_plots:

    # Define some important date for plots.
    # Define some dates when various changes were made to HARPS.
    dates_of_change = {"Secondary mirror unit changed":
                       {'x': dt.date(year=2004, month=8, day=8),
                        'color': 'MediumSeaGreen', 'linestyle': '--'},
                       "Flat field lamp changed":
                       {'x': dt.date(year=2008, month=8, day=22),
                        'color': 'OliveDrab', 'linestyle': '-'},
                       "Fibers changed":
                       {'x': dt.date(year=2015, month=6, day=1),
                        'color': 'SeaGreen', 'linestyle': '-.'}}

    date_plot_range = {'left': dt.date(year=2003, month=10, day=1),
                       'right': dt.date(year=2017, month=6, day=1)}
    folded_date_range = {'left': dt.date(year=2000, month=1, day=1),
                         'right': dt.date(year=2000, month=12, day=31)}


# Define a list of good "blend numbers" for chooosing which blends to look at.
blends_of_interest = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# Read the list of chosen transitions.
with open(final_selection_file, 'r+b') as f:
    transitions_list = pickle.load(f)

tqdm.write(f'Found {len(transitions_list)} individual transitions.')

all_transitions_set = set()
for transition in transitions_list:
    all_transitions_set.add(transition.label)

# Read the list of chosen pairs.
with open(pickle_pairs_file, 'r+b') as f:
    pairs_list = pickle.load(f)

tqdm.write(f'Found {len(pairs_list)} transition pairs (total) in list.')

# Set up lists of "good pairs" (ones with both features mostly unblended) and
# "good transitions" (the ones in the good pairs).

good_pairs = []
good_transitions_set = set()
for pair in pairs_list:
    if pair.blendTuple in blends_of_interest:
        good_pairs.append(pair)
        for transition in pair:
            good_transitions_set.add(transition.label)


tqdm.write('Found {} "good pairs"'.format(len(good_pairs)))
tqdm.write('and {} "good transitions" in those pairs.'.format(
           len(good_transitions_set)))
# Find the data in the given directory.
data_dir = Path(args.object_dir)
if not data_dir.exists():
    print(data_dir)
    raise RuntimeError('The given directory does not exist.')

# Search for pickle files in the given directory.
search_str = str(data_dir) + '/*/pickles_{}/*fits.pkl'.format(args.suffix)
tqdm.write(search_str)
pickle_files = [Path(path) for path in glob(search_str)]


# dictionary with entries per observation
# entries consist of dictionary with entries of pairs made from fits

# Set up the master dictionary to contain sub-entries per observation.
master_star_dict = {}

obs_name_re = re.compile('HARPS.*_e2ds_A')

for pickle_file in tqdm(pickle_files[:]):

    obs_name = obs_name_re.match(pickle_file.stem).group()

    tqdm.write('Analyzing results from {}'.format(obs_name))
    with open(pickle_file, 'r+b') as f:
        fits_list = pickle.load(f)

    # Set up a dictionary to map fits in this observation to transitions:
    fits_dict = {}
    for fit in fits_list:
        fits_dict[fit.transition.label] = fit

    pairs_dict = {}
    for pair in good_pairs:
        try:
            new_pair = [fits_dict[pair._higherEnergyTransition.label],
                        fits_dict[pair._lowerEnergyTransition.label]]
        except KeyError:
            continue
        pair_label = '_'.join([new_pair[0].transition.label,
                              new_pair[1].transition.label])

        if np.isnan(new_pair[0].medianErrVel) or \
           np.isnan(new_pair[1].medianErrVel):
            tqdm.write('{} in {} has a NaN velocity offset!'.format(
                    pair_label, obs_name))
            tqdm.write(str(new_pair[0].medianErrVel))
            tqdm.write(str(new_pair[1].medianErrVel))
            continue

        pairs_dict[pair_label] = new_pair

    master_star_dict[obs_name] = pairs_dict

#print(len(master_star_dict))
#for key, value in master_star_dict.items():
#    print(key, len(value))

for pair in tqdm(good_pairs[:]):
    if args.verbose:
        tqdm.write(f'Creating plot for pair {pair.label}')
    fitted_pairs = []
    date_obs = []
    for key, pair_dict in master_star_dict.items():
        try:
            # Grab the associated pair from each observation.
            fitted_pairs.append(pair_dict[pair.label])
            # Grab the observation date.
            date_obs.append(pair_dict[pair_label][0].dateObs)
        except KeyError:
            # If a particular pair isn't available, just continue.
            continue

    offsets, errors = [], []
    for fit_pair in fitted_pairs:
        offsets.append(wave2vel(fit_pair[0].median, fit_pair[1].median))
        error = np.sqrt(fit_pair[0].medianErrVel ** 2 +
                        fit_pair[1].medianErrVel ** 2)
        if np.isnan(error):
            print(fit_pair[0].medianErrVel)
            print(fit_pair[1].medianErrVel)
            raise ValueError
        errors.append(error)

    offsets = np.array(offsets) * u.m / u.s
    errors = np.array(errors) * u.m / u.s
    folded_dates = [obs_date.replace(year=2000) for obs_date in date_obs]

    weights = 1 / errors ** 2
    weighted_mean = np.average(offsets, weights=weights)

    tqdm.write('Weighted mean for {} is {:.2f}'.format(pair.label,
               weighted_mean))

    normalized_offsets = offsets - weighted_mean
    chi_squared = sum((normalized_offsets / errors) ** 2)

    weighted_mean_err = 1 / np.sqrt(sum(weights))

#    print(offsets)
#    print(errors)
#    print(np.median(offsets))
#    tqdm.write(str(np.mean(normalized_offsets)))
#    print(np.std(offsets))

    if args.create_plots:

        date_indices = []
        for value in dates_of_change.values():
            date_indices.append(date2index(value['x'], date_obs))

        style_params = {'marker': 'o', 'color': 'Chocolate',
                        'markeredgecolor': 'Black', 'ecolor': 'BurlyWood',
                        'linestyle': '', 'alpha': 0.7}
        weighted_mean_params = {'color': 'RoyalBlue', 'linestyle': '--'}
        weighted_err_params = {'color': 'SteelBlue', 'linestyle': ':'}

        plot_dir = data_dir / 'offset_plots'
        if not plot_dir.exists():
            mkdir(plot_dir)
        plot_name = plot_dir / '{}.png'.format(pair.label)
#        print(plot_name)

        fig, axes = plt.subplots(ncols=2, nrows=2,
                                 tight_layout=True,
                                 figsize=(9, 8))
        fig.autofmt_xdate()
        (ax1, ax2), (ax3, ax4) = axes
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylabel(r'$\Delta v_{\textrm{sep}} (\textrm{m/s})$')
            ax.axhline(y=0, **weighted_mean_params)
            ax.axhline(y=weighted_mean_err,
                       **weighted_err_params)
            ax.axhline(y=-1 * weighted_mean_err,
                       **weighted_err_params)
        for key, value in dates_of_change.items():
            ax3.axvline(label=key, **value)

        # Set up axis 1.
        ax1.errorbar(x=range(len(offsets)),
                     y=normalized_offsets,
                     yerr=errors, markersize=8, **style_params)
        for index, key in zip(date_indices, dates_of_change.keys()):
            if index is not None:
                ax1.axvline(x=index+0.5,
                            linestyle=dates_of_change[key]['linestyle'],
                            color=dates_of_change[key]['color'])

        # Set up axis 2.
        ax2.set_xlabel('Count')
        try:
            ax2.hist(normalized_offsets.value,
                     orientation='horizontal', color='White',
                     edgecolor='Black')
        except ValueError:
            print(fit_pair[0].medianErrVel)
            print(fit_pair[1].medianErrVel)
            print(offsets)
            print(errors)
            print(weights)
            raise

        # Set up axis 3.
        ax3.set_xlim(**date_plot_range)
        ax3.errorbar(x=date_obs, y=normalized_offsets,
                     yerr=errors, **style_params)

        # Set up axis 4.
        ax4.set_xlim(**folded_date_range)
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax4.errorbar(x=folded_dates, y=normalized_offsets,
                     yerr=errors, **style_params)

        fig.savefig(str(plot_name))
        plt.close(fig)
