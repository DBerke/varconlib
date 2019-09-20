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
import csv
import datetime as dt
from glob import glob
import lzma
from os import mkdir
from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
import unyt as u

from varconlib.miscellaneous import wavelength2velocity as wave2vel
from varconlib.miscellaneous import date2index


plt.rc('text', usetex=True)

# Read the config file and set up some paths:
base_path = Path(__file__).parent
config_file = base_path / '../config/variables.cfg'

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
desc = """A script to analyze fitted absorption features.
          Requires a directory where results for an object are stored to be
          given, and the suffix to use."""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('object_dir', action='store', type=str,
                    help='Object directory to search in')
parser.add_argument('suffix', action='store', type=str,
                    help='Suffix to add to directory names to search for.')
parser.add_argument('--create-plots', action='store_true', default=False,
                    help='Create plots of the offsets for each pair.')
parser.add_argument('--write-csv', action='store_true', default=False,
                    help='Create a CSV file of offsets for each pair.')
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

    style_params = {'marker': 'o', 'color': 'Chocolate',
                    'markeredgecolor': 'Black', 'ecolor': 'BurlyWood',
                    'linestyle': '', 'alpha': 0.7, 'markersize': 8}
    weighted_mean_params = {'color': 'RoyalBlue', 'linestyle': '--'}
    weighted_err_params = {'color': 'SteelBlue', 'linestyle': ':'}

# Define a list of good "blend numbers" for chooosing which blends to look at.
blends_of_interest = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# Find the data in the given directory.
data_dir = Path(args.object_dir)
if not data_dir.exists():
    print(data_dir)
    raise RuntimeError('The given directory does not exist.')

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

# Search for pickle files in the given directory.
search_str = str(data_dir) + '/*/pickles_{}/*fits.lzma'.format(args.suffix)
tqdm.write(search_str)
pickle_files = [Path(path) for path in glob(search_str)]


# dictionary with entries per observation
# entries consist of dictionary with entries of pairs made from fits

# Set up the master dictionary to contain sub-entries per observation.
master_star_dict = {}

obs_name_re = re.compile('HARPS.*_e2ds_A')

# Create a list to hold all the results for saving out as a csv file:
master_star_list = []
observation_names = []
for pickle_file in tqdm(pickle_files[:]):

    # Match the part of the pickle filename that is the observation name.
    obs_name = obs_name_re.match(pickle_file.stem).group()

    tqdm.write('Analyzing results from {}'.format(obs_name))
    with lzma.open(pickle_file, 'rb') as f:
        fits_list = pickle.loads(f.read())

    # Set up a dictionary to map fits in this observation to transitions:
    fits_dict = {}
    for fit in fits_list:
        fits_dict[fit.transition.label] = fit

    pairs_dict = {}
    observation_names.append(obs_name)
    pairs_list = [obs_name, dt.datetime.strftime(fits_list[0].dateObs,
                                                 '%Y-%m-%dT%H:%M:%S.%f')]
    column_names = ['Observation', 'Time']
    for pair in good_pairs:
        column_names.extend([pair.label, pair.label + '_err'])
        try:
            fits_pair = [fits_dict[pair._higherEnergyTransition.label],
                         fits_dict[pair._lowerEnergyTransition.label]]
        except KeyError:
            # Measurement of one or the other transition doesn't exist, so
            # skip it (but fill in the list to prevent getting out of sync).
            pairs_list.extend(['N/A', ' N/A'])
            continue

        if np.isnan(fits_pair[0].meanErrVel) or \
           np.isnan(fits_pair[1].meanErrVel):
            # Similar to above, fill in list with placeholder value.
            tqdm.write('{} in {} has a NaN velocity offset!'.format(
                    fits_pair.label, obs_name))
            tqdm.write(str(fits_pair[0].meanErrVel))
            tqdm.write(str(fits_pair[1].meanErrVel))
            pairs_list.extend(['NaN', ' NaN'])
            continue

        pairs_dict[pair.label] = fits_pair
        error = np.sqrt(fits_pair[0].meanErrVel ** 2 +
                        fits_pair[1].meanErrVel ** 2)
        velocity_separation = wave2vel(fits_pair[0].mean, fits_pair[1].mean)
        pairs_list.extend([velocity_separation.value, error.value])

    # This is for the script to use.
    master_star_dict[obs_name] = pairs_dict
    # This is to be written out.
    master_star_list.append(pairs_list)

if args.write_csv:
    # Write out a CSV file containing the pair separation values for all
    # observations of this star.
    csv_filename = data_dir / 'pair_separations_{}.csv'.format(data_dir.stem)
    if args.verbose:
        tqdm.write(f'Creating CSV file of separations for {data_dir.stem} '
                   f'at {csv_filename}')

    assert len(master_star_list[0]) == len(column_names)

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(column_names)
        for row in tqdm(master_star_list):
            csv_writer.writerow(row)

#    data = pd.DataFrame(data=master_star_list, columns=column_names)
#    data.to_csv(path_or_buf=csv_filename,
#                header=column_names,
#                index=False)

# Create the plots for each pair of transitions
if args.create_plots:
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
                date_obs.append(pair_dict[pair.label][0].dateObs)
            except KeyError:
                # If a particular pair isn't available, just continue.
                continue

        offsets, errors = [], []
        for fit_pair in fitted_pairs:
            offsets.append(wave2vel(fit_pair[0].mean, fit_pair[1].mean))
            error = np.sqrt(fit_pair[0].meanErrVel ** 2 +
                            fit_pair[1].meanErrVel ** 2)
            if np.isnan(error):
                print(fit_pair[0].meanErrVel)
                print(fit_pair[1].meanErrVel)
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

        date_indices = []
        for value in dates_of_change.values():
            date_indices.append(date2index(value['x'], date_obs))

        chi_squared = sum((normalized_offsets / errors) ** 2)
        chi_squared_nu = chi_squared / (len(normalized_offsets) - 1)

        plot_dir = data_dir / 'offset_plots'
        if not plot_dir.exists():
            mkdir(plot_dir)
        plot_name = plot_dir / '{}.png'.format(pair.label)

        fig, axes = plt.subplots(ncols=2, nrows=2,
                                 tight_layout=True,
                                 figsize=(10, 8))
        fig.autofmt_xdate()
        (ax1, ax2), (ax3, ax4) = axes
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylabel(r'$\Delta v_{\textrm{sep}}\textrm{ (m/s)}$')
            ax.axhline(y=0, **weighted_mean_params)
            ax.axhline(y=weighted_mean_err,
                       **weighted_err_params)
            ax.axhline(y=-weighted_mean_err,
                       **weighted_err_params)
        for key, value in dates_of_change.items():
            ax3.axvline(label=key, **value)

        # Set up axis 1.
        ax1.errorbar(x=range(len(offsets)),
                     y=normalized_offsets,
                     yerr=errors,
                     label=r'$\chi^2_\nu=${:.3f}'.format(chi_squared_nu.value),
                     **style_params)
        for index, key in zip(date_indices, dates_of_change.keys()):
            if index is not None:
                ax1.axvline(x=index+0.5,
                            linestyle=dates_of_change[key]['linestyle'],
                            color=dates_of_change[key]['color'])
        ax1.legend(loc='upper right')

        # Set up axis 2.
        ax2.set_xlabel('Count')
        try:
            ax2.hist(normalized_offsets.value,
                     orientation='horizontal', color='White',
                     edgecolor='Black')
        except ValueError:
            print(fit_pair[0].meanErrVel)
            print(fit_pair[1].meanErrVel)
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
