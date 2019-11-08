#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:33:52 2019

@author: dberke

A script to create transition pairs from lists of transition fits and return
information about them (via plots or otherwise).
"""

import argparse
import csv
import datetime as dt
from glob import glob
import lzma
import os
from pathlib import Path
import pickle
import re

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.miscellaneous import wavelength2velocity as wave2vel
from varconlib.miscellaneous import date2index
from varconlib.star import Star


# Where the analysis results live:
output_dir = Path(vcl.config['PATHS']['output_dir'])

# Set up CL arguments.
desc = """A script to analyze fitted absorption features.
          Requires a directory where results for an object are stored to be
          given, and the suffix to use."""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('object_dir', action='store', type=str,
                    help='Object directory to search in')

parser.add_argument('suffix', action='store', type=str,
                    help='Suffix to add to directory names to search for.')

parser.add_argument('--use-tex', action='store_true', default=False,
                    help='Use TeX rendering for fonts in plots (slow!).')

parser.add_argument('--create-transition-plots', action='store_true',
                    help='Create plots of the distributions of each individual'
                    'transition for a star.')

parser.add_argument('--create-offset-plots',
                    action='store_true', default=False,
                    help='Create plots of the offsets for each pair.')

parser.add_argument('--create-fit-plots', action='store_true', default=False,
                    help='Create plots of the fits for each transition.')

parser.add_argument('--create-berv-plots', action='store_true',
                    help="Create plots of each pair's separation vs. the"
                    " BERV at the time of observation.")

parser.add_argument('--create-airmass-plots', action='store_true',
                    default=False,
                    help="Plot transitions' scatter as a function of airmass.")

parser.add_argument('--link-fit-plots', action='store_true', default=False,
                    help='Hard-link fit plots by transition into new folders.')

parser.add_argument('--write-csv', action='store_true', default=False,
                    help='Create a CSV file of offsets for each pair.')

parser.add_argument('--recreate-star', action='store_true', default=False,
                    help='Force the star to be recreated from directories'
                    ' rather than be read in from saved data.')

parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Print more information about the process.')

args = parser.parse_args()

if args.use_tex or args.create_offset_plots:
    plt.rc('text', usetex=True)

# Define some important dates for plots.
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

style_params_pre = {'marker': 'o', 'color': 'Chocolate',
                    'markeredgecolor': 'Black', 'ecolor': 'BurlyWood',
                    'linestyle': '', 'alpha': 0.7, 'markersize': 6}

style_params_post = {'marker': 'o', 'color': 'CornFlowerBlue',
                     'markeredgecolor': 'Black', 'ecolor': 'LightSkyBlue',
                     'linestyle': '', 'alpha': 0.7, 'markersize': 6}

weighted_mean_params = {'color': 'RoyalBlue', 'linestyle': '--'}
weighted_err_params = {'color': 'SteelBlue', 'linestyle': ':'}


# Find the data in the given directory.
data_dir = Path(args.object_dir)
if not data_dir.exists():
    print(data_dir)
    raise RuntimeError('The given directory does not exist.')

# Read the list of chosen transitions.
with open(vcl.final_selection_file, 'r+b') as f:
    transitions_list = pickle.load(f)

tqdm.write(f'Found {len(transitions_list)} individual transitions.')

# Read the list of chosen pairs.
with open(vcl.final_pair_selection_file, 'r+b') as f:
    pairs_list = pickle.load(f)

tqdm.write(f'Found {len(pairs_list)} transition pairs (total) in list.')

# Read or create a Star object for this star.
obj_name = data_dir.stem
if args.recreate_star:
    load_data = False
else:
    load_data = True
star = Star(obj_name, data_dir, suffix=args.suffix, load_data=load_data)

if args.create_transition_plots:
    for transition in tqdm(star.transitionsList):
        for order in transition.ordersToFitIn:
            transition_label = '_'.join((transition.label, str(order)))

            fig = plt.figure(figsize=(6, 6), tight_layout=True)
            if star.fiberSplitIndex not in (0, None):
                gs = GridSpec(nrows=2, ncols=1, figure=fig,
                              height_ratios=[1, 1], hspace=0)
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax1.set_ylabel('Offset from expected position (m/s)')
                ax2.set_ylabel('Offset from expected position (m/s)')
                ax2.set_xlabel('Index number')

                offsets = star.fitOffsetsArray[:star.fiberSplitIndex,
                                               star._t_label(transition_label)]
                errors = star.fitErrorsArray[:star.fiberSplitIndex,
                                             star._t_label(transition_label)]
                w_mean, weight_sum = np.average(offsets, weights=(1/errors**2),
                                                returned=True)
                w_mean_err = 1 / np.sqrt(weight_sum)
                ax1.axhline(w_mean, color=style_params_pre['color'])

                indices = range(len(offsets))
                ax1.errorbar(x=indices,
                             y=offsets,
                             yerr=errors,
                             label=f'Mean: {w_mean:.2f}$\\pm${w_mean_err:.2f}',
                             **style_params_pre)
                ax1.legend(loc='lower right')

                offsets = star.fitOffsetsArray[star.fiberSplitIndex:,
                                               star._t_label(transition_label)]
                errors = star.fitErrorsArray[star.fiberSplitIndex:,
                                             star._t_label(transition_label)]
                w_mean, weight_sum = np.average(offsets, weights=(1/errors**2),
                                                returned=True)
                w_mean_err = 1 / np.sqrt(weight_sum)
                ax2.axhline(w_mean, color=style_params_post['color'])

                indices = range(len(offsets))
                ax2.errorbar(x=indices,
                             y=offsets,
                             yerr=errors,
                             label=f'Mean: {w_mean:.2f}$\\pm${w_mean_err:.2f}',
                             **style_params_post)
                ax2.legend(loc='lower right')

            else:
                ax = fig.add_subplot(1, 1, 1)
                if star.fiberSplitIndex == 0:
                    params = style_params_post
                else:
                    params = style_params_pre

                offsets = star.fitOffsetsArray[:,
                                               star._t_label(transition_label)]
                errors = star.fitErrorsArray[:,
                                             star._t_label(transition_label)]
                w_mean, weight_sum = np.average(offsets, weights=(1/errors**2),
                                                returned=True)
                w_mean_err = 1 / np.sqrt(weight_sum)
                ax1.axhline(w_mean, color=style_params_post['color'])

                indices = range(len(offsets))

                ax.errorbar(x=indices,
                            y=offsets,
                            yerr=errors,
                            label=f'Mean: {w_mean:.2f}$\\pm${w_mean_err:.2f}',
                            **params)
                ax.legend(loc='lower right')

            plots_dir = data_dir / 'transition_offset_plots'
            if not plots_dir.exists():
                os.mkdir(plots_dir)
            filename = plots_dir / '{}_{}.png'.format(
                    obj_name, transition_label)
            fig.savefig(filename)
            plt.close(fig)

if args.create_berv_plots:
    for pair_label in tqdm(star._pair_label_dict.keys()):
        fig = plt.figure(figsize=(6, 6), tight_layout=True)
        if star.fiberSplitIndex not in (0, None):
            gs = GridSpec(nrows=2, ncols=1, figure=fig,
                          height_ratios=[1, 1], hspace=0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax1.set_ylabel('Pair separation pre-change (km/s)')
            ax2.set_ylabel('Pair separation post-change (km/s)')
            ax2.set_xlabel('Radial velocity (km/s)')

            ax1.set_xlim(left=-100, right=100)
            ax1.tick_params(labelbottom=False)
            ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
            ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))

            separations = star.pairSeparationsArray[:star.fiberSplitIndex,
                                                    star._p_label(
                                                            pair_label)].to(
                                                                u.km/u.s)
            errors = star.pairSepErrorsArray[:star.fiberSplitIndex,
                                             star._p_label(pair_label)]

            velocities = star.bervArray[:star.fiberSplitIndex] +\
                star.radialVelocity

            ax1.errorbar(x=velocities, y=separations, yerr=errors,
                         **style_params_pre)

            separations = star.pairSeparationsArray[star.fiberSplitIndex:,
                                                    star._p_label(
                                                            pair_label)].to(
                                                                u.km/u.s)
            errors = star.pairSepErrorsArray[star.fiberSplitIndex:,
                                             star._p_label(pair_label)]

            velocities = star.bervArray[star.fiberSplitIndex:] +\
                star.radialVelocity

            ax2.errorbar(x=velocities, y=separations, yerr=errors,
                         **style_params_post)

            plots_dir = data_dir / 'BERV_plots'
            if not plots_dir.exists():
                os.mkdir(plots_dir)
            filename = plots_dir / '{}_{}.png'.format(
                    obj_name, pair_label)
            fig.savefig(filename)
            plt.close(fig)

    exit()

# Search for pickle files in the given directory.
search_str = str(data_dir) + '/*/pickles_{}/*fits.lzma'.format(args.suffix)
tqdm.write(search_str)
pickle_files = [Path(path) for path in glob(search_str)]


# dictionary with entries per observation
# entries consist of dictionary with entries of pairs made from fits

# Set up the master dictionary to contain sub-entries per observation.
master_star_dict = {}

obs_name_re = re.compile('HARPS.*_e2ds_A')

# Create some lists to hold all the results for saving out as a CSV file:
master_star_list = []
master_fits_list = []
for pickle_file in tqdm(pickle_files):

    # Match the part of the pickle filename that is the observation name.
    obs_name = obs_name_re.match(pickle_file.stem).group()

    tqdm.write('Analyzing results from {}'.format(obs_name))
    with lzma.open(pickle_file, 'rb') as f:
        fits_list = pickle.loads(f.read())

    # Set up a dictionary to map fits in this observation to transitions:
    fits_dict = {fit.label: fit for fit in fits_list}
    master_fits_list.append(fits_dict)

    # This can be used to remake plots for individual fits without rerunning
    # the fitting process in case the plot visual format changes.
    if args.create_fit_plots:
        closeup_dir = data_dir /\
            '{}/plots_{}/close_up'.format(obs_name, args.suffix)
        context_dir = data_dir /\
            '{}/plots_{}/context'.format(obs_name, args.suffix)
        tqdm.write('Creating plots of fits.')

        for transition in tqdm(transitions_list):
            for order_num in transition.ordersToFitIn:
                plot_closeup = closeup_dir / '{}_{}_{}_close.png'.format(
                        obs_name, transition.label, order_num)
                plot_context = context_dir / '{}_{}_{}_context.png'.format(
                        obs_name, transition.label, order_num)
                if args.verbose:
                    tqdm.write('Creating plots at:')
                    tqdm.write(str(plot_closeup))
                    tqdm.write(str(plot_context))
                fits_dict['_'.join((transition.label,
                                    str(order_num)))].plotFit(plot_closeup,
                                                              plot_context)

    pairs_dict = {}
    separations_list = [obs_name, fits_list[0].dateObs.
                        isoformat(timespec='milliseconds')]

    column_names = ['Observation', 'Time']
    for pair in pairs_list:
        for order_num in pair.ordersToMeasureIn:
            pair_label = '_'.join((pair.label, str(order_num)))
            column_names.extend([pair_label, pair_label + '_err'])
            try:
                fits_pair = [fits_dict['_'.join((pair._higherEnergyTransition.
                                                label, str(order_num)))],
                             fits_dict['_'.join((pair._lowerEnergyTransition.
                                                label, str(order_num)))]]
            except KeyError:
                # Measurement of one or the other transition doesn't exist, so
                # skip it (but fill in the list to prevent getting out of sync)
                separations_list.extend(['N/A', ' N/A'])
                continue

            assert fits_pair[0].order == fits_pair[1].order,\
                f"Orders don't match for {fits_pair}"
            if np.isnan(fits_pair[0].meanErrVel) or \
               np.isnan(fits_pair[1].meanErrVel):
                # Similar to above, fill in list with placeholder value.
                tqdm.write('{} in {} has a NaN velocity offset!'.format(
                        pair.label, obs_name))
                tqdm.write(str(fits_pair[0].meanErrVel))
                tqdm.write(str(fits_pair[1].meanErrVel))
                separations_list.extend(['NaN', ' NaN'])
                continue

            pairs_dict[pair_label] = fits_pair

            velocity_separation = wave2vel(fits_pair[0].mean,
                                           fits_pair[1].mean)
            error = np.sqrt(fits_pair[0].meanErrVel ** 2 +
                            fits_pair[1].meanErrVel ** 2)
            separations_list.extend([velocity_separation.value, error.value])

    # This is for the script to use.
    master_star_dict[obs_name] = pairs_dict
    if args.write_csv:
        # This is to be written out.
        master_star_list.append(separations_list)

# Write out a CSV file containing the pair separation values for all
# observations of this star.
# TODO: update
if args.write_csv:
    csv_filename = data_dir / 'pair_separations_{}.csv'.format(data_dir.stem)
    if args.verbose:
        tqdm.write(f'Creating CSV file of separations for {data_dir.stem} '
                   f'at {csv_filename}')

    assert len(master_star_list[0]) == len(column_names)

    with open(csv_filename, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(column_names)
        for row in tqdm(master_star_list):
            csv_writer.writerow(row)

    # Write out a series of CSV files containing information on the fits of
    # individual transitions for each star.
    column_names = ['ObsDate', 'Amplitude', 'Amplitude_err (A)', 'Mean (A)',
                    'Mean_err (A)', 'Mean_err_vel (m/s)', 'Sigma (A)',
                    'Sigma_err (A)', 'Offset (m/s)', 'Offset_err (m/s)',
                    'FWHM (m/s)', 'FWHM_err (m/s)', 'Chi-squared-nu',
                    'Order', 'Mean_airmass']

    csv_fits_dir = data_dir / 'fits_info_csv'
    if not csv_fits_dir.exists():
        os.mkdir(csv_fits_dir)
    if args.verbose:
        tqdm.write('Writing information on fits to files in {}'.format(
                   csv_fits_dir))
    for transition in tqdm(transitions_list):
        csv_filename = csv_fits_dir / '{}_{}.csv'.format(transition.label,
                                                         data_dir.stem)

        with open(csv_filename, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(column_names)
            for fits_dict in master_fits_list:
                csv_writer.writerow(fits_dict[transition.label].
                                    getFitInformation())


# Create the plots for each pair of transitions
if args.create_offset_plots:
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = '_'.join((pair.label, str(order_num)))
            if args.verbose:
                tqdm.write(f'Creating plot for pair {pair_label}')
            fitted_pairs = []
            date_obs = []
            for pair_dict in master_star_dict.values():
                try:
                    # Grab the associated pair from each observation.
                    fitted_pairs.append(pair_dict[pair_label])
                    # Grab the observation date.
                    date_obs.append(pair_dict[pair_label][0].dateObs)
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
            folded_dates = [obs_date.replace(year=2000) for obs_date
                            in date_obs]

            weights = 1 / errors ** 2
            weighted_mean = np.average(offsets, weights=weights)

            if args.verbose:
                tqdm.write('Weighted mean for {} is {:.2f}'.format(pair_label,
                           weighted_mean))

            normalized_offsets = offsets - weighted_mean
            chi_squared = sum((normalized_offsets / errors) ** 2)

            weighted_mean_err = 1 / np.sqrt(sum(weights))

            # Find the indices between dates of changes to make subsets of
            # points.
            date_indices = []
            for value in dates_of_change.values():
                date_indices.append(date2index(value['x'], date_obs))

            chi_squared_pre = sum((normalized_offsets[:date_indices[2]] /
                                   errors[:date_indices[2]]) ** 2)
            chi_squared_nu_pre = chi_squared_pre /\
                (len(normalized_offsets[:date_indices[2]]) - 1)

            chi_squared_post = sum((normalized_offsets[date_indices[2]:] /
                                   errors[date_indices[2]:]) ** 2)
            chi_squared_nu_post = chi_squared_post /\
                (len(normalized_offsets[date_indices[2]:]) - 1)

            plot_dir = data_dir / 'offset_plots'
            if not plot_dir.exists():
                os.mkdir(plot_dir)
            plot_name = plot_dir / '{}.png'.format(pair_label)

            fig, axes = plt.subplots(ncols=2, nrows=2,
                                     tight_layout=True,
                                     figsize=(10, 8),
                                     sharey='all')  # Share y-axis among all.
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
            ax1.grid(which='major', axis='y', color='Gray', alpha=0.6,
                     linestyle='--')
            # Plot pre-fiber change observations.
            pre_fiber_change_obs = len(offsets[:date_indices[2]])
            x_values = [x for x in range(pre_fiber_change_obs)]
            ax1.errorbar(x=x_values,
                         y=normalized_offsets[:date_indices[2]],
                         yerr=errors[:date_indices[2]],
                         label=r'$\chi^2_\nu=${:.3f}'.format(
                                 chi_squared_nu_pre.value),
                         **style_params_pre)
            # Plot post-fiber change observations.
            if date_indices[2] is not None:
                x_values = [x + pre_fiber_change_obs for x in
                            range(len(offsets[date_indices[2]:]))]
                ax1.errorbar(x=x_values,
                             y=normalized_offsets[date_indices[2]:],
                             yerr=errors[date_indices[2]:],
                             label=r'$\chi^2_\nu=${:.3f}'.format(
                                     chi_squared_nu_post.value),
                             **style_params_post)

            for index, key in zip(date_indices, dates_of_change.keys()):
                if index is not None:
                    ax1.axvline(x=index - 0.5,
                                linestyle=dates_of_change[key]['linestyle'],
                                color=dates_of_change[key]['color'])
            ax1.legend(loc='upper right', framealpha=0.6)

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
            ax3.xaxis.set_major_locator(mdates.YearLocator(base=1,
                                                           month=1, day=1))
            ax3.xaxis.set_minor_locator(mdates.YearLocator(base=1,
                                                           month=6, day=1))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax3.grid(which='major', axis='x', color='Gray', alpha=0.7,
                     linestyle='--')
            ax3.grid(which='minor', axis='x', color='LightGray', alpha=0.9,
                     linestyle=':')

            ax3.errorbar(x=date_obs, y=normalized_offsets,
                         yerr=errors, **style_params_pre)

            # Set up axis 4.
            ax4.set_xlim(**folded_date_range)
            ax4.xaxis.set_major_locator(mdates.MonthLocator())
            ax4.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

            ax4.grid(which='major', axis='x', color='Gray', alpha=0.7,
                     linestyle='--')
            ax4.grid(which='minor', axis='x', color='LightGray', alpha=1,
                     linestyle=':')

            ax4.errorbar(x=folded_dates, y=normalized_offsets,
                         yerr=errors, **style_params_pre)

            fig.savefig(str(plot_name))
            plt.close(fig)

if args.link_fit_plots or args.create_airmass_plots:
    transition_plots_dir = data_dir / 'plots_by_transition'
    if not transition_plots_dir.exists():
        os.mkdir(transition_plots_dir)


# Create hard links to all the fit plots by transition (across star) in their
# own directory, to make it easier to compare transitions across observations.
if args.link_fit_plots:

    tqdm.write('Linking fit plots to cross-observation directories.')

    for transition in tqdm(transitions_list):

        wavelength_str = transition.label
        transition_dir = transition_plots_dir / wavelength_str
        close_up_dir = transition_dir / 'close_up'
        context_dir = transition_dir / 'context'

        if args.link_fit_plots:
            for directory in (transition_dir, close_up_dir, context_dir):
                if not directory.exists():
                    os.mkdir(directory)

            for plot_type, directory in zip(('close_up', 'context'),
                                            (close_up_dir, context_dir)):

                search_str = str(data_dir) +\
                    '/HARPS*/plots_{}/{}/*{}*.png'.format(args.suffix,
                                                          plot_type,
                                                          wavelength_str)

                files_to_link = [Path(path) for path in glob(search_str)]
                for file_to_link in files_to_link:
                    dest_name = directory / file_to_link.name
                    if not dest_name.exists():
                        os.link(file_to_link, dest_name)

if args.create_airmass_plots:

    tqdm.write('Creating plots of airmass vs. dispersion for transitions.')
    airmass_dir = transition_plots_dir / 'airmass_plots'
    if not airmass_dir.exists():
        os.mkdir(airmass_dir)

    for transition in tqdm(transitions_list):
        for order_num in transition.ordersToFitIn:
            transition_label = '_'.join((transition.label, str(order_num)))
            filename = airmass_dir / 'Airmass_dispersion_{}.png'.format(
                    transition_label)

            # Get a list of all the observation dates.
            date_obs = []
            for fits_dict in master_fits_list:
                date_obs.append(fits_dict[transition_label].dateObs)

            date_indices = []
            for value in dates_of_change.values():
                date_indices.append(date2index(value['x'], date_obs))

            offsets_pre_list = []
            airmass_pre_list = []
            offsets_post_list = []
            airmass_post_list = []

            for fits_dict in master_fits_list[:date_indices[2]]:
                fit = fits_dict[transition_label]
                offsets_pre_list.append(fit.velocityOffset)
                airmass_pre_list.append(fit.airmass)

            for fits_dict in master_fits_list[date_indices[2]:]:
                fit = fits_dict[transition_label]
                offsets_post_list.append(fit.velocityOffset)
                airmass_post_list.append(fit.airmass)

            fig = plt.figure(figsize=(7, 7), tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)

            ax.set_xlabel('Airmass')
            ax.set_ylabel('Offset from expected wavelength (m/s)')
            ax.axhline(y=np.mean(offsets_pre_list),
                       label='Mean (pre-fiber change)',
                       color=style_params_pre['color'],
                       linestyle='--')
            ax.axhline(y=np.mean(offsets_post_list),
                       label='Mean (post-fiber change)',
                       color=style_params_post['color'],
                       linestyle='--')

            # Plot the pre-change points...
            ax.scatter(airmass_pre_list, offsets_pre_list, s=32,
                       c=style_params_pre['color'],
                       alpha=style_params_pre['alpha'],
                       edgecolor=style_params_pre['markeredgecolor'])
            # ...and the post-change ones.
            ax.scatter(airmass_post_list, offsets_post_list, s=32,
                       c=style_params_post['color'],
                       alpha=style_params_post['alpha'],
                       edgecolor=style_params_post['markeredgecolor'])

            ax.legend(framealpha=0.6)

            fig.savefig(str(filename))
            plt.close(fig)
