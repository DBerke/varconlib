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
from itertools import tee
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
import numpy.ma as ma
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.miscellaneous import wavelength2velocity as wave2vel
from varconlib.miscellaneous import date2index
from varconlib.star import Star


def pairwise(iterable):
    """Take a sequence and return pairs of successive values from it.
    s -> (s[0], s[1]), (s[1], s[2]),...,(s[-2], s[-1])

    Parameters
    ----------
    iterable : iterable
        Any iterable item to be iterated over

    Yields
    ------
    tuple
        A length-2 tuple of successive, overlapping pairs of values from
        `iterable`.

    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def link_fit_plots(transition_plots_dir):

    for transition in tqdm(transitions_list):

        wavelength_str = transition.label
        transition_dir = transition_plots_dir / wavelength_str
        close_up_dir = transition_dir / 'close_up'
        context_dir = transition_dir / 'context'

        for directory in (transition_dir, close_up_dir, context_dir):
            if not directory.exists():
                os.mkdir(directory)

        for plot_type, directory in zip(('close_up', 'context'),
                                        (close_up_dir, context_dir)):

            search_str = str(data_dir) +\
                '/HARPS*/plots_{}/{}/*{}*.png'.format(args.suffix,
                                                      plot_type,
                                                      wavelength_str)
            vprint(search_str)

            files_to_link = [Path(path) for path in glob(search_str)]
            for file_to_link in files_to_link:
                dest_name = directory / file_to_link.name
                if not dest_name.exists():
                    os.link(file_to_link, dest_name)


def create_transition_offset_plots(plots_dir):
    """Create plots of the offsets for each transition in the given star.

    Parameters
    ----------
    plots_dir : `pathlib.Path`
        The directory to save the output plots to.

    """

    def layout_plots():
        """Collect data from the star and layout the plots onto axes.

        """

        offsets = ma.masked_invalid(
                star.fitOffsetsNormalizedArray[time_slice, column_index].value)
        errors = ma.masked_invalid(star.fitErrorsArray[time_slice,
                                                       column_index].value)
        fit_chis = ma.masked_invalid(star.chiSquaredNuArray[time_slice,
                                                            column_index])
        mean_measured = np.average(ma.masked_invalid(
                                   star.fitMeansArray[time_slice,
                                                      column_index].value),
                                   weights=(1/errors**2))

        vprint(f'Mean for {star.t_label(column_index)} is'
               ' {mean_measured * u.angstrom:.4f}')

        w_mean, weight_sum = np.average(offsets,
                                        weights=(1/errors**2),
                                        returned=True)
#        w_mean_err = 1 / np.sqrt(weight_sum)

        chi_squared = np.sum(((offsets - w_mean) / errors) ** 2)
        if len(offsets) > 1:
            reduced_chi_squared = chi_squared / (len(offsets) - 1)
        else:
            reduced_chi_squared = chi_squared
        std_dev = np.std(offsets)

        # Set up the offset axis with grids.
        offset_ax.set_ylabel('Offset from expected position (m/s)')
        if not args.simplified:
            offset_ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
            offset_ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))
        plt.setp(offset_ax.get_xticklabels(), visible=False)

        offset_ax.xaxis.grid(which='major', color='Gray', alpha=0.7,
                             linestyle='-')
        offset_ax.xaxis.grid(which='minor', color='Gray', alpha=0.6,
                             linestyle='--')
        offset_ax.yaxis.grid(which='major', color='Gray', alpha=0.4,
                             linestyle='--')

        offset_ax.axhline(w_mean, color='Black')
        offset_ax.axhline(y=w_mean + std_dev, color='DimGray',
                          linestyle='-.')
        offset_ax.axhline(y=w_mean - std_dev, color='DimGray',
                          linestyle='-.')

        # Plot the values.
        indices = range(len(offsets))
        offset_ax.errorbar(x=indices, y=offsets, yerr=errors,
                           **params)

        # Plot the chi-squared values for the fits.
        chi_ax.set_xlabel('Index number')
        chi_ax.set_ylabel(r'$\chi^2_\nu$')

        chi_ax.axhline(y=1, color=params['color'], linestyle='-.')
        if not args.simplified:
            chi_ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
        chi_ax.yaxis.grid(which='major', color='Gray', alpha=0.6)
        chi_ax.xaxis.grid(which='major', color='Gray', alpha=0.8,
                          linestyle='-')
        chi_ax.xaxis.grid(which='minor', color='Gray', alpha=0.7,
                          linestyle='--')

        chi_ax.plot(indices, fit_chis, color='Black',
                    linestyle='-', marker='o', markersize=3)

        # Create the plot of chi-squared vs. offset.
        vs_ax.set_xlabel(r'$\chi^2_\nu$')
        vs_ax.set_ylabel('Offset from expected position (m/s)')

        vs_ax.axhline(w_mean, color='Black', alpha=0.7)
        vs_ax.axvline(1, color=params['color'], linestyle='-.')
        vs_ax.axhline(y=w_mean + std_dev, color='DimGray', linestyle='-.',
                      alpha=0.6)
        vs_ax.axhline(y=w_mean - std_dev, color='DimGray', linestyle='-.',
                      alpha=0.6)

        vs_ax.yaxis.grid(which='major', color='Gray', linestyle='--',
                         alpha=0.6)
        vs_ax.xaxis.grid(which='major', color='Gray', linestyle='--',
                         alpha=0.6)

        vs_ax.errorbar(x=fit_chis, y=offsets, yerr=errors,
                       label=f'$\mu$: {mean_measured * u.angstrom:.4f},'
                       f' {w_mean * u.m/u.s:.2f} $\\pm$'
                       f' {std_dev * u.m/u.s:.3f}\n'
                       r'$\chi^2_\nu=$'
                       f'{reduced_chi_squared:.3f},'
                       f' $N=${len(offsets)}',
                       **params)

        vs_ax.legend(loc='lower right', markerscale=0.7,
                         framealpha=1, mode='expand',
                         bbox_to_anchor=(-0.04, -0.35, 1.04, -0.35))

        hist_ax.yaxis.grid(which='major', color='Gray', alpha=0.6,
                           linestyle='--')
        hist_ax.axhline(w_mean, color='Black', alpha=0.7)
        hist_ax.axhline(y=w_mean + std_dev, color='DimGray', linestyle='-',
                        alpha=0.6)
        hist_ax.axhline(y=w_mean - std_dev, color='DimGray', linestyle='-',
                        alpha=0.6)
        hist_ax.hist(offsets, bins='fd', color='Black',
                     histtype='step', orientation='horizontal')

    # Define slice objects to capture time periods.
    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    for transition in tqdm(star.transitionsList):
        for order in transition.ordersToFitIn:
            transition_label = '_'.join((transition.label, str(order)))
            column_index = star.t_index(transition_label)

            if star.fiberSplitIndex not in (0, None):
                fig = plt.figure(figsize=(11, 8), tight_layout=True)
                gs = GridSpec(nrows=5, ncols=3, figure=fig, hspace=0,
                              height_ratios=[11, 3, 3, 11, 3],
                              width_ratios=[5, 5, 1])

                offset_ax1 = fig.add_subplot(gs[0, 0])
                offset_ax2 = fig.add_subplot(gs[3, 0])

                chi_ax1 = fig.add_subplot(gs[1, 0], sharex=offset_ax1)
                chi_ax2 = fig.add_subplot(gs[4, 0], sharex=offset_ax2,
                                          sharey=chi_ax1)

                vs_ax1 = fig.add_subplot(gs[0:1, 1], sharey=offset_ax1)
                vs_ax2 = fig.add_subplot(gs[3:4, 1], sharex=vs_ax1,
                                         sharey=offset_ax2)

                hist_ax1 = fig.add_subplot(gs[0, 2], sharey=offset_ax1)
                hist_ax2 = fig.add_subplot(gs[3, 2], sharey=offset_ax2)

                axes1 = (offset_ax1, chi_ax1, vs_ax1, hist_ax1)
                axes2 = (offset_ax2, chi_ax2, vs_ax2, hist_ax2)

                for axes, time_slice, params in zip((axes1, axes2),
                                                    (pre_slice, post_slice),
                                                    (style_params_pre,
                                                     style_params_post)):
                    offset_ax, chi_ax, vs_ax, hist_ax = axes
                    # Do the statistics and create the plots.
                    layout_plots()

            else:
                fig = plt.figure(figsize=(11, 5), tight_layout=True)
                gs = GridSpec(nrows=2, ncols=3, figure=fig, hspace=0,
                              height_ratios=[11, 3],
                              width_ratios=[5, 5, 1])
                offset_ax = fig.add_subplot(gs[0, 0])
                chi_ax = fig.add_subplot(gs[1, 0])
                vs_ax = fig.add_subplot(gs[0, 1], sharey=offset_ax)
                hist_ax = fig.add_subplot(gs[0, 2], sharey=offset_ax)

                if star.fiberSplitIndex == 0:
                    params = style_params_post
                else:
                    params = style_params_pre
                time_slice = slice(None, None)

                # Do the statistics and create the plots.
                layout_plots()

            filename = plots_dir / '{}_{}.png'.format(
                    obj_name, transition_label)
            fig.savefig(filename)
            plt.close(fig)


def create_BERV_plots():
    """Create plots of pair offset vs. the BERV at the time of the observation.

    """

    def layout_data(ax, time_slice, params):
        """Actually layout data onto plots.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` object
            An axis to plot the data onto.
        time_slice : slice object
            A slice objects indicating how many row of the various star arrays
            to use.
        params : dict
            A dictionary of plotting params to pass to `matplotlib.errorbar`.

        """

        # Gather the observations for the given time slice.
        separations = star.pairSeparationsArray[time_slice,
                                                column_index].to(u.km/u.s)
        errors = star.pairSepErrorsArray[time_slice, column_index]

        velocities = star.bervArray[time_slice] + star.radialVelocity
        velocities.convert_to_units('km/s')

        if args.verbose:
            # Generate some binned data to print out.
            vel_lower = velocities.min()
            vel_upper = velocities.max()
            bin_lims = np.linspace(vel_lower, vel_upper, num=6)
            binned_seps = []
            binned_chi_squareds = []
            for limits in pairwise(bin_lims):
                bin_seps = []
                bin_errs = []
                for vel, sep, err in zip(velocities, separations, errors):
                    if limits[0] < vel < limits[1]:
                        bin_seps.append(sep.to(u.m/u.s))
                        bin_errs.append(err.to(u.m/u.s))

                bin_errs = np.array(bin_errs)
                bin_mean = np.average(bin_seps,
                                      weights=1/bin_errs**2)
                chi_squared = np.sum(((bin_seps - bin_mean) / bin_errs) ** 2)
                if len(bin_seps) > 1:
                    reduced_chi_squared = chi_squared / (len(bin_seps) - 1)
                    binned_chi_squareds.append(reduced_chi_squared)
                else:
                    binned_chi_squareds.append(chi_squared)

                binned_seps.append(bin_mean)

            tqdm.write(str(tabulate(zip(binned_seps, binned_chi_squareds),
                                    headers=["Separation (km/s)",
                                             "Reduced χ²"])))

        ax.set_xlim(left=-100, right=100)
        ax.yaxis.grid(which='major', color='Gray', alpha=0.6, linestyle='--')
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.axhline(y=np.mean(separations), color='DimGray', linestyle='-')
        ax.errorbar(x=velocities, y=separations, yerr=errors,
                    **params)

    # Define slice objects to capture time periods.
    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    for pair_label in tqdm(star._pair_bidict.keys()):
        column_index = star.p_index(pair_label)

        if star.fiberSplitIndex not in (0, None):
            fig = plt.figure(figsize=(6, 6), tight_layout=True)
            gs = GridSpec(nrows=2, ncols=1, figure=fig,
                          height_ratios=[1, 1], hspace=0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax1.set_ylabel('Pair separation pre-change (km/s)')
            ax2.set_ylabel('Pair separation post-change (km/s)')
            ax2.set_xlabel('Radial velocity (km/s)')

            ax1.tick_params(labelbottom=False)

            for axis, time_slice, params in zip((ax1, ax2),
                                                (pre_slice, post_slice),
                                                (style_params_pre,
                                                 style_params_post)):

                layout_data(axis, time_slice, params)

        else:
            # BUG Only uses post-change, doesn't handle pre-change.
            fig = plt.figure(figsize=(6, 3), tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)

            ax.set_ylabel('Pair separation post-change (km/s)')
            ax.set_xlabel('Radial velocity (km/s)')

            if star.fiberSplitIndex == 0:
                params = style_params_post
            else:
                params = style_params_pre

            layout_data(ax, slice(None, None), params)

        plots_dir = data_dir / 'BERV_plots'
        if not plots_dir.exists():
            os.mkdir(plots_dir)
        filename = plots_dir / '{}_{}.png'.format(
                obj_name, pair_label)
        fig.savefig(filename)
        plt.close(fig)


def create_offset_plot(star):
    """Create a plot for this star showing the average offset in the measured
    wavelength of a transition compared to its laboratory expected value.

    Parameters
    ----------
    star : `star.Star`
        The star object to get data from.

    """

    def layout_plots(time_slice, time_str, color):
        """Layout data onto the current axis.

        Parameters
        ----------
        time_slice : slice
            A slice object specifying the range of observations to use for this
            axis.
        time_str : str, {'pre', 'post'}
            A string specifying the time period (pre- or post-fiber change).

        """

        ax.yaxis.grid(which='major', color='Gray', alpha=0.7,
                      linestyle='--')
        ax.yaxis.grid(which='minor', color='Gray', alpha=0.6,
                      linestyle='-.')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))
        ax.xaxis.grid(which='major', color='Gray', alpha=0.7,
                      linestyle='--')
        ax.xaxis.grid(which='minor', color='Gray', alpha=0.6,
                      linestyle=':')

        offset_pattern = star.getTransitionOffsetPattern(
            array_slice=time_slice,
            normalized=False)
        offsets, stddevs = offset_pattern[0], offset_pattern[1]

        ax.set_xlim(left=-1, right=len(offsets) + 1)

        median = np.nanmedian(offsets)
        mean_centered_offsets = offsets - median
        weighted_mean = np.average(offsets, weights=1/stddevs**2)
        vprint(f'weighted_mean = {weighted_mean:.3f}')
        vprint(f'mean = {np.mean(offsets):.3f}')

        indices = range(len(offsets))
        ax.errorbar(x=indices,
                    y=mean_centered_offsets,
                    yerr=stddevs,
                    label=f'{time_str.capitalize()}-fiber change\n'
                    f'median$=${median:.2f}\n'
                    f'N$={star.getNumObs(time_slice)}$,'
                    f' RV$=${star.radialVelocity}',
                    marker='_', linestyle='',
                    markeredgecolor=params['markeredgecolor'],
                    color=params['color'],
                    alpha=params['alpha'],
                    markersize=params['markersize'],
                    ecolor=color)
#        ax.set_ylabel(r'$\Delta\lambda_\mathrm{expected}$ (m/s)')
        ax.set_ylabel(r'$\Delta v-\mu_{\Delta v}$ (m/s)')
        ax.legend(loc='upper center')

    ecolor_pre = 'SaddleBrown'
    ecolor_post = 'MidnightBlue'

    plot_name = data_dir / f'{data_dir.stem}_offset_pattern.png'

    fig = plt.figure(figsize=(9, 7), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig,
                  height_ratios=[1, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    if star.fiberSplitIndex not in (0, None):

        pre_slice = slice(None, star.fiberSplitIndex)
        post_slice = slice(star.fiberSplitIndex, None)

        for ax, params, time_slice, time_str, color in zip((ax1, ax2),
                                                           (style_params_pre,
                                                            style_params_post),
                                                           (pre_slice,
                                                            post_slice),
                                                           ('pre', 'post'),
                                                           (ecolor_pre,
                                                            ecolor_post)):
            vprint(f'Creating "{time_str}" plot.')
            layout_plots(time_slice, time_str, color)

        ax1.tick_params(labelbottom=False)

    else:
        if star.fiberSplitIndex == 0:
            params = style_params_post
            time_str = 'post'
            ax = ax2
            color = ecolor_post
            vprint(f'Creating "{time_str}" plot.')
        else:
            params = style_params_pre
            time_str = 'pre'
            ax = ax1
            color = ecolor_pre
            vprint(f'Creating "{time_str}" plot.')
        time_slice = slice(None, None)

        layout_plots(time_slice, time_str, color)

    ax2.set_xlabel('Index number')

    fig.savefig(str(plot_name))
    plt.close(fig)
    # Link the plots to a common directory.
    dest_name = pattern_dir / plot_name.name
    vprint(f'Saving plot to {dest_name}.')
    if dest_name.exists():
        os.unlink(dest_name)
    os.link(plot_name, dest_name)


def create_chi_squared_plots():
    """Create plots of chi^2_nu distributions for the given star.

    """

    def layout_data():

        chi_squareds = star.chiSquaredNuArray[time_slice, column_index]
        means = star.fitOffsetsArray[time_slice, column_index]
        x_pos = np.std(means)
        # x_pos = [transition.normalizedDepth] * len(chi_squareds)

        axis.errorbar(x=x_pos, y=np.mean(chi_squareds),
                      # yerr=np.std(chi_squareds),
                      marker='o', markersize=4,
                      alpha=0.8)

    fig = plt.figure(figsize=(9, 8), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig,
                  height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)

    ax2.set_xlabel('Standard deviation (m/s)')
    for ax in (ax1, ax2):
        ax.set_ylabel(r'Mean $\chi^2_\nu$')
        ax.axhline(1, color='Black', linestyle='--')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax.xaxis.grid(which="major", color='Black', alpha=0.6,
                      linestyle=':')

    for transition in star.transitionsList:
        for order_num in transition.ordersToFitIn:
            transition_label = '_'.join((transition.label, str(order_num)))
            column_index = star.t_index(transition_label)

            if star.fiberSplitIndex not in (0, None):

                pre_slice = slice(None, star.fiberSplitIndex)
                post_slice = slice(star.fiberSplitIndex, None)

                for axis, time_slice in zip((ax1, ax2),
                                            (pre_slice, post_slice)):
                    layout_data()

            else:
                if star.fiberSplitIndex == 0:
                    axis = ax2
                else:
                    axis = ax1
                time_slice = slice(None, None)
                layout_data()

    # Set the limit after the data plotting to retain the auto-set upper
    # limit.
    ax1.set_ylim(bottom=0, top=ax1.get_ylim()[1])

    file_name = data_dir / f'{star.name}_chi_squared_stddev.png'
    tqdm.write(f'Saving plot to {file_name}')

    fig.savefig(str(file_name))
    plt.close(fig)
    dest_name = chi_squared_dir / file_name.name
    if dest_name.exists():
        os.unlink(dest_name)
    os.link(file_name, dest_name)


# Write out a CSV file containing the pair separation values for all
# observations of this star.
# TODO: update
def write_csv(column_names):

    csv_filename = data_dir / 'pair_separations_{}.csv'.format(data_dir.stem)
    vprint(f'Creating CSV file of separations for {data_dir.stem}'
           f' at {csv_filename}')

    assert len(master_star_list[0]) == len(column_names)

    with open(csv_filename, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(column_names)
        for row in tqdm(master_star_list):
            csv_writer.writerow(row)

    # Write out a series of CSV files containing information on the fits of
    # individual transitions for each star.
    column_headers = ['ObsDate', 'Amplitude', 'Amplitude_err (A)', 'Mean (A)',
                      'Mean_err (A)', 'Mean_err_vel (m/s)', 'Sigma (A)',
                      'Sigma_err (A)', 'Offset (m/s)', 'Offset_err (m/s)',
                      'FWHM (m/s)', 'FWHM_err (m/s)', 'Chi-squared-nu',
                      'Order', 'Mean_airmass']

    csv_fits_dir = data_dir / 'fits_info_csv'
    if not csv_fits_dir.exists():
        os.mkdir(csv_fits_dir)
    vprint(f'Writing information on fits to files in {csv_fits_dir}')
    for transition in tqdm(transitions_list):
        csv_filename = csv_fits_dir / '{}_{}.csv'.format(transition.label,
                                                         data_dir.stem)

        with open(csv_filename, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(column_headers)
            for fits_dict in master_fits_list:
                csv_writer.writerow(fits_dict[transition.label].
                                    getFitInformation())


def create_pair_offset_plots(plot_dir):
    """Create plots showing the pair separations for all observations in the
    given star.

    Parameters
    ----------
    plot_dir : `pathlib.Path`
        The directory where the output plots should be saved.

    """

    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = '_'.join((pair.label, str(order_num)))
            vprint(f'Creating plot for pair {pair_label}')
            fitted_pairs = []
            date_obs = []
            for pair_dict in master_star_dict.values():
                if pair_dict[pair_label] is not None:
                    # Grab the associated pair from each observation.
                    fitted_pairs.append(pair_dict[pair_label])
                    # Grab the observation date.
                    date_obs.append(pair_dict[pair_label][0].dateObs)
                else:
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

            vprint(f"Weighted mean for {pair_label} is"
                   " {weighted_mean:.2f}")

            normalized_offsets = offsets - weighted_mean
#            chi_squared = sum((normalized_offsets / errors) ** 2)

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

            plot_name = plot_dir / '{}.png'.format(pair_label)

            fig, axes = plt.subplots(ncols=2, nrows=2,
                                     tight_layout=True,
                                     figsize=(10, 8),
                                     sharey='all')  # Share y-axis among all.
            fig.autofmt_xdate()
            (ax1, ax2), (ax3, ax4) = axes
            for ax in (ax1, ax2, ax3, ax4):
                ax.set_ylabel(r'$\Delta v_{\mathrm{sep}}\mathrm{ (m/s)}$')
                ax.axhline(y=0, **weighted_mean_params)
                ax.axhline(y=weighted_mean_err,
                           **weighted_err_params)
                ax.axhline(y=-weighted_mean_err,
                           **weighted_err_params)
            for key, value in dates_of_change.items():
                ax3.axvline(label=key, **value)

            # Set up axis 1.
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))
            ax1.grid(which='major', axis='y', color='Gray', alpha=0.6,
                     linestyle='--')
            ax1.grid(which='major', axis='x', color='Gray', alpha=0.6,
                     linestyle='-')
            ax1.grid(which='minor', axis='x', color='Gray', alpha=0.6,
                     linestyle=':')
            # Plot pre-fiber change observations.
            # TODO: This all needs to take into account the three possible
            # cases for pre-/post-fiber change distributions.
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
                             label=r'$\chi^2_\nu=${:.3f},'
                                   r' $\mu=${:.2f}'.format(
                                     chi_squared_nu_post.value,
                                     ),
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


def create_airmass_plots(transition_plots_dir):
    """Create plots of the velocity offsets of each transition vs. the airmass
    of the observation.

    Parameters
    ----------
    transition_plots_dir : `pathlib.Path`
        The directory to use as the parent for the plots created by this
        function.

    """

    def get_airmasses(time_slice):
        """Get the airmasses for a given time slice.

        Parameters
        ----------
        time_slice : slice
            A slice object representing the time period to get airmasses from
            observations in.

        """

        return star.airmassArray[time_slice]

    def get_offsets(time_slice):
        """Get the offsets from expected position for a given time_slice.

        """

        return star.fitOffsetsArray[time_slice, column_index]

    airmass_dir = transition_plots_dir / 'airmass_plots'
    if not airmass_dir.exists():
        os.mkdir(airmass_dir)

    for transition_label in tqdm(star._transition_bidict.keys()):
        column_index = star.t_index(transition_label)
        filename = airmass_dir / 'Airmass_dispersion_{}.png'.format(
                transition_label)

        pre_slice = slice(None, star.fiberSplitIndex)
        post_slice = slice(star.fiberSplitIndex, None)

        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('Airmass')
        ax.set_ylabel('Offset from expected wavelength (m/s)')

        if star.fiberSplitIndex not in (0, None):

            for time_slice, params, time in zip((pre_slice, post_slice),
                                                (style_params_pre,
                                                style_params_post),
                                                ('pre', 'post')):
                airmasses = get_airmasses(time_slice)
                offsets = get_offsets(time_slice)

                ax.axhline(y=np.mean(offsets),
                           label=f'Mean ({time}-fiber change)',
                           color=params['color'],
                           linestyle='--')

                ax.scatter(airmasses, offsets, s=32,
                           c=params['color'],
                           alpha=params['alpha'],
                           edgecolor=params['markeredgecolor'])

        else:
            if star.fiberSplitIndex == 0:
                params = style_params_post
                time = 'post'
            else:
                params = style_params_pre
                time = 'pre'
            time_slice = slice(None, None)

            airmasses = get_airmasses(time_slice)
            offsets = get_offsets(time_slice)

            ax.axhline(y=np.mean(offsets),
                       label=f'Mean ({time}-fiber change)',
                       color=params['color'],
                       linestyle='--')

            ax.scatter(airmasses, offsets, s=32,
                       c=params['color'],
                       alpha=params['alpha'],
                       edgecolor=params['markeredgecolor'])

        ax.legend(framealpha=0.6)

        fig.savefig(str(filename))
        plt.close(fig)


def create_dir_if_necessary(func):
    """Create a directory returned from a function if it doesn't yet exist.

    Parameters
    func : function which returns a `pathlib.Path` object
        A function which returns a directory to be created if it doesn't
        already exist. Directory creation will still fail if the directory's
        immediate parent doesn't exist.

    """

    def wrapper():
        directory = func()

        if not directory.exists():
            vprint(directory)
            vprint('Creating missing directory.')
            os.mkdir(directory)

        return directory

    return wrapper


@create_dir_if_necessary
def transition_dir():
    """Return the directory to put transitions-speciic plots in, and create it
    if it doesn't exist.

    Returns
    -------
    `pathlib.Path`
        The directory to put transition-specific plots in.

    """

    return data_dir / 'plots_by_transition'


@create_dir_if_necessary
def pattern_dir():
    """Return the directory to link star pattern plots in, creating it first if
    necessary.

    Returns
    -------
    `pathlib.Path`
        The directory to hard link the pattern plots into.

    """

    return data_dir.parent / 'star_pattern_plots'


@create_dir_if_necessary
def chi_squared_plots_dir():
    """Return the path to the directory to put plots of chi^2 vs. standard
    deviation for each star in.

    """

    return data_dir.parent / 'chi^2_stddev_plots'


if __name__ == '__main__':

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

    parser.add_argument('--create-transition-offset-plots',
                        action='store_true',
                        help='Create plots of the distributions of each'
                        ' individual  transition for a star.')

    parser.add_argument('--create-pair-offset-plots',
                        action='store_true',
                        help='Create plots of the offsets for each pair.')

    parser.add_argument('--create-fit-plots', action='store_true',
                        help='Create plots of the fits for each transition.')

    parser.add_argument('--create-berv-plots', action='store_true',
                        help="Create plots of each pair's separation vs. the"
                        ' BERV at the time of observation.')

    parser.add_argument('--create-airmass-plots', action='store_true',
                        help="Plot transitions' scatter as a function of"
                        ' airmass.')

    parser.add_argument('--create-offset-plot', action='store_true',
                        help='Create a plot of the average offset for all'
                        ' transitions from their expected laboratory position.')

    parser.add_argument('--create-chi-squared-plots', action='store_true',
                        help='Create plots showing the distribution of chi^2'
                        ' values for the given star.')

    parser.add_argument('--link-fit-plots', action='store_true', default=False,
                        help='Hard-link fit plots by transition into new'
                        ' individual folders for each transition.')

    parser.add_argument('--write-csv', action='store_true', default=False,
                        help='Create a CSV file of offsets for each pair.')

    parser.add_argument('--recreate-star', action='store_true', default=False,
                        help='Force the star to be recreated from directories'
                        ' rather than be read in from saved data.')

    parser.add_argument('--simplified', action='store_true', default=False,
                        help='Forego usual tick markers for stars with'
                        ' thousands of observations.')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print more information about the process.')

    args = parser.parse_args()

    # Define vprint to write out messages (using tqdm.write, after having been
    # string-ified) if the "verbose" flag is passed.
    vprint = vcl.verbose_print(args.verbose)

#    if args.use_tex or args.create_pair_offset_plots:
#        plt.rc('text', usetex=True)

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
        raise RuntimeError('The given directory does not exist:'
                           f' {data_dir}')

    # Read the list of chosen transitions.
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    tqdm.write(f'Found {len(transitions_list)} individual transitions.')

    # Read the list of chosen pairs.
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    tqdm.write(f'Found {len(pairs_list)} transition pairs (total) in list.')

    ###############
    # Stuff that needs a Star.

    if args.create_transition_offset_plots or args.create_berv_plots\
            or args.create_offset_plot or args.create_airmass_plots\
            or args.create_chi_squared_plots:

        # Read or create a Star object for this star.
        obj_name = data_dir.stem
        if args.recreate_star:
            load_data = False
        else:
            load_data = None
        star = Star(obj_name, data_dir, suffix=args.suffix,
                    load_data=load_data)

        if args.create_offset_plot:
            tqdm.write('Creating plot of mean fit offsets.')
            pattern_dir = pattern_dir()
            create_offset_plot(star)

        if args.create_transition_offset_plots:
            tqdm.write('Creating transition offset plots.')
            transition_plots_dir = data_dir / 'transition_offset_plots'
            if not transition_plots_dir.exists():
                os.mkdir(transition_plots_dir)
            create_transition_offset_plots(transition_plots_dir)

        if args.create_berv_plots:
            tqdm.write('Creating plots of pair offsets vs. BERV.')
            create_BERV_plots()

        if args.create_airmass_plots:
            tqdm.write('Creating plots of offsets vs. airmass.')
            t_plots_dir = transition_dir()
            create_airmass_plots(t_plots_dir)

        if args.create_chi_squared_plots:
            tqdm.write('Creating plots of chi^2 distributions.')
            chi_squared_dir = chi_squared_plots_dir()
            if star.getNumObs(slice(None, None)) > 1:
                create_chi_squared_plots()
            else:
                tqdm.write('Only one observation for this star.')

    # Stuff that needs analysis of pickle files.
    if args.write_csv or args.create_pair_offset_plots\
            or args.create_fit_plots:

        # Search for pickle files in the given directory.
        search_str = str(data_dir)+f'/HARPS*/pickles_{args.suffix}/*fits.lzma'
        vprint(f'Searching for pickle files using string: {search_str}')
        pickle_files = [Path(path) for path in glob(search_str)]

        # dictionary with entries per observation
        # entries consist of dictionary with entries of pairs made from fits

        # Set up the master dictionary to contain sub-entries per observation.
        master_star_dict = {}

        obs_name_re = re.compile('HARPS.*_e2ds_A')

        # Create some lists to hold all the results for saving out as a CSV
        # file:
        master_star_list = []
        master_fits_list = []
        for pickle_file in tqdm(pickle_files):

            # Match the part of the pickle filename that is the observation
            # name.
            obs_name = obs_name_re.match(pickle_file.stem).group()

            tqdm.write('Analyzing results from {}'.format(obs_name))
            with lzma.open(pickle_file, 'rb') as f:
                fits_list = pickle.loads(f.read())

            # Set up a dictionary to map fits in this observation to
            # transitions:
            fit_labels = []
            for transition in transitions_list:
                for order_num in transition.ordersToFitIn:
                    fit_labels.append(f'{transition.label}_{order_num}')
            fits_dict = {}
            for fit, label in zip(fits_list, fit_labels):
                if fit is not None:
                    assert fit.label == label, f'{fit.label} does not match'\
                                                'generated label: {label}'
                    fits_dict[label] = fit
                else:
                    fits_dict[label] = None
#            fits_dict = {fit.label: fit for fit in fits_list}
            master_fits_list.append(fits_dict)

            # This can be used to remake plots for individual fits without
            # rerunning the fitting process in case the plot visual format
            # changes.
            if args.create_fit_plots:
                closeup_dir = data_dir /\
                    '{}/plots_{}/close_up'.format(obs_name, args.suffix)
                context_dir = data_dir /\
                    '{}/plots_{}/context'.format(obs_name, args.suffix)
                tqdm.write('Creating plots of fits.')

                for transition in tqdm(transitions_list):
                    for order_num in transition.ordersToFitIn:
                        plot_closeup = closeup_dir /\
                            f'{obs_name}_{transition.label}'\
                            f'_{order_num}_close.png'
                        plot_context = context_dir /\
                            f'{obs_name}_{transition.label}'\
                            f'_{order_num}_context.png'
                        vprint('Creating plots at:')
                        vprint(plot_closeup)
                        vprint(plot_context)
                        tr_label = f'{transition.label}_{order_num}'
                        if fits_dict[tr_label] is not None:
                            fits_dict[tr_label].plotFit(plot_closeup,
                                                        plot_context)
                        else:
                            err_msg = f'The fit for {tr_label} failed for'\
                                      f' observation {obs_name}, and'\
                                      ' will need to be re-reduced to be'\
                                      ' created.'
                            tqdm.write(err_msg)

            pairs_dict = {}
            for fit in fits_list:
                # Iterate through the list until a non-None observation is
                # found.
                try:
                    separations_list = [obs_name, fit.dateObs.
                                        isoformat(timespec='milliseconds')]
                    break
                except AttributeError:
                    # If the fit is None, this will catch it.
                    pass

            column_names = ['Observation', 'Time']
            for pair in pairs_list:
                for order_num in pair.ordersToMeasureIn:
                    pair_label = '_'.join((pair.label, str(order_num)))
                    column_names.extend([pair_label, pair_label + '_err'])
                    high_E_label = '_'.join((pair._higherEnergyTransition.
                                            label, str(order_num)))
                    low_E_label = '_'.join((pair._lowerEnergyTransition.
                                            label, str(order_num)))

                    # Check that fits for both transitions exist.
                    if fits_dict[high_E_label] and fits_dict[low_E_label]:
                        fits_pair = [fits_dict[high_E_label],
                                     fits_dict[low_E_label]]

                        assert fits_pair[0].order == fits_pair[1].order,\
                            f"Orders don't match for {fits_pair}"
                        if np.isnan(fits_pair[0].meanErrVel) or \
                           np.isnan(fits_pair[1].meanErrVel):
                            # Similar to above, fill in list with placeholder
                            # values.
                            vprint(f'{pair.label} in {obs_name} has a'
                                   ' NaN velocity offset!')
                            vprint(fits_pair[0].meanErrVel)
                            vprint(fits_pair[1].meanErrVel)
                            separations_list.extend(['NaN', ' NaN'])
                            pairs_dict[pair_label] = None
                            continue

                    else:
                        # Measurement of one or both transition doesn't
                        # exist, so skip it (but fill in the list to prevent
                        # getting out of sync)
                        separations_list.extend(['N/A', ' N/A'])
                        pairs_dict[pair_label] = None
                        continue

                    # If both transitions have non-NaN values, add the velocity
                    # separation and error:
                    pairs_dict[pair_label] = fits_pair

                    velocity_separation = wave2vel(fits_pair[0].mean,
                                                   fits_pair[1].mean)
                    error = np.sqrt(fits_pair[0].meanErrVel ** 2 +
                                    fits_pair[1].meanErrVel ** 2)
                    separations_list.extend([velocity_separation.value,
                                             error.value])

            # This is for the script to use.
            master_star_dict[obs_name] = pairs_dict
            if args.write_csv:
                # This is to be written out.
                master_star_list.append(separations_list)

    if args.write_csv:
        tqdm.write('Writing out CSV files.')
        write_csv(column_names)

    if args.create_pair_offset_plots:
        # Create the plots for each pair of transitions
        pair_plots_dir = data_dir / 'pair_offset_plots'
        if not pair_plots_dir.exists():
            os.mkdir(pair_plots_dir)
        create_pair_offset_plots(pair_plots_dir)

    ################
    # Stuff that doesn't need a Star or analysis of pickle files.

    # Create hard links to all the fit plots by transition (across star) in
    # their own directory, to make it easier to compare transitions across
    # observations.
    if args.link_fit_plots:
        t_plots_dir = transition_dir()
        tqdm.write('Linking fit plots to cross-observation directories.')
        link_fit_plots(t_plots_dir)
