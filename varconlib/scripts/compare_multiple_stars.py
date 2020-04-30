#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:07:04 2019

@author: dberke
A script to compare multiple stars in various ways, such as plotting the
transition offset pattern for multiple stars.

"""

import argparse
import csv
from itertools import zip_longest
from pathlib import Path
import pickle
from pprint import pprint
import sys

import h5py
import hickle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import HDF5FileNotFoundError
import varconlib.fitting.fitting
from varconlib.miscellaneous import get_params_file
from varconlib.star import Star


# Define style parameters to use for stellar parameter plots.
style_pre = {'color': 'Chocolate',
             'ecolor_thick': 'DarkOrange',
             'ecolor_thin': 'BurlyWood'}
style_post = {'color': 'DodgerBlue',
              'ecolor_thick': 'CornFlowerBlue',
              'ecolor_thin': 'LightSkyBlue'}
style_ref = {'color': 'DarkGreen',
             'ecolor_thick': 'ForestGreen',
             'ecolor_thin': 'DarkSeaGreen'}
style_markers = {'markeredgecolor': 'Black',
                 'markeredgewidth': 1,
                 'alpha': 0.7,
                 'markersize': 4}
style_caps = {'capsize_thin': 4,
              'capsize_thick': 7,
              'linewidth_thin': 2,
              'linewidth_thick': 3,
              'cap_thin': 1.5,
              'cap_thick': 2.5}


def get_star(star_path, verbose=False):
    """Return a varconlib.star.Star object based on its name.

    Parameters
    ----------
    star_path : str
        A string representing the name of the directory where the HDF5 file
        containing a `star.Star`'s data can be found.

    Optional
    --------
    verbose : bool, Default: False
        If *True*, write out additional information.

    Returns
    -------
    `star.Star`
        A Star object from the directory. Note that this will only use already-
        existing stars, it will not create ones which do not already exist from
        their observations.

    """

    assert star_path.exists(), FileNotFoundError('Star directory'
                                                 f' {star_path}'
                                                 ' not found.')
    try:
        return Star(star_path.stem, star_path, load_data=True)
    except IndexError:
        if verbose:
            tqdm.write(f'Excluded {star_path.stem}.')
        pass
    except HDF5FileNotFoundError:
        if verbose:
            tqdm.write(f'No HDF5 file for {star_path.stem}.')
        pass
    except AttributeError:
        if verbose:
            tqdm.write(f'Affected star is {star_path.stem}.')
        raise


def create_parameter_comparison_figures(ylims=None,
                                        temp_lims=(5300 * u.K, 6200 * u.K),
                                        mtl_lims=(-0.75, 0.4),
                                        mag_lims=(4, 5.8),
                                        logg_lims=(4.1, 4.6)):
    """Creates and returns a figure with pre-set subplots.

    This function creates the background figure and subplots for use with the
    --compare-stellar-parameter-* flags.

    Optional
    ----------
    ylims : 2-tuple of floats or ints
        A tuple of length 2 containing the upper and lower limits of the
        subplots in the figure.
    temp_lims : 2-tuple of floats or ints (optional dimensions of temperature)
        A tuple of length containing upper and lower limits for the x-axis of
        the temperature subplot.
    mtl_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the metallicity subplot.
    mag_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the absolute magnitude subplot.
    logg_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the log(g) subplot.

    Returns
    -------
    tuple
        A tuple containing the figure itself and the various axes of the
        subplots within it.

    """

    comp_fig = plt.figure(figsize=(16, 8), tight_layout=True)
    gs = GridSpec(ncols=5, nrows=2, figure=comp_fig,
                  width_ratios=(5, 5, 5, 5, 3))

    temp_ax_pre = comp_fig.add_subplot(gs[0, 0])
    temp_ax_post = comp_fig.add_subplot(gs[1, 0],
                                        sharex=temp_ax_pre,
                                        sharey=temp_ax_pre)
    mtl_ax_pre = comp_fig.add_subplot(gs[0, 1],
                                      sharey=temp_ax_pre)
    mtl_ax_post = comp_fig.add_subplot(gs[1, 1],
                                       sharex=mtl_ax_pre,
                                       sharey=mtl_ax_pre)
    mag_ax_pre = comp_fig.add_subplot(gs[0, 2],
                                      sharey=temp_ax_pre)
    mag_ax_post = comp_fig.add_subplot(gs[1, 2],
                                       sharex=mag_ax_pre,
                                       sharey=mag_ax_pre)
    logg_ax_pre = comp_fig.add_subplot(gs[0, 3],
                                       sharey=temp_ax_pre)
    logg_ax_post = comp_fig.add_subplot(gs[1, 3],
                                        sharex=logg_ax_pre,
                                        sharey=logg_ax_pre)
    hist_ax_pre = comp_fig.add_subplot(gs[0, 4],
                                       sharey=temp_ax_pre)
    hist_ax_post = comp_fig.add_subplot(gs[1, 4],
                                        sharex=hist_ax_pre,
                                        sharey=hist_ax_pre)

    all_axes = (temp_ax_pre, temp_ax_post, mtl_ax_pre, mtl_ax_post,
                mag_ax_pre, mag_ax_post, logg_ax_pre, logg_ax_post,
                hist_ax_pre, hist_ax_post)
    # Set the plot limits here. The y-limits for temp_ax1 are
    # used for all subplots.
    if ylims is not None:
        temp_ax_pre.set_ylim(bottom=ylims[0],
                             top=ylims[1])
    temp_ax_pre.set_xlim(left=temp_lims[0],
                         right=temp_lims[1])
    mtl_ax_pre.set_xlim(left=mtl_lims[0],
                        right=mtl_lims[1])

    # Axis styles for all subplots.
    for ax in all_axes:
        if not args.full_range:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(
                                       base=100))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(
                                       base=50))
        else:
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.axhline(y=0, color='Black', linestyle='--')
        ax.yaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.85)
        ax.yaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.75)
        if ax not in (hist_ax_pre, hist_ax_post):
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='--', alpha=0.85)

    for ax in (temp_ax_pre, temp_ax_post):
        ax.set_xlabel('Temperature (K)')
    for ax in (mtl_ax_pre, mtl_ax_post):
        ax.set_xlabel('Metallicity [Fe/H]')
    for ax in (mag_ax_pre, mag_ax_post):
        ax.set_xlabel('Absolute Magnitude')
    for ax in (logg_ax_pre, logg_ax_post):
        ax.set_xlabel(r'$\log(g)$')

    # Just label the left-most two subplots' y-axes.
    for ax in (temp_ax_pre, temp_ax_post):
        ax.set_ylabel('Pre-fiber change offset (m/s)')

    axes_dict = {'temp_pre': temp_ax_pre, 'temp_post': temp_ax_post,
                 'mtl_pre': mtl_ax_pre, 'mtl_post': mtl_ax_post,
                 'mag_pre': mag_ax_pre, 'mag_post': mag_ax_post,
                 'logg_pre': logg_ax_pre, 'logg_post': logg_ax_post,
                 'hist_pre': hist_ax_pre, 'hist_post': hist_ax_post}

    return comp_fig, axes_dict


def plot_data_points(axis, x_pos, y_pos, thick_err, thin_err, era=None,
                     ref=False):
    """Plot a data point for a star.

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        An axes to plot the data on.
    x_pos : iterable of floats or `unyt.unyt_quantity`
        The x-positions of the points to plot.
    y_pos : iterable of floats or `unyt.unyt_quantity`
        The y-positions of the points to plot.
    thick_err : iterable of floats or `unyt.unyt_quantity`
        The values of the thick error bars to plot.
    thin_err : iterable of floats or `unyt.unyt_quantity`
        The values of the thin error bars to plot.
    era : string, ['pre', 'post'], Default : None
        Whether the time period of the plot is pre- or post-fiber
        change. Only allowed values are 'pre' and 'post'. Controls
        color of the points. If `ref` is *True*, the value of `era` is
        ignored, and can be left unspecified, otherwise it needs a
        value to be given.
    ref : bool, Default : False
        Whether this data point is for the reference star. If *True*,
        will use a special separate color scheme.

    Returns
    -------
    None.

    """
    if ref:
        params = style_ref
    elif era == 'pre':
        params = style_pre
    elif era == 'post':
        params = style_post
    else:
        raise ValueError("Keyword 'era' received an unknown value"
                         f" (valid values are 'pre' & 'post'): {era}")

    axis.errorbar(x=x_pos, y=y_pos,
                  yerr=thin_err, linestyle='',
                  marker='', capsize=style_caps['capsize_thin'],
                  color=params['color'],
                  ecolor=params['ecolor_thin'],
                  elinewidth=style_caps['linewidth_thin'],
                  capthick=style_caps['cap_thin'])
    axis.errorbar(x=x_pos, y=y_pos,
                  yerr=thick_err, linestyle='',
                  marker='o', markersize=style_markers['markersize'],
                  markeredgewidth=style_markers['markeredgewidth'],
                  markeredgecolor=style_markers['markeredgecolor'],
                  alpha=style_markers['alpha'],
                  capsize=style_caps['capsize_thick'],
                  color=params['color'],
                  ecolor=params['ecolor_thick'],
                  elinewidth=style_caps['linewidth_thick'],
                  capthick=style_caps['cap_thick'])


def get_pair_data_point(star, time_slice, pair_label):
    """Return the pair separation for a given star and pair.

    The returned values will be the weighted mean value of the pair
    separation, the standard deviation of all the pair separation
    values for that star in the given time period (pre- or post-fiber
    chage), and the error on the weighted mean.

    Parameters
    ----------
    star : `star.Star`
        The star get the data from.
    time_slice : slice
        A slice object specifying the data to use from the star.
    pair_label : str
        The label to use to select a particular pair.

    Returns
    -------
    tuple
        Returns a 3-tuple of the weighted mean, the error on the
        weighted mean, and the standard deviation.

    """

    col_index = star.p_index(pair_label)

    separations = star.pairSeparationsArray[time_slice, col_index]
    errs = star.pairSepErrorsArray[time_slice, col_index]
    weighted_mean, weight_sum = np.average(separations,
                                           weights=errs**-2,
                                           returned=True)
    weighted_mean.convert_to_units(u.m / u.s)
    error_on_weighted_mean = 1 / np.sqrt(weight_sum)
    error_on_mean = np.std(separations) / np.sqrt(
        star.getNumObs(time_slice))

    return (weighted_mean, error_on_weighted_mean, error_on_mean)


def get_transition_data_point(star, time_slice, transition_label,
                              fit_params=None):
    """Return the weighted mean of a transition for a star across observations.

    The returned values will be the weighted mean value of the transition
    velocity offset from expected wavelength, the error on the weighted mean,
    and the error on the mean.

    Parameters
    ----------
    star : `star.Star`
        The star get the data from.
    time_slice : slice
        A slice object specifying the data to use from the star.
    transition_label : str
        The label to use to select a particular transition.

    Optional
    --------
    fit_params : dict
        Should be the results of a varconlib.miscellaneous.get_params_file()
        call, a dictionary containing various information about a fitting model.

    Returns
    -------
    tuple
        Returns a 3-tuple of the weighted mean, the error on the
        weighted mean, and the standard deviation.

    """
    # print(star.name)
    # print(star.getNumObs())

    col_index = star.t_index(transition_label)
    errs = star.fitErrorsArray[time_slice, col_index]

    if fit_params is None:
        offsets = star.fitOffsetsNormalizedArray[time_slice, col_index]
        weighted_mean, weight_sum = np.average(offsets,
                                               weights=errs.value**-2,
                                               returned=True)

        error_on_weighted_mean = 1 / np.sqrt(weight_sum)
        error_on_mean = np.nanstd(offsets) /\
            np.sqrt(star.getNumObs(time_slice))

    else:
        corrected_array, mask_array = star.getOutliersMask(fit_params,
                                                           n_sigma=2.5)
        offsets = ma.array(corrected_array.value, mask=mask_array)[time_slice,
                                                             col_index]
        weighted_mean, weight_sum = ma.average(offsets,
                                               weights=errs.value**-2,
                                               returned=True)
        # print(f'Weighted mean: {weighted_mean}')
        # print(f'Weight sum: {weight_sum}')
        error_on_weighted_mean = 1 / np.sqrt(weight_sum)
        # print(offsets)
        # print(type(offsets))
        # print(f'STDDEV: {ma.std(offsets)}')
        # print(np.sqrt(star.getNumObs(time_slice)))
        error_on_mean = ma.std(offsets) /\
            np.sqrt(star.getNumObs(time_slice))
        # print(f'EotWM: {error_on_weighted_mean}')
        # print(f'EotM: {error_on_mean}')
        # raise RuntimeError

    return (weighted_mean, error_on_weighted_mean, error_on_mean)


def main():
    """Run the main function for the script."""

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    main_dir = Path(args.main_dir[0])
    if not main_dir.exists():
        raise FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    if args.fit_params_file:
        vprint(f'Reading params file {args.fit_params_file}...')

        params_file = main_dir / f'fit_params/{args.fit_params_file}'
        fit_results = get_params_file(params_file)
        model_func = fit_results['model_func']
        coeffs = fit_results['coeffs']
        sigma_sys = fit_results['sigmas_sys']

        apply_corrections = True
    else:
        fit_results = None
        apply_corrections = False

    if args.reference_star:
        ref_star = get_star(main_dir / args.reference_star)
        tqdm.write(f'Reference star is {ref_star.name}.')

    star_list = []
    tqdm.write('Collecting stars...')
    for star_dir in tqdm(args.star_names):
        star = get_star(main_dir / star_dir)
        if star is None:
            pass
        else:
            if args.casagrande2011:
                vprint('Applying values from Casagrande et al. 2011.')
                star.getStellarParameters('Casagrande2011')
            elif args.nordstrom2004:
                vprint('Applying values from Nordstrom et al. 2004.')
                star.getStellarParameters('Nordstrom2004')
            star_list.append(star)
            vprint(f'Added {star.name}.')

    tqdm.write(f'Found {len(star_list)} usable stars in total.')

    if args.compare_offset_patterns:

        offset_patterns_pre = []
        offset_patterns_post = []
        stars_pre = []
        stars_post = []

        for star in star_list:

            if star.hasObsPre:
                pre_slice = slice(None, star.fiberSplitIndex)
                offset_patterns_pre.append(star.getTransitionOffsetPattern(
                    pre_slice))
                stars_pre.append(star.name)
            if star.hasObsPost:
                post_slice = slice(star.fiberSplitIndex, None)
                offset_patterns_post.append((star.getTransitionOffsetPattern(
                    post_slice)))
                stars_post.append(star.name)

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        gs = GridSpec(nrows=2, ncols=2, figure=fig,
                      height_ratios=[1, 1], width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)

        for ax in (ax1, ax2, ax3, ax4):
            ax.axhline(y=0, color='Black')

        ax1.set_xlim(left=-2, right=len(offset_patterns_pre[0][0])+1)

        ax1.set_ylabel('Offset from expected position (m/s)')
        ax3.set_ylabel('Offset from expected position (m/s)')
        ax2.set_ylabel('Standard deviation (m/s)')
        ax4.set_ylabel('Standard deviation (m/s)')

        for pattern, star_name in zip(offset_patterns_pre, stars_pre):
            indices = [x for x in range(len(pattern[0]))]

            ax1.plot(indices, pattern[0], label=star_name, alpha=1,
                     marker='D', markersize=1.5, linestyle='')
            ax2.plot(indices, pattern[1], label=star_name, alpha=1,
                     marker='D', markersize=1.5, linestyle='')

        for pattern, star_name in zip(offset_patterns_post, stars_post):
            indices = [x for x in range(len(pattern[0]))]
            ax3.plot(indices, pattern[0], label=star_name, alpha=1,
                     marker='D', markersize=1.5, linestyle='')
            ax4.plot(indices, pattern[1], label=star_name, alpha=1,
                     marker='D', markersize=1.5, linestyle='')

        for ax in (ax1, ax2, ax3, ax4):
            ax.legend(ncol=3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=100))

            ax.xaxis.grid(which='major', color='Gray', alpha=0.7,
                          linestyle='-')
            ax.xaxis.grid(which='minor', color='Gray', alpha=0.4,
                          linestyle='--')
            ax.yaxis.grid(which='major', color='Gray', alpha=0.4,
                          linestyle='--')
            ax.yaxis.grid(which='minor', color='Gray', alpha=0.4,
                          linestyle='--')

        plt.show()

    if args.compare_stellar_parameters_pairs:
        tqdm.write('Unpickling pairs list...')
        with open(vcl.final_pair_selection_file, 'r+b') as f:
            pairs_list = pickle.load(f)

        plots_folder = main_dir / "star_comparisons/pairs"
        if not plots_folder.exists():
            import os
            os.makedirs(plots_folder)

        tqdm.write('Creating plots for each pair...')
        for pair in tqdm(pairs_list):
            blend1 = pair._higherEnergyTransition.blendedness
            blend2 = pair._lowerEnergyTransition.blendedness
            for order_num in pair.ordersToMeasureIn:
                pair_label = '_'.join([pair.label, str(order_num)])

                offsets_pre, offsets_post = [], []
                errs_pre, errs_post = [], []
                stds_pre, stds_post = [], []

                temp_pre, temp_post = [], []
                mtl_pre, mtl_post = [], []
                mag_pre, mag_post = [], []
                logg_pre, logg_post = [], []

                # Get the reference star properties.
                pre_slice = slice(None, ref_star.fiberSplitIndex)
                post_slice = slice(ref_star.fiberSplitIndex, None)
                ref_mean_pre, ref_err_pre, ref_std_pre =\
                    get_pair_data_point(ref_star, pre_slice, pair_label)
                ref_mean_post, ref_err_post, ref_std_post =\
                    get_pair_data_point(ref_star, post_slice, pair_label)

                # Collect the data points for each star:
                for star in tqdm(star_list):

                    # Ignore the reference star.
                    if star.name == ref_star.name:
                        vprint(f'Skipping over reference star {star.name}.')
                        continue

                    pre_slice = slice(None, star.fiberSplitIndex)
                    post_slice = slice(star.fiberSplitIndex, None)

                    if star.hasObsPre:
                        star_mean_pre, star_err_pre, star_std_pre =\
                            get_pair_data_point(star, pre_slice, pair_label)

                        offset = ref_mean_pre - star_mean_pre

                        offsets_pre.append(offset)
                        errs_pre.append(star_err_pre)
                        stds_pre.append(star_std_pre)
                        temp_pre.append(star.temperature)
                        mtl_pre.append(star.metallicity)
                        mag_pre.append(star.absoluteMagnitude)
                        logg_pre.append(star.logG)

                    if star.hasObsPost:
                        star_mean_post, star_err_post, star_std_post =\
                            get_pair_data_point(star, post_slice, pair_label)

                        offset = ref_mean_post - star_mean_post

                        offsets_post.append(offset)
                        errs_post.append(star_err_post)
                        stds_post.append(star_std_post)
                        temp_post.append(star.temperature)
                        mtl_post.append(star.metallicity)
                        mag_post.append(star.absoluteMagnitude)
                        logg_post.append(star.logG)

                # Create the figure and subplots:
                comp_fig, axes_dict = create_parameter_comparison_figures(
                        ylims=(-300 * u.m / u.s, 300 * u.m / u.s),
                        temp_lims=(5400 * u.K, 6300 * u.K),
                        mtl_lims=(-0.63, 0.52))

                for ax in (axes_dict.values()):
                    ax.annotate(f'Blendedness: ({blend1}, {blend2})',
                                (0.01, 0.95),
                                xycoords='axes fraction')

                for ax, attr in zip(('temp_pre', 'mtl_pre',
                                     'mag_pre', 'logg_pre'),
                                    (np.array(temp_pre)+97,
                                     np.array(mtl_pre)+0.12,
                                     mag_pre, logg_pre)):
                    plot_data_points(axes_dict[ax], attr,
                                     offsets_pre, errs_pre,
                                     stds_pre, era='pre')

                for ax, attr in zip(('temp_post', 'mtl_post',
                                     'mag_post', 'logg_post'),
                                    (np.array(temp_post)+97,
                                     np.array(mtl_post)+0.12,
                                     mag_post, logg_post)):
                    plot_data_points(axes_dict[ax], attr,
                                     offsets_post, errs_post,
                                     stds_post, era='post')

                # Plot the reference star points last so they're on top.
                plot_data_points(axes_dict['temp_pre'], ref_star.temperature,
                                 0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_points(axes_dict['temp_post'], ref_star.temperature,
                                 0, ref_err_post, ref_std_post, ref=True)
                plot_data_points(axes_dict['mtl_pre'], ref_star.metallicity,
                                 0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_points(axes_dict['mtl_post'], ref_star.metallicity,
                                 0, ref_err_post, ref_std_post, ref=True)
                plot_data_points(axes_dict['mag_pre'],
                                 ref_star.absoluteMagnitude,
                                 0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_points(axes_dict['mag_post'],
                                 ref_star.absoluteMagnitude,
                                 0, ref_err_post, ref_std_post, ref=True)
                plot_data_points(axes_dict['logg_pre'], ref_star.logG,
                                 0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_points(axes_dict['logg_post'], ref_star.logG,
                                 0, ref_err_post, ref_std_post, ref=True)

                file_name = plots_folder / f'{pair_label}.png'
                vprint(f'Saving file {pair_label}.png')

                comp_fig.savefig(str(file_name))
                plt.close('all')

    if args.compare_stellar_parameters_transitions:
        tqdm.write('Unpickling transitions list..')
        with open(vcl.final_selection_file, 'r+b') as f:
            transitions_list = pickle.load(f)

        plots_folder = main_dir / "star_comparisons/transitions"
        if apply_corrections:
            model_name = '_'.join(model_func.__name__.split('_')[:-1])
        else:
            model_name = 'uncorrected'
        plots_folder /= model_name
        if not plots_folder.exists():
            import os
            os.makedirs(plots_folder)

        index_nums = []
        index_num = 0
        sigma_list_pre, sigma_list_post = [], []
        sigma_sys_pre, sigma_sys_post = [], []

        tqdm.write('Creating plots for each transition...')
        for transition in tqdm(transitions_list):
            for order_num in transition.ordersToFitIn:
                transition_label = '_'.join([transition.label, str(order_num)])
                vprint(f'Analysing {transition_label}...')

                index_nums.append(index_num)
                index_num += 1

                means_pre, means_post = [], []
                errs_pre, errs_post = [], []
                stds_pre, stds_post = [], []

                temp_pre, temp_post = [], []
                mtl_pre, mtl_post = [], []
                mag_pre, mag_post = [], []
                logg_pre, logg_post = [], []

                for star in tqdm(star_list):
                    pre_slice = slice(None, star.fiberSplitIndex)
                    post_slice = slice(star.fiberSplitIndex, None)

                    if star.hasObsPre:
                        star_mean_pre, star_err_pre, star_std_pre =\
                            get_transition_data_point(star, pre_slice,
                                                      transition_label,
                                                      fit_params=fit_results)
                        means_pre.append(star_mean_pre)
                        errs_pre.append(star_err_pre)
                        stds_pre.append(star_std_pre)
                        temp_pre.append(star.temperature)
                        mtl_pre.append(star.metallicity)
                        mag_pre.append(star.absoluteMagnitude)
                        logg_pre.append(star.logG)

                    if star.hasObsPost:
                        star_mean_post, star_err_post, star_std_post =\
                            get_transition_data_point(star, post_slice,
                                                      transition_label,
                                                      fit_params=fit_results)
                        means_post.append(star_mean_post)
                        errs_post.append(star_err_post)
                        stds_post.append(star_std_post)
                        temp_post.append(star.temperature)
                        mtl_post.append(star.metallicity)
                        mag_post.append(star.absoluteMagnitude)
                        logg_post.append(star.logG)

                # Correct for trends in stellar parameters here.
                if apply_corrections:
                    # vprint(f'Applying corrections from {model_name} model')
                    # data_pre = np.stack((temp_pre, mtl_pre, mag_pre),
                    #                     axis=0)
                    # params_pre = coeffs[transition_label + '_pre']
                    # corrections = u.unyt_array(model_func(data_pre,
                    #                                       *params_pre),
                    #                            units=u.m/u.s)
                    # for mean in means_pre:
                    #     print(mean)
                    #     print(mean.units)
                    # means_pre = ma.masked_invalid(means_pre)
                    # means_pre -= corrections
                    sigma_sys_pre.append(sigma_sys[transition_label +
                                                    '_pre'].value)

                    # data_post = np.stack((temp_post, mtl_post, mag_post),
                    #                      axis=0)
                    # params_post = coeffs[transition_label + '_post']
                    # corrections = u.unyt_array(model_func(data_post,
                    #                                       *params_post),
                    #                            units=u.m/u.s)
                    # means_post = ma.masked_invalid(means_post)
                    # means_post -= corrections
                    sigma_sys_post.append(sigma_sys[transition_label +
                                                    '_post'].value)
                else:
                    sigma_sys_pre.append(0)
                    sigma_sys_post.append(0)

                # means_pre = u.unyt_array(means_pre, units=u.m/u.s)
                # means_post = u.unyt_array(means_post, units=u.m/u.s)

                # print('Units are:')
                # print(type(means_pre))
                # print(type(means_post))

                sigma_pre = np.nanstd(means_pre)
                sigma_post = np.nanstd(means_post)
                sigma_list_pre.append(sigma_pre)
                sigma_list_post.append(sigma_post)

                # print(sigma_pre.units)
                # print(sigma_post.units)

                # Write out data into a CSV file for checking.
                csv_file = plots_folder /\
                    f'Data_{transition_label}_{model_name}.csv'
                with open(csv_file, 'w', newline='') as f:
                    datawriter = csv.writer(f)
                    header = ('weighted_means_pre', 'EoWM_pre', 'EoM_pre',
                              'weighted_means_post', 'EoWM_post', 'EoM_post')
                    datawriter.writerow(header)
                    for row in zip_longest(means_pre, errs_pre, stds_pre,
                                           means_post, errs_post, stds_post):
                        datawriter.writerow(row)

                # Create the figure and subplots:
                if not apply_corrections:
                    total_means = np.concatenate((means_pre, means_post))
                    median = np.nanmedian(total_means)
                    y_limits = (median - 300, median + 300)
                else:
                    y_limits = (-300, 300)

                comp_fig, axes_dict = create_parameter_comparison_figures(
                        ylims=None if args.full_range else y_limits,
                        temp_lims=(5400 * u.K, 6300 * u.K),
                        mtl_lims=(-0.63, 0.52))

                for ax, attr in zip(('temp_pre', 'mtl_pre',
                                     'mag_pre', 'logg_pre'),
                                    (temp_pre, mtl_pre,
                                     mag_pre, logg_pre)):
                    plot_data_points(axes_dict[ax], attr,
                                     means_pre, errs_pre,
                                     stds_pre, era='pre')
                    axes_dict[ax].annotate(
                            f'Blendedness: {transition.blendedness}'
                            '\n'
                            fr'$\sigma$: {sigma_pre:.2f}',
                            (0.01, 0.99),
                            xycoords='axes fraction',
                            horizontalalignment='left',
                            verticalalignment='top')
                data = np.array(ma.masked_invalid(means_pre).compressed())
                axes_dict['hist_pre'].hist(data,
                                           bins='fd',
                                           color='Black',
                                           histtype='step',
                                           orientation='horizontal')

                for ax, attr in zip(('temp_post', 'mtl_post',
                                     'mag_post', 'logg_post'),
                                    (temp_post, mtl_post,
                                     mag_post, logg_post)):
                    plot_data_points(axes_dict[ax], attr,
                                     means_post, errs_post,
                                     stds_post, era='post')
                    axes_dict[ax].annotate(
                            f'Blendedness: {transition.blendedness}'
                            '\n'
                            fr'$\sigma$: {sigma_post:.2f}',
                            (0.01, 0.99),
                            xycoords='axes fraction',
                            horizontalalignment='left',
                            verticalalignment='top')
                data = np.array(ma.masked_invalid(means_post).compressed())
                axes_dict['hist_post'].hist(data,
                                            bins='fd',
                                            color='Black',
                                            histtype='step',
                                            orientation='horizontal')

                file_name = plots_folder /\
                    f'{transition_label}_{model_name}.png'
                vprint(f'Saving file {transition_label}.png')

                comp_fig.savefig(str(file_name))
                plt.close('all')

        csv_file = plots_folder / f'{model_name}_sigmas.csv'
        with open(csv_file, 'w', newline='') as f:
            datawriter = csv.writer(f)
            header = ('#index', 'sigma_pre', 'sigma_sys_pre',
                      'sigma_post', 'sigma_sys_post')
            datawriter.writerow(header)
            for row in zip(index_nums, sigma_list_pre, sigma_sys_pre,
                           sigma_list_post, sigma_sys_post):
                datawriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a plot of the'
                                     ' transition offset pattern for multiple'
                                     ' stars.')
    parser.add_argument('main_dir', action='store', type=str, nargs=1,
                        help='The main directory within which to find'
                        ' additional star directories.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')
    parser.add_argument('--reference-star', action='store', type=str,
                        metavar='star_name',
                        help='The star to be used as a reference when using'
                        ' the --compare-stellar-parameters-pairs flag'
                        ' (unnecessary otherwise).')

    parser.add_argument('--compare-offset-patterns', action='store_true',
                        help='Create a plot of all the transition offset'
                        ' patterns for the given stars.')

    parser.add_argument('--compare-stellar-parameters-pairs',
                        action='store_true',
                        help='Create plots for each pair of transitions'
                        ' with stars sorted by parameters such as temperature'
                        ' or metallicity.')
    parser.add_argument('--compare-stellar-parameters-transitions',
                        action='store_true',
                        help='Create plots for each transition with stars'
                        ' sorted by parameters such as temperature or'
                        ' metallicity.')
    parser.add_argument('--correct-transitions', action='store',
                        type=str, dest='fit_params_file',
                        help='The name of the file containing the fitting'
                        ' function and parameters for each transition. It will'
                        ' automatically be looked for in the fit_params folder'
                        ' in the output data directory.')
    parser.add_argument('--full-range', action='store_true',
                        help='Plot the full vertical range of transition'
                        ' offsets instead of a fixed range.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")

    paper = parser.add_mutually_exclusive_group()
    paper.add_argument('--casagrande2011', action='store_true',
                       help='Use values from Casagrande et al. 2011.')
    paper.add_argument('--nordstrom2004', action='store_true',
                       help='Use values from Nordstrom et al. 2004.')

    args = parser.parse_args()

    main()
