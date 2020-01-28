#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:07:04 2019

@author: dberke
A script to compare multiple stars in various ways, such as plotting the
transition offset pattern for multiple stars.

"""

import argparse
from pathlib import Path
import pickle

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import HDF5FileNotFoundError
from varconlib.star import Star


def get_star(star_name):
    """Return a varconlib.star.Star object based on its name.

    Parameters
    ----------
    star_name : str
        A string representing the name of the directory within the main given
        directory where a star's observations can be found.

    Returns
    -------
    `star.Star`
        A Star object from the directory. Note that this will only use already-
        existing stars, it will not create ones which do not already exist from
        their observations.

    """
    star_path = main_dir / star_name
    assert star_path.exists(), FileNotFoundError('Star directory'
                                                 f' {star_path}'
                                                 ' not found.')
    try:
        return Star(star_path.stem, star_path, load_data=True)
    except IndexError:
        vprint(f'Excluded {star_path.stem}.')
        pass
    except HDF5FileNotFoundError:
        vprint(f'No HDF5 file for {star_path.stem}.')
        pass


def create_parameter_comparison_figures():
    """Creates and returns a figure with pre-set subplots.

    This function creates the background figure and subplots for use with the
    --compare-stellar-parameter-* flags.

    Returns
    -------
    tuple
        A tuple containing the figure itself and the various axes of the
        subplots within it.

    """

    comp_fig = plt.figure(figsize=(14, 10), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=2, figure=comp_fig)

    temp_ax1 = comp_fig.add_subplot(gs[0, 0])
    temp_ax2 = comp_fig.add_subplot(gs[1, 0],
                                    sharex=temp_ax1,
                                    sharey=temp_ax1)
    mtl_ax1 = comp_fig.add_subplot(gs[0, 1],
                                   sharey=temp_ax1)
    mtl_ax2 = comp_fig.add_subplot(gs[1, 1],
                                   sharex=mtl_ax1,
                                   sharey=mtl_ax1)
    # Set the plot limits here. The y-limits for temp_ax1 are
    # used for all subplots.
#    temp_ax1.set_ylim(bottom=-300 * u.m / u.s,
#                      top=300 * u.m / u.s)
    temp_ax1.set_xlim(left=5300 * u.K,
                      right=6200 * u.K)
    mtl_ax1.set_xlim(left=-0.75,
                     right=0.4)

    # Axis styles for all subplots.
    for ax in (temp_ax1, temp_ax2, mtl_ax1, mtl_ax2):
#        ax.yaxis.set_major_locator(ticker.MultipleLocator(
#                                   base=50))
#        ax.yaxis.set_minor_locator(ticker.MultipleLocator(
#                                   base=25))
        ax.axhline(y=0, color='Black', linestyle='--')
        ax.yaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.85)
        ax.xaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.85)
        ax.yaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.75)

    for ax in (temp_ax1, temp_ax2):
        ax.set_xlabel('Temperature (K)')
    for ax in (mtl_ax1, mtl_ax2):
        ax.set_xlabel('Metallicity [Fe/H]')

    for ax in (temp_ax1, mtl_ax1):
        ax.set_ylabel('Pre-fiber change offset (m/s)')
    for ax in (temp_ax2, mtl_ax2):
        ax.set_ylabel('Post-fiber change offset (m/s)')

    return (comp_fig, temp_ax1, temp_ax2, mtl_ax1, mtl_ax2)


def plot_data_point(axis, xpos, mean, err, std, era=None,
                    ref=False):
    """Plot a data point for a star.

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        An axes to plot the data on.
    xpos : float
        The x-position of the point to plot.
    mean : `unyt.unyt_quantity`
        The offset of the weighted mean for a star from the value of
        the weighted mean for the same transition pair from the
        reference star. Units of velocity, m/s by default.
    err : `unyt.unyt_quantity`
        The error on the weighted mean, in units of velocity, m/s by
        default.
    std : `unyt.unyt_quantity`
        The standard deviation of the distribution of pair separation
        values for this star.
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

    axis.errorbar(x=xpos, y=mean,
                  yerr=std,
                  marker='', capsize=style_caps['capsize_thin'],
                  color=params['color'],
                  ecolor=params['ecolor_thin'],
                  elinewidth=style_caps['linewidth_thin'],
                  capthick=style_caps['cap_thin'])
    axis.errorbar(x=xpos, y=mean,
                  yerr=err,
                  marker='o', capsize=style_caps['capsize_thick'],
                  color=params['color'],
                  ecolor=params['ecolor_thick'],
                  elinewidth=style_caps['linewidth_thick'],
                  capthick=style_caps['cap_thick'])


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
                        ' the --compare-stellar-parameters flag (unnecessary'
                        ' otherwise).')

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
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")

    args = parser.parse_args()

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    main_dir = Path(args.main_dir[0])
    if not main_dir.exists():
        raise FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    if args.reference_star:
        ref_star = get_star(args.reference_star)
        tqdm.write(f'Reference star is {ref_star.name}.')

    star_list = []
    tqdm.write('Collecting stars...')
    for star_dir in tqdm(args.star_names):
        if args.reference_star:
            star = get_star(star_dir)
            if star is None:
                pass
            elif star.name != ref_star.name:
                vprint(f'Added star {star.name}.')
                star_list.append(star)
            else:
                vprint(f'Found reference star! {star.name}')
        else:
            star_list.append(get_star(star_dir))
    tqdm.write(f'Found {len(star_list)} usable stars in total.')

    # Define style parameters to use for stellar parameter plots.
    style_pre = {'color': 'DodgerBlue',
                 'ecolor_thick': 'CornFlowerBlue',
                 'ecolor_thin': 'LightSkyBlue'}
    style_post = {'color': 'Chocolate',
                  'ecolor_thick': 'DarkOrange',
                  'ecolor_thin': 'BurlyWood'}
    style_ref = {'color': 'DarkGreen',
                 'ecolor_thick': 'ForestGreen',
                 'ecolor_thin': 'DarkSeaGreen'}
    style_caps = {'capsize_thin': 4,
                  'capsize_thick': 7,
                  'linewidth_thin': 2,
                  'linewidth_thick': 3.4,
                  'cap_thin': 1.5,
                  'cap_thick': 2.5}

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

            means = star.pairSeparationsArray[time_slice, col_index]
            stds = star.pairSepErrorsArray[time_slice, col_index]
            weighted_mean = np.average(means,
                                       weights=1/stds**2).to(u.m / u.s)
            weighted_error = np.std(means) / np.sqrt(
                star.getNumObs(time_slice))

            return (weighted_mean, weighted_error, np.std(means))

        # -----------------
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

                # Create the figure and subplots:
                comp_fig, temp_ax1, temp_ax2, mtl_ax1, mtl_ax2 =\
                    create_parameter_comparison_figures()

                # Get the reference star properties.
                pre_slice = slice(None, ref_star.fiberSplitIndex)
                post_slice = slice(ref_star.fiberSplitIndex, None)
                ref_mean_pre, ref_err_pre, ref_std_pre =\
                    get_pair_data_point(ref_star, pre_slice, pair_label)
                ref_mean_post, ref_err_post, ref_std_post =\
                    get_pair_data_point(ref_star, post_slice, pair_label)

                for ax in (temp_ax1, temp_ax2, mtl_ax1, mtl_ax2):
                    ax.annotate(f'Blendedness: ({blend1}, {blend2})',
                                (0.01, 0.95),
                                xycoords='axes fraction')

                # Plot the data points for each star:
                for star in tqdm(star_list):
                    pre_slice = slice(None, star.fiberSplitIndex)
                    post_slice = slice(star.fiberSplitIndex, None)

                    if star.hasObsPre:
                        star_mean_pre, star_err_pre, star_std_pre =\
                            get_pair_data_point(star, pre_slice, pair_label)

                        offset = ref_mean_pre - star_mean_pre

                        for ax, attr in zip((temp_ax1, mtl_ax1),
                                            ('temperature', 'metallicity')):
                            plot_data_point(ax, getattr(star, attr),
                                            offset, star_err_pre,
                                            star_std_pre, era='pre')

                    if star.hasObsPost:
                        star_mean_post, star_err_post, star_std_post =\
                            get_pair_data_point(star, post_slice, pair_label)

                        offset = ref_mean_post - star_mean_post

                        for ax, attr in zip((temp_ax2, mtl_ax2),
                                            ('temperature', 'metallicity')):
                            plot_data_point(ax, getattr(star, attr),
                                            offset, star_err_post,
                                            star_std_post, era='post')

                # Plot the reference star points last so they're on top.
                plot_data_point(temp_ax1, ref_star.temperature,
                                0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_point(temp_ax2, ref_star.temperature,
                                0, ref_err_post, ref_std_post, ref=True)
                plot_data_point(mtl_ax1, ref_star.metallicity,
                                0, ref_err_pre, ref_std_pre, ref=True)
                plot_data_point(mtl_ax2, ref_star.metallicity,
                                0, ref_err_post, ref_std_post, ref=True)

                file_name = plots_folder / f'{pair_label}.png'
                vprint(f'Saving file {pair_label}.png')

                comp_fig.savefig(str(file_name))
                plt.close('all')

    if args.compare_stellar_parameters_transitions:

        def get_transition_data_point(star, time_slice, transition_label):
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
            transition_label : str
                The label to use to select a particular transition.

            Returns
            -------
            tuple
                Returns a 3-tuple of the weighted mean, the error on the
                weighted mean, and the standard deviation.

            """

            col_index = star.t_index(transition_label)

            offsets = star.fitOffsetsArray[time_slice, col_index]
            stds = star.fitErrorsArray[time_slice, col_index]
            weighted_mean = np.average(offsets,
                                       weights=1/stds**2).to(u.m/u.s)
            weighted_error = np.std(offsets) / np.sqrt(
                star.getNumObs(time_slice))

            return (weighted_mean, weighted_error, np.std(offsets))

        # ------------------
        tqdm.write('Unpickling transitions list..')
        with open(vcl.final_selection_file, 'r+b') as f:
            transitions_list = pickle.load(f)

        plots_folder = main_dir / "star_comparisons/transitions"
        if not plots_folder.exists():
            import os
            os.makedirs(plots_folder)

        tqdm.write('Creating plots for each transition...')
        for transition in tqdm(transitions_list):
            for order_num in transition.ordersToFitIn:
                transition_label = '_'.join([transition.label, str(order_num)])

                # Create the figure and subplots:
                comp_fig, temp_ax1, temp_ax2, mtl_ax1, mtl_ax2 =\
                    create_parameter_comparison_figures()

                for ax in (temp_ax1, temp_ax2, mtl_ax1, mtl_ax2):
                    ax.annotate(f'Blendedness: ({transition.blendedness})',
                                (0.01, 0.95),
                                xycoords='axes fraction')

                for star in tqdm(star_list):
                    pre_slice = slice(None, star.fiberSplitIndex)
                    post_slice = slice(star.fiberSplitIndex, None)

                    if star.hasObsPre:
                        star_mean_pre, star_err_pre, star_std_pre =\
                            get_transition_data_point(star, pre_slice,
                                                      transition_label)

                        for ax, attr in zip((temp_ax1, mtl_ax1),
                                            ('temperature', 'metallicity')):
                            plot_data_point(ax, getattr(star, attr),
                                            star_mean_pre, star_err_pre,
                                            star_std_pre, era='pre')

                    if star.hasObsPost:
                        star_mean_post, star_err_post, star_std_post =\
                            get_transition_data_point(star, post_slice,
                                                      transition_label)

                        for ax, attr in zip((temp_ax2, mtl_ax2),
                                            ('temperature', 'metallicity')):
                            plot_data_point(ax, getattr(star, attr),
                                            star_mean_post, star_err_post,
                                            star_std_post, era='post')

                file_name = plots_folder / f'{transition_label}.png'
                vprint(f'Saving file {transition_label}.png')

                comp_fig.savefig(str(file_name))
                plt.close('all')
