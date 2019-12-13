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
from varconlib.star import Star

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a plot of the'
                                     'transition offset pattern for multiple'
                                     'stars.')
    parser.add_argument('main_dir', action='store', type=str, nargs=1,
                        help='The main directory within which to find'
                        ' additional star directories.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')

    parser.add_argument('--compare-offset-patterns', action='store_true',
                        help='Create a plot of all the transition offset'
                        'patterns for the given stars.')

    parser.add_argument('--compare-stellar-parameters', action='store_true',
                        help='Create plots for each pair of transitions'
                        'with stars sorted by parameters such as temperature'
                        'or metallicity.')

    args = parser.parse_args()

    main_dir = Path(args.main_dir[0])
    assert main_dir.exists(), FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    star_list = []
    for star_dir in tqdm(args.star_names):
        star_path = main_dir / star_dir
        assert star_path.exists(), FileNotFoundError('Star directory'
                                                     f' {star_path}'
                                                     ' not found.')
        try:
            tqdm.write(f'Getting star {star_path.stem}')
            star_list.append(Star(star_path.stem, star_path))
        except IndexError:
            tqdm.write(f'Excluded {star_path.stem}')

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

    if args.compare_stellar_parameters:

        def get_data_point(star, time_slice, pair_label):
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
            pair_label : int
                The column index to use to select a particular pair.

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

        def plot_data_point(axis, star, attr, mean, err, std,
                            color):
            """Plot a data point for a star.

            Parameters
            ----------
            axis : `matplotlib.axes.Axes`
                An axes to plot the data on.
            star : `star.Star`
                The star to get the temperature or metallicity from.
            attr : str, ["temperature", "metallicity"]
                The attribute to get the value of. Currently can be either
                "temperature" or "metallicity".
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
            color : str
                A `matplotlib`-recognized color string to color the point.

            Returns
            -------
            None.

            """

            axis.errorbar(x=getattr(star, attr), y=mean,
                          yerr=std,
                          marker='', capsize=4, color=color,
                          ecolor=color, elinewidth=2,
                          capthick=1.5)
            axis.errorbar(x=getattr(star, attr), y=mean,
                          yerr=err,
                          marker='o', capsize=7, color=color,
                          ecolor=color, elinewidth=4,
                          capthick=2.5)

        plots_folder = main_dir / "star_comparisons(pairs)"
        if not plots_folder.exists():
            import os
            os.mkdir(plots_folder)

        tqdm.write('Unpickling pairs list...')
        with open(vcl.final_pair_selection_file, 'r+b') as f:
            pairs_list = pickle.load(f)

        point_color = 'DodgerBlue'

        star_labels, metal_values, temp_values = [], [], []

        for star in tqdm(star_list):
            star_labels.append(star.name)
            metal_values.append(star.metallicity)
            temp_values.append(star.temperature)

        tqdm.write('Creating plots for each pair...')
        for pair in tqdm(pairs_list):
            for order_num in pair.ordersToMeasureIn:
                pair_label = '_'.join([pair.label, str(order_num)])

                # Create the figure and set it up here.
                temp_fig = plt.figure(figsize=(11, 10), tight_layout=True)
                metal_fig = plt.figure(figsize=(11, 10), tight_layout=True)

                temp_ax1 = temp_fig.add_subplot(2, 1, 1)
                temp_ax2 = temp_fig.add_subplot(2, 1, 2)

                mtl_ax1 = metal_fig.add_subplot(2, 1, 1)
                mtl_ax2 = metal_fig.add_subplot(2, 1, 2)

                # Axis styles for all subplots.
                for ax in (temp_ax1, temp_ax2, mtl_ax1, mtl_ax2):
                    ax.axhline(y=0, color='Black', linestyle='--')
                    ax.yaxis.grid(which='major', color='Gray',
                                  linestyle=':', alpha=0.8)
                    ax.xaxis.grid(which='major', color='Gray',
                                  linestyle=':', alpha=0.8)

                for ax in (temp_ax1, temp_ax2):
                    ax.set_xlabel('Temperature (K)')
                for ax in (mtl_ax1, mtl_ax2):
                    ax.set_xlabel('Metallicity [Fe/H]')

                for ax in (temp_ax1, mtl_ax1):
                    ax.set_ylabel('Pre-fiber change offset (m/s)')
                for ax in (temp_ax2, mtl_ax2):
                    ax.set_ylabel('Post-fiber change offset (m/s)')

                ref_star = star_list[0]
                pre_slice = slice(None, ref_star.fiberSplitIndex)
                post_slice = slice(ref_star.fiberSplitIndex, None)
                ref_mean_pre, ref_err_pre, ref_std_pre = get_data_point(
                    ref_star, pre_slice, pair_label)
                ref_mean_post, ref_err_post, ref_std_post = get_data_point(
                    ref_star, post_slice, pair_label)

                plot_data_point(temp_ax1, ref_star, "temperature",
                                0, ref_err_pre, ref_std_pre, color='Green')
                plot_data_point(temp_ax2, ref_star, "temperature",
                                0, ref_err_post, ref_std_post, color='Green')
                plot_data_point(mtl_ax1, ref_star, "metallicity",
                                0, ref_err_pre, ref_std_pre, color='Green')
                plot_data_point(mtl_ax2, ref_star, "metallicity",
                                0, ref_err_post, ref_std_post, color='Green')

                for star in tqdm(star_list[1:]):

                    pre_slice = slice(None, star.fiberSplitIndex)
                    post_slice = slice(star.fiberSplitIndex, None)

                    if star.hasObsPre:
                        star_mean_pre, star_err_pre, star_std_pre =\
                            get_data_point(star, pre_slice, pair_label)

                        offset = ref_mean_pre - star_mean_pre

                        plot_data_point(temp_ax1, star, "temperature",
                                        offset, star_err_pre, star_std_pre,
                                        point_color)
                        plot_data_point(mtl_ax1, star, "metallicity",
                                        offset, star_err_pre, star_std_pre,
                                        point_color)

                    if star.hasObsPost:
                        star_mean_post, star_err_post, star_std_post =\
                            get_data_point(star, post_slice, pair_label)

                        offset = ref_mean_post - star_mean_post

                        plot_data_point(temp_ax2, star, "temperature",
                                        offset, star_err_post, star_std_post,
                                        point_color)
                        plot_data_point(mtl_ax2, star, "metallicity",
                                        offset, star_err_pre, star_std_post,
                                        point_color)

#            temp_ax1.set_xticks(temp_values)
#            temp_ax1.set_xticklabels(star_labels, horizontalalignment='right',
#                                     rotation='vertical')
#            temp_ax2.set_xticks(temp_values)
#            temp_ax2.set_xticklabels(star_labels, horizontalalignment='right',
#                                     rotation='vertical')
#            mtl_ax1.set_xticks(metal_values)
#            mtl_ax1.set_xticklabels(star_labels, horizontalalignment='right',
#                                    rotation='vertical')
#            mtl_ax2.set_xticks(metal_values)
#            mtl_ax2.set_xticklabels(star_labels, horizontalalignment='right',
#                                    rotation='vertical')
            temperature_file = plots_folder / f"Temperature_{pair_label}.png"
            metallicity_file = plots_folder / f"Metallicity_{pair_label}.png"

            temp_fig.savefig(str(temperature_file))
            metal_fig.savefig(str(metallicity_file))
            plt.close('all')
