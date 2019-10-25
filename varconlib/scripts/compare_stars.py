#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:22:33 2019

@author: dberke

A script for comparing measured results between two stars and calculating
an alpha shift difference between them.

"""

import argparse
import datetime as dt
from pathlib import Path
import pickle

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

import varconlib as vcl
from varconlib.star import Star


style_params_pre = {'marker': 'o', 'color': 'Chocolate',
                    'markeredgecolor': 'Black', 'ecolor': 'DarkGoldenRod',
                    'linestyle': '', 'alpha': 0.6, 'markersize': 4}

style_params_post = {'marker': 'o', 'color': 'CornFlowerBlue',
                     'markeredgecolor': 'Black', 'ecolor': 'DodgerBlue',
                     'linestyle': '', 'alpha': 0.6, 'markersize': 4}

plt.rc('text', usetex=True)


def create_pair_comparison_plot(main_ax, chi_ax, star1, star2,
                                time_period, params):
    """Create an errorbar plot of given data on a given axis.

    Parameters
    ----------
    main_ax : `matplotlib.Axes` object
        The axis to plot the pair average offsets on.
    chi_ax :  `matplotlib.Axes` object
        The smaller axis to plot the chi-squared values of each pair on
    time_period : str
        Possible values are 'pre' and 'post' for the time periods before and
        after the HARPS fiber change in 2015.
    star1, star2 : `varconlib.star.Star` objects
        Two instances of the Star class to be compared.
    params : dict
        A dictionary of parameter values to pass to `matplotlib.errorbar`.

    """

    differences = np.empty(shape=star1.pairSeparationsArray.shape[1])
    errors = np.empty(shape=star1.pairSepErrorsArray.shape[1])
    reduced_chi_squared1 = np.empty(shape=star1.pairSepErrorsArray.shape[1])
    reduced_chi_squared2 = np.empty(shape=star2.pairSepErrorsArray.shape[1])

    for pair in star1._pair_label_dict.keys():
        if time_period == 'pre':
            separations1 = star1.pairSeparationsArray[:star1.fiberSplitIndex,
                                                      star1._p_label(pair)]
            separations2 = star2.pairSeparationsArray[:star2.fiberSplitIndex,
                                                      star2._p_label(pair)]
            errors1 = star1.pairSepErrorsArray[:star1.fiberSplitIndex,
                                               star1._p_label(pair)]
            errors2 = star2.pairSepErrorsArray[:star2.fiberSplitIndex,
                                               star2._p_label(pair)]
        elif time_period == 'post':
            separations1 = star1.pairSeparationsArray[star1.fiberSplitIndex:,
                                                      star1._p_label(pair)]
            separations2 = star2.pairSeparationsArray[star2.fiberSplitIndex:,
                                                      star2._p_label(pair)]
            errors1 = star1.pairSepErrorsArray[star1.fiberSplitIndex:,
                                               star1._p_label(pair)]
            errors2 = star2.pairSepErrorsArray[star2.fiberSplitIndex:,
                                               star2._p_label(pair)]
        else:
            raise ValueError('time_period must be "pre" or "post".')

        weights1 = 1 / errors1 ** 2
        weights2 = 1 / errors2 ** 2

        w_mean1 = np.average(separations1, weights=weights1)
        w_mean2 = np.average(separations2, weights=weights2)

        w_mean_err1 = 1 / np.sqrt(np.sum(weights1))
        w_mean_err2 = 1 / np.sqrt(np.sum(weights2))

        differences[star1._p_label(pair)] = w_mean1 - w_mean2
        errors[star1._p_label(pair)] = np.sqrt(w_mean_err1 ** 2 +
                                               w_mean_err2 ** 2)
        chi_squared1 = np.sum(((separations1 - w_mean1) / errors1) ** 2)
        chi_squared2 = np.sum(((separations2 - w_mean2) / errors2) ** 2)

        reduced_chi_squared1[star1._p_label(pair)] =\
            chi_squared1 / (len(separations1) - 1)
        reduced_chi_squared2[star1._p_label(pair)] =\
            chi_squared2 / (len(separations2) - 1)

    sigma = np.std(differences)

    main_ax.axhline(y=0, color='Black', alpha=0.9)
    for i, color in zip((1, 2, 3), ('DimGray', 'DarkGray', 'LightGray')):
        for j in (-1, 1):
            main_ax.axhline(y=i * j * sigma, color=color, linestyle='--')

    weighted_mean = np.average(differences, weights=(1/errors**2))
    main_ax.axhline(y=weighted_mean, color=params['color'],
                    label=f'Weighted mean: {weighted_mean:.2f} m/s')

#    main_ax.annotate(f'{star1.name}: {len(separations1)} observations\n'
#                     f'{star2.name}: {len(separations2)} observations',
#                     xy=(0.01, 0.92), xycoords='axes fraction', va='center')

    # Plot the pair differences data.
    indices = [x for x in range(len(star1._pair_label_dict.keys()))]
    main_ax.errorbar(x=indices, y=differences, yerr=errors,
                     label=f'{time_period.capitalize()}-fiber change',
                     **params)

    main_ax.legend(loc='upper right')

    # Plot the reduced chi-squared values on the lower axis.
    chi_ax.axhline(y=1, color='Black')

    chi_ax.plot(indices, reduced_chi_squared1, color='DarkOliveGreen',
                label=f'{star1.name}: {len(separations1)} observations',
                alpha=0.9)
    chi_ax.plot(indices, reduced_chi_squared2, color='Orange',
                label=f'{star2.name}: {len(separations2)} observations',
                alpha=0.8)

    chi_ax.legend(loc='upper right', framealpha=0.4)


if __name__ == '__main__':
    # Where the analysis results live:
    output_dir = Path(vcl.config['PATHS']['output_dir'])

    desc = """A script to compare fitting results between pairs of transitions
              across pairs of stars."""

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('object_dir1', action='store', type=str,
                        help='First object directory to search in.')

    parser.add_argument('object_dir2', action='store', type=str,
                        help='Second object directory to search in.')

    parser.add_argument('suffix', action='store', type=str, default='int',
                        help='Suffix to add to directory names to search for.'
                        ' Defaults to "int".')

    args = parser.parse_args()

    # Date of fiber change in HARPS:
    fiber_change_date = dt.datetime(year=2015, month=6, day=1,
                                    hour=0, minute=0, second=0)

    # Find the data in the given directories.
    data_dir1 = Path(args.object_dir1)
    data_dir2 = Path(args.object_dir2)

    if not data_dir1.exists():
        raise(RuntimeError(f'The directory {data_dir1} could not be found.'))
    elif not data_dir2.exists():
        raise RuntimeError(f'The directory {data_dir2} could not be founnd.')

    # Read the list of chosen transitions.
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    tqdm.write(f'Found {len(transitions_list)} individual transitions.')

    # Read the list of chosen pairs.
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    tqdm.write(f'Found {len(pairs_list)} transition pairs (total) in list.')

    # Initialize Star objects from the given directories.
    tqdm.write('Reading in first star...')
    star1 = Star(name=data_dir1.name, star_dir=data_dir1, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list)
    tqdm.write('Reading in second star...')
    star2 = Star(name=data_dir2.name, star_dir=data_dir2, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list)

    # Create a plot comparing the two stars.
    tqdm.write('Creating plot...')
    if (star1.fiberSplitIndex not in (0, None))\
       and (star2.fiberSplitIndex not in (0, None)):
        fig = plt.figure(figsize=(12, 9), dpi=100, tight_layout=True)
        gs = GridSpec(nrows=5, ncols=1, figure=fig,
                      height_ratios=[9, 3, 1, 9, 3], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[3], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[4], sharex=ax1, sharey=ax2)
        ax1.set_ylabel('Pre-fiber change\npair separations (m/s)')
        ax3.set_ylabel('Post-fiber change\n pair separations (m/s)')
        ax4.set_xlabel('Pair index number (increasing wavelength)')

        ax2.set_ylabel('$\\chi^2_\\nu$')
        ax4.set_ylabel('$\\chi^2_\\nu$')
        ax1.set_xlim(left=-2,
                     right=star1.pairSeparationsArray.shape[1]+2)

        base_tick_major = 10
        base_tick_minor = 2
        for ax in (ax1, ax2, ax3, ax4):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(
                    base=base_tick_major))
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(
                    base=base_tick_minor))

            ax.grid(which='major', axis='x',
                    color='Gray', alpha=0.55, linestyle='--')
            ax.grid(which='minor', axis='x',
                    color='Gray', alpha=0.7, linestyle=':')
        for ax in (ax2, ax4):
            ax.grid(which='both', axis='y',
                    color='LightGray', alpha=0.9, linestyle='-')

        create_pair_comparison_plot(ax1, ax2, star1, star2, 'pre',
                                    style_params_pre)
        create_pair_comparison_plot(ax3, ax4, star1, star2, 'post',
                                    style_params_post)

    temp_filename = '/Users/dberke/Pictures/Pre-post-change.png'
    fig.savefig(temp_filename)
