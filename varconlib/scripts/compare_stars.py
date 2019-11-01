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
import unyt as u

import varconlib as vcl
from varconlib.star import Star


style_params_pre = {'marker': 'o', 'color': 'Chocolate',
                    'markeredgecolor': 'Black', 'ecolor': 'DarkGoldenRod',
                    'linestyle': '', 'alpha': 0.6, 'markersize': 4}

style_params_post = {'marker': 'o', 'color': 'CornFlowerBlue',
                     'markeredgecolor': 'Black', 'ecolor': 'DodgerBlue',
                     'linestyle': '', 'alpha': 0.6, 'markersize': 4}

plt.rc('text', usetex=True)


low_energy_pairs = ('4652.593Cr1_4653.460Cr1_29',
                    '4652.593Cr1_4653.460Cr1_30',
                    '4759.449Ti1_4760.600Ti1_32',
                    '4759.449Ti1_4760.600Ti1_33',
                    '4799.873Ti2_4800.072Fe1_33',
                    '4799.873Ti2_4800.072Fe1_34',
                    '5138.510Ni1_5143.171Fe1_42',
                    '5187.346Ti2_5200.158Fe1_43',
                    '6123.910Ca1_6138.313Fe1_60',
                    '6123.910Ca1_6139.390Fe1_60',
                    '6138.313Fe1_6139.390Fe1_60',
                    '6153.320Fe1_6155.928Na1_61',
                    '6153.320Fe1_6162.452Na1_61',
                    '6153.320Fe1_6168.150Ca1_61',
                    '6155.928Na1_6162.452Na1_61',
                    '6162.452Na1_6168.150Ca1_61',
                    '6162.452Na1_6175.044Fe1_61',
                    '6168.150Ca1_6175.044Fe1_61',
                    '6192.900Ni1_6202.028Fe1_61',
                    '6242.372Fe1_6244.834V1_62')


def get_star_results(star1, star2, time_period):
    """Calculate and return pair separations, errors, and chi-squared values
    for the given stars.

    Parameters
    ----------
    star1, star2 : `varconlib.star.Star` objects
        Two `Star` class objects which have been initialized from measured
        fits.
    time_period : str
        Possible values are 'pre' and 'post' for the time periods before and
        after the HARPS fiber change in 2015.

    Returns
    -------
    dict
        A dictionary of results, consisting of:
            1) the `differences` between each pair.
            2) the associated `errors`.
            3) `reduced_chi_squared1` and `reduced_chi_squared2`, the reduced
               chi-squared value for the two stars.
            5) The `standard deviation` of the differences distribution.
            6) `num_obs1` and `num_obs2, the number of observations during this
               time period for the two stars.
            8) `w_means1` and `w_means2`, the weighted mean values of each
               collection of pair separations in the given time period.
            9) `w_mean_err1` and `w_mean_err2`, the errors on the weighted mean
               for each collection of pair separations in the given time
               period.
    """

    differences = np.empty(shape=star1.pairSeparationsArray.shape[1])
    errors = np.empty(shape=star1.pairSeparationsArray.shape[1])
    reduced_chi_squared1 = np.empty(shape=star1.pairSeparationsArray.shape[1])
    reduced_chi_squared2 = np.empty(shape=star2.pairSeparationsArray.shape[1])
    weighted_means1 = np.empty(shape=star1.pairSeparationsArray.shape[1])
    weighted_means2 = np.empty(shape=star2.pairSeparationsArray.shape[1])
    weighted_errs1 = np.empty(shape=star1.pairSeparationsArray.shape[1])
    weighted_errs2 = np.empty(shape=star2.pairSeparationsArray.shape[1])

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

        w_mean1, sum_weights1 = np.average(separations1, weights=weights1,
                                           returned=True)
        w_mean2, sum_weights2 = np.average(separations2, weights=weights2,
                                           returned=True)

        weighted_means1[star1._p_label(pair)] = w_mean1
        weighted_means2[star1._p_label(pair)] = w_mean2

        w_mean_err1 = 1 / np.sqrt(sum_weights1)
        w_mean_err2 = 1 / np.sqrt(sum_weights2)

        weighted_errs1[star1._p_label(pair)] = w_mean_err1
        weighted_errs2[star1._p_label(pair)] = w_mean_err2

        differences[star1._p_label(pair)] = w_mean1 - w_mean2
        errors[star1._p_label(pair)] = np.sqrt(w_mean_err1 ** 2 +
                                               w_mean_err2 ** 2)
        chi_squared1 = np.sum(((separations1 - w_mean1) / errors1) ** 2)
        chi_squared2 = np.sum(((separations2 - w_mean2) / errors2) ** 2)

        reduced_chi_squared1[star1._p_label(pair)] =\
            chi_squared1 / (len(separations1) - 1)
        reduced_chi_squared2[star1._p_label(pair)] =\
            chi_squared2 / (len(separations2) - 1)

    standard_deviation = np.std(differences)
    num_obs1 = len(separations1)
    num_obs2 = len(separations2)

    results_dict = {'differences': differences * u.m / u.s,
                    'errors': errors * u.m / u.s,
                    'chi_squared1': reduced_chi_squared1,
                    'chi_squared2': reduced_chi_squared2,
                    'standard_deviation': standard_deviation * u.m / u.s,
                    'num_obs1': num_obs1,
                    'num_obs2': num_obs2,
                    'w_means1': weighted_means1 * u.m / u.s,
                    'w_means2': weighted_means2 * u.m / u.s,
                    'w_mean_err1': weighted_errs1 * u.m / u.s,
                    'w_mean_err2': weighted_errs2 * u.m / u.s}

    return results_dict


def plot_chi_squared(ax, x_values, chi_squared1, chi_squared2,
                     num_obs1, num_obs2):

    """Plot the chi-squared distributions for the two stars on a given axis.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        The axis to plot on.
    x_values : 1-D iterable
        A list, 1-D array, or similar iterable structure to plot as the _x_
        values.
    chi_squared1, chi_squared2 : list or 1-D iterable
        The values of chi-squared to plot for the two stars, as a 1-D iterable.
    num_obs1, num_obs2 : int
        The number of observations that went into this plot for each star.
    """

    # Plot the reduced chi-squared values on the lower axis.
    ax.axhline(y=1, color='Black')

    ax.plot(x_values, chi_squared1, color='DarkOliveGreen',
            label=f'{star1.name.replace("_", "-")}: {num_obs1}'
            ' observations', alpha=1)
    ax.plot(x_values, chi_squared2, color='Orange',
            label=f'{star2.name.replace("_", "-")}: {num_obs2}'
            ' observations', alpha=0.8)

    ax.legend(loc='upper left', framealpha=0.4)


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
        after the HARPS fiber change in 2015. Passed to `get_star_results`.
    star1, star2 : `varconlib.star.Star` objects
        Two instances of the Star class to be compared.
    params : dict
        A dictionary of parameter values to pass to `matplotlib.errorbar`.

    """

    results = get_star_results(star1, star2, time_period)

    main_ax.axhline(y=0, color='Black', alpha=0.9)
    for i, color in zip((1, 2, 3), ('DimGray', 'DarkGray', 'LightGray')):
        for j in (-1, 1):
            main_ax.axhline(y=i * j * results['standard_deviation'],
                            color=color, linestyle='--')

    weighted_mean = np.average(results['differences'],
                               weights=(1/results['errors']**2))
    main_ax.axhline(y=weighted_mean, color=params['color'],
                    label=f'Weighted mean: {weighted_mean:.2f} m/s')

    # Show which pairs are low energy.
    le_indices = [star1._p_label(pair_label) for pair_label
                  in low_energy_pairs]
    for i in le_indices:
        main_ax.axvline(i, ymin=0, ymax=0.3, linestyle='-',
                        color='DarkOrchid', alpha=0.6)

    # Plot the pair differences data.
    indices = [x for x in range(len(star1._pair_label_dict.keys()))]
    main_ax.errorbar(x=indices, y=results['differences'],
                     yerr=results['errors'],
                     label=f'{time_period.capitalize()}-fiber change',
                     **params)

    main_ax.legend(loc='upper right')

    # Plot the reduced chi-squared values on the lower axis.
    plot_chi_squared(chi_ax, indices,
                     results['reduced_chi_squared1'],
                     results['reduced_chi_squared2'],
                     results['num_obs1'],
                     results['num_obs2'])


def create_pair_blend_comparison_plot(main_ax, chi_ax, star1, star2,
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
        after the HARPS fiber change in 2015. Passed to `get_star_results`.
    star1, star2 : `varconlib.star.Star` objects
        Two instances of the Star class to be compared.
    params : dict
        A dictionary of parameter values to pass to `matplotlib.errorbar`.

    """

    blend_tuples = set()
    pair_blends_dict = {}
    for pair in star1.pairsList:
        for order_num in pair.ordersToMeasureIn:
            pair_label = '_'.join((pair.label, str(order_num)))
            pair_blends_dict[pair_label] = pair.blendTuple
            blend_tuples.add(pair.blendTuple)

    sorted_blend_tuples = sorted(blend_tuples)
    print(sorted_blend_tuples)

    results = get_star_results(star1, star2, time_period)

    sorted_differences = []
    sorted_errors = []
    sorted_chi_squared1 = []
    sorted_chi_squared2 = []
    divisions = []
    total = 0
    for blend_tuple in sorted_blend_tuples:
        for key, value in pair_blends_dict.items():
            if blend_tuple == value:
                total += 1
                sorted_differences.append(results['differences']
                                          [star1._p_label(key)])
                sorted_errors.append(results['errors'][star1._p_label(key)])
                sorted_chi_squared1.append(results['reduced_chi_squared1'][
                        star1._p_label(key)])
                sorted_chi_squared2.append(results['reduced_chi_squared2'][
                        star1._p_label(key)])
        divisions.append(total)

    print(len(sorted_differences))
    print(len(sorted_errors))

    for div, b_tuple in zip(divisions, sorted_blend_tuples):
        main_ax.axvline(x=div, color='DarkSlateGray', alpha=0.8)
        main_ax.annotate(str(b_tuple),
                         xy=(div - 5, 4 * results['standard_deviation']),
                         rotation=90)

    main_ax.axhline(y=0, color='Black', alpha=0.9)
    for i, color in zip((1, 2, 3), ('DimGray', 'DarkGray', 'LightGray')):
        for j in (-1, 1):
            main_ax.axhline(y=i * j * results['standard_deviation'],
                            color=color, linestyle='--')

    weighted_mean = np.average(results['differences'],
                               weights=(1/results['errors']**2))
    main_ax.axhline(y=weighted_mean, color=params['color'],
                    label=f'Weighted mean: {weighted_mean:.2f} m/s')

    # Show which pairs are low energy.
#    le_indices = [star1._p_label(pair_label) for pair_label
#                  in low_energy_pairs]
#    for i in le_indices:
#        main_ax.axvline(i, ymin=0, ymax=0.3, linestyle='-',
#                        color='DarkOrchid', alpha=0.6)

    # Plot the pair differences data.
    indices = [x for x in range(len(star1._pair_label_dict.keys()))]
    main_ax.errorbar(x=indices, y=sorted_differences, yerr=sorted_errors,
                     label=f'{time_period.capitalize()}-fiber change',
                     **params)

    main_ax.legend(loc='lower left')

    # Plot the reduced chi-squared values on the lower axis.
    plot_chi_squared(chi_ax, indices, sorted_chi_squared1,
                     sorted_chi_squared2,
                     results['num_obs1'], results['num_obs2'])


def create_pair_separations_plot(ax, star1, star2,
                                 time_period, params):
    """Create an errorbar plot of pair differences vs. absolute pair separation
    values on the given axis.

    Parameters
    ----------
    main_ax : `matplotlib.Axes` object
        The axis to plot the pair average offsets on.
    chi_ax :  `matplotlib.Axes` object
        The smaller axis to plot the chi-squared values of each pair on
    time_period : str
        Possible values are 'pre' and 'post' for the time periods before and
        after the HARPS fiber change in 2015. Passed to `get_star_results`.
    star1, star2 : `varconlib.star.Star` objects
        Two instances of the Star class to be compared.
    params : dict
        A dictionary of parameter values to pass to `matplotlib.errorbar`.

    """

    results = get_star_results(star1, star2, time_period)

    ax.axhline(y=0, color='Black', alpha=0.9)
    for i, color in zip((1, 2, 3), ('DimGray', 'DarkGray', 'LightGray')):
        for j in (-1, 1):
            ax.axhline(y=i * j * results['standard_deviation'],
                       color=color, linestyle='--')

    ax.errorbar(x=results['w_means1'].to(u.km/u.s), y=results['differences'],
                xerr=results['w_mean_err1'], yerr=results['errors'],
                marker=params['marker'], color=params['color'],
                ecolor=params['ecolor'],
                markeredgecolor=params['markeredgecolor'],
                linestyle='', markersize=8)
    ax.errorbar(x=results['w_means2'].to(u.km/u.s), y=results['differences'],
                xerr=results['w_mean_err2'], yerr=results['errors'],
                marker='D', color='Gainsboro',
                markeredgecolor='LightSlateGray', linestyle='',
                markersize=4)
#    for w_mean, w_err in zip(results['w_means1'], results['w_means2']),\
#                         zip(results['w_mean_err1'], results['w_mean_err2']):
#        ax.plot(w_mean, w_err, marker='', linestyle='', color='Black')


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
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--compare-pairs', action='store_true',
                            help='Create plots comparing all pairs in the two'
                            'stars together.')
    plot_group.add_argument('--sort-blended', action='store_true',
                            help='Create plots of pairs sorted by the'
                            ' blendedness of their component transitions.')
    plot_group.add_argument('--pair-separations', action='store_true',
                            help='Create plots of pairs plotted by their'
                            ' absolution separation values.')

    parser.add_argument('--recreate-stars', action='store_false', default=True,
                        help='Force the stars to be recreated from directories'
                        ' rather than be read in from saved data.')

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
    if args.recreate_stars:
        tqdm.write('Creating stars from directories...')
    tqdm.write('Reading in first star...')
    star1 = Star(name=data_dir1.name, star_dir=data_dir1, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list, load_data=args.recreate_stars)
    tqdm.write('Reading in second star...')
    star2 = Star(name=data_dir2.name, star_dir=data_dir2, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list, load_data=args.recreate_stars)

    if args.compare_pairs or args.sort_blended:

        # Create a plot comparing the two stars.
        tqdm.write('Creating plot...')
        if (star1.fiberSplitIndex not in (0, None))\
           and (star2.fiberSplitIndex not in (0, None)):
            fig = plt.figure(figsize=(12, 9), dpi=100, tight_layout=True)
            gs = GridSpec(nrows=5, ncols=1, figure=fig,
                          height_ratios=[10, 3, 1, 10, 3], hspace=0)
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

            if args.compare_pairs:
                create_pair_comparison_plot(ax1, ax2, star1, star2, 'pre',
                                            style_params_pre)
                create_pair_comparison_plot(ax3, ax4, star1, star2, 'post',
                                            style_params_post)

                temp_filename = '/Users/dberke/Pictures/' +\
                                f'{star1.name}-{star2.name}_comparison.png'
                fig.savefig(temp_filename)

            elif args.sort_blended:
                create_pair_blend_comparison_plot(ax1, ax2, star1, star2,
                                                  'pre', style_params_pre)
                create_pair_blend_comparison_plot(ax3, ax4, star1, star2,
                                                  'post', style_params_post)

                temp_filename = '/Users/dberke/Pictures/' +\
                                f'{star1.name}-{star2.name}_blend_sorted.png'
                fig.savefig(temp_filename)

    elif args.pair_separations:
        tqdm.write('Creating plot...')
        if (star1.fiberSplitIndex not in (0, None))\
           and (star2.fiberSplitIndex not in (0, None)):
            fig = plt.figure(figsize=(12, 9), dpi=100, tight_layout=True)
            gs = GridSpec(nrows=3, ncols=1, figure=fig,
                          height_ratios=[12, 1, 12], hspace=0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[2], sharex=ax1)
            ax1.set_ylabel('Pre-fiber change\npair separations (m/s)')
            ax2.set_ylabel('Post-fiber change\n pair separations (m/s)')
            ax2.set_xlabel('Absolute pair separation (km/s)')

#            ax1.set_xlim(left=-2,
#                         right=star1.pairSeparationsArray.shape[1]+2)

#            base_tick_major = 10
#            base_tick_minor = 2

            create_pair_separations_plot(ax1, star1, star2,
                                         'pre', style_params_pre)
            create_pair_separations_plot(ax2, star1, star2,
                                         'post', style_params_post)

            plt.show()
