#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:26:27 2021

@author: dberke

Generate statistics on transition outlier incidence for different sigma cutoffs
"""

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

import varconlib as vcl
from varconlib.star import Star


def get_star(star_path, suffix=''):
    """Return a varconlib.star.Star object based on its name.

    Parameters
    ----------
    star_path : str
        A string representing the name of the directory where the HDF5 file
        containing a `star.Star`'s data can be found.

    Optional
    --------
    suffix : str
        A suffix to append to the star name to get different ones, such as
        '_quadratic_shifted' or '_linear_unshifted'.

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
    star_name = ''.join((star_path.stem, suffix))
    vprint(star_name)
    try:
        return Star(star_name, star_path)
    except IndexError:
        vprint(f'Excluded {star_path.stem}.')
        pass
    except AttributeError:
        vprint(f'Affected star is {star_path.stem}.')
        raise


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate statistics on how'
                                     ' many outliers there are for a given'
                                     ' sigma limit.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')
    parser.add_argument('--suffix', action='store', type=str, default='',
                        help='The suffix to be appended to the data to'
                        ' determine which values to use. Must'
                        ' include leading underscore.')
    parser.add_argument('--sigma', action='store', type=float,
                        default=2.5,
                        help='The sigma limit to use.')
    parser.add_argument('--wrong-errors', action='store_true',
                        help='Use the incorrect significance calculation not'
                        ' taking sigma_** into account.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")
    args = parser.parse_args()

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    main_dir = vcl.output_dir

    star_list = []
    for star_name in args.star_names:
        star = get_star(main_dir / star_name, suffix=args.suffix)
        if star is None:
            continue
        star_list.append(star)
        vprint(f'Added {star.name}.')

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    transition_labels = []
    for transition in transitions_list:
        if transition.blendedness < 3:
            for order_num in transition.ordersToFitIn:
                transition_labels.append('_'.join([transition.label,
                                                   str(order_num)]))

    num_rows = len(transition_labels)
    num_cols = len(star_list)

    included_fraction_pre = []
    included_fraction_post = []
    for transition_label in tqdm(transition_labels):
        star_fractions_pre = []
        star_fractions_post = []
        for star in star_list:
            col = star.t_index(transition_label)

            pre_slice = slice(None, star.fiberSplitIndex)
            # Only use if star has obs post!!
            post_slice = slice(star.fiberSplitIndex, None)

            if star.hasObsPre:
                model_value = star.transitionModelArray[0, col]
                if args.wrong_errors:
                    total_err_array = star.fitErrorsArray[pre_slice, col]
                else:
                    total_err_array =\
                        np.sqrt(star.fitErrorsArray[pre_slice, col] ** 2 +
                                star.transitionSysErrorsArray[0, col] ** 2)

                significances = abs((star.fitOffsetsNormalizedArray[
                        pre_slice, col] - model_value) / total_err_array)

                included = np.count_nonzero(significances
                                            < args.sigma) / len(significances)
                assert 0 <= included <= 1, f'Incorrect fraction! {included}'

                star_fractions_pre.append(included)
            else:
                star_fractions_pre.append(np.nan)

            # Get the post-change values:
            if star.hasObsPost:
                model_value = star.transitionModelArray[1, col]
                if args.wrong_errors:
                    total_err_array = star.fitErrorsArray[post_slice, col]
                else:
                    total_err_array =\
                        np.sqrt(star.fitErrorsArray[post_slice, col] ** 2 +
                                star.transitionSysErrorsArray[1, col] ** 2)

                significances = abs((star.fitOffsetsNormalizedArray[
                        post_slice, col] - model_value) / total_err_array)

                included = np.count_nonzero(significances
                                            < args.sigma) / len(significances)
                assert 0 <= included <= 1, f'Incorrect fraction! {included}'

                star_fractions_post.append(included)
            else:
                star_fractions_post.append(np.nan)

        included_fraction_pre.append(star_fractions_pre)
        included_fraction_post.append(star_fractions_post)

    # Columns are stars, rows are transitions.
    fractions_pre = np.array(included_fraction_pre)
    fractions_post = np.array(included_fraction_post)

    # Set up plot.
    fig = plt.figure(figsize=(13, 7), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig,
                  width_ratios=(5, 1))
    ax_frac_pre = fig.add_subplot(gs[0, 0])
    ax_frac_post = fig.add_subplot(gs[1, 0], sharex=ax_frac_pre,
                                   sharey=ax_frac_pre)

    ax_hist_pre = fig.add_subplot(gs[0, 1], sharey=ax_frac_pre)
    ax_hist_post = fig.add_subplot(gs[1, 1], sharey=ax_frac_pre)

    ax_frac_pre.set_ylim(bottom=-0.05, top=1.05)
    ax_frac_pre.set_xlim(left=-3, right=num_rows+3)
    ax_hist_pre.xaxis.set_major_locator(ticker.LogLocator(subs='all'))
    ax_hist_post.xaxis.set_major_locator(ticker.LogLocator(subs='auto'))
    ax_hist_pre.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    ax_hist_post.xaxis.set_minor_locator(ticker.LogLocator(subs='auto'))

    bins = np.linspace(0, 1, 25)

    for i in range(num_rows):
        ax_frac_pre.plot([i] * num_cols, fractions_pre[i, :],
                         color='Chocolate', alpha=0.6,
                         marker='o', markersize=4, linestyle='')
        ax_frac_post.plot([i] * num_cols, fractions_post[i, :],
                          color='RoyalBlue', alpha=0.6,
                          marker='o', markersize=4, linestyle='')

    ax_hist_pre.hist(fractions_pre.ravel(), bins=bins, histtype='step',
                     color='Black', linewidth=1.5, orientation='horizontal',
                     log=True)
    ax_hist_post.hist(fractions_post.ravel(), bins=bins, histtype='step',
                      color='Black', linewidth=1.5, orientation='horizontal',
                      log=True)

    plot_dir = Path('/Users/dberke/Pictures/outlier_investigation')
    insert = '_wrong_errors' if args.wrong_errors else ''
    file_name = f'Outliers_{args.sigma}-sigma{args.suffix}{insert}' + \
                '_solar_twins.png'
    fig.savefig(str(plot_dir / file_name), bbox_inches='tight',
                pad_inches=0.01)
#    plt.show()
