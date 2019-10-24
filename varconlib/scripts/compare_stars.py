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

import matplotlib.pyplot as plt
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
    star1 = Star(name=data_dir1.name, star_dir=data_dir1, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list)
    star2 = Star(name=data_dir2.name, star_dir=data_dir2, suffix=args.suffix,
                 transitions_list=transitions_list,
                 pairs_list=pairs_list)

    # Create a plot comparing the two stars.
    if (star1.fiberSplitIndex not in (0, None))\
       and (star2.fiberSplitIndex not in (0, None)):
        fig, axes = plt.subplots(ncols=1, nrows=2,
                                 tight_layout=True,
                                 figsize=(10, 7),
                                 sharey='all',
                                 sharex='all')
        fig.suptitle(f'{data_dir1.name} {data_dir2.name}')
        ax1, ax2 = axes
        ax1.set_ylabel('Pre-fiber change pair separations (m/s)')
        ax2.set_ylabel('Post-fiber change pair separations (m/s)')
        ax2.set_xlabel('Pair index number (increasing wavelength)')
        ax1.set_xlim(left=-2,
                     right=star1.pSeparationsArray.shape[1]+2)

        differences_pre = np.empty(shape=star1.pSeparationsArray.shape[1])
        differences_post = np.empty(shape=star1.pSeparationsArray.shape[1])
        errors_pre = np.empty(shape=star1.pSepErrorsArray.shape[1])
        errors_post = np.empty(shape=star1.pSepErrorsArray.shape[1])

        for pair in star1._pair_label_dict.keys():
            separations1 = star1.pSeparationsArray[:star1.fiberSplitIndex,
                                                   star1._p_label(pair)]
            separations2 = star2.pSeparationsArray[:star2.fiberSplitIndex,
                                                   star2._p_label(pair)]
            errors1 = star1.pSepErrorsArray[:star1.fiberSplitIndex,
                                            star1._p_label(pair)]
            errors2 = star2.pSepErrorsArray[:star2.fiberSplitIndex,
                                            star2._p_label(pair)]
            weights1 = 1 / errors1 ** 2
            weights2 = 1 / errors2 ** 2

            w_mean1 = np.average(separations1, weights=weights1)
            w_mean2 = np.average(separations2, weights=weights2)

            w_mean_err1 = 1 / np.sqrt(np.sum(weights1))
            w_mean_err2 = 1 / np.sqrt(np.sum(weights2))

            differences_pre[star1._p_label(pair)] = w_mean1 - w_mean2
            errors_pre[star1._p_label(pair)] = np.sqrt(w_mean_err1 ** 2 +
                                                       w_mean_err2 ** 2)

            separations1 = star1.pSeparationsArray[star1.fiberSplitIndex:,
                                                   star1._p_label(pair)]
            separations2 = star2.pSeparationsArray[star2.fiberSplitIndex:,
                                                   star2._p_label(pair)]
            errors1 = star1.pSepErrorsArray[star1.fiberSplitIndex:,
                                            star1._p_label(pair)]
            errors2 = star2.pSepErrorsArray[star2.fiberSplitIndex:,
                                            star2._p_label(pair)]
            weights1 = 1 / errors1 ** 2
            weights2 = 1 / errors2 ** 2

            w_mean1 = np.average(separations1, weights=weights1)
            w_mean2 = np.average(separations2, weights=weights2)

            w_mean_err1 = 1 / np.sqrt(np.sum(weights1))
            w_mean_err2 = 1 / np.sqrt(np.sum(weights2))

            differences_post[star1._p_label(pair)] = w_mean1 - w_mean2
            errors_post[star1._p_label(pair)] = np.sqrt(w_mean_err1 ** 2 +
                                                        w_mean_err2 ** 2)

        sigma_pre = np.std(differences_pre)
        sigma_post = np.std(differences_post)

        ax1.axhline(y=0, color='Black', alpha=0.9)
        ax1.axhline(y=sigma_pre, color='DimGray', linestyle='--')
        ax1.axhline(y=-sigma_pre, color='DimGray', linestyle='--')
        ax1.axhline(y=2 * sigma_pre, color='DarkGray', linestyle='--')
        ax1.axhline(y=-2 * sigma_pre, color='DarkGray', linestyle='--')
        ax1.axhline(y=3 * sigma_pre, color='LightGray', linestyle='--')
        ax1.axhline(y=-3 * sigma_pre, color='LightGray', linestyle='--')

        weighted_mean_pre = np.average(differences_pre,
                                       weights=(1/errors_pre**2))
        ax1.axhline(y=weighted_mean_pre, color='DarkSalmon',
                    label=f'Weighted mean: {weighted_mean_pre:.2f} m/s')

        ax2.axhline(y=0, color='Black', alpha=0.9)
        ax2.axhline(y=sigma_post, color='DimGray', linestyle='--')
        ax2.axhline(y=-sigma_post, color='DimGray', linestyle='--')
        ax2.axhline(y=2 * sigma_post, color='DarkGray', linestyle='--')
        ax2.axhline(y=-2 * sigma_post, color='DarkGray', linestyle='--')
        ax2.axhline(y=3 * sigma_post, color='LightGray', linestyle='--')
        ax2.axhline(y=-3 * sigma_post, color='LightGray', linestyle='--')

        weighted_mean_post = np.average(differences_post,
                                        weights=(1/errors_post**2))
        ax2.axhline(y=weighted_mean_post, color='DarkTurquoise',
                    label=f'Weighted mean: {weighted_mean_post:.2f} m/s')

        indices = range(len(star1._pair_label_dict.keys()))
        ax1.errorbar(x=indices, y=differences_pre, yerr=errors_pre,
                     label='Pre-fiber change', barsabove=False,
                     **style_params_pre)
        ax2.errorbar(x=indices, y=differences_post, yerr=errors_post,
                     label='Post-fiber change', barsabove=False,
                     **style_params_post)

        ax1.legend()
        ax2.legend()

    plt.show()
