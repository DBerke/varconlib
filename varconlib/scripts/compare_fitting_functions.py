#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:49:20 2020

@author: dberke

A script to compare the results of fitting transition velocity offsets as a
function of stellar parameters using different functions.
"""

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
import unyt as u
import sys

import varconlib as vcl

colors = {'linear': 'Gold',
          'quadratic': 'ForestGreen',
          'cross_term': 'RoyalBlue',
          'quadratic_mag': 'FireBrick'}


corr_colors = {'pre_uncorr': 'SaddleBrown',
               'pre_corr': 'LightSalmon',
               'post_uncorr': 'RoyalBlue',
               'post_corr': 'LightSkyBlue'}


def plot_histograms():
    """Create plots of histograms for the various quantities of interest.


    Returns
    -------
    None.

    """

    cols = {'index': 0,
            'sigma_pre': 1,
            'sigma_sys_pre': 2,
            'sigma_post': 3,
            'sigma_sys_post': 4}

    main_dir = Path(vcl.config['PATHS']['output_dir']) /\
        'star_comparisons/transitions'

    functions = {#'uncorrected': 'Uncorrected',
                 'linear': 'Linear',
                 'quadratic': 'Quadratic',
                 'cross_term': 'Linear, [Fe/H]/T$_{eff}$',
                 'quadratic_mag': r'Linear, cross term, $\mathrm{M}_{v}^2$'}
    files = [main_dir / f'{x}/{x}_sigmas.csv' for x in functions.keys()]

    fig = plt.figure(figsize=(11, 5), tight_layout=True)
    ax_pre = fig.add_subplot(1, 2, 1)
    # ax_pre.set_xscale('log')
    ax_pre.set_xlabel(r'Pre-fiber change $\sigma_\mathrm{sys}$ (m/s)')
    ax_pre.set_xlim(left=0, right=100)
    ax_post = fig.add_subplot(1, 2, 2)
    # ax_post.set_xscale('log')
    ax_post.set_xlabel(r'Post-fiber change $\sigma_\mathrm{sys}$ (m/s)')
    ax_post.set_xlim(left=0, right=100)

    bin_edges = [x for x in range(0, 1005, 3)]

    for file, function in zip(files, functions.keys()):
        with open(file, 'r', newline='') as f:
            data = np.loadtxt(f, delimiter=',')
        ax_pre.hist(data[:, cols['sigma_sys_pre']],
                    cumulative=False, histtype='step',
                    label=functions[function], bins=bin_edges)
        ax_post.hist(data[:, cols['sigma_sys_post']],
                     cumulative=False, histtype='step',
                     label=functions[function], bins=bin_edges)

    ax_pre.legend(loc='upper right')
    ax_post.legend(loc='upper right')

    plt.show(fig)
    sys.exit()


def plot_per_transition():
    """Create plots showing quantities of interest based on transition index
    number.


    Returns
    -------
    None.

    """

    plots_dir = Path('/Users/dberke/Pictures/fitting_comparisons')

    cols = {'index': 0,
            'chi_squared_pre': 1,
            'sigma_pre': 2,
            'sigma_sys_pre': 3,
            'chi_squared_post': 4,
            'sigma_post': 5,
            'sigma_sys_post': 6}

    quantities = {#'chi_squared': r'$\chi^2_\nu$',
                  'sigma': r'$\sigma$ (m/s)',
                  'sigma_sys': r'$\sigma_{\mathrm{sys}} (m/s)$'}

    main_dir = Path(vcl.config['PATHS']['output_dir']) /\
        'stellar_parameter_fits'

    functions = {'linear': 'Linear',
                 'quadratic': 'Quadratic',
                 'cross_term': 'Linear, [Fe/H]/T$_{eff}$',
                 'quadratic_mag': r'Linear, cross term, $\mathrm{M}_{v}^2$'}
    files = [main_dir / f'{x}/{x}_fit_results.csv' for x in functions.keys()]
    corr_files = [main_dir /
                  f'{x}_corrected/{x}_fit_results.csv' for x in
                  functions.keys()]

    tqdm.write('Unpickling transitions list...')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    for quantity in quantities.keys():

        for file, corr_file, function in zip(files, corr_files,
                                            functions.keys()):
            with open(file, 'r', newline='') as f:
                data = np.loadtxt(f, delimiter=',')
            with open(corr_file, 'r', newline='') as f:
                corr_data = np.loadtxt(f, delimiter=',')


            fig = plt.figure(figsize=(11, 7), tight_layout=True)
            ax_pre = fig.add_subplot(2, 1, 1)
            ax_post = fig.add_subplot(2, 1, 2)

            x = data[:, 0]
            corr_x = corr_data[:, 0]
            for ax, time in zip((ax_pre, ax_post), ('pre', 'post')):
                ax.set_xlabel(f'{time.capitalize()}-fiber change index')
                # ax.set_yscale('log')
                ax.set_ylabel(f'{quantities[quantity]} ({functions[function]})')
                ax.set_xlim(left=-1, right=len(transitions_list)+2)

                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))

                ax.xaxis.grid(which='both', color='Gray',
                              linestyle='-', alpha=0.6)

                y = data[:, cols[quantity + f'_{time}']]
                corr_y = corr_data[:, cols[quantity + f'_{time}']]

                ax.fill_between(x, y, corr_y,
                                color='Gray',
                                alpha=0.5)

                ax.plot(x, y, color=corr_colors[time + '_uncorr'],
                        marker='o',
                        label='No outlier rejection',
                        markeredgecolor='Black',
                        markersize=6)
                ax.plot(corr_x, corr_y, color=corr_colors[time + '_corr'],
                        marker='o',
                        label='Outlier rejection',
                        markeredgecolor='Black',
                        markersize=6)

            ax_pre.legend(loc='upper right')
            ax_post.legend(loc='upper right')

            file_name = plots_dir / f'{quantity}_{function}.png'
            # plt.show(fig)
            fig.savefig(str(file_name))
    sys.exit()


def main():
    """Run the main routine for this script.

    Returns
    -------
    None.

    """

    if args.histogram:
        plot_histograms()
    elif args.per_transition:
        plot_per_transition()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create various plots in order'
                                     ' to compare various fitting models with'
                                     ' each other.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')

    plot_type = parser.add_mutually_exclusive_group(required=True)
    plot_type.add_argument('--histogram', action='store_true',
                           help='Plot histograms of various quantities for'
                           ' different fitting functions.')
    plot_type.add_argument('--per-transition', action='store_true',
                           help='Create plots which show various quantities as'
                           ' a function of transition number.')
    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
