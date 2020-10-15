#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:49:20 2020

@author: dberke

A script to compare the results of fitting transition velocity offsets and pairs
as a function of stellar parameters using different functions.
"""

import argparse
import os
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


def plot_histograms(target):
    """Create plots of histograms for the various quantities of interest.

    Parameters
    ----------
    target : str, ['transitions', 'pairs']
        A string denoting whether to compare the results for transitions, or for
        pairs.

    Returns
    -------
    None.

    """

    cols = {'index': 0,
            'chi_squared_pre': 1,
            'sigma_pre': 2,
            'sigma_sys_pre': 3,
            'chi_squared_post': 4,
            'sigma_post': 5,
            'sigma_sys_post': 6}

    main_dir = vcl.output_dir / f'stellar_parameter_fits_{target}'

    functions = {'linear': 'Linear',
                 'quadratic': 'Quadratic',
                 'cross_term': 'Linear, [Fe/H]/T$_{eff}$',
                 'quadratic_mag': r'Linear, cross term, $\mathrm{M}_{v}^2$'}

    files = {x: main_dir / f'{x}/{x}_{target}_fit_results.csv' for
             x in functions.keys()}

    x_lims = {'left': -15, 'right': 15}

    fig = plt.figure(figsize=(11, 5), tight_layout=True)
    ax_pre = fig.add_subplot(1, 2, 1)
    ax_pre.set_yscale('log')
    ax_pre.set_xlabel(r'Pre-change $\Delta\sigma_\mathrm{sys}$ vs. linear'
                      ' (m/s)')
    ax_pre.set_xlim(**x_lims)
    ax_post = fig.add_subplot(1, 2, 2)
    ax_post.set_yscale('log')
    ax_post.set_xlabel(r'Post-change $\Delta\sigma_\mathrm{sys}$ vs. linear'
                       ' (m/s)')
    ax_post.set_xlim(**x_lims)

    bin_edges = np.linspace(x_lims['left'], x_lims['right'], num=70)

    data_dict = {}
    for function in functions.keys():
        with open(files[function], 'r', newline='') as f:
            data_dict[function] = np.loadtxt(f, delimiter=',')

    linear_sigma_sys_pre = np.array(data_dict['linear']
                                    [:, cols['sigma_sys_pre']])
    linear_sigma_sys_post = np.array(data_dict['linear']
                                     [:, cols['sigma_sys_post']])

    for function in ('cross_term', 'quadratic', 'quadratic_mag'):
        data_pre = np.array(data_dict[function]
                            [:, cols['sigma_sys_pre']])
        data_post = np.array(data_dict[function]
                             [:, cols['sigma_sys_post']])

        diffs_pre = data_pre - linear_sigma_sys_pre
        diffs_post = data_post - linear_sigma_sys_post

        ax_pre.hist(diffs_pre,
                    cumulative=False, histtype='step',
                    label=functions[function], bins=bin_edges)
        ax_post.hist(diffs_post,
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
    if not plots_dir.exists():
        os.mkdir(plots_dir)

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

    # tqdm.write('Unpickling transitions list...')
    # with open(vcl.final_selection_file, 'r+b') as f:
    #     transitions_list = pickle.load(f)

    for quantity in tqdm(quantities.keys()):

        for file, corr_file, function in tqdm(zip(files, corr_files,
                                              functions.keys())):
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
                ax.set_xlim(left=-1, right=len(x)+1)
                if quantity == 'sigma':
                    ax.set_ylim(bottom=0, top=250)
                elif quantity == 'sigma_sys':
                    ax.set_ylim(bottom=-1, top=85)

                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))

                ax.xaxis.grid(which='both', color='Gray',
                              linestyle='-', alpha=0.6)
                ax.yaxis.grid(which='major', color='Gray',
                              linestyle='--', alpha=0.4)

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

            ax_pre.legend(loc='best')
            ax_post.legend(loc='best')

            file_name = plots_dir / f'{quantity}_{function}.png'
            # plt.show(fig)
            fig.savefig(str(file_name))

    for file, corr_file, function in tqdm(zip(files, corr_files,
                                              functions.keys())):
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
            ax.set_xlabel(f'{time.capitalize()}-fiber change index, {function}')
            ax.set_ylabel(r'$\sigma_\mathrm{sys}/\sigma$')
            ax.set_xlim(left=-1, right=len(x)+1)

            ax.axhline(y=1, color='Black')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=2))

            ax.xaxis.grid(which='both', color='Gray',
                          linestyle='-', alpha=0.6)
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='--', alpha=0.4)

            y_sig = data[:, cols[f'sigma_{time}']]
            y_sig_sys = data[:, cols[f'sigma_sys_{time}']]
            # y_sig_corr = corr_data[:, cols[f'sigma_{time}']]
            # y_sig_sys_corr = corr_data[:, cols[f'sigma_sys_{time}']]

            ax.plot(x, y_sig_sys / y_sig, color='LightCoral',
                    marker='+',
                    label=r'$\sigma_\mathrm{sys}/\sigma$',
                    markeredgecolor='Black',
                    markersize=6)
            # ax.plot(x, y_sig_sys, color='Green',
            #         marker='+',
            #         label=quantities['sigma_sys'],
            #         markeredgecolor='Black',
            #         markersize=6)

        ax_pre.legend(loc='best')
        ax_post.legend(loc='best')

        file_name = plots_dir / f'sigma-sigma_sys_{function}.png'
        # plt.show(fig)
        fig.savefig(str(file_name))

    sys.exit()


def main():
    """Run the main routine for this script.

    Returns
    -------
    None.

    """

    if args.transitions:
        target = 'transitions'
    elif args.pairs:
        target = 'pairs'

    if args.histogram:
        plot_histograms(target)
    elif args.per_transition:
        plot_per_transition(target)


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

    target_type = parser.add_mutually_exclusive_group(required=True)
    target_type.add_argument('-T', '--transitions', action='store_true',
                             help='Plot for individual transitions.')
    target_type.add_argument('-P', '--pairs', action='store_true',
                             help='Plot for pairs.')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
