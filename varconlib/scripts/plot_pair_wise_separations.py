#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:38:30 2020

@author: dberke

A script to plot the per-star pair-wise velocity separations for each pair of
tansitions by various parameters.

"""

import argparse
import csv
import os
import pickle
import sys

from astropy.coordinates import SkyCoord
import astropy.units as units
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy import cos, sin
import pandas as pd
from tqdm import tqdm
import unyt as u

import varconlib as vcl

params_dict = {'temperature': 'Teff (K)',
               'metallicity': '[Fe/H]',
               'logg': 'log(g)'}


types_dict = {'#star_name': str,
              'delta(v)_pair (m/s)': np.float,
              'err_stat_pair (m/s)': np.float,
              'err_sys_pair (m/s)': np.float,
              'transition1 (m/s)': np.float,
              't_stat_err1 (m/s)': np.float,
              't_sys_err1 (m/s)': np.float,
              'chi^2_nu1': np.float,
              'transition2 (m/s)': np.float,
              't_stat_err2 (m/s)': np.float,
              't_sys_err2 (m/s)': np.float,
              'chi^2_nu2': np.float}


def read_csv_file(pair_label, csv_dir, era):
    """
    Import data on a pair from a CSV file.

    Parameters
    ----------
    pair_label : str
        The label of a pair for which the data is to be read.
    csv_dir : `pathlib.Path`
        The path to the main directory where the data files are kept.
    era : str, ['pre', 'post']
        A string denoting whether to read the data for the pre- or post-fiber
        change era.

    Returns
    -------
    None.

    """

    infile = csv_dir / f'{era}/{pair_label}_pair_separations_{era}.csv'
    return pd.read_csv(infile)


def plot_vs(parameter):
    """
    Plot pair-wise velocity separations as a function of the given parameter.

    Parameters
    ----------
    parameter : str
        The name of the parameter to plot against.

    Returns
    -------
    None.

    """

    plot_axis_labels = {'temperature': r'T_\text{eff}',
                        'metallicity': '[Fe/H]',
                        'logg': r'\log(g)'}
    tqdm.write('Writing out data for each pair.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir / f'pair_result_plots/{parameter}'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)

        ax_pre.set_xlabel(plot_axis_labels[parameter])
        ax_post.set_xlabel(plot_axis_labels[parameter])
        ax_pre.set_ylabel(r'$\Delta V$ (pair, m/s)')
        ax_post.set_ylabel(r'$\Delta V$ (pair, m/s)')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)

        ax_pre.errorbar(star_data[params_dict[parameter]],
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(star_data[params_dict[parameter]],
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


def plot_distance():
    """Plot pair separations as a function of distance from the Sun."""

    tqdm.write('Writing out data for each pair.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir /\
                f'pair_result_plots/heliocentric_distance'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)
        ax_pre.set_xlim(left=-1, right=54)
        ax_pre.set_xlabel('Heliocentric distance (pc)')
        ax_post.set_xlabel('Heliocentric distance (pc)')
        ax_pre.set_ylabel(r'$[\Delta v_\mathrm{pair}]^\odot_*$ (m/s, pre)')
        ax_post.set_ylabel(r'$[\Delta v_\mathrm{pair}]^\odot_*$ (m/s, post)')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.axhline(0, color='Black')
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)

        ax_pre.errorbar(star_data['distance (pc)'],
                        data_pre['delta(v)_pair (m/s)'].iloc[-1] -
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(star_data['distance (pc)'],
                         data_pre['delta(v)_pair (m/s)'].iloc[-1] -
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


def plot_galactic_distance():
    """Plot pair separations as a function of distance from the Galactic center.


    Returns
    -------
    None

    """

    RA = star_data['RA'][:-1]
    DEC = star_data['DEC'][:-1]
    dist = star_data['distance (pc)'][:-1]
    coordinates = SkyCoord(ra=RA, dec=DEC, distance=dist,
                           unit=(units.hourangle, units.degree,
                                 units.pc))

    distances = [np.sqrt(c.galactocentric.x**2 + c.galactocentric.y**2).value
                 for c in coordinates]
    # Add Sun's galactocentric distance at the end manually.
    distances.append(8300)
    distances *= u.pc

    tqdm.write('Writing out data for each pair.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir /\
                f'pair_result_plots/galactocentric_distance'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)
        ax_pre.set_xlim(left=8240, right=8340)
        ax_pre.set_xlabel('Galactocentric distance (pc)')
        ax_post.set_xlabel('Galactocentric distance (pc)')
        ax_pre.set_ylabel(r'$[\Delta v_\mathrm{pair}]^{\odot}_*$ (m/s, pre)')
        ax_post.set_ylabel(r'$[\Delta v_\mathrm{pair}]^{\odot}_*$ (m/s, post)')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.axhline(0, color='Black')

        ax_pre.errorbar(distances,
                        data_pre['delta(v)_pair (m/s)'].iloc[-1] -
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(distances,
                         data_pre['delta(v)_pair (m/s)'].iloc[-1] -
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


# Main script body.
parser = argparse.ArgumentParser(description="Plot results for each pair"
                                 " of transitions for various parameters.")

parameter_options = parser.add_argument_group(
    title="Plot parameter options",
    description="Select what parameters to plot the pairs-wise velocity"
    " separations by.")
parser.add_argument('--heliocentric-distance', action='store_true',
                    help='Plot as a function of distance from the Sun.')
parser.add_argument('--galactocentric-distance', action='store_true',
                    help='Plot as a function of distance from galactic center.')

parameter_options.add_argument('-T', '--temperature',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='temperature',
                               help="Plot as a function of stellar"
                               "temperature.")
parameter_options.add_argument('-M', '--metallicity',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='metallicity',
                               help="Plot as a function of stellar"
                               " metallicity.")
parameter_options.add_argument('-G', '--logg',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='logg',
                               help="Plot as a function of stellar"
                               " surface gravity.")

parser.add_argument('-v', '--verbose', action='store_true',
                    help="Print more output about what's happening.")

args = parser.parse_args()

# Define vprint to only print when the verbose flag is given.
vprint = vcl.verbose_print(args.verbose)


csv_dir = vcl.output_dir / 'pair_separation_files'

star_properties_file = csv_dir / 'star_properties.csv'

star_data = pd.read_csv(star_properties_file)

# Import the list of pairs to use.
with open(vcl.final_pair_selection_file, 'r+b') as f:
    pairs_list = pickle.load(f)

if args.parameters_to_plot:
    for parameter in tqdm(args.parameters_to_plot):
        vprint(f'Plotting vs. {parameter}')
        plot_vs(parameter)

if args.heliocentric_distance:
    plot_distance()

if args.galactocentric_distance:
    plot_galactic_distance()
