#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:29:57 2020

@author: dberke

Script to plot a histogram of the BERV - RV distribution for stars in our
sample.

"""

import argparse
from pathlib import Path

import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.star import Star


def plot_histogram(rv_diff_list):

    bins = [n for n in range(-100, 101)]

    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(left=-15, right=15)
    ax.set_xlabel('BERV - RV (km/s)')
    ax.set_yscale('log')

    ax.axvspan(xmin=-8, xmax=-2, color='Red',
               alpha=0.4)
    ax.axvspan(xmin=2, xmax=8, color='Red',
               alpha=0.4)

    ax.hist(rv_diff_list,
            bins=bins,
            alpha=0.7,
            label='BERV - RV (km/s)')

    ax.legend()

    plt.show(fig)


def plot_apparent_magnitude(rv_diff_list, app_mag_list):

    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('BERV - RV (km/s)')
    ax.set_ylabel('Apparent Magnitude')

    ax.axvspan(xmin=-8, xmax=-2, color='Red',
               alpha=0.4)
    ax.axvspan(xmin=2, xmax=8, color='Red',
               alpha=0.4)
    ax.axhline(y=7, color='RoyalBlue')

    ax.set_ylim(bottom=6.95, top=8.4)
    ax.set_xlim(left=-15, right=15)

    ax.scatter(rv_diff_list, app_mag_list,
               color='PeachPuff', alpha=0.7,
               s=14, edgecolors='Black')

    plt.show(fig)


def main():
    """Run the main routine for this script.


    Returns
    -------
    None.

    """

    main_dir = Path(args.main_dir[0])
    if not main_dir.exists():
        raise FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    rv_diff_list = []
    app_mag_list = []
    tqdm.write('Collecting stars...')
    for star_dir in tqdm(args.star_names):
        star_path = main_dir / star_dir
        star = Star(star_path.stem, star_path, load_data=True)
        vprint(f'Collecting info from {star_dir}.')
        rv = star.radialVelocity
        berv_list = star.bervArray
        rv_diff = [d.to(u.km/u.s) for d in berv_list - rv]
        app_mag = [star.apparentMagnitude] * len(rv_diff)
        vprint(f'Added {len(rv_diff)} observations.')
        rv_diff_list.extend(rv_diff)
        app_mag_list.extend(app_mag)

    tqdm.write(f'Found {len(rv_diff_list)} total observations.')

    # plot_histogram(rv_diff_list)

    plot_apparent_magnitude(rv_diff_list, app_mag_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a plot of the'
                                     ' BERV - RV for multiple'
                                     ' stars.')
    parser.add_argument('main_dir', action='store', type=str, nargs=1,
                        help='The main directory within which to find'
                        ' additional star directories.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()