#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:12:56 2020

@author: dberke

A script to plot values of the binned residuals for HARPS calibration across
multiple blocks of the CCD.

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import varconlib as vcl


def main():
    """Run the main routine for the script."""

    residuals_dir = vcl.data_dir / 'residual_data'

    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax3.set_xlabel('x-position (pixels)')
    ax2.set_ylabel('Residuals (m/s)')

    for num, ax in enumerate((ax1, ax2, ax3)):
        data_file64 = residuals_dir /\
            f'residuals_block_{num+1}_forDB_64bins.txt'
        data_file128 = residuals_dir /\
            f'residuals_block_{num+1}_forDB_128bins.txt'
        # Columns are bin center, mean of residuals in bin, error on the mean
        data64 = np.loadtxt(data_file64, skiprows=2, dtype=float)
        data128 = np.loadtxt(data_file128, skiprows=2, dtype=float)

        ax.set_xlim(left=0, right=4096)
        ax.set_ylim(bottom=-13, top=13)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
        ax.yaxis.grid(which='major', color='Gray',
                      alpha=0.9)
        ax.yaxis.grid(which='minor', color='Gray',
                      alpha=0.4, linestyle='--')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=512))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=64))
        ax.xaxis.grid(which='major', color='Gray',
                      alpha=0.9)
        ax.xaxis.grid(which='minor', color='Gray',
                      alpha=0.2, linestyle='--')

        ax.errorbar(data64[:, 0], data64[:, 1], yerr=data64[:, 2],
                    marker='x', markersize=8, markeredgecolor='FireBrick',
                    color='FireBrick')
        ax.errorbar(data128[:, 0], data128[:, 1], yerr=data128[:, 2],
                    marker='+', markersize=8, markeredgecolor='RoyalBlue',
                    color='RoyalBlue')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the measured binned'
                                     ' residuals for HARPS calibration ')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
