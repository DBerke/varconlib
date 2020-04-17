#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:49:20 2020

@author: dberke

A script to compare the results of fitting transition velocity offsets as a
function of stellar parameters using different functions.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import unyt as u
import sys

import varconlib as vcl

cols = {'index': 0,
        'chi_squared_pre': 1,
        'sigma_pre': 2,
        'sigma_sys_pre': 3,
        'chi_squared_post': 4,
        'sigma_post': 5,
        'sigma_sys_post': 6}


def main():
    """Run the main routine for this script.

    Returns
    -------
    None.

    """

    main_dir = Path(vcl.config['PATHS']['output_dir']) /\
        'star_comparisons/transitions'

    functions = {'uncorrected': 'Uncorrected',
                 'linear': 'Linear',
                 'quadratic': 'Quadratic',
                 'cross_term': 'Linear, [Fe/H]/T$_{eff}$',
                 'quadratic_mag': r'Linear, cross term, $\mathrm{M}_{v}^2$'}
    files = [main_dir / f'{x}/{x}_sigmas.csv' for x in functions.keys()]

    fig = plt.figure(figsize=(11, 5), tight_layout=True)
    ax_pre = fig.add_subplot(1, 2, 1)
    # ax_pre.set_xscale('log')
    ax_pre.set_xlabel(r'Pre-fiber change $\sigma$ (m/s)')
    ax_pre.set_xlim(left=0, right=100)
    ax_post = fig.add_subplot(1, 2, 2)
    # ax_post.set_xscale('log')
    ax_post.set_xlabel(r'Post-fiber change $\sigma$ (m/s)')
    ax_post.set_xlim(left=0, right=100)

    bin_edges = [x for x in range(0, 1005, 3)]

    for file, function in zip(files, functions.keys()):
        with open(file, 'r', newline='') as f:
            data = np.loadtxt(f, delimiter=',')
        ax_pre.hist(data[:, cols['chi_squared_pre']],
                    cumulative=False, histtype='step',
                    label=functions[function], bins=bin_edges)
        ax_post.hist(data[:, cols['chi_squared_post']],
                     cumulative=False, histtype='step',
                     label=functions[function], bins=bin_edges)

    ax_pre.legend(loc='upper right')
    ax_post.legend(loc='upper right')

    plt.show(fig)
    sys.exit()


if __name__ == '__main__':

    main()
