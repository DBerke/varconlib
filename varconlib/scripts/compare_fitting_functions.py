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

import varconlib as vcl


def main():
    """The main routine for this script.

    Returns
    -------
    None.

    """

    main_dir = Path(vcl.config['PATHS']['output_dir']) /\
        'star_comparisons/transitions'

    functions = {'uncorrected': 'Uncorrected',
                 'linear': 'Linear',
                 'quadratic': 'Quadratic',
                 'quadratic_cross_term': 'Quad., [Fe/H]/T$_{eff}$',
                 'quadratic_mag': r'Quad., cross term, $\mathrm{M}_{v}^2$'}
    files = [main_dir / f'{x}/{x}_sigmas.csv' for x in functions.keys()]

    fig = plt.figure(figsize=(11, 5), tight_layout=True)
    ax_pre = fig.add_subplot(1, 2, 1)
    ax_pre.set_xscale('log')
    ax_pre.set_xlabel(r'Pre-fiber change $\sigma$ (m/s)')
    ax_post = fig.add_subplot(1, 2, 2)
    ax_post.set_xscale('log')
    ax_post.set_xlabel(r'Post-fiber change $\sigma$ (m/s)')

    for file, function in zip(files, functions.keys()):
        with open(file, 'r', newline='') as f:
            data = np.loadtxt(f, delimiter=',')
        ax_pre.hist(data[:, 1], cumulative=True, histtype='step',
                    label=functions[function], bins='sturges')
        ax_post.hist(data[:, 2], cumulative=True, histtype='step',
                     label=functions[function], bins='sturges')

    ax_pre.legend(loc='lower right')
    ax_post.legend(loc='lower right')


if __name__ == '__main__':

    main()
