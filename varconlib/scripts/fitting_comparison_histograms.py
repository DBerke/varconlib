#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:10:53 2020

@author: dberke

A script for comparing the sigma and sigma_sys distributions between fitting
runs using absolute magnitude and log(g).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import varconlib as vcl


parser = argparse.ArgumentParser()
era = parser.add_mutually_exclusive_group(required=True)
era.add_argument('--pre', action='store_true',
                 help="Plot the pre-fiber-change values.")
era.add_argument('--post', action='store_true',
                 help="Plot the pre-fiber-change values.")

args = parser.parse_args()

models = ('linear', 'quadratic', 'cross_term', 'quadratic_mag')

base_dir = vcl.output_dir

abs_mag_dir = base_dir / 'stellar_parameter_fits_abs_mag'
logg_dir = base_dir / 'stellar_parameter_fits'

fig = plt.figure(figsize=(8, 8), tight_layout=True)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

axes = (ax1, ax2, ax3, ax4)

bins = [x for x in range(0, 45, 4)]

for model, ax in tqdm(zip(models, axes)):

    if args.pre:
        ax.set_xlabel(r'$\sigma_\mathrm{sys}\mathrm{pre}$ (m/s)')
    elif args.post:
        ax.set_xlabel(r'$\sigma_\mathrm{sys}\mathrm{post}$ (m/s)')

    abs_mag_file = abs_mag_dir / f'{model}_corrected/{model}_fit_results.csv'
    logg_file = logg_dir / f'{model}_corrected/{model}_fit_results.csv'

    # sigma_pre, sigma_sys_pre, sigma_post, sigma_sys_post
    abs_mag_data = np.loadtxt(abs_mag_file, dtype=float, usecols=(2, 3, 5, 6),
                              delimiter=',')
    logg_data = np.loadtxt(logg_file, dtype=float, usecols=(2, 3, 5, 6),
                           delimiter=',')

    if args.pre:
        col = 1
    elif args.post:
        col = 3

    abs_mag_values = abs_mag_data[:, col]
    logg_values = logg_data[:, col]

    mag_median = np.median(abs_mag_values)
    logg_median = np.median(logg_values)

    ax.hist(abs_mag_values, bins=bins, color='Green', label='abs. mag.',
            histtype='step')
    ax.hist(logg_values, bins=bins, color='Blue', label='log(g)',
            linestyle='--',
            histtype='step')
    ax.axvline(x=mag_median, color='Green', label='abs. mag. median',
               linestyle='--', alpha=0.7)
    ax.axvline(x=logg_median, color='Blue', label='log(g) median',
               linestyle='-', alpha=0.7)
    ax.legend(title=model.capitalize())

plt.show()
