#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:14:10 2020

@author: dberke

Script to compare the values found for each transition between the
pre- and post-fiber change eras.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, ttest_ind
from tqdm import tqdm

import varconlib as vcl
from varconlib.miscellaneous import remove_nans
from varconlib.star import Star

parser = argparse.ArgumentParser(description='Compare the values between pre-'
                                 ' and post-fiber change observations in'
                                 ' various ways.')

parser.add_argument('star_name', action='store', type=str,
                    help='The name of the star (directory) where the star to be'
                    ' used is.')

args = parser.parse_args()

plots_dir = vcl.output_dir / 'pair_result_plots'

star = Star(args.star_name, vcl.output_dir / args.star_name)

# KS stands for the 2-sided Kolmogorov-Smirnov test, TT for Welch's t-test.

ks_p_values = []
ks_stats = []
tt_p_values = []
tt_stats = []

pre_slice = slice(None, star.fiberSplitIndex)
post_slice = slice(star.fiberSplitIndex, None)


for pair_label, num in tqdm(star._pair_bidict.items()):

    ks_stat, ks_p_value = ks_2samp(
        remove_nans(star.pairSeparationsArray[pre_slice, num]),
        remove_nans(star.pairSeparationsArray[post_slice, num]))
    tt_stat, tt_p_value = ttest_ind(
        star.pairSeparationsArray[pre_slice, num],
        star.pairSeparationsArray[post_slice, num],
        equal_var=False, nan_policy='omit')

    ks_p_values.append(ks_p_value)
    ks_stats.append(ks_stat)
    tt_p_values.append(tt_p_value)
    tt_stats.append(tt_stat)


fig1 = plt.figure(figsize=(10, 8), tight_layout=True)
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)

ax2.set_xlabel('Pair ordinal number')
ax1.set_ylabel('KS p-value')
ax2.set_ylabel('KS test statistic')

x = [x for x in range(len(star._pair_bidict.values()))]

ax1.plot(x, ks_p_values, color='Blue', linestyle='',
         marker='o', label='2 sided-KS')
ax1.legend()

ax2.plot(x, ks_stats, color='Blue', linestyle='',
         marker='o', label='2 sided-KS')
ax2.legend()

ks_file = plots_dir / f'2s-KS_test_{star.name}.png'
fig1.savefig(str(ks_file))

fig2 = plt.figure(figsize=(10, 8), tight_layout=True)
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)

ax2.set_xlabel('Pair ordinal number')
ax1.set_ylabel("Welch's t-test p-value")
ax2.set_ylabel("Welch's t-test statistic")

ax1.plot(x, tt_p_values, color='Red', linestyle='',
         marker='o', label="Welch's t-test")
ax1.legend()

ax2.plot(x, tt_stats, color='Red', linestyle='',
         marker='o', label="Welch's t-test")
ax2.legend()

tt_file = plots_dir / f"Welch's_t-test_{star.name}.png"
fig2.savefig(str(tt_file))
plt.close('all')


# models = ('linear', 'quadratic', 'cross_term', 'quadratic_mag')

# fig = plt.figure(figsize=(8, 8), tight_layout=True)

# line_lims = np.linspace(-1, 70, 2)

# for num, model in enumerate(models):
#     ax = fig.add_subplot(2, 2, num+1)
#     ax.set_xlim(*line_lims)
#     ax.set_ylim(*line_lims)
#     ax.set_xlabel(r'Pre-change $\sigma_\mathrm{sys}$ (m/s)')
#     ax.set_ylabel(r'Post-change $\sigma_\mathrm{sys}$ (m/s)')
#     file_name = base_dir / f'stellar_parameter_fits/{model}_corrected/'\
#         f'{model}_fit_results.csv'

#     data = np.loadtxt(file_name, dtype=float, usecols=(2, 3, 5, 6),
#                       delimiter=',')

#     ax.scatter(data[:, 1], data[:, 3], s=20, label=model.capitalize(),
#                c='DodgerBlue',
#                edgecolors='Black')
#     ax.plot(line_lims, line_lims, color='Black')

#     ax.legend()

# plt.show()
