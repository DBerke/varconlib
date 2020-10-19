#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:35:01 2020

@author: dberke

A script to compare the sigma_sys values calculated for each pair using 2.5
and 5 sigma outlier rejection.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import varconlib as vcl


sigma_lims = ('2.5', '5.0')
models = ('linear', 'cross_term', 'quadratic_mag', 'quadratic',
          'quad_cross_terms', 'cubic')


data_dict = {}
data_types = {'#index': int, 'chi_squared_pre': float, 'sigma_pre': float,
              'sigma_sys_pre': float, 'chi_squared_post': float,
              'sigma_post': float, 'sigma_sys_post': float}
col_names = ('#index', 'chi_squared_pre', 'sigma_pre',
             'sigma_sys_pre', 'chi_squared_post',
             'sigma_post', 'sigma_sys_post')

plots_dir = Path('/Users/dberke/Pictures/sigma_sys_sigma_limit_comparison')

for model in models:

    fig = plt.figure(figsize=(9, 7), tight_layout=True)
    fig.suptitle(model)
    ax_pre = fig.add_subplot(2, 1, 1)
    ax_pre.set_ylim(bottom=-1, top=45)
    ax_pre.set_xlim(left=-1, right=230)
    ax_post = fig.add_subplot(2, 1, 2, sharex=ax_pre, sharey=ax_pre)

    for ax in (ax_pre, ax_post):
        ax.set_ylabel(r'$\sigma_\mathrm{sys}$ (m/s)')
    ax_post.set_xlabel('Pair index number')

    for sigma_lim in sigma_lims:
        file_path = vcl.output_dir /\
            f'stellar_parameter_fits_pairs_{sigma_lim}sigma/{model}' /\
            f'{model}_pairs_fit_results.csv'
        print(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'Could not be found: {file_path}')

        with open(file_path, 'r', newline='') as f:
            # lines = f.readlines()
            # print(lines)
            data = np.loadtxt(f, delimiter=',')
        # data = pd.read_csv(file_path, sep=',', engine='python',
        #                    names=col_names, skiprows=0)

        ax_pre.plot(data[:, 0], data[:, 3],
                    label=f'{sigma_lim}-sigma')

        ax_post.plot(data[:, 0], data[:, 6],
                     label=f'{sigma_lim}-sigma')

        ax_pre.legend(loc='upper right')
        ax_post.legend(loc='upper right')

    # plt.show()
    outfile = plots_dir / f'{model}.png'
    fig.savefig(str(outfile))
    plt.close()
