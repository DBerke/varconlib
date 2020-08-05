#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:14:10 2020

@author: dberke

This script plots the sigma_sys values found for each transition between the
pre- and post-fiber change eras.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import varconlib as vcl

base_dir = vcl.output_dir

models = ('linear', 'quadratic', 'cross_term', 'quadratic_mag')

fig = plt.figure(figsize=(8, 8), tight_layout=True)

line_lims = np.linspace(-1, 70, 2)

for num, model in enumerate(models):
    ax = fig.add_subplot(2, 2, num+1)
    ax.set_xlim(*line_lims)
    ax.set_ylim(*line_lims)
    ax.set_xlabel(r'Pre-change $\sigma_\mathrm{sys}$ (m/s)')
    ax.set_ylabel(r'Post-change $\sigma_\mathrm{sys}$ (m/s)')
    file_name = base_dir / f'stellar_parameter_fits/{model}_corrected/'\
        f'{model}_fit_results.csv'

    data = np.loadtxt(file_name, dtype=float, usecols=(2, 3, 5, 6),
                      delimiter=',')

    ax.scatter(data[:, 1], data[:, 3], s=20, label=model.capitalize(),
               c='DodgerBlue',
               edgecolors='Black')
    ax.plot(line_lims, line_lims, color='Black')

    ax.legend()

plt.show()
