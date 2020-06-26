#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:28:52 2020

@author: dberke

A script for plotting absolute magnitude vs. log(g) for stars to look for
additional degrees of freedom.
"""

from pathlib import Path

import h5py
import hickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


file_path = Path('/Users/dberke/data_output/stellar_databases/'
                 'stellar_db_uncorrected.hdf5')

outfile = Path('/Users/dberke/Pictures/magnitude_vs_gravity_plots/'
               'Magnitude_vs_Gravity.png')

with h5py.File(file_path, mode='r') as f:
    magnitudes = hickle.load(f, path='/star_magnitudes')
    gravities = hickle.load(f, path='/star_gravities')
    column_dict = hickle.load(f, path='/transition_column_index')


fig = plt.figure(figsize=(8, 8), tight_layout=True)
ax1 = fig.add_subplot(2, 1, 1)

limits = (4, 5.7)
# ax.set_xlim(*limits)
# ax.set_ylim(*limits)

ax1.set_xlabel('Absolute magnitudes')
ax1.set_ylabel('Surface gravities')



# ax.plot(limits, y, color='Blue')
ax1.plot(magnitudes, gravities,
         linestyle='', color='Black', marker='o')

fit = np.polyfit(magnitudes.ravel(), gravities.ravel(), 1)
poly = np.poly1d(fit)

x = np.linspace(*limits, 200)
y = poly(x)

ax1.plot(x, y)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Absolute magnitudes')
ax2.set_ylabel('Residuals, log(g) - fit')

residuals = gravities - poly(magnitudes)

ax2.plot(magnitudes, residuals, color='Black', linestyle='',
         marker='o')
ax2.axhline(y=0)


fig.savefig(outfile)
plt.close(fig)
