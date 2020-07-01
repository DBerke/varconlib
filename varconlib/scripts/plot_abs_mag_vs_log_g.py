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
import unyt as u


file_path = Path('/Users/dberke/data_output/stellar_databases/'
                 'stellar_db_uncorrected.hdf5')

outfile = Path('/Users/dberke/Pictures/magnitude_vs_gravity_plots/'
               'Magnitude_vs_Gravity.png')

temperatures = u.unyt_array.from_hdf5(
        file_path, dataset_name='star_temperatures')

with h5py.File(file_path, mode='r') as f:
    magnitudes = hickle.load(f, path='/star_magnitudes')
    gravities = hickle.load(f, path='/star_gravities')
    metallicities = hickle.load(f, path='/star_metallicities')
    column_dict = hickle.load(f, path='/transition_column_index')


fig = plt.figure(figsize=(12, 6), tight_layout=True)
ax1 = fig.add_subplot(2, 3, 1)
limits = (4, 5.7)

ax1.set_xlabel('Absolute magnitudes')
ax1.set_ylabel('Surface gravities')

ax1.plot(magnitudes, gravities,
         linestyle='', color='Black', marker='o')

fit = np.polyfit(magnitudes.ravel(), gravities.ravel(), 1)
poly_logg = np.poly1d(fit)

x = np.linspace(*limits, 200)
y = poly_logg(x)

ax1.plot(x, y)


ax2 = fig.add_subplot(2, 3, 2)

ax2.set_xlabel('Absolute magnitudes')
ax2.set_ylabel('Metallicities')

ax2.plot(magnitudes, metallicities,
         linestyle='', color='Black', marker='o')

fit = np.polyfit(magnitudes.ravel(), metallicities.ravel(), 1)
poly_feh = np.poly1d(fit)

y = poly_feh(x)

ax2.plot(x, y)

ax3 = fig.add_subplot(2, 3, 3)

ax3.set_xlabel('Absolute magnitudes')
ax3.set_ylabel('Temperatures')

ax3.plot(magnitudes, temperatures,
         linestyle='', color='Black', marker='o')

fit = np.polyfit(magnitudes.ravel(), temperatures.ravel(), 1)
poly_T = np.poly1d(fit)

y = poly_T(x)

ax3.plot(x, y)

ax4 = fig.add_subplot(2, 3, 4)
ax4.set_xlabel('Absolute magnitudes')
ax4.set_ylabel('Residuals, log(g) - fit')

residuals = gravities - poly_logg(magnitudes)

ax4.plot(magnitudes, residuals, color='Black', linestyle='',
         marker='o')
ax4.axhline(y=0)

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_xlabel('Metallicities')
ax5.set_ylabel('Residuals, [Fe/H] - fit')
ax5.set_ylabel('Residuals, log(g) - fit')

# residuals = metallicities - poly_feh(magnitudes)

ax5.plot(metallicities, residuals, color='Black', linestyle='',
         marker='o')
ax5.axhline(y=0)

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_xlabel('Temperatures')
ax6.set_ylabel('Residuals, Teff - fit')
ax6.set_ylabel('Residuals, log(g) - fit')

# residuals = temperatures.value - poly_T(magnitudes)

ax6.plot(temperatures, residuals, color='Black', linestyle='',
         marker='o')
ax6.axhline(y=0)

plt.show()
# fig.savefig(outfile)
plt.close(fig)
