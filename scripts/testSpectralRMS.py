#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:42:51 2018

@author: dberke

Script to check whether RMS of a flat continuum region of a given HARPS
spectrum matches the error array, to see if the error array is well-estimated.
"""

import numpy as np
from pathlib import Path
from glob import glob
import varconlib as vcl
import matplotlib.pyplot as plt

baseDir = Path('/Volumes/External Storage/HARPS/HD146233')

files = glob(str(baseDir / '*.fits'))

total_fluxes_stddev = []
total_median_errors = []

for file in files:
    data = vcl.readHARPSfile(file, radvel=True)

    errors = []
    fluxes = []

    for wl, flux, error in zip(data['w'], data['f'], data['e']):
        if 6075.0 <= wl <= 6077.5:
            errors.append(error)
            fluxes.append(flux)

#    flux_rms = np.sqrt(np.mean(np.square(np.array(fluxes))))
    flux_rms = np.std(fluxes)
    med_err = np.median(error)
    filename = Path(file).stem
    print('For {}:'.format(filename))
    print('Flux RMS: {}, median error: {}'.format(flux_rms, med_err))
    total_fluxes_stddev.append(flux_rms)
    total_median_errors.append(med_err)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(total_fluxes_stddev, total_median_errors)
plt.show()

plt.close(fig)
