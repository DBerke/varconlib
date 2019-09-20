#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:42:27 2019

@author: dberke

A script for quantifying the fringing seen in HARPS e2ds spectra at redder
orders (and whatever is causing similar short-range ripples at the blue orders)

"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from tqdm import trange

from varconlib.obs2d import HARPSFile2D

flat_file = HARPSFile2D('/Users/dberke/HARPS/Calibration/2012-02-25/data/'
                        'reduced/2012-02-25/'
                        'HARPS.2012-02-25T22:07:11.413_flat_A.fits')

data = flat_file._rawData

stddev = np.std(data, axis=1)

fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.set_xlabel('Order')
ax1.set_ylabel(r'$\sigma$ (normalized flux)')

ax1.plot(stddev, marker='o', linestyle='--')


for box_radius in trange(10, 60, 10):
    start_point = box_radius
    end_point = 4095 - box_radius
    smoothed_data = []
    for j in range(0, 72):
        smoothed_order = []
        for i in range(start_point, end_point, 5):
            smoothed_order.append(np.median(data[j, i - box_radius:
                                  i + box_radius]))
        smoothed_data.append(smoothed_order)

    smoothed_data = np.array(smoothed_data)
    smoothed_std = np.std(smoothed_data, axis=1)

    ax2.plot(smoothed_std, marker='+', linestyle='--',
             label=r'box radius = {}'.format(box_radius))

ax2.legend()

plt.show(fig)
