#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:28:29 2019

@author: dberke

A script to test whether the overlaps/gaps between pixels based on their
sizes and positions given in their geometry files are likely the result of
limited precision or not.

"""

import configparser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from obs2d import HARPSFile2D


config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

pixel_geom_files_dir = Path(config['PATHS']['pixel_geom_files_dir'])
pixel_size_file = pixel_geom_files_dir / 'pixel_geom_size_HARPS_2004_A.fits'
pixel_pos_file = pixel_geom_files_dir / 'pixel_geom_pos_HARPS_2004_A.fits'

pixel_size = HARPSFile2D(pixel_size_file)._rawData
pixel_pos = HARPSFile2D(pixel_pos_file)._rawData

pixel_lower = pixel_pos - pixel_size / 2
pixel_upper = pixel_pos + pixel_size / 2


groups = ((507, 508, 509, 510), (1019, 1020, 1021, 1022),
          (1531, 1532, 1533, 1534), (2043, 2044, 2045, 2046),
          (2555, 2556, 2557, 2558), (3067, 3068, 3069, 3070),
          (3579, 3580, 3581, 3582))

overlaps = []
overlaps2 = []

for order in range(72):
    edges = []
    for group in groups:
        edges.append(pixel_upper[order, group[0]] -
                     pixel_lower[order, group[1]])
        edges.append(pixel_upper[order, group[1]] -
                     pixel_lower[order, group[2]])
        edges.append(pixel_upper[order, group[2]] -
                     pixel_lower[order, group[3]])

    overlaps.append(edges)
    overlaps2.extend(edges)
    print(edges)

overlaps = np.array(overlaps)
overlaps2 = np.array(overlaps2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_xlabel('Overlap (nominal pixel units)')
ax1.set_ylabel('Count')

ax1.hist(overlaps2, bins=21)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)

ax2.set_xlabel('Pixel edges across orders')
ax2.set_ylabel('Number')

num_positive = np.zeros(21)
num_negative = np.zeros(21)

for order in range(72):
    for i, overlap in enumerate(overlaps[order]):
        if overlap > 0:
            num_positive[i] += 1
        elif overlap < 0:
            num_negative[i] += 1

#print(num_positive)
#print(num_negative)
#print(sum(num_positive))
#print(sum(num_negative))

x = [x for x in range(21)]

ax2.bar(x, num_positive, color='Red', label='# positive overlaps')
ax2.bar(x, num_negative, color='Blue', label='# negative overlaps (gaps)')

ax2.legend()

plt.show()
