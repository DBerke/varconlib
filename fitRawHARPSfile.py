#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:08:35 2018

@author: dberke

Script to fit Gaussian functions to the various echelle orders
in a HARPS raw image cross-wise to the perpendicular direction to
generate points to fit higher-order polynomial functions to the
dispersion.

"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.polynomial.polynomial import polyfit


filepath = Path('/Users/dberke/HARPS/HARPS.2017-01-01T20_20_49.449.fits')

with fits.open(filepath) as hdul:
    bluedata = hdul[1].data
    reddata = hdul[2].data

bluepeaks = []
redpeaks = []
arrays = (bluedata, reddata)
colors = ('Blue', 'Red')
thresholds = (800, 10000)
peaklists = (bluepeaks, redpeaks)

slicepoints = [int(n) for n in np.linspace(1023, 3071, num=20, endpoint=True)]

for array, color, threshold, peaklist in zip(arrays, colors, thresholds,
                                             peaklists):
    for i in (1023, 2047, 3071):
#    for i in slicepoints:
        data_slice = array[i]
        peaks = []

        for j in range(len(data_slice)):
            if (j == 0) or (j == len(data_slice)-1):
                continue
            if data_slice[j] > threshold:
                if (data_slice[j-1] < data_slice[j]) and\
                   (data_slice[j+1] < data_slice[j]):
                    peaks.append(j)

        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=45, right=2103)
        #ax.set_ylim(bottom=220, top=3000)

        ax.plot(data_slice, marker='.', markersize=4, linestyle='-',
                color='Gray')
        ax.plot(peaks, data_slice[peaks], marker='.', markersize=8,
                color=color,
                linestyle='')

        outfile = '/Users/dberke/Pictures/Fit_{0}_{1}.png'.format(color, i)
        fig.savefig(outfile)
        plt.close(fig)

        slicelist = []

        if color == 'Red':
            for k in range(0, len(peaks), 2):
                slicelist.append((i, peaks[k]))
        elif color == 'Blue':
            for k in range(1, len(peaks), 2):
                slicelist.append((i, peaks[k]))
        peaklist.append(slicelist)

plotrange = np.linspace(0, 4096, num=20)
print(plotrange)
for peaklist, color in zip(peaklists, colors):
    slice1, slice2, slice3 = peaklist
    fig2 = plt.figure(figsize=(16, 14))
    ax2 = fig2.add_subplot(1, 1, 1)
    for point1, point2, point3 in zip(slice1, slice2, slice3):
        points = (point1, point2, point3)
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        coeffs = polyfit(x, y, 5)
#        poly = np.polynomial.polynomial.Polynomial(coeffs,
#                                                   domain=[0, 4095],
#                                                   window=[0, 4095])
        ax2.plot(plotrange, np.polynomial.polynomial.polyval(plotrange,
                                                             coeffs),
                 marker='o')
    outfile = '/Users/dberke/Pictures/Fit_{}.png'.format(color)
    fig2.savefig(outfile)
    plt.close(fig2)
