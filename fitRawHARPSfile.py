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
import csv


def threshold(ypixel, xpixel, color):
    """Return a threshold above which to choose peak pixels based on position.

    """
    if color is 'Red':
        return 10000
    elif color is 'Blue':
        if 1100 < xpixel < 3100:
            if 0 < ypixel <= 100:
                return 500
            elif 100 < ypixel <= 107:
                if xpixel == 1509:
                    return 4100
                else:
                    return 3100
            elif 107 < ypixel <= 120:
                return 1000
            elif 120 < ypixel <= 125:
                return 1200
            elif 125 < ypixel <= 400:
                return 1300
            elif 400 < ypixel <= 800:
                return 1400
            elif 800 < ypixel <= 1400:
                return 3400
            elif 1400 < ypixel < 2080:
                return 3500
            else:
                return 400
        else:
            if 0 < ypixel <= 100:
                return 340
            elif 100 < ypixel <= 105:
                return 410
            elif 105 < ypixel <= 125:
                return 550
            elif 125 < ypixel <= 400:
                return 500
            elif 400 < ypixel <= 800:
                return 900
            elif 800 < ypixel <= 1400:
                return 3400
            elif 1400 < ypixel < 2080:
                return 3500
            else:
                return 400


filepath = Path('/Users/dberke/HARPS/HARPS.2017-01-01T20_20_49.449.fits')

with fits.open(filepath) as hdul:
    bluedata = hdul[1].data
    reddata = hdul[2].data

bluepeaks = []
redpeaks = []
arrays = (bluedata, reddata)
colors = ('Blue', 'Red')
peaklists = (bluepeaks, redpeaks)

slicepoints = [int(n) for n in np.linspace(4, 4091, num=20, endpoint=True)]
#slicepoints = (4, 1023, 2047, 3071, 4091)
for array, color, peaklist in zip(arrays, colors, peaklists):
    for i in slicepoints:
        # Take a thin slice of the array around each of the points,
        # then take its mean in the dispersion direction to help avoid
        # the effects of noise spikes.
        data_slice = array[i-4:i+4, :].mean(axis=0)
        peaks = []

        for j in range(len(data_slice)):
            if (j == 0) or (j == len(data_slice)-1):
                continue
            if data_slice[j] > threshold(j, i, color):
                if (data_slice[j-1] < data_slice[j]) and\
                   (data_slice[j+1] < data_slice[j]):
                    peaks.append(j)

        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=45, right=2103)
#        ax.set_xlim(left=45, right=250)
#        ax.set_ylim(bottom=220, top=3000)

        ax.plot(data_slice, marker='.', markersize=4, linestyle='-',
                color='Gray')
        ax.plot(peaks, data_slice[peaks], marker='.', markersize=12,
                color=color, markeredgecolor='Black', markeredgewidth=1,
                linestyle='')
        xrange = np.array([x for x in range(0, 4095, 2)])
        y = [threshold(j, i, color) for j in xrange]
        ax.plot(xrange, y, color=color,
                linestyle='-', linewidth=1, marker='')

        outfile = '/Users/dberke/Pictures/Fitting_CCDs/Fit_{0}_{1}.png'.\
                  format(color, i)
        fig.savefig(outfile)
        plt.close(fig)

        slicelist = []

        if color == 'Red':
            start_on = 0
        elif color == 'Blue':
            if 860 < i < 3650:
                start_on = 1
            else:
                start_on = 0
        for k in range(start_on, len(peaks), 2):
            slicelist.append((i, peaks[k]))
        peaklist.append(slicelist)
print('Created individual slice plots.')

plotrange = np.linspace(0, 4096, num=100)
#plotrange = slicepoints
for peaklist, color, array in zip(peaklists, colors, arrays):
    coeffslist = []
    slices = zip(*peaklist)
    fig2 = plt.figure(figsize=(40.96, 20.48), dpi=100, tight_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlim(left=0, right=4095)
    ax2.imshow(np.log(np.fliplr(np.rot90(array, k=3))), cmap='inferno')
#    ax2.imshow(np.log(array), cmap='magma')
    labels = []
    for points in slices:
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        coeffs = polyfit(x, y, 4)
        coeffslist.append(coeffs)
        ax2.plot(x, y, marker='+', color='Cyan', markersize=8, alpha=1,
                 linestyle='')
        for xpos, ypos in zip(x, y):
            labels.append(plt.text(xpos, ypos+10, '{}, {}'.format(xpos, ypos),
                                   fontsize=10, color='Red'))
        ax2.plot(plotrange, np.polynomial.polynomial.polyval(plotrange,
                                                             coeffs),
                 marker='x', markersize=6,
                 linewidth=1, linestyle='--', color='White',
                 alpha=1)

    outfile = '/Users/dberke/Pictures/Fitting_CCDs/Fit_{}.png'.format(color)
    fig2.savefig(outfile)
    plt.close(fig2)

    datafile = 'data/HARPS_Fit_Coeffs_{}.txt'.format(color)
    with open(datafile, 'w', newline='') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for item in coeffslist:
            csvwriter.writerow(item)
print('Created CDD overplots.')
