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
from adjustText import adjust_text


def threshold(pixel, color):
    if color is 'Red':
        return 10000
    elif color is 'Blue':
        if 0 < pixel <= 100:
            return 340
        elif 100 < pixel <= 105:
            return 410
        elif 105 < pixel <= 125:
            return 600
        elif 125 < pixel <= 400:
            return 500
        elif 400 < pixel <= 800:
            return 900
        elif 800 < pixel <= 1400:
            return 3400
        elif 1400 < pixel < 2080:
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

slicepoints = [int(n) for n in np.linspace(1023, 3071, num=20, endpoint=True)]
slicepoints = (4, 1023, 2047, 3071, 4091)
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
            if data_slice[j] > threshold(j, color):
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
        ax.plot(peaks, data_slice[peaks], marker='.', markersize=9,
                color=color, markeredgecolor='Black', markeredgewidth=1,
                linestyle='')
        xrange = np.array([x for x in range(0, 4095, 2)])
        y = [threshold(x, color) for x in xrange]
        ax.plot(xrange, y, color=color,
                linestyle='-', linewidth=1, marker='')

        outfile = '/Users/dberke/Pictures/Fit_{0}_{1}.png'.format(color, i)
        fig.savefig(outfile)
        plt.close(fig)

        slicelist = []

        if color == 'Red':
            for k in range(0, len(peaks), 2):
                slicelist.append((i, peaks[k]))
        elif color == 'Blue':
            if (i == 4) or (i == 4091):
                for k in range(0, len(peaks), 2):
                    slicelist.append((i, peaks[k]))
            else:
                for k in range(1, len(peaks), 2):
                    slicelist.append((i, peaks[k]))
        peaklist.append(slicelist)

plotrange = np.linspace(0, 4096, num=100)
#plotrange = slicepoints
for peaklist, color, array in zip(peaklists, colors, arrays):
    slices = zip(*peaklist)
    fig2 = plt.figure(figsize=(40.96, 20.48), dpi=100, tight_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlim(left=0, right=4095)
    ax2.imshow(np.log(np.fliplr(np.rot90(array, k=3))), cmap='inferno')
    labels = []
    for points in slices:
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        coeffs = polyfit(x, y, 5)
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

#    adjust_text(labels)
    outfile = '/Users/dberke/Pictures/Fit_{}.png'.format(color)
    fig2.savefig(outfile)
    plt.close(fig2)
