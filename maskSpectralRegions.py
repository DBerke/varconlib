#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:59:10 2018

@author: dberke

Script to step through a TAPAS synthetic spectrum of telluric lines and
mark pixels that are likely part of a line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import varconlib as vcl
from pathlib import Path
from tqdm import tqdm, trange


def find_telluric_lines(wavelength, flux, smallwindow, largewindow, threshold,
                          start=None, end=None):
    """
    Steps through a spectrum returning pixels likely to be part of telluric
    absorption lines.

    Parameters
    ----------
    wavelength : array_like
        A list or array containing wavelengths, in ascending order.
    flux : array_like
        A list or array containing flux values, to go along with `wavelength`.
    smallwindow : int
        The number of pixels to include in the search either side of the
        central pixel (a radius, rather than diameter, e.g., a `smallwindow`
        of 2 would result in analyzing a region 5 pixels across).
    largewindow : int
        The number of pixels to consider on each side when determining the
        continnum level (a radius, rather than diameter).
    threshold : float
        The difference in flux compared to the median flux in `largewindow`
        above which a pixel will be considered to be part of a line.
    start : float
        The wavelength to begin at.
    end : float
    """

    masked_pixels = []
    for i in trange(start_point, end_point):
        s_win_start, s_win_end = i - smallwindow, i + smallwindow
        l_win_start, l_win_end = i - largewindow, i + largewindow
        continuum = np.median(flux[l_win_start:l_win_end])
        chunk = np.median(flux[s_win_start:s_win_end])
        if continuum - chunk > threshold:
            masked_pixels.append(i)

    tqdm.write('Found {0} possible line locations.'.format(len(masked_pixels)))
    tqdm.write('Marking masked pixels...')
    wl_ranges = []
    range_start = masked_pixels[0]
    for x in trange(len(masked_pixels[:-1])):
        if masked_pixels[x+1] == masked_pixels[x]+1:
            continue
        else:
            range_end = masked_pixels[x]
            wl_ranges.append((range_start, range_end))
            range_start = masked_pixels[x+1]
    print('Found {0} restricted wavelength ranges.'.format(len(wl_ranges)))
    return wl_ranges


############
start, end = None, None
smallwindow = 2
largewindow = 50
tolerance = 0.001

infile = Path('/Users/dberke/code/spectra/tapas_HARPS_res.ipac')
data = pd.read_csv(infile, sep='\s+', header=28, names=('wavelength',
                                                        'transmittance'))

wl = np.array([w for w in tqdm(reversed(data['wavelength']))])
flux = np.array([f for f in tqdm(reversed(data['transmittance']))])

if start is not None:
    start_point = vcl.wavelength2index(wl, start)
else:
    start_point = largewindow + 1
if end is not None:
    end_point = vcl.wavelength2index(wl, end)
else:
    end_point = len(wl) - largewindow

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(left=wl[start_point-1], right=wl[end_point+1])
ax.set_ylim(bottom=0., top=1.02)
ax.plot(wl, flux, marker='', color='DodgerBlue', linestyle='-',
        linewidth=1)

wl_ranges = find_telluric_lines(wl, flux, smallwindow, largewindow,
                                tolerance, start=start, end=end)

for wl_range in tqdm(wl_ranges):

    ax.axvspan(xmin=wl[wl_range[0]], xmax=wl[wl_range[1]], ymin=0.50,
               ymax=0.95, color='ForestGreen', alpha=0.5)
ax.grid(which='both')

outfile = Path('/Users/dberke/Pictures/TAPAS_spectrum.png')
fig.savefig(str(outfile))
plt.close(fig)
