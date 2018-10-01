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
        The wavelength to end at.
    """

    masked_pixels = []
    for i in trange(start, end):
        s_win_start, s_win_end = i - smallwindow, i + smallwindow
        l_win_start, l_win_end = i - largewindow, i + largewindow
        continuum = np.median(flux[l_win_start:l_win_end])
        chunk = np.median(flux[s_win_start:s_win_end])
        if continuum - chunk > threshold:
            masked_pixels.append(i)

    tqdm.write('Found {0} possible line locations.'.format(len(masked_pixels)))
    tqdm.write('Marking masked pixels...')
    wl_ranges = []
    if len(masked_pixels) > 0:
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
    else:
        return None


def find_CCD_boundaries(proximity, blueformat, redformat):
    """Finds wavelengths close to the CCD boundaries in HARPS.

    Parameters
    ----------
    proximity : int
        How close a pixel can be to a CCD (internal) boundary before it is
        flagged as potentially problematic.
    blueformat : :class:`~pandas.DataFrame`
        A DataFrame containing the spectral format of the HARPS blue CCD.
    redformat : pandas DataFrame
        A DataFrame containing the spectral format of the HARPS red CCD.
    """
    coeffs_file = '/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits'
    coeffs_dict = vcl.readHARPSfile(coeffs_file, coeffs=True)
    masked_wls = []
    x_boundaries = [512, 1024, 1536, 2048, 2560, 3072, 3584]
    for order in trange(0, 72, 1):
        order_num = vcl.map_spectral_order(order)
        if order < 46:
            order_row = blueformat[blueformat['order'] == order_num]
        else:
            order_row = redformat[redformat['order'] == order_num]
        for pixel in range(0, 4096, 1):
            distances = [abs(bound - pixel) for bound in x_boundaries]
            if any(dist <= proximity for dist in distances):
                wl = vcl.pix_order_to_wavelength(pixel, order, coeffs_dict)
                fsrmin = float(order_row['FSRmin'])
                fsrmax = float(order_row['FSRmax'])
                if (wl < fsrmin) or (wl > fsrmax):
                    continue
                else:
                    masked_wls.append(wl)

    wl_ranges = []
    range_start = masked_wls[0]
    for x in range(len(masked_wls[:-1])):
        if abs(masked_wls[x+1] - masked_wls[x]) <= 0.01:
            continue
        else:
            range_end = masked_wls[x]
            wl_ranges.append((range_start, range_end))
            range_start = masked_wls[x+1]
    print('Found {0} restricted wavelength ranges.'.format(len(wl_ranges)))
    return wl_ranges


############

start, end = 650, 692

# The small window size
smallwindow = 2
# The larger window size to check for the continuum
largewindow = 50
# How much the median fluxes in the two windows can vary before the central
# pixel in the small window is flagged
tolerance = 0.001
# How close pixels can be to CCD boundaries before they get flagged
proximity = 5

# The maximum and minimum radial velocities of stars in our sample
maxradvel = 143500
minradvel = -68800

blueCCDdata = pd.read_csv(vcl.blueCCDpath, header=0, engine='c')
redCCDdata = pd.read_csv(vcl.redCCDpath, header=0, engine='c')

spectral_windows = [(377, 400), (400, 425), (425, 450), (450, 475), (475, 500),
                    (500, 525), (525, 550), (550, 575), (575, 600), (600, 625),
                    (625, 650), (650, 675), (675, 692)]

bound_wl_ranges = find_CCD_boundaries(proximity, blueCCDdata, redCCDdata)

specfile = Path('/Users/dberke/code/spectra/tapas_HARPS_res.ipac')
data = pd.read_csv(specfile, sep='\s+', header=28, names=('wavelength',
                                                          'transmittance'))

wl = [w for w in tqdm(reversed(data['wavelength']))]
flux = [f for f in tqdm(reversed(data['transmittance']))]

for i, window in enumerate(spectral_windows, start=1):
    start, end = window
    if start is not None:
        start_point = max(vcl.wavelength2index(wl, start), largewindow + 1)
    else:
        start_point = largewindow + 1
    if end is not None:
        end_point = min(vcl.wavelength2index(wl, end), len(wl) - largewindow)
    else:
        end_point = len(wl) - largewindow

    tl_wl_ranges = find_telluric_lines(wl, flux, smallwindow, largewindow,
                                       tolerance, start=start_point,
                                       end=end_point)

    fig = plt.figure(figsize=(10, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=wl[start_point-1], right=wl[end_point+1])
    ax.set_ylim(bottom=0.94, top=1.05)
    ax.plot(wl, flux, marker='', color='DodgerBlue', linestyle='-',
            linewidth=2, zorder=0)
    ax.grid(which='both')

    for wl_range in bound_wl_ranges:
        ax.axvspan(xmin=wl_range[0], xmax=wl_range[1],
                   ymin=0.6, ymax=0.65,
                   color='DarkViolet', alpha=0.7, zorder=2,
                   edgecolor=None)

    if tl_wl_ranges is not None:
        for wl_range in tl_wl_ranges:
            ax.axvspan(xmin=wl[wl_range[0]], xmax=wl[wl_range[1]],
                       ymin=0.55, ymax=0.6,
                       color='ForestGreen', alpha=0.7, zorder=1,
                       edgecolor=None)

        tl_wl = [(wl[x[0]], wl[x[1]]) for x in tl_wl_ranges]

        sorted_by_lower_bound = sorted(tl_wl + bound_wl_ranges,
                                       key=lambda tup: tup[0])
    else:
        sorted_by_lower_bound = sorted(bound_wl_ranges, key=lambda tup: tup[0])

    expanded = []
    for region in sorted_by_lower_bound:
        blueshift = vcl.getwlseparation(-30000 + minradvel,
                                        region[0]) + region[0]
        redshift = vcl.getwlseparation(30000 + maxradvel,
                                       region[1]) + region[1]

        expanded.append((blueshift, redshift))

    merged = []
#    for higher in sorted_by_lower_bound:
    for higher in expanded:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)
            else:
                merged.append(higher)

    for wl_range in merged:
        ax.axvspan(xmin=wl_range[0], xmax=wl_range[1],
                   ymin=0.65, ymax=0.7,
                   color='Crimson', alpha=0.7, zorder=3,
                   edgecolor=None)

    outdir = Path('/Users/dberke/Pictures/spectral_masking')
    outfile = outdir / 'TAPAS_spectrum_velocity_spread_BERV_{0}nm_{1}nm.png'.\
                   format(start, end)
    fig.savefig(str(outfile))
    plt.close(fig)

    outfile = Path('/Users/dberke/Documents/usable_spectrum.txt')
    with open(outfile, 'w') as f:
        for wl_range in merged:
            f.write('{:.3f},{:.3f}\n'.format(*wl_range))

total_range = 691.225 - 378.122
ranges = [x[1] - x[0] for x in merged]
total = sum(ranges)
print('Total spectral range is {} nm.'.format(total_range))
print('Total unuseable spectrum is {:.4f} nm, or {:.4f}%.'.format(total,
      total / total_range * 100))
