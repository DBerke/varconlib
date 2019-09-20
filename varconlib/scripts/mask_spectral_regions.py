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
from adjustText import adjust_text


def find_telluric_lines(wavelength, flux, smallwindow, largewindow, threshold,
                        start=None, end=None):
    """Steps through a spectrum returning pixels likely to be part of telluric
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

    tqdm.write('Found {0} possible telluric line locations.'.
               format(len(masked_pixels)))
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


def find_CCD_boundaries(proximity, blueformat, redformat,
                        bluecoeffs, redcoeffs):
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
    bluecoeffs : array-like
        A list of coefficients for fits to the x-y pixel positions of the
        spectral orders on the blue CCD.
    redcoeffs : array-like
        A list of coefficients for fits to the x-y pixel positions of the
        spectral orders on the red CCD.

    """

    coeffs_file = '/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits'
    coeffs_dict = vcl.readHARPSfile(coeffs_file, coeffs=True)
    masked_wls = []
    x_boundaries = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    y_boundaries = [50, 1074, 2148]

    # Create a figure for consistency checking.
    fig = plt.figure(figsize=(40.96, 20.48), dpi=100, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(bottom=0, top=2148)
    ax.set_xlim(left=0, right=4096)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.hlines(1024, xmin=0, xmax=4096, color='Black', linewidth=1)
    ax.vlines([x_boundaries[1:-1]], ymin=0, ymax=2148, color='Black',
              linewidth=1)
    ax.axhspan(ymin=1024-proximity, ymax=1024+proximity, xmin=0, xmax=1,
               color='Gray', alpha=0.5)

    for order in trange(0, 72, 1):
        order_num = vcl.map_spectral_order(order)
        if order < 46:
            order_row = blueformat[blueformat['order'] == order_num]
            y_fit_coeffs = bluecoeffs[order]
            color = 'Blue'
        else:
            order_row = redformat[redformat['order'] == order_num]
            y_fit_coeffs = redcoeffs[order - 47]
            color = 'Red'

#        print(y_fit_coeffs)
        fsrmin = float(order_row['FSRmin'])
        fsrmax = float(order_row['FSRmax'])
#        print('fsrmin: {}, fsrmax: {}'.format(fsrmin, fsrmax))
        pix_range = [i for i in range(0, 4096, 1)]
        y_fit = np.polynomial.polynomial.polyval(pix_range, y_fit_coeffs)
        for xpos in pix_range:
            ypos = int(y_fit[xpos])
            x_distances = [abs(boundx - xpos) for boundx in x_boundaries]
            y_distances = [abs(boundy - ypos) for boundy in y_boundaries]
#            print(y_distances)
            if any([distx <= proximity for distx in x_distances]) or\
               any([disty <= proximity for disty in y_distances]):

                wl = vcl.pix_order_to_wavelength(xpos, order, coeffs_dict)

                if (wl < fsrmin) or (wl > fsrmax):
                    continue
                else:
                    if any([disty <= proximity for disty in y_distances]):
                        if not xpos % 100:
                            print(xpos, ypos)
                            print('{} too close in y: {}'.format(wl, ypos))
                    masked_wls.append(wl)
        ax.plot(pix_range, y_fit, color=color, alpha=0.7, linewidth=2)
#        FSRleft, FSRright = np.polynomial.polynomial.polyval([fsrmin, fsrmax],
#                                                             y_fit_coeffs)
#        ax.plot([fsrmin, fsrmax], [FSRleft, FSRright], linestyle='',
#                marker='|', color=color, markersize=18)

    outfile = '/Users/dberke/Pictures/spectral_masking/Fitting_check.png'
    fig.savefig(outfile)
    plt.close(fig)

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


def merge_list_entries(list_to_merge):
    """Merge overlapping tuples in a list.


    This function takes a list of tuples containing exactly two numbers (as
    floats) with the smaller number first. It sorts them by lower bound, and
    then compares them to see if any overlap. Ultimately it returns a list of
    tuples containing the union of any tuples that overlap in range.

    Parameters
    ----------
    list_to_merge : list
        A list of tuples of floats, denoting regions on the number line.

    Returns
    -------
    list
        A list containing all the overlapping regions found in the input list.
    """

    merged = []
    sorted_by_lower_bound = sorted(list_to_merge, key=lambda tup: tup[0])
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)
            else:
                merged.append(higher)
    return merged


############

start, end = 650, 692

# The small window size to check for the continuum
smallwindow = 2
# The larger window size to check for the continuum
largewindow = 50
# How much the median fluxes in the two windows can vary before the central
# pixel in the small window is flagged
tolerance = 0.001


# Whether to take CCD sub-boundaries into account or not. This will not
# affect taking into account the gap between CCDs, which will always happen.
CCD_bounds = True
# How close pixels can be to CCD boundaries before they get flagged
proximity = 5
# The wavelength range between CCDs to avoid.
CCD_gap = (530.51, 533.81)

# The maximum and minimum radial velocities of stars in our sample
maxradvel = 143500
minradvel = -68800

# Values to use as cutoffs in radial veclocity.
maxradvel = 70000
minradvel = -70000

blueCCDdata = pd.read_csv(vcl.blueCCDpath, header=0, engine='c')
redCCDdata = pd.read_csv(vcl.redCCDpath, header=0, engine='c')

blueCoeffsFile = Path('data/HARPS_Fit_Coeffs_Blue.txt')
redCoeffsFile = Path('data/HARPS_Fit_Coeffs_Red.txt')

blueCoeffs = np.genfromtxt(blueCoeffsFile, delimiter=',')
redCoeffs = np.genfromtxt(redCoeffsFile, delimiter=',')

spectral_windows = [(377, 400), (400, 425), (425, 450), (450, 475), (475, 500),
                    (500, 525), (525, 550), (550, 575), (575, 600), (600, 625),
                    (625, 650), (650, 675), (675, 692)]
if CCD_bounds:
    bound_wl_ranges = find_CCD_boundaries(proximity, blueCCDdata, redCCDdata,
                                          blueCoeffs, redCoeffs)
    bound_wl_ranges.append(CCD_gap)
else:
    bound_wl_ranges = []

specfile = Path('/Users/dberke/code/spectra/tapas_HARPS_res.ipac')
data = pd.read_csv(specfile, sep='\s+', header=28, names=('wavelength',
                                                          'transmittance'))

wl = [w for w in tqdm(reversed(data['wavelength']))]
flux = [f for f in tqdm(reversed(data['transmittance']))]

# List to put all the merged wavelength avoidance regions in.
all_merged = []

for i, window in enumerate(spectral_windows, start=1):
    start, end = window
    print('Analyzing {}nm-{}nm'.format(start, end))
    if start is not None:
        start_point = max(vcl.wavelength2index(wl, start), largewindow + 1)
    else:
        start_point = largewindow + 1
    if end is not None:
        end_point = min(vcl.wavelength2index(wl, end), len(wl) - largewindow)
    else:
        end_point = len(wl) - largewindow

    if not CCD_bounds:
        if (start <= CCD_gap[0]) and (end >= CCD_gap[1]):
            bound_wl_ranges.append(CCD_gap)

    tl_wl_ranges = find_telluric_lines(wl, flux, smallwindow, largewindow,
                                       tolerance, start=start_point,
                                       end=end_point)

    fig = plt.figure(figsize=(12, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=wl[start_point-1], right=wl[end_point+1])
    ax.set_ylim(bottom=0.94, top=1.05)
    ax.plot(wl, flux, marker='', color='DodgerBlue', linestyle='-',
            linewidth=2, zorder=0)
    ax.grid(which='both')

    temp_sort_list = []

    # Add any CCD boundaries in the window to the list of all excluded ranges.
    temp_bound_ranges = []
    for wl_range in bound_wl_ranges:
        if (wl_range[0] >= start) and (wl_range[1] <= end):
            ax.axvspan(xmin=wl_range[0], xmax=wl_range[1],
                       ymin=0.6, ymax=0.65,
                       color='DarkViolet', alpha=0.7, zorder=2,
                       edgecolor=None)
            temp_bound_ranges.append(wl_range)
    # Extend the temporary list used for sorting with the boundary
    # avoidance regions.
    temp_sort_list.extend(temp_bound_ranges)

    # If there are telluric lines in the window, add them to the list of
    # excluded ranges.
    if tl_wl_ranges is not None:
        for wl_range in tl_wl_ranges:
            ax.axvspan(xmin=wl[wl_range[0]], xmax=wl[wl_range[1]],
                       ymin=0.55, ymax=0.6,
                       color='ForestGreen', alpha=0.7, zorder=1,
                       edgecolor=None)

        # Do a quick conversion from indices to actual wavelengths.
        tl_wl = [(wl[x[0]], wl[x[1]]) for x in tl_wl_ranges]
        # Extend the temporary list used for sorting with the telluric line
        # avoidance regions.
        temp_sort_list.extend(tl_wl)

    expanded = []
    for region in tqdm(temp_sort_list):
        # Value used here is BERVMAX from HARPS files (m/s)
        blueshift = vcl.getwlseparation(-31984 + -1*maxradvel,
                                        region[0]) + region[0]
        redshift = vcl.getwlseparation(31984 + -1*minradvel,
                                       region[1]) + region[1]

        expanded.append((blueshift, redshift))

    merged = merge_list_entries(expanded)

    # Plot the combined spectral regions to avoid.
    for wl_range in merged:
        ax.axvspan(xmin=wl_range[0], xmax=wl_range[1],
                   ymin=0.65, ymax=0.7,
                   color='Tomato', alpha=0.7, zorder=3,
                   edgecolor=None)
    all_merged.extend(merged)

    labels = []
    # Plot the spectral lines chosen.
    for linepair in vcl.pairlist:
        for line in linepair:
            line_wl = float(line)
            if start < line_wl < end:
                ax.axvline(line_wl, ymin=0.54, ymax=0.73, color='Black',
                           linestyle='-', linewidth=1, zorder=4)
                labels.append(ax.text(line_wl, 1.02, '{}'.format(line),
                                      fontsize=10, color='Navy',
                                      ha='center', va='center'))

    if labels:
        adjust_text(labels, arrowprops=dict(arrowstyle='->', color='gray',
                                            alpha=0.4))
    outdir = Path('/Users/dberke/Pictures/spectral_masking')
    if CCD_bounds:
        outfile = outdir / 'Spectral_mask_CCD_{0}nm_{1}nm.png'.\
                            format(start, end)
    else:
        outfile = outdir / 'Spectral_mask_no_CCD_{0}nm_{1}nm.png'.\
                        format(start, end)
    fig.savefig(str(outfile))
    plt.close(fig)

# Output the results to a file.
datafile_base = Path('/Users/dberke/code/data/')

if CCD_bounds:
    file_append = '_CCDbounds'
else:
    file_append = '_noCCDbounds'

datafile = datafile_base / 'unusable_spectrum{}.txt'.format(file_append)

fully_merged = merge_list_entries(all_merged)

print('File written out as {}'.format(datafile))
with open(datafile, 'w') as f:
    if CCD_bounds:
        f.write('# CCD boundaries included\n')
    else:
        f.write('# No CCD boundaries included\n')
    for wl_range in fully_merged:
        f.write('{:.3f},{:.3f}\n'.format(*wl_range))

total_range = 691.225 - 378.122

ranges = [x[1] - x[0] for x in fully_merged]
total = sum(ranges)
print('Total spectral range is {} nm.'.format(total_range))
print('Total unuseable spectrum is {:.3f} nm, or {:.2f}%.'.format(total,
      total / total_range * 100))
print('Total usable spectrum is {:.2f}%'.format(((total_range - total) /
      total_range) * 100))
