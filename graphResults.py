#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:32:43 2018

@author: dberke
"""

# Script to plot cached results from lineFind.py

import datetime as dt
from pathlib import Path
import varconlib as vcl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import matplotlib.dates as dates
from scipy.optimize import curve_fit
from tqdm import tqdm

def polynomial1D(x, m, b):
    """Return the value of a line with slope m and offset b

    x: the independent variable
    m: the slope of the line
    b: the offset of the line
    """
    return (m * x) + b


def fitCCDslope(CCDdata):
    """Return the least-squares fit of the CCD wavelength-to-y(pix) function

    CCDdata: a spectral format for one of HARPS' CCDs, in pandas table format
    """
    x = CCDdata['centcolwl']
    y = CCDdata['centcolypix']
    popt, pcov = curve_fit(polynomial1D, x, y)
    return popt


def readHARPSspectralformat(filename):
    """Return a pandas dataframe containing the spectral format of a HARPS CCD

    filename: the csv file to read in

    returns: a dataframe containing the contents of the file
    """
    return pd.read_csv(filename, header=0, engine='c')


def getHARPSxypos(wl, data, slope, offset):
    """Return the x,y pixel on HARPS CCD given a wavelength [378.113, 691.219]

    wl: wavelength to be mapped back to x,y coordinates. Must be within
        378.113nm and 530.43nm or 533.73nm and 691.219nm

    returns: a tuple of x, y pixel values
    """

    CCD_x_width = 4096
    CCD_y_height = 2048

#    if not ((378.113 <= wl <= 530.43) or (533.73 <= wl <= 691.219)):
#        error_string1 = "Given wavelength ({}) not in HARPS' spectral range".\
#                        format(wl)
#        error_string2 = " (378.113nm - 530.43nm or 533.73 - 691.219nm)"
#        raise ValueError(error_string1+error_string2)

    for minwl, maxwl, minfsr, maxfsr in zip(data['startwl'], data['endwl'],
                                            data['FSRmin'], data['FSRmax']):
        if minfsr <= wl <= maxfsr:
            lowerwl, upperwl = minwl, maxwl
            break

    try:
        xfrac = (wl - lowerwl) / (upperwl - lowerwl)
    except UnboundLocalError:
        print(wl, data, slope, offset)
        raise
    xpos = xfrac * CCD_x_width
    ypos = polynomial1D(wl, slope, offset)
    return (xpos, ypos)


def plot_HARPS_CCDs(pairlist):
    """Plot the HARPS CCDs at 1-to-1 pixel scale, with wavelengths
    """

    maxradvel = 143500
    minradvel = -68800

    mpl.rcParams['font.size'] = 24
    fig_blue = plt.figure(figsize=(40.96, 20.48), dpi=100, tight_layout=True)
    fig_red = plt.figure(figsize=(40.96, 20.48), dpi=100, tight_layout=True)
    ax_blue = fig_blue.add_subplot(1, 1, 1)
    ax_red = fig_red.add_subplot(1, 1, 1)
    axes = (ax_blue, ax_red)

    bluetable = readHARPSspectralformat(blueCCDpath)
    redtable = readHARPSspectralformat(redCCDpath)
    tables = (bluetable, redtable)

    blueparams = fitCCDslope(bluetable)
    redparams = fitCCDslope(redtable)
    parameters = (blueparams, redparams)

    colors = ({'main': 'Blue',
               'mid': 'DarkCyan'},
              {'main': 'Red',
               'mid': 'Maroon'})

    for ax in axes:
        ax.set_xlim(left=0, right=4096)
        ax.set_ylim(bottom=0, top=2048)
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
        vert_joins = [x for x in range(512, 4096, 512)]
        ax.vlines(vert_joins, 0, 2048, color='black', linewidth=1)
        ax.hlines(1024, 0, 4096, color='black', linewidth=1)

    for ax, table, params, color in zip(axes, tables, parameters, colors):
        # Plot a bunch of evenly-spaced point to outline the location of the
        # light.
        for wl in tqdm(np.linspace(table['FSRmin'].min(),
                                   table['FSRmax'].max(), 10000),
                       unit='Reference points'):
            x, y = getHARPSxypos(wl, table, *params)
            ax.plot(x, y, color=color['main'], linestyle='',
                    marker='.', markersize=2)

        # Plot the positions of the central columns
        for wl in table['centcolwl']:
            x, y, = getHARPSxypos(wl, table, *params)
            ax.plot(x, y, color=color['mid'], linestyle='',
                    marker='|', markersize=24)

    # Plot the locations of each line in each pair
    for pair in tqdm(pairlist, unit='Line pairs'):
        line1 = float(pair[0])
        line2 = float(pair[1])
        if line2 < 530.43:
            axis = ax_blue
            table = bluetable
            params = blueparams
        elif line1 > 533.73:
            axis = ax_red
            table = redtable
            params = redparams

        x1, y1 = getHARPSxypos(line1, table, *params)
        x2, y2 = getHARPSxypos(line2, table, *params)
        # Plot the first line of the pair
        axis.plot(x1, y1, color='Purple', linestyle='', marker='P',
                  markersize=12, alpha=1)
        # Annotate it with its wavelength
        axis.annotate(pair[0], xy=(x1, y1), xytext=(x1-55, y1+11),
                      fontsize=15)
        # Plot the maximum limits of where it falls on the detector, assuming
        # a maximum radial velocity shift of ±30 km/s
        blueshift1 = vcl.getwlseparation(-30000+minradvel, line1) + line1
        redshift1 = vcl.getwlseparation(30000+maxradvel, line1) + line1
        x3, y3 = getHARPSxypos(blueshift1, table, *params)
        x4, y4 = getHARPSxypos(redshift1, table, *params)
        lims1 = ((x3, y3), (x4, y4))
        for lims in lims1:
            axis.plot(lims[0], lims[1], color='Purple', linestyle='',
                      marker='|', markersize=24)
        bluerad1 = vcl.getwlseparation(minradvel, line1) + line1
        redrad1 = vcl.getwlseparation(maxradvel, line1) + line1
        x7, y7 = getHARPSxypos(bluerad1, table, *params)
        x8, y8 = getHARPSxypos(redrad1, table, *params)
        axis.plot(x7, y7, color='Purple', linestyle='',
                  marker=8, markersize=8, alpha=1)
        axis.plot(x8, y8, color='Purple', linestyle='',
                  marker=9, markersize=8, alpha=1)

        # Plot the second line of the pair.
        axis.plot(x2, y2, color='Green', linestyle='', marker='P',
                  markersize=12, alpha=1)
        # Annotate it with its wavelength
        axis.annotate(pair[1], xy=(x2, y2), xytext=(x2-55, y2-31),
                      fontsize=15)
        # Plot the maximum limits of where it falls on the detector, assuming
        # a maximum radial velocity shift of ±30 km/s
        blueshift2 = vcl.getwlseparation(-30000+minradvel, line2) + line2
        redshift2 = vcl.getwlseparation(30000+maxradvel, line2) + line2
        x5, y5 = getHARPSxypos(blueshift2, table, *params)
        x6, y6 = getHARPSxypos(redshift2, table, *params)
        lims2 = ((x5, y5), (x6, y6))
        for lims in lims2:
            axis.plot(lims[0], lims[1], color='Green', linestyle='',
                      marker='|', markersize=24)
        bluerad2 = vcl.getwlseparation(minradvel, line2) + line2
        redrad2 = vcl.getwlseparation(maxradvel, line2) + line2
        x9, y9 = getHARPSxypos(bluerad2, table, *params)
        x10, y10 = getHARPSxypos(redrad2, table, *params)
        axis.plot(x9, y9, color='Green', linestyle='',
                  marker=8, markersize=8, alpha=1)
        axis.plot(x10, y10, color='Green', linestyle='',
                  marker=9, markersize=8, alpha=1)

    outfile_blue = '/Users/dberke/Pictures/CCD_blue.png'
    outfile_red = '/Users/dberke/Pictures/CCD_red.png'
    fig_blue.savefig(outfile_blue)
    fig_red.savefig(outfile_red)
    plt.close(fig_blue)
    plt.close(fig_red)
    mpl.rcdefaults()


def plot_as_func_of_date(data, pairlist, filepath, folded=False):
    """Plot separations as a function of date.

    """

    for linepair in tqdm(pairlist, unit='Line pairs'):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{}'.format(file.name))
        ax.set_ylabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                      format(linepair[0], linepair[1]), fontsize=18)
        if folded:
            xlabel = 'Date of observation (folded by month)'
        else:
            xlabel = 'Date of observation'
        ax.set_xlabel(xlabel, fontsize=18)
    #        ax.set_ylim(bottom=-200, top=200)

        # Select the data to plot
        filtdata = data[(data['line1_nom_wl'] == float(linepair[0])) &
                        (data['line2_nom_wl'] == float(linepair[1]))]
        gaussvel = filtdata['vel_diff_gauss']
        gaussvel -= np.median(gaussvel)
        gausserr = filtdata['diff_err_gauss']

        # Find the RMS of the scatter
        gauss_rms = np.sqrt(np.mean(np.square(gaussvel)))
        # Find the mean error
        mean_err = np.mean(gausserr)

        gaussdatetimes = filtdata['date']
        gausspy = []
        # Convert to Python datetimes so matplotlib's errorbar will work --
        # it doesn't like pandas' native Timestamp object apparently
        for date in gaussdatetimes:
            gausspy.append(date.to_pydatetime())

        if folded:
            gaussdates = [date.replace(year=2000) for date in gausspy]
            format_str = '%b'
            # Set the dates with a slight margin on each side to show points
            # right at the turn of the year.
            ax.set_xlim(left=dt.date(year=1999, month=12, day=29),
                        right=dt.date(year=2001, month=1, day=2))
        else:
            format_str = '%Y%m%d'
            gaussdates = gausspy

        ax.xaxis.set_major_locator(dates.AutoDateLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter(format_str))

        # Plot the RMS
        ax.axhspan(-1*gauss_rms, gauss_rms, color='gray', alpha=0.3)
        # Plot the median-subtracted velocity differences
        ax.errorbar(gaussdates, gaussvel, yerr=gausserr,
                    markerfacecolor='Black', markeredgecolor='Black',
                    linestyle='', marker='o',
                    markersize=5, elinewidth=2, ecolor='Green',
                    capsize=2, capthick=2)

        fig.subplots_adjust(bottom=0.16, wspace=0.0, hspace=0.0)
        fig.autofmt_xdate(bottom=0.16, rotation=30, ha='right')

        ax.annotate('RMS: {:.4f} m/s\nmean error: {:.4f} m/s'.format(gauss_rms,
                    mean_err), xy=(0.35, 0.07), xycoords='axes fraction',
                    size=14, ha='right', va='center',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.2))
    #    ax.legend(framealpha=0.4, fontsize=18, numpoints=0)
    #    plt.show()

        if folded:
            outfile = filepath / 'graphs' / 'Linepair_{}_{}_folded.png'.format(
                                                             linepair[0],
                                                             linepair[1])
        else:
            outfile = filepath / 'graphs' / 'Linepair_{}_{}.png'.format(
                                                                linepair[0],
                                                                linepair[1])
        fig.savefig(str(outfile), format='png')
        plt.close(fig)


#########################

blueCCDpath = Path('/Users/dberke/code/tables/HARPS_CCD_blue.csv')
redCCDpath = Path('/Users/dberke/code/tables/HARPS_CCD_red.csv')

baseDir = Path('/Volumes/External Storage/HARPS')
obj = 'HD146233'

filepath = Path('/Users/dberke/HD146233')  # 18 Sco, G2 (7 files)
filepath = Path('/Volumes/External Storage/HARPS/HD146233')
filepath = Path('/Volumes/External Storage/HARPS/HD78660')
filepath = Path('/Volumes/External Storage/HARPS/HD183658')
filepath = Path('/Volumes/External Storage/HARPS/HD45184')
file = filepath / '{}.csv'.format(filepath.stem)

data = pd.read_csv(file, header=0, parse_dates=[1], engine='c')

plot_as_func_of_date(data, vcl.pairlist, filepath, folded=False)
plot_as_func_of_date(data, vcl.pairlist, filepath, folded=True)

#plot_HARPS_CCDs(vcl.pairlist)
