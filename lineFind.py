#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:41:58 2018

@author: dberke
"""

# Script to automatically find given transitions line from a list and the known
# radial velocity of the star, then fit them and measure their positions.

import numpy as np
from scipy.optimize import curve_fit
import varconlib as vcl
import os.path
import matplotlib.pyplot as plt
from scipy.constants import c
from astropy.modeling import models, fitting


def getpairlist(listfile):
    """Return a list of pairs of lines to check from the given file
    
    The format for the given file should be
    
    pair1a
    pair1b
    
    pair2a
    pair2b
    
    etc. Any columns after the first are ignore.
    """
    pairlist = []
    print(listfile)
    with open(listfile, 'r') as f:
        lines = f.readlines()

    temppair = []
    for line in lines:
        if '#' in line:
            pass
        else:
            if not line == '\n':
                temppair.append(float(line.split()[0]))
            else:
                pairlist.append((temppair[0], temppair[1]))
                temppair = []

    return pairlist



def lineshift(line, radvel):
    """Find the new position of a line given the radial velocity of a star
    
    line: line position. Can be nm or Angstroms, will return in same units
    radvel: radial velocity in km/s
    
    returns: the new line position
    """
    shiftedpos = round(((float(radvel) * 1000 / c) * line) + line, 4)
    print("Original: {}, shifted: {:.4f}".format(line, shiftedpos))
    return shiftedpos


def linefind(line, vac_wl, flux, err, radvel, pixrange=3, plot=False):
    """Find a given line in a HARPS spectrum after correcting for rad. vel.
    
    line: the line to look for, in nm
    vac_wl: an array of vacuum wavelengths
    flux: an array of fluxes
    err: an error array
    radvel: the radial velocity of the star in km/s
    pixrange: the number of "pixels" to search either side of the main wl
    plot: create a plot of the area surrounding the line    
    """

    shiftedline = lineshift(line, radvel)
    central_wl = vcl.wavelength2index(vac_wl, shiftedline)
    lowerlim, upperlim = central_wl - pixrange, central_wl + pixrange +1

    x = vac_wl[lowerlim:upperlim]
    y = flux[lowerlim:upperlim]
#    e = np.array(err[lowerlim:upperlim])
#    weights = 1.0/e

    # Use Astropy to do the fitting
    p_init = models.Polynomial1D(2)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x, y)

    # Fit the returned parabola with a number of points, then find the minimum
    linrange = np.linspace(x[0], x[-1], num=30)
    minwl_p = linrange[p(linrange).argmin()]
    
    print("Found line {:.4f} at {:.4f}".format(shiftedline, minwl_p))

#    g_init = models.Gaussian1D()
#    fit_g = fitting.LevMarLSQFitter()
#    g = fit_g(g_init, x, y)

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=shiftedline-0.02, right=shiftedline+0.02)
        #ax.set_ylim(top=45000, bottom=20000)
        ax.errorbar(vac_wl, flux, yerr=err, color='blue', marker='.',
                    linestyle='')
        ax.vlines(shiftedline, color='green', ymin=0, ymax=50000,
                  linestyle='--')
        ax.vlines(minwl_p, color='magenta', ymin=0, ymax=50000,
                  linestyle='--')
        ax.axvspan(xmin=x[0], xmax=x[-1],
                   color='red', alpha=0.2)

        ax.plot(x, p(x), color='black')
    #    ax.plot(x, g(x), color='cyan')

        plt.show()

    return minwl_p


def measurepairsep(linepair, vac_wl, flux, err, radvel):
    """Return the distance between a pair of lines
    
    """
    params = (vac_wl, flux, err, radvel)
    pos1 = linefind(linepair[0], *params)
    pos2 = linefind(linepair[1], *params)

    medpos = np.median((pos1, pos2))

    return abs(pos1 - pos2)
     

def searchFITSfile(FITSfile, pairlist):
    """Measure line pair separations in given file with given list
    
    """
    
    data = vcl.readHARPSfile(FITSfile)
    vac_wl = vcl.air2vacESO(data['w']) / 10 #Convert from Angstroms to nm AFTER
    flux = data['f']
    err = data['e']
    radvel = data['radvel']

    params = (vac_wl, flux, err, radvel)

    measuredseps = []
    calcseps = []
    for linepair in pairlist:
        measuredseps.append(measurepairsep(linepair, *params))
        calcseps.append(abs(linepair[1] - linepair[0]))
    for item, linepair in zip(measuredseps, pairlist):
        print("{}, {}: measured separation {:.7f} nm".format(*linepair, item))
    #foundlinepos = linefind(line1, *params, plot=True)
    return np.array(measuredseps), np.array(calcseps)


############

pairlistfile = "/Users/dberke/Documents/GoldStandardLineList_vac_eV.txt"
print(os.path.exists(pairlistfile))
pairlist = getpairlist(pairlistfile)

baseDir = "/Volumes/External Storage/HARPS/"

line1 = 600.4673

# RV = -0.1
infile = os.path.join(baseDir, 'HD177758/ADP.2014-09-16T11:07:40.490.fits')
# RV = 43.2
#infile = os.path.join(baseDir, 'HD197818/ADP.2014-09-25T15:36:30.170.fits')
# RV = 0.5 (HD 219482)
#infile = "/Users/dberke/Documents/ADP.2014-09-17T11:22:04.440.fits"
# RV = -21.6 (HD 190248, Delta Pavonis G8)
infile = "/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"

files = ["/Users/dberke/Documents/ADP.2014-09-17T11:22:04.440.fits",
         "/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for infile in files:
    mseps, cseps = searchFITSfile(infile, pairlist)

    diffs = cseps - mseps

    ax.plot(diffs, label='{}'.format(infile[-28:]), linestyle='',
            marker='o')

ax.legend()
plt.show()





