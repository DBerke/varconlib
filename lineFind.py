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


def lineshift(line, radvel):
    """Find the new position of a line given the radial velocity of a star
    
    line: line position. Can be nm or Angstroms, will return in same units
    radvel: radial velocity in km/s
    
    returns: the new line position
    """
    shiftedpos = ((float(radvel) * 1000 / c) * line) + line
    print("Original: {}, shifted: {}".format(line, shiftedpos))
    return shiftedpos


def linefind(line, FITSfile):
    """Find a given line in a HARPS spectrum after correcting for rad. vel.
    
    line: the line to look for, in nm
    FITSfile: the HARPS FITS file to read
    """
    
    data = vcl.readHARPSfile(FITSfile)
    vac_wl = vcl.air2vacESO(data['w']) / 10 # Convert from Angstroms to nm
    flux = data['f']
    err = data['e']

    pix_range = 3

    shiftedline = lineshift(line, data['radvel'])
    central_wl = vcl.wavelength2index(vac_wl, shiftedline)
    lowerlim, upperlim = central_wl - pix_range, central_wl + pix_range +1

    x = vac_wl[lowerlim:upperlim]
    y = flux[lowerlim:upperlim]
#    e = np.array(err[lowerlim:upperlim])
#    weights = 1.0/e

    p_init = models.Polynomial1D(2)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x, y)

    linrange = np.linspace(x[0], x[-1], num=10)
    print(linrange)
    minwl = linrange[p(linrange).argmin()]
    
    print(minwl)
    print("Found line {:.3f} at {:.3f}".format(shiftedline, minwl))

#    g_init = models.Gaussian1D()
#    fit_g = fitting.LevMarLSQFitter()
#    g = fit_g(g_init, x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=shiftedline-0.02, right=shiftedline+0.02)
    #ax.set_ylim(top=45000, bottom=20000)
    ax.errorbar(vac_wl, flux, yerr=err, color='blue', marker='.', linestyle='')
    ax.vlines(shiftedline, color='green', ymin=0, ymax=50000, linestyle='--')
    ax.vlines(minwl, color='magenta', ymin=0, ymax=50000, linestyle='--')
    ax.axvspan(xmin=vac_wl[lowerlim], xmax=vac_wl[upperlim],
               color='red', alpha=0.2)

    
    ax.plot(x, p(x), color='black')
#    ax.plot(x, g(x), color='cyan')

    plt.show()



    return vac_wl, flux, err





baseDir = "/Volumes/External Storage/HARPS/"

line1 = 600.4673

# RV = -0.1
infile = os.path.join(baseDir, 'HD177758/ADP.2014-09-16T11:07:40.490.fits')
# RV = 43.2
#infile = os.path.join(baseDir, 'HD197818/ADP.2014-09-25T15:36:30.170.fits')
# RV = 0.5
#infile = "/Users/dberke/Documents/ADP.2014-09-17T11:22:04.440.fits"
# RV = -21.6 (HD 190248, Delta Pavonis)
#infile = "/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"


vac_wl, flux, err = linefind(line1, infile)


#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(1, 1, 1)

#x1 = np.array([-3, -2, -1, 0, 1, 2, 3])
#y1 = np.array([9, 4, 1, 0, 1, 4, 9])

#popt, pcov = curve_fit(parabola, x1, y1+500, p0=(1, 0, 500))

#domain = np.linspace(-3, 3)
#fitted_curve = [parabola(x, *popt) for x in domain]
#ax.plot(domain, fitted_curve, color='red')
#plt.show()



