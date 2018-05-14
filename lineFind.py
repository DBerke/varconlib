#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:41:58 2018

@author: dberke
"""

# Script to automatically find given transitions line from a list and the known
# radial velocity of the star, then fit them and measure their positions.

import numpy as np
import varconlib as vcl
import os.path
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from glob import glob
import sys

def parabola(x, c0, c1, c2):
    """Return the value of a parabola of order c0 + c1 * x + c2 * x^2
    
    x: independent variable
    c0: zeroth-order coefficient
    c1: first-order coefficient
    c2: second-order coefficient
    """
    return c0 + c1 * x + c2 * x * x


def gaussian(x, a, b, c):
    """Return the value of a Gaussian function with parameters a, b, and c
    
    x: independent variable
    a: amplitude of Gaussian
    b: center of Gaussian
    c: standard deviation of Gaussian
    """
    return a * np.exp(-1 * ((x - b)**2 / (2 * c * c)))


def getpairlist(listfile):
    """Return a list of pairs of lines to check from the given file
    
    The format for the given file should be
    
    pair1a
    pair1b
    
    pair2a
    pair2b
    
    etc. Any columns after the first are ignored.
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


def linefind(line, vac_wl, flux, err, radvel, pixrange=3, velsep=5000,
             plot=False):
    """Find a given line in a HARPS spectrum after correcting for rad. vel.
    
    line: the line to look for, in nm
    vac_wl: an array of vacuum wavelengths
    flux: an array of fluxes
    err: an error array
    radvel: the radial velocity of the star in km/s
    pixrange: the number of "pixels" to search either side of the main wl
    plot: create a plot of the area surrounding the line    
    """

    radvel = float(radvel)
    # Figure out location of line given radial velocity of the star (in km/s)
    shiftedlinewl = vcl.lineshift(line, radvel) # In nm here.
    print('Given radial velocity {} km/s, line {} should be at {}'.\
          format(radvel, line, shiftedlinewl))
    wlrange = vcl.getwlseparation(velsep, shiftedlinewl) # 5 km/s by default
    upperwllim = shiftedlinewl + wlrange
    lowerwllim = shiftedlinewl - wlrange
    upperguess = vcl.wavelength2index(vac_wl, upperwllim)
    lowerguess = vcl.wavelength2index(vac_wl, lowerwllim)
    centralpos = flux[lowerguess:upperguess].argmin() + lowerguess
    centralwl = vac_wl[centralpos]
    continuum = flux[lowerguess:upperguess].max()
    lowerlim, upperlim = centralpos - pixrange, centralpos + pixrange +1

    x = np.array(vac_wl[lowerlim:upperlim])
    y = np.array(flux[lowerlim:upperlim])
    if not y.all():
        print('Found zero flux for line {}'.format(line))
        return None
    e = np.array(err[lowerlim:upperlim])
    fluxrange = y.max() - y.min()
    #print(x)
    #print(y)

    # Use Astropy to do the fitting
    p_init = models.Polynomial1D(2)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x, y)
    #print(p)
    
    # Find minimum analytically
    foundcenterwl = -1 * p.c1.value / (2 * p.c2.value)
    print('Astropy center wl: {}'.format(foundcenterwl))
    if not x.min() <= foundcenterwl <= x.max():
        print('Line center not found within search bounds!')
        #return None

    # Normalize data for Scipy fitting
    xnorm = x - centralwl
    xnorm *= 1000
    ynorm = y - y.min()
    ynorm /= fluxrange

    # Fit a parabola to the line center
    popt_par, pcov_par = curve_fit(parabola, xnorm, ynorm, sigma=e)
    #print(popt_par)

    # Find curve_fit minimum analytically (line of symmetry = -b/2a)
    # Correct for fitting normalized data
    parcenterwl = (-1 * popt_par[1] / (2 * popt_par[2])) / 1000 + centralwl
    print('Scipy central wl: {}'.format(parcenterwl))

    # Fit a Gaussian to the line center
    print("Continuum level is {}".format(continuum))
    popt_gauss, pcov_gauss = curve_fit(gaussian, xnorm, ynorm-continuum,
                                       p0=(-1*continuum, 0, 1e3),
                                       sigma=e, maxfev=800)
    #print(popt_gauss)
    
    # Find center of Gaussian &
    # correct for fitting normalized data
    gausscenterwl = popt_gauss[1] / 1000 + centralwl
    print('Gaussian central wl: {}'.format(gausscenterwl))

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=lowerwllim, right=upperwllim)
        ax.errorbar(vac_wl[lowerguess:upperguess],
                    flux[lowerguess:upperguess],
                    yerr=err[lowerguess:upperguess],
                    color='blue', marker='.', linestyle='')
        ax.vlines(shiftedlinewl, color='crimson',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=flux[lowerguess:upperguess].max(),
                  linestyle='-', label='Line expected position')
        ax.vlines(foundcenterwl, color='green',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=flux[lowerguess:upperguess].max(),
                  linestyle='-', label='Astropy fit center')
        ax.vlines(parcenterwl, color='red',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=flux[lowerguess:upperguess].max(),
                  linestyle='--', label='Parabola center')
        ax.vlines(gausscenterwl, color='cyan',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=flux[lowerguess:upperguess].max(),
                  linestyle=':', label='Gaussian center')
        ax.axvspan(xmin=x[0], xmax=x[-1],
                   color='grey', alpha=0.3)

        ax.plot(x, p(x), color='black', label='Astropy fit')
        ax.plot((xnorm/1000)+centralwl,
                (parabola(xnorm, *popt_par)*fluxrange)+y.min(),
                color='magenta', linestyle='--',
                label='Parabola fit')
        ax.plot((xnorm/1000)+centralwl,
                ((gaussian(xnorm, *popt_gauss)+continuum)*fluxrange)+y.min(),
                color='orange', linestyle=':',
                label='Gaussian fit')
        ax.legend()

        plt.show()

#        ax2 = fig2.add_subplot(1, 1, 1)
#        ax2.plot(xnorm, ynorm, color='blue', marker='o')
#        ax2.plot(xnorm, parabola(xnorm, *popt_par), color='red',
#                 label='parabola')
#        ax2.plot(xnorm, gaussian(xnorm, *popt_gauss)+continuum, color='green',
#                 linestyle='--', label='Gaussian')
#        ax2.legend()
#        plt.show()

    print("Found line center at {} using parabola.".format(parcenterwl))
    parrestframeline = vcl.lineshift(parcenterwl, -1*radvel)
    print("Corrected for radial velocity: {}".format(parrestframeline))
    print("Found line center at {} using Gaussian.".format(gausscenterwl))
    gaussrestframeline = vcl.lineshift(gausscenterwl, -1*radvel)
    print("Corrected for radial velocity: {}".format(gaussrestframeline))
    print("--------------------------------")

    return (parrestframeline, gaussrestframeline)


def measurepairsep(linepair, vac_wl, flux, err, radvel, FITSfile):
    """Return the distance between a pair of lines
    
    """
    global unfittablelines    
    params = (vac_wl, flux, err, radvel)
    line1wl = linefind(linepair[0], *params,
                       plot=True, velsep=9000, pixrange=3)
    line2wl = linefind(linepair[1], *params,
                       plot=True, velsep=9000, pixrange=3)

    if line1wl == None:
        unfittablelines += 1
    if line2wl == None:
        unfittablelines += 1

    if (line1wl and line2wl) != None:

        parveldiff = abs(vcl.getvelseparation(line1wl[0], line2wl[0]))
        gaussveldiff = abs(vcl.getvelseparation(line1wl[1], line2wl[1]))
        
#        if veldiff > calcveldiff:
#            linefind(linepair[0], *params, plot=True, pixrange=3, velsep=10000)
#            linefind(linepair[1], *params, plot=True, pixrange=3, velsep=10000)
#            print(veldiff)
#            print(calcveldiff)
#            print(FITSfile)
#            sys.exit(0)
    
        return (parveldiff, gaussveldiff)
    else:
        return None
     

def searchFITSfile(FITSfile, pairlist):
    """Measure line pair separations in given file with given list
    
    """
    
    data = vcl.readHARPSfile(FITSfile)
    vac_wl = vcl.air2vacESO(data['w']) / 10 #Convert from Angstroms to nm AFTER
                                            #converting to vacuum wavelengths
    flux = data['f']
    err = data['e']
    radvel = data['radvel']

    params = (vac_wl, flux, err, radvel)

    #foundlinepos = linefind(line1, *params, plot=True)

    measuredseps = []
    for linepair in pairlist:
        msep = measurepairsep(linepair, *params, FITSfile)
        if msep != None:
            measuredseps.append(msep)
        else:
            measuredseps.append(math.nan)
    for item, linepair in zip(measuredseps, pairlist):
        if item != None:
            print("{}, {}: measured separation {:.3f} m/s".format(*linepair,
                  item))
        else:
            print("Couldn't measure separation for {}, {}".format(*linepair))

    return np.array(measuredseps)
#    return 0, 0

def plotstarseparations(mseps):
    """
    """


    for i in range(len(mseps[0])):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        #ax = fig_main.add_subplot(i, j, n+1)
        histlist = []
        for seplist in mseps:
            histlist.append(seplist[i])
        print(histlist)
        histlist = np.array(histlist)
        print(histlist.min(), histlist.max())
        ax.hist(histlist, bins=10)#, range=(-70000, 70000))
        outfile = '/User/dberke/Pictures/Line_pair_{}.png'.format(i)
        #fig.savefig(outfile, format='png')
        plt.show()



############

pairlistfile = "/Users/dberke/code/GoldStandardLineList_vac_working.txt"
pairlist = getpairlist(pairlistfile)

#pairlist = [(537.5203, 538.1069)]
#pairlist = [(579.4679, 579.9464)]

baseDir = "/Volumes/External Storage/HARPS/"
global unfittablelines


line1 = 600.4673

# RV = -0.1
#infile = os.path.join(baseDir, 'HD177758/ADP.2014-09-16T11:07:40.490.fits')
# RV = 43.2
#infile = os.path.join(baseDir, 'HD197818/ADP.2014-09-25T15:36:30.170.fits')
# RV = 0.5 (HD 219482 F6)
#infile = "/Users/dberke/Documents/ADP.2014-09-17T11:22:04.440.fits"
# RV = -21.6 (HD 190248, Delta Pavonis G8)
#infile = "/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"

#files = ["/Users/dberke/Documents/ADP.2014-09-17T11:22:04.440.fits",
#         "/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"]
#files =["/Users/dberke/Documents/ADP.2016-10-02T01:02:55.839.fits"]

#files = glob(os.path.join(baseDir, 'HD208704/*.fits')) # G1 (17 files)
#files = glob(os.path.join(baseDir, 'HD138573/*.fits')) # G5
#files = ['/Volumes/External Storage/HARPS/HD138573/ADP.2014-09-26T16:51:15.230.fits']
#files = glob('/Users/dberke/HD146233/*.fits') # 18 Sco, G2 (7 files)
files = ['/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits']


results = []

for infile in files:
    unfittablelines = 0
    mseps = searchFITSfile(infile, pairlist)
    print(mseps)
    results.append(mseps)

#    ax_main.plot(mseps, label='{}'.format(infile[-28:]), linestyle='',
#            marker='o')
    print('Found {} unfittable lines.'.format(unfittablelines))

#ax_main.legend()

print("#############")
print(len(results[0]))
#plotstarseparations(results)




