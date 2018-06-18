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
import datetime as dt
import math
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import matplotlib.dates as dates
from scipy.optimize import curve_fit
from astropy.visualization import hist as astrohist
from glob import glob
from pathlib import Path
plt.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)


def parabola(x, c0, c1, c2, x0):
    """Return the value of a parabola of order c0 + c1 * x + c2 * x^2

    x: independent variable
    c0: zeroth-order coefficient
    c1: first-order coefficient
    c2: second-order coefficient
    """
    return c0 + (c1 * (x - x0)) + (c2 * (x - x0)**2)


def simpleparabola(x, x0, a, c):
    """Return the value of a parabola constrained to move along the x-axis

    x: independent variable
    x0: point of line of symmetry of parabola
    a: second-order coefficient
    c: zeroth-order coefficient
    """
    return a * (x - x0)**2 + c


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


def fitParabola(xnorm, ynorm, enorm, centralwl, radvel, verbose=False):
    """Fit a parabola to given data

    verbose: Prints a bunch of diagnostic information about the fitting process
    """

    # Fit a parabola to the line center
    popt_par, pcov_par = curve_fit(parabola, xnorm, ynorm, sigma=enorm,
                                   absolute_sigma=True)
    perr_par = np.sqrt(np.diag(pcov_par))
    r_par = ynorm - parabola(xnorm, *popt_par)
    chisq_par = sum((r_par / enorm) ** 2)
    chisq_nu_par = chisq_par / 3  # nu = 7 - 4

    # Find curve_fit minimum analytically (line of symmetry = -b/2a)
    # Correct for fitting normalized data
    parcenterwl = (-1 * popt_par[1] / (2 * popt_par[2])) / 1000 + centralwl
    parcenterwl = popt_par[3] / 1000 + centralwl

    # Find the error in the parabolic fit.
    delta_lambda = np.sqrt((perr_par[1]/1000/popt_par[1]/1000)**2 +
                           (2*perr_par[2]/1000/popt_par[2]/1000)**2)
    wl_err_par = perr_par[3] / 1000

    if chisq_nu_par > 1:
        wl_err_par *= math.sqrt(chisq_nu_par)
    vel_err_par = vcl.getvelseparation(parcenterwl*1e-9,
                                       (parcenterwl+wl_err_par)*1e-9)

    # Shift to stellar rest frame by correcting radial velocity.
    parrestframeline = vcl.lineshift(parcenterwl, -1*radvel)

    if verbose:
            print('Covariance matrix for parabola:')
            print(pcov_par)
            print('popt_par = {}'.format(popt_par))
            print('perr_par = {}'.format(perr_par))
            print(u'χ^2 (parabola) = {:.7f}'.format(chisq_par))
            print(u'χ_ν^2 (parabola) = {:.7f}'.format(chisq_nu_par))
            print('Parabola central wl: {:.6f}'.format(parcenterwl))
            print('δλ/λ = {:.6e}'.format(delta_lambda))
            print('δλ = {:.6e}'.format(wl_err_par))
            print("1 stddev parabola = {:.6e} nm".format(wl_err_par))
            print("1 stddev parabola velspace = {:.7f} m/s".
                  format(vel_err_par))
            print("Found line center at {} using parabola.".
                  format(parcenterwl))
            print("Corrected for radial velocity: {}".format(parrestframeline))

    return {'parrestframeline': parrestframeline,
            'vel_err_par': vel_err_par,
            'parcenterwl': parcenterwl,
            'popt_par': popt_par}


def fitGaussian(xnorm, ynorm, enorm, centralwl, radvel, continuum,
                verbose=False):
    """Fit a Gaussian to the given data

    verbose: prints out diagnostic info on the process
    """

    # Fit a Gaussian to the line center

    popt_gauss, pcov_gauss = curve_fit(gaussian, xnorm, ynorm-continuum,
                                       p0=(-1*continuum, 0, 1e3),
                                       sigma=enorm,
                                       absolute_sigma=True)

    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    r_gauss = (ynorm - continuum) - gaussian(xnorm, *popt_gauss)
    chisq_gauss = sum((r_gauss / enorm) ** 2)
    chisq_nu_gauss = chisq_gauss / 4  # nu = 7 - 3

    wl_err_gauss = perr_gauss[1] / 1000

    # Find center of Gaussian &
    # correct for fitting normalized data
    gausscenterwl = popt_gauss[1] / 1000 + centralwl

    if chisq_nu_gauss > 1:
        wl_err_gauss *= math.sqrt(chisq_nu_gauss)
    vel_err_gauss = vcl.getvelseparation(gausscenterwl*1e-9,
                                         (gausscenterwl+wl_err_gauss)*1e-9)
    # Shift line to stellar rest frame
    gaussrestframeline = vcl.lineshift(gausscenterwl, -1*radvel)

    if verbose:
        print("Continuum level is {}".format(continuum))
        print("Covariance matrix for Gaussian:")
        print(pcov_gauss)
        print('popt_gauss = {}'.format(popt_gauss))
        print('perr_gauss = {}'.format(perr_gauss))
        print(u'χ^2 (Gaussian) = {:.7f}'.format(chisq_gauss))
        print(u'χ_ν^2 (Gaussian) = {:.7f}'.format(chisq_nu_gauss))
        print('Gaussian central wl: {:.6f} nm'.format(gausscenterwl))
        print("1 stddev Gaussian = {:.6e} nm".format(wl_err_gauss))
        print("1 stddev Gaussian velspace = {:.7f} m/s".format(vel_err_gauss))
        print("Found line center at {} using Gaussian.".format(gausscenterwl))
        print("Corrected for radial velocity: {}".format(gaussrestframeline))

    return {'gaussrestframeline': gaussrestframeline,
            'vel_err_gauss': vel_err_gauss,
            'gausscenterwl': gausscenterwl,
            'popt_gauss': popt_gauss}


def fitSimpleParabola(xnorm, ynorm, enorm, centralwl, radvel, verbose=False):
    """Fit a parabola constrained to shift along the x-axis

    verbose: prints out a bunch of diagnostic info about the fitting process
    """

    # Fit constrained parabola
    popt_spar, pcov_spar = curve_fit(simpleparabola, xnorm, ynorm,
                                     sigma=enorm,
                                     absolute_sigma=True)

    perr_spar = np.sqrt(np.diag(pcov_spar))
    r_spar = ynorm - simpleparabola(xnorm, *popt_spar)
    chisq_spar = sum((r_spar / enorm) ** 2)
    chisq_nu_spar = chisq_spar / 4  # nu = 7 - 3

    wl_err_spar = perr_spar[0] / 1000
    sparcenterwl = popt_spar[0] / 1000 + centralwl

    if chisq_nu_spar > 1:
        wl_err_spar *= math.sqrt(chisq_nu_spar)

    vel_err_spar = vcl.getvelseparation(sparcenterwl*1e-9,
                                        (sparcenterwl+wl_err_spar)*1e-9)
    # Convert to restframe of star
    sparrestframeline = vcl.lineshift(sparcenterwl, -1*radvel)

    if verbose:
        print("Covariance matrix for constrained parabola:")
        print(pcov_spar)
        print('popt_spar = {}'.format(popt_spar))
        print('perr_spar = {}'.format(perr_spar))
        print(u'χ^2 (constrained parabola) = {:.7f}'.format(chisq_spar))
        print(u'χ_ν^2 (constrained parabola) = {:.7f}'.format(chisq_nu_spar))
        print("Constrained parabola central wl: {:.6f}".format(sparcenterwl))
        print("1 stddev constrained parabola = {:.6e} nm".format(wl_err_spar))
        print("1 stddev constrained parabola velspace = {:.7f} m/s".
              format(vel_err_spar))
        print("Found line center at {} using constrained parabola.".
              format(sparcenterwl))
        print("Corrected for radial velocity: {}".format(sparrestframeline))

    return {'sparrestframeline': sparrestframeline,
            'vel_err_spar': vel_err_spar,
            'sparcenterwl': sparcenterwl,
            'popt_spar': popt_spar}


def linefind(line, vac_wl, flux, err, radvel, pixrange=3, velsep=5000,
             plot=False, par_fit=False, gauss_fit=False, spar_fit=False):
    """Find a given line in a HARPS spectrum after correcting for rad. vel.

    line: the line to look for, in nm
    vac_wl: an array of vacuum wavelengths
    flux: an array of fluxes
    err: an error array
    radvel: the radial velocity of the star in km/s
    pixrange: the number of "pixels" to search either side of the main wl
    plot: create a plot of the area surrounding the line
    """

    # Create a dictionary to store the results
    results = {}

    radvel = float(radvel)
    # Figure out location of line given radial velocity of the star (in km/s)
    shiftedlinewl = vcl.lineshift(line, radvel)  # In nm here.
#    print('Given radial velocity {} km/s, line {} should be at {:.4f}'.
#          format(radvel, line, shiftedlinewl))
    wlrange = vcl.getwlseparation(velsep, shiftedlinewl)  # 5 km/s by default
    continuumrange = vcl.getwlseparation(velsep+2e4, shiftedlinewl)  # +20 km/s
    upperwllim = shiftedlinewl + wlrange
    lowerwllim = shiftedlinewl - wlrange
    upperguess = vcl.wavelength2index(vac_wl, upperwllim)
    lowerguess = vcl.wavelength2index(vac_wl, lowerwllim)
    uppercont = vcl.wavelength2index(vac_wl, shiftedlinewl+continuumrange)
    lowercont = vcl.wavelength2index(vac_wl, shiftedlinewl-continuumrange)
    centralpos = flux[lowerguess:upperguess].argmin() + lowerguess
    centralwl = vac_wl[centralpos]
    continuum = flux[lowercont:uppercont].max()
    lowerlim, upperlim = centralpos - pixrange, centralpos + pixrange + 1

    x = np.array(vac_wl[lowerlim:upperlim])
    y = np.array(flux[lowerlim:upperlim])
    if not y.all():
        print('Found zero flux for line {}'.format(line))
        return None
    e = np.array(err[lowerlim:upperlim])
    fluxrange = y.max() - y.min()
#    print("fluxrange = {}".format(fluxrange))
#    print(x)
#    print(y)
#    print(e)

    # Normalize data for Scipy fitting
    xnorm = x - centralwl
    xnorm *= 1000
    ynorm = y - y.min()
    ynorm /= fluxrange
    enorm = e / fluxrange

    # Fit a parabola to the normalized data
    if par_fit:
        parData = fitParabola(xnorm, ynorm, enorm, centralwl, radvel,
                              verbose=False)
        results['line_par'] = parData['parrestframeline']
        results['err_par'] = parData['vel_err_par']

    # Fit a Gaussian to the normalized data
    if gauss_fit:
        gaussData = fitGaussian(xnorm, ynorm, enorm, centralwl, radvel,
                                continuum, verbose=False)
        results['line_gauss'] = gaussData['gaussrestframeline']
        results['err_gauss'] = gaussData['vel_err_gauss']

    # Fit a constrained parabola to the normalized data
    if spar_fit:
        sparData = fitSimpleParabola(xnorm, ynorm, enorm, centralwl, radvel,
                                     verbose=False)
        results['line_spar'] = sparData['sparrestframeline']
        results['err_spar'] = sparData['vel_err_spar']

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=lowerwllim, right=upperwllim)
        ax.set_xlim(left=vac_wl[lowercont], right=vac_wl[uppercont])
        ax.errorbar(vac_wl[lowercont:uppercont],
                    flux[lowercont:uppercont],
                    yerr=err[lowercont:uppercont],
                    color='blue', marker='.', linestyle='')
        ax.vlines(shiftedlinewl, color='crimson',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=continuum,
                  linestyle='-', label='Line expected position')
        if par_fit:
            ax.vlines(parData['parcenterwl'], color='green',
                      ymin=flux[lowerguess:upperguess].min(),
                      ymax=continuum,
                      linestyle='--', label='Parabola center')
            ax.plot((xnorm / 1000) + centralwl,
                    (parabola(xnorm, *parData['popt_par']) * fluxrange)
                    + y.min(),
                    color='magenta', linestyle='--',
                    label='Parabola fit')
        if gauss_fit:
            ax.vlines(gaussData['gausscenterwl'], color='blue',
                      ymin=flux[lowerguess:upperguess].min(),
                      ymax=continuum,
                      linestyle=':', label='Gaussian center')
            ax.plot((xnorm / 1000) + centralwl,
                    ((gaussian(xnorm, *gaussData['popt_gauss']) + continuum)
                    * fluxrange) + y.min(),
                    color='black', linestyle=':',
                    label='Gaussian fit')
        if spar_fit:
            ax.vlines(sparData['sparcenterwl'], color='orange',
                      ymin=flux[lowerguess:upperguess].min(),
                      ymax=continuum,
                      linestyle='-.', label='Constrained parabola center')
            ax.plot((xnorm / 1000) + centralwl,
                    (simpleparabola(xnorm, *sparData['popt_spar']) * fluxrange)
                    + y.min(),
                    color='purple', linestyle='-.',
                    label='Constrained parabola fit')

        ax.axvspan(xmin=lowerwllim, xmax=upperwllim,
                   color='grey', alpha=0.25)
        ax.legend()
        plt.show()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.errorbar(xnorm, ynorm, yerr=enorm,
                     color='blue', marker='o', linestyle='')
        if par_fit:
            ax2.plot(xnorm, parabola(xnorm, *parData['popt_par']), color='red',
                     label='parabola')
        if gauss_fit:
            ax2.plot(xnorm, gaussian(xnorm, *gaussData['popt_gauss'])
                     + continuum, color='green', linestyle='--',
                     label='Gaussian')
        if spar_fit:
            ax2.plot(xnorm, simpleparabola(xnorm, *sparData['popt_spar']),
                     color='black', linestyle=':', label='Const. parabola')
        ax2.legend()
        plt.show()

#    print("--------------------------------")

    return results


def measurepairsep(linepair, vac_wl, flux, err, radvel, FITSfile, plot=False):
    """Return the distance between a pair of lines

    """

    # Create dictionary to store results
    results = {}

    global unfittablelines
    params = (vac_wl, flux, err, radvel)
    line1 = linefind(linepair[0], *params,
                     plot=plot, velsep=5000, pixrange=3,
                     par_fit=False, gauss_fit=True, spar_fit=False)
    line2 = linefind(linepair[1], *params,
                     plot=plot, velsep=5000, pixrange=3,
                     par_fit=False, gauss_fit=True, spar_fit=False)

    if line1 is None:
        unfittablelines += 1
    if line2 is None:
        unfittablelines += 1

    if (line1 and line2) is not None:

        if 'line_par' in (line1 and line2):
            parveldiff = abs(vcl.getvelseparation(line1['line_par'],
                                                  line2['line_par']))
            err_par = np.sqrt((line1['err_par'])**2 +
                              (line2['err_par'])**2)
            results['parveldiff'] = parveldiff
            results['pardifferr'] = err_par

        if 'line_gauss' in (line1 and line2):
            gaussveldiff = abs(vcl.getvelseparation(line1['line_gauss'],
                                                    line2['line_gauss']))
            err_gauss = np.sqrt((line1['err_gauss'])**2 +
                                (line2['err_gauss'])**2)
            results['gaussveldiff'] = gaussveldiff
            results['gaussdifferr'] = err_gauss

        if 'line_spar' in (line1 and line2):
            sparveldiff = abs(vcl.getvelseparation(line1['line_spar'],
                                                   line2['line_spar']))
            err_spar = np.sqrt((line1['err_spar'])**2 +
                               (line2['err_spar'])**2)
            results['sparveldiff'] = sparveldiff
            results['spardifferr'] = err_spar

        return results
    else:
        return None


def searchFITSfile(FITSfile, pairlist):
    """Measure line pair separations in given file with given list

    """

    data = vcl.readHARPSfile(str(FITSfile), radvel=True, date_obs=True,
                             hdnum=True)
    vac_wl = vcl.air2vacESO(data['w']) / 10 #Convert from Angstroms to nm AFTER
                                            #converting to vacuum wavelengths
    flux = data['f']
    err = data['e']
    radvel = data['radvel']
    hdnum = data['hdnum']

    params = (vac_wl, flux, err, radvel)

    #foundlinepos = linefind(line1, *params, plot=True)

    measuredseps = []
    for linepair in pairlist:
        msepdict = measurepairsep(linepair, *params, FITSfile, plot=False)
        if msepdict != None:
            measuredseps.append(msepdict)
        else:
            measuredseps.append(math.nan)
    for item, linepair in zip(measuredseps, pairlist):
        if item != None:
            pass
#            print("{}, {}: measured separation {:.3f} m/s".format(
#                    *linepair, item['gaussveldiff']))
        else:
            print("Couldn't measure separation for {}, {}".format(*linepair))

    file_parent = FITSfile.parent
    date_str = data['date_obs'].strftime('%Y%m%dT%H%M%S') + '.csv'
    csvfilePath = file_parent / date_str
    with open(csvfilePath, 'w', newline='') as csvfile:
        fieldnames = ['date_obs']
        for i in range(len(pairlist)):
            fieldnames.append('pair_{}_vel_diff'.format(i))
            fieldnames.append('pair_{}_err'.format(i))
        print(fieldnames)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        row = [data['date_obs']]
        for msepdict in measuredseps:
            row.append(msepdict['gaussveldiff'])
            row.append(msepdict['gaussdifferr'])
        print(row)
        rowdict = dict(zip(fieldnames, row))
        print(rowdict)
#        writer.writerow(rowdict)

    return rowdict
#    return measuredseps


def plotstarseparations(mseps):
    """
    """

    fig_par = plt.figure(figsize=(8, 6))
    fig_gauss = plt.figure(figsize=(8, 6))
    for i in range(len(mseps[0])):
            ax_par = fig_par.add_subplot(5, 7, i+1)
            ax_gauss = fig_gauss.add_subplot(5, 7, i+1)
            parhistlist = []
            gausshistlist = []
            for seplist in mseps:
                parhistlist.append(seplist[i][0])
                gausshistlist.append(seplist[i][1])
            parhistlist = np.array(parhistlist)
            parhistlist -= np.median(parhistlist)
            gausshistlist = np.array(gausshistlist)
            gausshistlist -= np.median(gausshistlist)
            parmax = parhistlist.max()
            parmin = parhistlist.min()
#            print(min, max)
            if parmax > abs(parmin):
                parlim = parmax
            else:
                parlim = abs(parmin)
#            print(lim)
            gaussmax = parhistlist.max()
            gaussmin = parhistlist.min()
#            print(min, max)
            if gaussmax > abs(gaussmin):
                gausslim = gaussmax
            else:
                gausslim = abs(gaussmin)
#            ax.hist(parhistlist, range=(-1.05*lim, 1.05*lim))
            astrohist(parhistlist, ax=ax_par, bins=10,
                      range=(-1.05*parlim, 1.05*parlim))
            astrohist(gausshistlist, ax=ax_gauss, bins=10,
                      range=(-1.05*gausslim, 1.05*gausslim))
#    outfile
#    outfile = '/Users/dberke/Pictures/.png'.format(i)
#    fig.savefig(outfile, format='png')
    plt.tight_layout(pad=0.5)
    plt.show()


def plot_line_comparisons(mseps, linepairs):
    """
    """

    for i, linepair in zip(range(len(mseps[0])), linepairs):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                      format(linepair[0],
                      linepair[1]), fontsize=18)
        ax.set_ylabel('Observation number', fontsize=18)
        parlist = []
        parerr = []
        gausslist = []
        gausserr = []
        sparlist = []
        sparerr = []
        for seplist in mseps:
                parlist.append(seplist[i]['parveldiff'])
                parerr.append(seplist[i]['pardifferr'])
                gausslist.append(seplist[i]['gaussveldiff'])
                gausserr.append(seplist[i]['gaussdifferr'])
                sparlist.append(seplist[i]['sparveldiff'])
                sparerr.append(seplist[i]['spardifferr'])
        parlist = np.array(parlist)
        parlist -= np.median(parlist)
        gausslist = np.array(gausslist)
        gausslist -= np.median(gausslist)
        sparlist = np.array(sparlist)
        sparlist -= np.median(sparlist)
        par_rms = np.sqrt(np.mean(np.square(parlist)))
        gauss_rms = np.sqrt(np.mean(np.square(gausslist)))
        spar_rms = np.sqrt(np.mean(np.square(sparlist)))
        y_par = np.arange(0, len(parlist), 1)
        y_gauss = np.arange(0.3, len(parlist)+0.3, 1)
        y_spar = np.arange(0.6, len(parlist)+0.6, 1)
        ymin, ymax = -5, len(parlist)+5
        lim = max(abs(parlist.min()), abs(gausslist.min()),
                  parlist.max(), gausslist.max())
        if lim == 0:
            lim = 10
        ax.vlines(x=0, ymin=ymin, ymax=ymax+2.3, color='black',
                  linestyle='--', alpha=1, zorder=1)
#        ax.set_xlim(left=-1.05*lim, right=1.05*lim)
        ax.set_xlim(left=-100, right=100)
        ax.set_ylim(bottom=ymin, top=ymax)
        mean_rms = np.mean([par_rms, gauss_rms, spar_rms])
        mean_rms = np.mean(gauss_rms)
        ax.axvspan(-1*mean_rms, mean_rms, color='gray', alpha=0.3)
#        ax.errorbar(parlist, y_par, xerr=parerr,
#                    color='DodgerBlue', marker='o', elinewidth=2, linestyle='',
#                    capsize=2, capthick=2, markersize=4,
#                    label='Parabola fit (RMS: {:.2f}, mean err: {:.2f})'.
#                    format(par_rms, np.mean(parerr)),
#                    zorder=2)
        ax.errorbar(gausslist, y_gauss, xerr=gausserr,
                    color='ForestGreen', marker='o', elinewidth=2,
                    linestyle='', capsize=2, capthick=2, markersize=4,
                    label='Gaussian fit (RMS: {:.1f}, mean err: {:.1f})'.
                    format(gauss_rms, np.mean(gausserr)),
                    zorder=3)
        ax.errorbar(0, len(gausslist)+1.3, xerr=gauss_rms,
                    color='FireBrick', marker='', elinewidth=4,
                    capsize=0, label='RMS')
        ax.errorbar(0, len(gausslist)+2.3, xerr=np.mean(gausserr),
                    color='DodgerBlue', marker='', elinewidth=4,
                    capsize=0, label='Mean error')
#        ax.errorbar(sparlist, y_spar, xerr=sparerr,
#                    color='FireBrick', marker='o', elinewidth=2, linestyle='',
#                    capsize=2, capthick=2, markersize=4,
#                    label='Const. parabola fit (RMS: {:.1f}, mean err: {:.1f})'.
#                    format(spar_rms, np.mean(sparerr)),
#                    zorder=2)

        ax.xaxis.set_minor_locator(ticks.AutoMinorLocator(5))
        ax.legend(framealpha=0.4, fontsize=18)
        plt.show()
        outfile = '/Users/dberke/Pictures/HD146233/Linepair{}.png'.\
                  format(i+1)
        fig.savefig(outfile, format='png')
        plt.close(fig)


def plot_as_func_of_date(mseps, linepairs, folded=False):
    """Plot separations as a function of date.

    """

    for i, linepair in zip(range(len(mseps[0])), linepairs):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('HD146233')
        ax.set_ylabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                      format(linepair[0],
                      linepair[1]), fontsize=18)
        xlabel = 'Date of observation'
        if folded:
            xlabel = 'Date of observation (folded by year)'
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylim(bottom=-200, top=200)


        gausslist = []
        gausserr = []
        datelist = []
        for seplist in mseps:
                gausslist.append(seplist[i]['gaussveldiff'])
                gausserr.append(seplist[i]['gaussdifferr'])
                datelist.append(seplist[i]['date_obs'])
        gausslist = np.array(gausslist)
        gausslist -= np.median(gausslist)

        if folded:
            for j in range(0, len(datelist), 1):
                datelist[j] = datelist[j].replace(year=2000)
            format_str = '%b'
            ax.set_xlim(left=dt.date(year=2000, month=1, day=1),
                         right=dt.date(year=2000, month=12, day=31))
        else:
            format_str = '%Y%m%d'

        ax.xaxis.set_major_locator(dates.AutoDateLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter(format_str))

        ax.errorbar(datelist, gausslist, yerr=gausserr,
                    markerfacecolor='Black', markeredgecolor='Black',
                    linestyle='', marker='o',
                    markersize=5, elinewidth=2, ecolor='Green',
                    capsize=2, capthick=2)
        fig.subplots_adjust(bottom=0.16, wspace=0.0, hspace=0.0)
        fig.autofmt_xdate(bottom=0.16, rotation=30, ha='right')
#        plt.show()
        outfile = '/Users/dberke/Pictures/HD146233/Linepair_{}_folded_date.png'.\
                 format(i+1)
        fig.savefig(outfile, format='png')
        plt.close(fig)


############

#pairlistfile = "/Users/dberke/code/GoldStandardLineList_vac_working.txt"
#pairlist = getpairlist(pairlistfile)

pairlist = [(443.9589, 444.1128), (450.0151, 450.3467), (459.9405, 460.329),
            (460.5846, 460.6877), (465.8889, 466.284), (473.3122, 473.378),
            (475.9448, 476.0601), (480.0073, 480.0747), (484.0896, 484.4496),
            (488.6794, 488.7696), (490.9102, 491.0754), (497.1304, 497.4489),
            (500.5115, 501.142), (506.8562, 507.4086), (507.3492, 507.4086),
            (513.2898, 513.8813), (514.8912, 515.3619), (524.851, 525.167),
            (537.5203, 538.1069), (554.4686, 554.5475), (563.551, 564.3),
            (571.3716, 571.9418), (579.4679, 579.9464), (579.5521, 579.9779),
            (580.8335, 581.0828), (593.1823, 593.6299), (595.4367, 595.8344),
            (600.4673, 601.022), (616.3002, 616.8146), (617.2213, 617.5042),
            (617.7065, 617.8498), (623.9045, 624.6193), (625.9833, 626.0427),
            (625.9833, 626.2831), (647.098, 647.7413)]


#pairlist = [(537.5203, 538.1069)]
#pairlist = [(579.4679, 579.9464)]
#pairlist = [(507.3498, 507.4086)]

baseDir = Path("/Volumes/External Storage/HARPS/")
global unfittablelines


line1 = 600.4673

#files = glob(os.path.join(baseDir, '4Vesta/*.fits')) # Vesta (6 files)
#files = glob(os.path.join(baseDir, 'HD126525/*.fits')) # G4 (133 files))
#files = glob(os.path.join(baseDir, 'HD208704/*.fits')) # G1 (17 files)
#files = glob(os.path.join(baseDir, 'HD138573/*.fits')) # G5
#files = glob(os.path.join(baseDir, 'HD146233/*.fits')) # G2 (151 files)
filepath = Path('/Users/dberke/HD146233')
filepath = baseDir / 'HD146233'
files = [file for file in filepath.glob('*.fits')] # 18 Sco, G2 (7 files)
files = [Path('/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits')]

results = []

num_file = 1
for infile in files:
    print('Processing file {} of {}.'.format(num_file, len(files)))
    unfittablelines = 0
    mseps = searchFITSfile(infile, pairlist)
    results.append(mseps)

    print('Found {} unfittable lines.'.format(unfittablelines))
    num_file += 1

print("#############")
print("{} files analyzed total.".format(len(files)))

file_parent = files[0].parent
target = file_parent.stem + '.csv'
csvfilePath = file_parent / target
with open(csvfilePath, 'w', newline='') as csvfile:
    fieldnames = ['date_obs']
    for i in range(len(pairlist)):
        fieldnames.append('pair_{}_vel_diff'.format(i))
        fieldnames.append('pair_{}_err'.format(i))
    print(fieldnames)
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    row = [data['date_obs']]
    for msepdict in measuredseps:
        row.append(msepdict['gaussveldiff'])
        row.append(msepdict['gaussdifferr'])
    print(row)
    rowdict = dict(zip(fieldnames, row))
    print(rowdict)
    writer.writerow(rowdict)

#plotstarseparations(results)
#plot_line_comparisons(results, pairlist)
#plot_as_func_of_date(results, pairlist, folded=True)