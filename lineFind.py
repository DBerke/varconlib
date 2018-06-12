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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
from scipy.optimize import curve_fit
from astropy.visualization import hist as astrohist
from glob import glob
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
    shiftedlinewl = vcl.lineshift(line, radvel)  # In nm here.
    print('Given radial velocity {} km/s, line {} should be at {}'.
          format(radvel, line, shiftedlinewl))
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
    print("fluxrange = {}".format(fluxrange))
    print(x)
    print(y)
    print(e)

    # Normalize data for Scipy fitting
    xnorm = x - centralwl
    xnorm *= 1000
    ynorm = y - y.min()
    ynorm /= fluxrange
    enorm = e / fluxrange

    # Fit a parabola to the line center
    popt_par, pcov_par = curve_fit(parabola, xnorm, ynorm, sigma=enorm,
                                   absolute_sigma=True)
    print('Covariance matrix for parabola:')
    print(pcov_par)
    perr_par = np.sqrt(np.diag(pcov_par))
    r_par = ynorm - parabola(xnorm, *popt_par)
    chisq_par = sum((r_par / enorm) ** 2)
    chisq_nu_par = chisq_par / 3  # nu = 7 - 4
    print('popt_par = {}'.format(popt_par))
    print('perr_par = {}'.format(perr_par))
    print(u'χ^2 (parabola) = {:.7f}'.format(chisq_par))
    print(u'χ_ν^2 (parabola) = {:.7f}'.format(chisq_nu_par))

    # Find curve_fit minimum analytically (line of symmetry = -b/2a)
    # Correct for fitting normalized data
    parcenterwl = (-1 * popt_par[1] / (2 * popt_par[2])) / 1000 + centralwl
    parcenterwl = popt_par[3] / 1000 + centralwl
    print('Parabola central wl: {:.6f}'.format(parcenterwl))

    # Find the error in the parabolic fit.
    delta_lambda = np.sqrt((perr_par[1]/1000/popt_par[1]/1000)**2 +
                           (2*perr_par[2]/1000/popt_par[2]/1000)**2)
    print('δλ/λ = {:.6e}'.format(delta_lambda))
    wl_err_par = parcenterwl * delta_lambda
    wl_err_par = perr_par[3] / 1000
#    wl_err_par =  parcenterwl * np.sqrt((perr_par[1]/1000/popt_par[1]/1000)**2 +
#                         (2*perr_par[2]/1000/popt_par[2]/1000)**2)

    print('δλ = {:.6e}'.format(wl_err_par))
    if chisq_nu_par > 1:
        wl_err_par *= math.sqrt(chisq_nu_par)
    print("1 stddev parabola = {:.6e} nm".format(wl_err_par))
    vel_err_par = vcl.getvelseparation(parcenterwl*1e-9,
                                       (parcenterwl+wl_err_par)*1e-9)
    print("1 stddev parabola velspace = {:.7f} m/s".format(vel_err_par))

    # Fit a Gaussian to the line center
    print("Continuum level is {}".format(continuum))
    popt_gauss, pcov_gauss = curve_fit(gaussian, xnorm, ynorm-continuum,
                                       p0=(-1*continuum, 0, 1e3),
                                       sigma=enorm,
                                       absolute_sigma=True)
    print("Covariance matrix for Gaussian:")
    print(pcov_gauss)
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    r_gauss = (ynorm - continuum) - gaussian(xnorm, *popt_gauss)
    chisq_gauss = sum((r_gauss / enorm) ** 2)
    chisq_nu_gauss = chisq_gauss / 4  # nu = 7 - 3
    print('popt_gauss = {}'.format(popt_gauss))
    print('perr_gauss = {}'.format(perr_gauss))
    print(u'χ^2 (Gaussian) = {:.7f}'.format(chisq_gauss))
    print(u'χ_ν^2 (Gaussian) = {:.7f}'.format(chisq_nu_gauss))
    wl_err_gauss = perr_gauss[1] / 1000

    # Find center of Gaussian &
    # correct for fitting normalized data
    gausscenterwl = popt_gauss[1] / 1000 + centralwl
    print('Gaussian central wl: {:.6f} nm'.format(gausscenterwl))

    if chisq_nu_gauss > 1:
        wl_err_gauss *= math.sqrt(chisq_nu_gauss)
    print("1 stddev Gaussian = {:.6e} nm".format(wl_err_gauss))
    vel_err_gauss = vcl.getvelseparation(gausscenterwl*1e-9,
                                         (gausscenterwl+wl_err_gauss)*1e-9)
    print("1 stddev Gaussian velspace = {:.7f} m/s".format(vel_err_gauss))

    # Fit constrained parabola
    popt_spar, pcov_spar = curve_fit(simpleparabola, xnorm, ynorm,
                                     sigma=enorm,
                                     absolute_sigma=True)
    print("Covariance matrix for constrained parabola:")
    print(pcov_spar)
    perr_spar = np.sqrt(np.diag(pcov_spar))
    r_spar = ynorm - simpleparabola(xnorm, *popt_spar)
    chisq_spar = sum((r_spar / enorm) ** 2)
    chisq_nu_spar = chisq_spar / 4  # vu = 7 - 3
    print('popt_spar = {}'.format(popt_spar))
    print('perr_spar = {}'.format(perr_spar))
    print(u'χ^2 (constrained parabola) = {:.7f}'.format(chisq_spar))
    print(u'χ_ν^2 (constrained parabola) = {:.7f}'.format(chisq_nu_spar))
    wl_err_spar = perr_spar[0] / 1000

    sparcenterwl = popt_spar[0] / 1000 + centralwl
    print("Constrained parabola central wl: {:.6f}".format(sparcenterwl))

    if chisq_nu_spar > 1:
        wl_err_spar *= math.sqrt(chisq_nu_spar)
    print("1 stddev constrained parabola = {:.6e} nm".format(wl_err_spar))
    vel_err_spar = vcl.getvelseparation(sparcenterwl*1e-9,
                                        (sparcenterwl+wl_err_spar)*1e-9)
    print("1 stddev constrained parabola velspace = {:.7f} m/s".
          format(vel_err_spar))

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
        ax.vlines(parcenterwl, color='green',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=continuum,
                  linestyle='--', label='Parabola center')
        ax.vlines(gausscenterwl, color='blue',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=continuum,
                  linestyle=':', label='Gaussian center')
        ax.vlines(sparcenterwl, color='orange',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=continuum,
                  linestyle='-.', label='Constrained parabola center')
        ax.axvspan(xmin=lowerwllim, xmax=upperwllim,
                   color='grey', alpha=0.25)

        ax.plot((xnorm/1000)+centralwl,
                (parabola(xnorm, *popt_par)*fluxrange)+y.min(),
                color='magenta', linestyle='--',
                label='Parabola fit')
        ax.plot((xnorm/1000)+centralwl,
                ((gaussian(xnorm, *popt_gauss)+continuum)*fluxrange)+y.min(),
                color='black', linestyle=':',
                label='Gaussian fit')
        ax.plot((xnorm/1000)+centralwl,
                (simpleparabola(xnorm, *popt_spar)*fluxrange)+y.min(),
                color='purple', linestyle='-.',
                label='Constrained parabola fit')
        ax.legend()

        plt.show()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.errorbar(xnorm, ynorm, yerr=enorm,
                     color='blue', marker='o', linestyle='')
        ax2.plot(xnorm, parabola(xnorm, *popt_par), color='red',
                 label='parabola')
        ax2.plot(xnorm, gaussian(xnorm, *popt_gauss)+continuum, color='green',
                 linestyle='--', label='Gaussian')
        ax2.plot(xnorm, simpleparabola(xnorm, *popt_spar), color='black',
                 linestyle=':', label='Const. parabola')
        ax2.legend()
        plt.show()

    print("Found line center at {} using parabola.".format(parcenterwl))
    parrestframeline = vcl.lineshift(parcenterwl, -1*radvel)
    print("Corrected for radial velocity: {}".format(parrestframeline))
    print("Found line center at {} using Gaussian.".format(gausscenterwl))
    gaussrestframeline = vcl.lineshift(gausscenterwl, -1*radvel)
    print("Corrected for radial velocity: {}".format(gaussrestframeline))
    print("Found line center at {} using constrained parabola.".
          format(sparcenterwl))
    sparrestframeline = vcl.lineshift(sparcenterwl, -1*radvel)
    print("Corrected for radial velocity: {}".format(sparrestframeline))
    print("--------------------------------")

    return {'line_par': parrestframeline, 'err_par': vel_err_par,
            'line_gauss': gaussrestframeline, 'err_gauss': vel_err_gauss,
            'line_spar': sparrestframeline, 'err_spar': vel_err_spar}


def measurepairsep(linepair, vac_wl, flux, err, radvel, FITSfile, plot=False):
    """Return the distance between a pair of lines

    """
    global unfittablelines
    params = (vac_wl, flux, err, radvel)
    line1 = linefind(linepair[0], *params,
                     plot=plot, velsep=5000, pixrange=3)
    line2 = linefind(linepair[1], *params,
                     plot=plot, velsep=5000, pixrange=3)

    if line1 is None:
        unfittablelines += 1
    if line2 is None:
        unfittablelines += 1

    if (line1 and line2) is not None:

        parveldiff = abs(vcl.getvelseparation(line1['line_par'],
                                              line2['line_par']))
        gaussveldiff = abs(vcl.getvelseparation(line1['line_gauss'],
                                                line2['line_gauss']))
        sparveldiff = abs(vcl.getvelseparation(line1['line_spar'],
                                               line2['line_spar']))

        err_par = np.sqrt((line1['err_par'])**2 + (line2['err_par'])**2)
        err_gauss = np.sqrt((line1['err_gauss'])**2 + (line2['err_gauss'])**2)
        err_spar = np.sqrt((line1['err_spar'])**2 + (line2['err_spar'])**2)

        return {'parveldiff': parveldiff, 'pardifferr': err_par,
                'gaussveldiff': gaussveldiff, 'gaussdifferr': err_gauss,
                'sparveldiff': sparveldiff, 'spardifferr': err_spar}
    else:
        return None


def searchFITSfile(FITSfile, pairlist):
    """Measure line pair separations in given file with given list

    """

    data = vcl.readHARPSfile(FITSfile, radvel=True, date_obs=True)
    vac_wl = vcl.air2vacESO(data['w']) / 10 #Convert from Angstroms to nm AFTER
                                            #converting to vacuum wavelengths
    flux = data['f']
    err = data['e']
    radvel = data['radvel']

    params = (vac_wl, flux, err, radvel)

    #foundlinepos = linefind(line1, *params, plot=True)

    measuredseps = []
    for linepair in pairlist:
        msepdict = measurepairsep(linepair, *params, FITSfile, plot=False)
        msepdict['date_obs'] = data['date_obs']
        if msepdict != None:
            measuredseps.append(msepdict)
        else:
            measuredseps.append(math.nan)
    for item, linepair in zip(measuredseps, pairlist):
        if item != None:
            print("{}, {}: measured separation {:.3f}/{:.3f} m/s".format(
                    *linepair, item['parveldiff'], item['gaussveldiff']))
        else:
            print("Couldn't measure separation for {}, {}".format(*linepair))

    return measuredseps


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


def plot_as_func_of_date(mseps, linepairs):
    """Plot separations as a function of date.

    """

    for i, linepair in zip(range(len(mseps[0])), linepairs):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                      format(linepair[0],
                      linepair[1]), fontsize=18)
        ax.set_xlabel('Date of observation', fontsize=18)
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

        ax.errorbar(datelist, gausslist, yerr=gausserr, color='green',
                    linestyle='', marker='o', markersize=4, elinewidth=2,
                    capsize=2, capthick=2)
        fig.subplots_adjust(bottom=0.26, wspace=0.0, hspace=0.0)
        fig.autofmt_xdate(bottom=0.26, rotation=30, ha='right')
        plt.show()
        outfile = '/Users/dberke/Pictures/HD146233/Linepair_{}_date.png'.\
                  format(i+1)
        fig.savefig(outfile, format='png')
        plt.close(fig)


############

pairlistfile = "/Users/dberke/code/GoldStandardLineList_vac_working.txt"
pairlist = getpairlist(pairlistfile)

#pairlist = [(537.5203, 538.1069)]
#pairlist = [(579.4679, 579.9464)]
#pairlist = [(507.3498, 507.4086)]

baseDir = "/Volumes/External Storage/HARPS/"
global unfittablelines


line1 = 600.4673

#files = glob(os.path.join(baseDir, '4Vesta/*.fits')) # Vesta (6 files)
files = glob(os.path.join(baseDir, 'HD126525/*.fits')) # G4 (133 files))
#files = glob(os.path.join(baseDir, 'HD208704/*.fits')) # G1 (17 files)
#files = glob(os.path.join(baseDir, 'HD138573/*.fits')) # G5
files = glob(os.path.join(baseDir, 'HD146233/*.fits')) # G2 (151 files)
#files = glob('/Users/dberke/HD146233/*.fits') # 18 Sco, G2 (7 files)
#files = ['/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits']

results = []

for infile in files:
    unfittablelines = 0
    mseps = searchFITSfile(infile, pairlist)
    results.append(mseps)

    print('Found {} unfittable lines.'.format(unfittablelines))


print("#############")
print("{} files analyzed total.".format(len(files)))
#plotstarseparations(results)
#plot_line_comparisons(results, pairlist)
plot_as_func_of_date(results, pairlist)
