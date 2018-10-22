#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke
"""

# Module to contain functions potentially useful across multiple programs

import numpy as np
import datetime as dt
import pandas as pd
from astropy.io import fits
from astropy.constants import c
from scipy.optimize import curve_fit
from math import sqrt, log
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Some generic information useful in different scripts

pairlist = [('443.9589', '444.1128'), ('450.0151', '450.3467'),
            ('459.9405', '460.3290'), ('460.5846', '460.6877'),
            ('465.8889', '466.2840'), ('473.3122', '473.3780'),
            ('475.9448', '476.0601'), ('480.0073', '480.0747'),
            ('484.0896', '484.4496'), ('488.6794', '488.7696'),
            ('490.9102', '491.0754'), ('497.1304', '497.4489'),
            ('500.5115', '501.1420'), ('506.8562', '507.4086'),
            ('507.3492', '507.4086'), ('513.2898', '513.8813'),
            ('514.8912', '515.3619'), ('524.8510', '525.1670'),
            ('537.5203', '538.1069'), ('554.4686', '554.5475'),
            ('563.5510', '564.3000'), ('571.3716', '571.9418'),
            ('579.4679', '579.9464'), ('579.5521', '579.9779'),
            ('580.8335', '581.0828'), ('593.1823', '593.6299'),
            ('595.4367', '595.8344'), ('600.4673', '601.0220'),
            ('616.3002', '616.8146'), ('617.2213', '617.5042'),
            ('617.7065', '617.8498'), ('623.9045', '624.6193'),
            ('625.9833', '626.0427'), ('625.9833', '626.2831'),
            ('647.0980', '647.7413')]

elemdict = {('443.9589', '444.1128'): 26,
            ('450.0151', '450.3467'): 25,
            ('459.9405', '460.3290'): 26,
            ('460.5846', '460.6877'): 26,
            ('465.8889', '466.2840'): 26,
            ('473.3122', '473.3780'): 28,
            ('475.9448', '476.0601'): 22,
            ('480.0073', '480.0747'): 26,
            ('484.0896', '484.4496'): 26,
            ('488.6794', '488.7696'): 26,
            ('490.9102', '491.0754'): 26,
            ('497.1304', '497.4489'): 26,
            ('500.5115', '501.1420'): 28,
            ('506.8562', '507.4086'): 26,
            ('507.3492', '507.4086'): 26,
            ('513.2898', '513.8813'): 26,
            ('514.8912', '515.3619'): 22,
            ('524.8510', '525.1670'): 26,
            ('537.5203', '538.1069'): 26,
            ('554.4686', '554.5475'): 26,
            ('563.5510', '564.3000'): 26,
            ('571.3716', '571.9418'): 26,
            ('579.4679', '579.9464'): 14,
            ('579.5521', '579.9779'): 26,
            ('580.8335', '581.0828'): 26,
            ('593.1823', '593.6299'): 26,
            ('595.4367', '595.8344'): 26,
            ('600.4673', '601.0220'): 26,
            ('616.3002', '616.8146'): 20,
            ('617.2213', '617.5042'): 26,
            ('617.7065', '617.8498'): 28,
            ('623.9045', '624.6193'): 14,
            ('625.9833', '626.0427'): 22,
            ('625.9833', '626.2831'): 22,
            ('647.0980', '647.7413'): 26}

# Lines known to be compromised by telluric lines.
badlines = frozenset(['506.8562', '507.4086', '593.1823', '593.6299',
                      '595.4367', '595.8344', '600.4673', '601.0220',
                      '647.0980', '647.7413'])

# Spectral format files for HARPS blue and red CCDs.
blueCCDpath = Path('/Users/dberke/code/data/HARPS_CCD_blue.csv')
redCCDpath = Path('/Users/dberke/code/data/HARPS_CCD_red.csv')


# Functions
def map_spectral_order(order):
    """
    Converts from HARPS' internal 0-71 order numbers to those in the HARPS
    spectral format (89-114, 116-161).

    Parameters
    ----------
    order : int
        An order number in the range [0, 71]
    """
    if not type(order) is int:
        raise ValueError("order parameter must be an integer in [0, 71]")
    if 0 <= order <= 45:
        new_order = 161 - order
        return new_order
    elif 46 <= order <= 71:
        new_order = 160 - order
        return new_order
    else:
        raise ValueError("Given order not in range [0, 71].")


def readHARPSfile(FITSfile, obj=False, wavelenmin=False, date_obs=False,
                  spec_bin=False, med_snr=False, hdnum=False, radvel=False,
                  coeffs=False):
    """Read a HARPS FITS file and return a dictionary of information.

    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    obj : bool, Default: False
        If *True*, the output will contain the contents of the OBJECT FITS
        header card.
    wavelenmin : bool, Default: False
        If *True*, the output will contain the contents of the WAVELMIN FITS
        header card.
    date_obs : bool, Default: False
        If *True*, the output will contain the contents of the DATE-OBS FITS
        header card.
    spec_bin : bool, Default: False
        If *True*, the output will contain the contents of the SPEC_BIN FITS
        header card.
    med_snr : bool, Default: False
        If *True*, the output will contain the contents of the SNR FITS header
        card.
    hdnum : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        HDNUM FITS header card. (Added to unify object identifiers across all
        stars, some of which were occasionally identified by things other than
        HD number.)
    radvel : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        RADVEL FITS header card. (Added to unify the radial velocity for each
        star, as a small minority of stars had different radial velocity
        information in their HIERARCH ESO TEL TAFG RADVEL header cards.)
    coeffs : bool, Default: False
        If *True*, the output will contain the contents of the various
        *ESO DRS CAL TH COEFF LLX* header cards, where *X* ranges from 0 to
        287.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:

        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            The estimated error array (HARPS returns no error array by
            default).

        Optionally
        ==========
        obj : str
            The object name from the 'OBJECT' flag.
        wlmin : float
            The minimum wavelength.
        date_obs : datetime object
            The date the file was observed.
        spec_bin : float
            The wavelength bin size.
        med_snr : float
            The median SNR of the flux array.
        hd_num : str
            The HD identifier of the star in the format "HDxxxxxx".
        radvel : float
            The radial velocity of the star in km/s.
        If the `coeffs` keyword argument is *True*, there will be 288 entries
        of the form "ESO DRS CAL TH COEFF LLX": *value*, where X will range
        from 0 to 287.
    """

    result = {}
    with fits.open(FITSfile) as hdulist:
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        data = hdulist[1].data
        w = data.WAVE[0]
        gain = header0['GAIN']
        # Multiply by the gain to convert from ADUs to photoelectrons
        f = data.FLUX[0] * gain
        e = 1.e6 * np.absolute(f)
        # Construct an error array by taking the square root of each flux value
        try:
            # First assume no negative flux values and use Numpy array
            # magic to speed up the process.
            e = np.sqrt(f)
        except ValueError:
            # If that raises an error, do it element-by-element.
            for i in np.arange(0, len(f), 1):
                if (f[i] > 0.0):
                    e[i] = np.sqrt(f[i])
        result['w'] = w
        result['f'] = f
        result['e'] = e
        if obj:
            result['obj'] = header1['OBJECT']
        if wavelenmin:
            result['wavelmin'] = header0['WAVELMIN']
        if date_obs:
            result['date_obs'] = dt.datetime.strptime(header0['DATE-OBS'],
                                                      '%Y-%m-%dT%H:%M:%S.%f')
        if spec_bin:
            result['spec_bin'] = header0['SPEC_BIN']
        if med_snr:
            result['med_snr'] = header0['SNR']
        if hdnum:
            result['hdnum'] = header0['HDNUM']
        if radvel:
            result['radvel'] = header0['RADVEL']

        # If the coeffs keyword is given, returna all 288 wavelength solution
        # coefficients.
        if coeffs:
            for i in range(0, 288, 1):
                key_string = 'ESO DRS CAL TH COEFF LL{0}'.format(str(i))
                result[key_string] = header0[key_string]

    return result


def readESPRESSOfile(ESPfile):
    """Read an ESPRESSO file and return a dictionary of information

    ESPfile: a path to the ESPRESSO file to be read

    output: a dictionary containing the following information:
        obj: the name from the OBJECT card
        w: the wavelength array
        f: the flux array
        e: the error array
    """
    with fits.open(ESPfile) as hdulist:
        data = hdulist[1].data
        obj = hdulist[0].header['OBJECT']
        w = data['wavelength']
        f = data['flux']
        e = data['error']
    return {'obj': obj, 'w': w, 'f': f, 'e': e}


def air_indexEdlen53(l, t=15., p=760.):
    """Return the index of refraction of air at given temp, pressures, and wl (A)

    l: vacuum wavelength in Angstroms
    t: temperature (don't use)
    p: pressure (don't use)

    Formula is Edlen 1953, provided directly by ESO
    """
    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t)\
        * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n


def vac2airESO(ll):
    """Return a vacuum wavlength from an air wavelength (A) using Edlen 1953

    ll: air wavlength in Angstroms

    """
    llair = ll/air_indexEdlen53(ll)
    return llair


def air2vacESO(air_arr):
    """Take an array of air wls (A) and return an array of vacum wls

    Parameters
    ----------
    air_arr: array-like
        A list of wavelengths in air, in Angstroms.

    Returns
    -------
    array
        An array of wavelengths in vacuum, in Angstroms.
    """
    if not type(air_arr) == np.ndarray:
        air_arr = np.array(air_arr)

    tolerance = 1e-12

    vac = []

    for i in range(0, len(air_arr)):
        newwl = air_arr[i]
        oldwl = 0.
        while abs(oldwl - newwl) > tolerance:
            oldwl = newwl
            n = air_indexEdlen53(newwl)
            newwl = air_arr[i] * n

        vac.append(round(newwl, 2))
    vac_arr = np.array(vac)

    return vac_arr


def vac2airMorton00(wl_vac):
    """Take an input vacuum wavelength in Angstroms and return the air wavelength.

    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    from Morton (2000, ApJ. Suppl., 130, 403) (IAU standard)
    """
    s = 1e4 / wl_vac
    n = 1 + 0.0000834254 + (0.02406147 / (130 - s**2)) +\
        (0.00015998 / (38.9 - s**2))
    return wl_vac / n


def air2vacMortonIAU(wl_air):
    """Take an input air wavelength in Angstroms and return the vacuum wavelength.

    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    """
    s = 1e4 / wl_air
    n = 1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s**2))\
        + (0.0001599740894897 / (38.92568793293 - s**2))
    return wl_air * n


def index2wavelength(index, step, min_wl):
    """Return the wavelength associated with an index.

    index: index position of the spectrum list
    step: the step in wavelength per index, in nm
    min_wl: the minimum wavelength of the spectrum, in nm
    """
    return round((step * index + min_wl), 2)


def wavelength2index(wl_arr, wl, reverse=False):
    """Find the index in a list associated with a given wavelength

    wl_arr: an iterable object of wavelengths, *in increasing order*
    wl: the wavelength to search for
    reverse: a Boolean for if the wavelength array is listed from large to
             small. Will first re

    returns: the index for which the wavelength is closest to the given
    """
    length = len(wl_arr)
    for i in range(0, length, 1):
        # First find the index for which the value is greater than the given wl
        if wl_arr[i] >= wl:
            # Then check if it's closest to this index or the previous one
            # Note that the way it's set up it should always be
            # wl_arr[i-1] <= wl <= wl_arr[i]
            if wl_arr[i] - wl > wl - wl_arr[i-1]:
                return i-1
            else:
                return i

    print("Couldn't find the given wavelength: {}".format(wl))
    raise ValueError


def lineshift(line, radvel):
    """Find the new position of a line given the radial velocity of a star

    line: line position. Can be nm or Angstroms, will return in same units
    radvel: radial velocity in km/s

    returns: the new line position
    """
    return ((radvel * 1000 / c.value) * line) + line


def getwlseparation(v, wl):
    """Return wavelength separation for a given velocity separation at a point

    v: the velocity separation. Should be in m/s
    wl: the wavelength at which the function should be evaluated, since
        it's also a function of wavelength. Returned in whatever units it's
        given in.

    """
    return (v * wl) / c.value


def getvelseparation(wl1, wl2):
    """Return velocity separation for a pair of points in wavelength space

    wl1 & wl2: wavelengths to get the velocity separation between in m/s.
               Should be in meters.
    """
    return (wl2 - wl1) * c.value / ((wl1 + wl2) / 2)


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


def fitGaussian(xnorm, ynorm, enorm, centralwl, radvel, continuum, linebottom,
                fluxrange, verbose=False):
    """
    Fit a Gaussian to the given data

    Parameters
    ----------
    xnorm : array_like
        An array of x-values (wavelength), normalized from -0.03 to 0.03.
    ynorm : array_like
        An array of y-values (photon counts) normalized from 0 to 1.
    enorm : array_like
        An array of error values for the y-values, normalized the same way as
        for `ynorm`.
    centralwl : float
        The wavelength of the pixel with the lowest flux value in the
        absorption line.
    radvel : float
        The radial velocity of the source in km/s, to the nearest tenth.
    continuum : float
        The flux value of the highest pixel within 20 km/s of the pixel with
        wavelength given by `centralwl`.
    linebottom : float
        The flux of the lowest pixel in the feature (i.e., the pixel at
        `centralwl`).
    fluxrange : float
        The (unnormalized) flux range between the highest pixel in the
        wavelength range selected (± 3 pixels around `centralwl`) and the
        lowest (given by `linebottom`).
    verbose : bool. Default: False
        If *True*, the function will print out diagnostic info on the process.

    Returns
    -------
    dict
        Returns a dictionary containing information about and relevant to
        the fit found.
    """

    # Fit a Gaussian to the line center
    linedepth = continuum - linebottom
    neg_linedepth = -1 * linedepth
    gauss_params = (neg_linedepth, 0, 1e2)
    try:
        popt_gauss, pcov_gauss = curve_fit(gaussian, xnorm,
                                           ynorm-continuum+linebottom,
                                           p0=gauss_params, sigma=enorm,
                                           absolute_sigma=True)
    except RuntimeError:
#        print(continuum)
#        print(linebottom)
#        print(linedepth)
#        print(neg_linedepth)
#        print(gauss_params)
#        fig = plt.figure(figsize=(8, 8))
#        ax = fig.add_subplot(1, 1, 1)
#        ax.errorbar(xnorm, ynorm, yerr=enorm,
#                    color='blue', marker='o', linestyle='')
#        ax.plot(xnorm, (gaussian(xnorm, *gauss_params)), color='Black')
#        outfile = Path('/Users/dberke/Pictures/debug.png')
#        fig.savefig(str(outfile))
#        plt.close(fig)
        raise

    # Get the errors in the fitted parameters from the covariance matrix
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    r_gauss = (ynorm - continuum + linebottom) - gaussian(xnorm, *popt_gauss)
    chisq_gauss = sum((r_gauss / enorm) ** 2)
    chisq_nu_gauss = chisq_gauss / 4  # nu = 7 - 3

    # Find center of Gaussian &
    # correct for fitting normalized data
    gausscenterwl = popt_gauss[1] / 1000 + centralwl
    wl_err_gauss = perr_gauss[1] / 1000

    if chisq_nu_gauss > 1:
        wl_err_gauss *= sqrt(chisq_nu_gauss)

    # Multiply by 1e-9 to get nm to m for getvelseparation which requires m
    vel_err_gauss = getvelseparation(gausscenterwl*1e-9,
                                     (gausscenterwl+wl_err_gauss)*1e-9)
    # Shift line to stellar rest frame
    gaussrestframeline = lineshift(gausscenterwl, -1*radvel)

    # Get the width (sigma) of the Gaussian
    gauss_sigma = abs(popt_gauss[2] / 1000)
    gauss_sigma_err = perr_gauss[2] / 1000

    # Get the full width at half maximum (approximately 2.355 * sigma)
    gauss_fwhm = 2 * sqrt(2 * log(2)) * gauss_sigma
    gauss_fwhm_err = 2 * sqrt(2 * log(2)) * gauss_sigma_err

    # Convert sigma and FWHM to velocity space
    sigma_vel = getvelseparation(gausscenterwl*1e-9,
                                 (gausscenterwl+gauss_sigma)*1e-9)
    sigma_vel_err = getvelseparation(gausscenterwl*1e-9,
                                     (gausscenterwl+gauss_sigma_err)*1e-9)

    fwhm_vel = getvelseparation(gausscenterwl*1e-9,
                                (gausscenterwl+gauss_fwhm)*1e-9)
    fwhm_vel_err = getvelseparation(gausscenterwl*1e-9,
                                    (gausscenterwl+gauss_fwhm_err)*1e-9)

    # Get the amplitude of the Gaussian
    amp = popt_gauss[0]
    amp_err = perr_gauss[0]

    if verbose:
        print('-----------')
        print("Continuum level = {}".format(continuum))
        print('Depth of line = {}'.format(continuum - linebottom))
        print('fluxrange = {}'.format(fluxrange))
        print("Covariance matrix for Gaussian:")
        print(pcov_gauss)
        print('popt_gauss = {}'.format(popt_gauss))
        print('perr_gauss = {}'.format(perr_gauss))
        print(u'χ^2 (Gaussian) = {:.7f}'.format(chisq_gauss))
        print(u'χ_ν^2 (Gaussian) = {:.7f}'.format(chisq_nu_gauss))
        print('Gaussian central wl: {:.6f} nm'.format(gausscenterwl))
        print("1 stddev Gaussian = {:.6e} nm".format(wl_err_gauss))
        print("1 stddev Gaussian velspace = {:.7f} m/s".format(vel_err_gauss))
        print('1 sigma = {:.6f} nm'.format(gauss_sigma))
        print('1 sigma velspace = {:.7f} m/s'.format(sigma_vel))
        print('FWHM = {:.6f}'.format(gauss_fwhm))
        print('FWHM velspace = {:.7f} m/s'.format(fwhm_vel))
        print('Gaussian amplitude = {:.6f} photons'.format(amp))
        print('Gaussian amp err = {:.6f} photons'.format(amp_err))
        print("Found line center at {:.6f} nm.".format(gausscenterwl))
        print("Corrected for rad vel: {:.6f} nm".format(gaussrestframeline))

    return {'restframe_line_gauss': gaussrestframeline,
            'vel_err_gauss': vel_err_gauss,
            'amplitude_gauss': amp,
            'amplitude_err_gauss': amp_err,
            'width_gauss': sigma_vel,
            'width_err_gauss': sigma_vel_err,
            'fwhm_gauss': fwhm_vel,
            'fwhm_gauss_err': fwhm_vel_err,
            'chisq_nu_gauss': chisq_nu_gauss,
            'gausscenterwl': gausscenterwl,
            'popt_gauss': popt_gauss}


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
        wl_err_par *= sqrt(chisq_nu_par)
    vel_err_par = getvelseparation(parcenterwl*1e-9,
                                   (parcenterwl+wl_err_par)*1e-9)

    # Shift to stellar rest frame by correcting radial velocity.
    parrestframeline = lineshift(parcenterwl, -1*radvel)

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
        wl_err_spar *= sqrt(chisq_nu_spar)

    vel_err_spar = getvelseparation(sparcenterwl*1e-9,
                                    (sparcenterwl+wl_err_spar)*1e-9)
    # Convert to restframe of star
    sparrestframeline = lineshift(sparcenterwl, -1*radvel)

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


def linefind(line, vac_wl, flux, err, radvel, filename, starname,
             pixrange=3, velsep=5000, plot=True, par_fit=False,
             gauss_fit=False, spar_fit=False, plot_dir=None, date_obs=None,
             save_arrays=False):
    """
    Find a given line in a HARPS spectrum after correcting for the star's
    radial velocity.

    Parameters
    ----------
    line : string
        The wavelength of the line to look for, in nanometers, (*vacuum*
        wavelength).
    vac_wl : array_like
        The wavelength array of the spectrum to search (in vacuum wavelengths).
    flux : array_like
        The flux array of the spectrum to search.
    err : array_like
        The error array of the spectrum to search.
    radvel : float
        The radial velocity of the star in km/s (to nearest tenth is fine).
    filename : Path object
        The name of the file being analyzed, without suffix. Used to create
        the directory to store information related to that specific file.
        Example: ADP.2014-09-16T11:06:39.660
    starname : str
        The string identifier used for the star being analyzed, e.g. HD146233.
    pixrange : int
        The number of pixels to search either side of the main wavelength given
        as `line`.
    velsep : int or float
        The range in m/s in velocity space around the central wavelength given
        as `line` to search for the deepest point.

    Optional
    --------
    plot : bool. Default: True
        If *True*, save two plots per function used to fit the line: one of the
        area within ± velsep m/s around the line, and another showing the
        normalized pixels from the line core.
    par_fit : bool. Default: False
        If *True*, fit the line with a parabola.
    gauss_fit : bool. Default: False
        If *True, fit the line with a Gaussian.
    spar_fit : bool. Default: True
        If *True*, fit the line with a parabola constrained to translate on the
        x-axis only.
    plot_dir : string. Default: *None*
        A directory to store output plots in. Will have no effect if plot is
        *False*. If left as *None* will show the plots instead of saving them.
    date_obs : datetime: Default: *None*
        A datetime object representing the date the observation was taken, to
        be used as part of the output filename. Has no effect if plot_dir is
        not defined.
    save_arrays : bool. Default: False
        If *True*, write out the normalized arrays and information necessary to
        reconstruct a fit from them in a CSV file. If *True*, `plot_dir` should
        also be given.

    Returns
    -------
    dict
        A dictionary containing the results of the various fitting actions
        performed. Results from different fitting functions have different
        name prefixes to prevent duplication.
    """

    # Create a dictionary to store the results
    results = {}

    radvel = float(radvel)
    # Figure out location of line given radial velocity of the star (in km/s)
    shiftedlinewl = lineshift(float(line), radvel)  # In nm here.
#    print('Given radial velocity {} km/s, line {} should be at {:.4f}'.
#          format(radvel, line, shiftedlinewl))
    wlrange = getwlseparation(velsep, shiftedlinewl)  # 5 km/s by default
    continuumrange = getwlseparation(velsep+2e4, shiftedlinewl)  # +20 km/s
    upperwllim = shiftedlinewl + wlrange
    lowerwllim = shiftedlinewl - wlrange
    upperguess = wavelength2index(vac_wl, upperwllim)
    lowerguess = wavelength2index(vac_wl, lowerwllim)
    uppercont = wavelength2index(vac_wl, shiftedlinewl+continuumrange)
    lowercont = wavelength2index(vac_wl, shiftedlinewl-continuumrange)
    centralpos = flux[lowerguess:upperguess].argmin() + lowerguess
    centralwl = vac_wl[centralpos]
    continuum = flux[lowercont:uppercont].max()
    lowerlim, upperlim = centralpos - pixrange, centralpos + pixrange + 1
    results['continuum'] = continuum

    x = np.array(vac_wl[lowerlim:upperlim])
    y = np.array(flux[lowerlim:upperlim])
    if not y.all():
        print('Found zero flux for line {}'.format(line))
        raise
    e = np.array(err[lowerlim:upperlim])
    linebottom = y.min()
    fluxrange = y.max() - linebottom

    normdepth = (continuum - linebottom) / continuum
    results['norm_depth'] = normdepth

    # Normalize data for Scipy fitting
    xnorm = x - centralwl
    xnorm *= 1000
    ynorm = y - linebottom
    ynorm /= fluxrange
    enorm = e / fluxrange

    if save_arrays:
        if plot_dir is None:
            raise
            print('No directory given for plot dir, no plots will be output.')
        else:
            outfile = plot_dir / '{}/line_{}.csv'.format(filename, line)
            df = pd.DataFrame({'xnorm': xnorm, 'ynorm': ynorm, 'enorm': enorm,
                               'centralwl': centralwl, 'radvel': radvel,
                               'continuum': continuum,
                               'linebottom': linebottom,
                               'fluxrange': fluxrange})
            if not outfile.parent.exists():
                outfile.parent.mkdir()
            df.to_csv(path_or_buf=outfile, header=True, index=False, mode='w',
                      float_format='%.5f')

    # Fit a parabola to the normalized data
    if par_fit:
        parData = fitParabola(xnorm, ynorm, enorm, centralwl, radvel,
                              verbose=False)
        results['line_par'] = parData['parrestframeline']
        results['err_par'] = parData['vel_err_par']

    # Fit a Gaussian to the normalized data
    if gauss_fit:
        gaussData = fitGaussian(xnorm, ynorm, enorm, centralwl, radvel,
                                continuum, linebottom, fluxrange,
                                verbose=False)
        results['restframe_line_gauss'] = gaussData['restframe_line_gauss']
        results['vel_err_gauss'] = gaussData['vel_err_gauss']
        results['amplitude_gauss'] = gaussData['amplitude_gauss']
        results['amplitude_err_gauss'] = gaussData['amplitude_err_gauss']
        results['width_gauss'] = gaussData['width_gauss']
        results['width_err_gauss'] = gaussData['width_err_gauss']
        results['fwhm_gauss'] = gaussData['fwhm_gauss']
        results['fwhm_gauss_err'] = gaussData['fwhm_gauss_err']
        results['chisq_nu_gauss'] = gaussData['chisq_nu_gauss']
        results['gauss_vel_offset'] = getvelseparation(shiftedlinewl*1e-9,
                                           gaussData['gausscenterwl']*1e-9)

    # Fit a constrained parabola to the normalized data
    if spar_fit:
        sparData = fitSimpleParabola(xnorm, ynorm, enorm, centralwl, radvel,
                                     verbose=False)
        results['line_spar'] = sparData['sparrestframeline']
        results['err_spar'] = sparData['vel_err_spar']

    if plot:
        # Create the plots for the line
        datestring = date_obs.strftime('%Y%m%dT%H%M%S.%f')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(left=lowerwllim, right=upperwllim)
        ax.set_xlim(left=vac_wl[lowercont], right=vac_wl[uppercont])
        ax.set_title('{} nm'.format(line), fontsize=14)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Photons', fontsize=14)
        ax.errorbar(vac_wl[lowercont:uppercont],
                    flux[lowercont:uppercont],
                    yerr=err[lowercont:uppercont],
                    color='blue', marker='.', linestyle='')
        ax.vlines(shiftedlinewl, color='crimson',
                  ymin=flux[lowerguess:upperguess].min(),
                  ymax=continuum,
                  linestyle='--', label='Line expected position')
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
                    ((gaussian(xnorm, *gaussData['popt_gauss'])
                     + continuum - linebottom)
                     * fluxrange) + linebottom,
                    color='black', linestyle='-',
                    label='Gaussian fit')
#            ax.plot(vac_wl[lowerguess:upperguess],
#                    ((gaussian((vac_wl[lowerguess:upperguess]-centralwl)*1000,
#                     *gaussData['popt_gauss'])
#                     + continuum - linebottom) * fluxrange) + linebottom,
#                    color='black', linestyle='--', alpha=0.3)
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

        if plot_dir:
            try:
                line_dir = plot_dir / filename
            except TypeError:
                try:
                    line_dir = Path(plot_dir + '/' + filename)
                except TypeError:
                    raise
            if not line_dir.exists():
                line_dir.mkdir()
            filepath1 = line_dir / '{}_{}_{:.3f}nm.png'.format(
                                                    plot_dir.parent.stem,
                                                    datestring, float(line))
            filepath2 = line_dir / '{}_{}_norm_{:.3f}nm.png'.format(
                                                    plot_dir.parent.stem,
                                                    datestring, float(line))
            plt.savefig(str(filepath1), format='png')
        else:
            plt.show()
        plt.close(fig)

        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_title('{} nm'.format(line), fontsize=14)
        ax2.set_xlabel('Normalized wavelength', fontsize=14)
        ax2.set_ylabel('Normalized flux', fontsize=14)
        ax2.errorbar(xnorm, ynorm, yerr=enorm,
                     color='blue', marker='o', linestyle='')
        if par_fit:
            ax2.plot(xnorm, parabola(xnorm, *parData['popt_par']), color='red',
                     label='parabola')
        if gauss_fit:
            ax2.plot(xnorm, gaussian(xnorm, *gaussData['popt_gauss'])
                     + continuum - linebottom, color='green', linestyle='--',
                     label='Gaussian')
        if spar_fit:
            ax2.plot(xnorm, simpleparabola(xnorm, *sparData['popt_spar']),
                     color='black', linestyle=':', label='Const. parabola')
        ax2.legend()
        if plot_dir:
            plt.savefig(str(filepath2), format='png')
        else:
            plt.show()
        plt.close(fig2)

    return results


def measurepairsep(linepair, vac_wl, flux, err, radvel, filename, starname,
                   progressbar, plot=False, plot_dir=None, date=None,
                   save_arrays=False):
    """Return the distance between a pair of absorption lines.

    Parameters
    ----------
    linepair : tuple
        A tuple containing the wavelengths of a pair of absorption lines to
        analyze, as strings, in nanometers.
    vac_wl : array_like
        A wavelength array (in vacuum wavelengths) in which to search for the
        given lines.
    flux : array_like
        A flux array corresponding to the wavelength array.
    err : array_like
        An error array corresponding to the flux array.
    radvel : float
        The radial velocity of the star, in km/s.
    filename : Path object or str
        The name of the file from whence the arrays came.
    starname : str
        A string identifier for the star being analyzed, usually its HD number
        (e.g., HD146233).
    progressbar : a tqdm progress bar instance
        A progress bar to update on a per-line basis.
    plot : bool, Default: False
        A flag to pass to `~linefind` to indicate whether to plot the results
        of analysis or not.
    plot_dir : Path object
        A path representing the base directory for where to put any created
        plots, to be passed to `~linefind`.
    date : str
        A date representing the time of observation of the file being analyzed.
    save_arrays : bool, Default: False
        A flag to pass to `~linefind` to tell it whether to save out data
        about the measured lines or not.

    """

    # Create dictionary to store results
    results = {}

    global unfittablelines
    params = (vac_wl, flux, err, radvel, filename, starname)
    line1 = linefind(linepair[0], *params,
                     plot=plot, velsep=5000, pixrange=3,
                     par_fit=False, gauss_fit=True, spar_fit=False,
                     plot_dir=plot_dir, date_obs=date,
                     save_arrays=save_arrays)
    progressbar.update(1)
    line2 = linefind(linepair[1], *params,
                     plot=plot, velsep=5000, pixrange=3,
                     par_fit=False, gauss_fit=True, spar_fit=False,
                     plot_dir=plot_dir, date_obs=date,
                     save_arrays=save_arrays)
    progressbar.update(1)

    if line1 is None:
        unfittablelines += 1
    if line2 is None:
        unfittablelines += 1

    if (line1 and line2) is not None:

        if 'line_par' in (line1 and line2):
            parveldiff = abs(getvelseparation(line1['line_par'],
                                              line2['line_par']))
            err_par = np.sqrt((line1['err_par'])**2 +
                              (line2['err_par'])**2)
            results['parveldiff'] = parveldiff
            results['pardifferr'] = err_par

        if 'restframe_line_gauss' in (line1 and line2):
            gaussveldiff = abs(getvelseparation(line1['restframe_line_gauss'],
                                                line2['restframe_line_gauss']))
            err_gauss = np.sqrt((line1['vel_err_gauss'])**2 +
                                (line2['vel_err_gauss'])**2)

            # Populate results dict with linepair specific data...
            results['vel_diff_gauss'] = gaussveldiff
            results['diff_err_gauss'] = err_gauss

            # ... and line 1 specific data...
            results['line1_wl_gauss'] = line1['restframe_line_gauss']
            results['line1_wl_err_gauss'] = line1['vel_err_gauss']
            results['line1_amp_gauss'] = line1['amplitude_gauss']
            results['line1_amp_err_gauss'] = line1['amplitude_err_gauss']
            results['line1_width_gauss'] = line1['width_gauss']
            results['line1_width_err_gauss'] = line1['width_err_gauss']
            results['line1_fwhm_gauss'] = line1['fwhm_gauss']
            results['line1_fwhm_err_gauss'] = line1['fwhm_gauss_err']
            results['line1_chisq_nu_gauss'] = line1['chisq_nu_gauss']
            results['line1_continuum'] = line1['continuum']
            results['line1_norm_depth'] = line1['norm_depth']
            results['line1_gauss_vel_offset'] = line1['gauss_vel_offset']

            # ... and line 2 specific data.
            results['line2_wl_gauss'] = line2['restframe_line_gauss']
            results['line2_wl_err_gauss'] = line2['vel_err_gauss']
            results['line2_amp_gauss'] = line2['amplitude_gauss']
            results['line2_amp_err_gauss'] = line2['amplitude_err_gauss']
            results['line2_width_gauss'] = line2['width_gauss']
            results['line2_width_err_gauss'] = line2['width_err_gauss']
            results['line2_fwhm_gauss'] = line2['fwhm_gauss']
            results['line2_fwhm_err_gauss'] = line2['fwhm_gauss_err']
            results['line2_chisq_nu_gauss'] = line2['chisq_nu_gauss']
            results['line2_continuum'] = line2['continuum']
            results['line2_norm_depth'] = line2['norm_depth']
            results['line2_gauss_vel_offset'] = line2['gauss_vel_offset']

        if 'line_spar' in (line1 and line2):
            sparveldiff = abs(getvelseparation(line1['line_spar'],
                                               line2['line_spar']))
            err_spar = np.sqrt((line1['err_spar'])**2 +
                               (line2['err_spar'])**2)
            results['sparveldiff'] = sparveldiff
            results['spardifferr'] = err_spar

        return results
    else:
        return None


def searchFITSfile(FITSfile, pairlist, index, plot=False, save_arrays=False):
    """
    Measure line pair separations in a given file from a given list.

    Parameters
    ----------
    FITSfile : Path object
        The path to the FITS file to analyze.
    pairlist : tuple
        An arbitrary-length tuple of tuples each containing a pair of strings
        capable of being converted to floats, representing the wavelengths
        in nanometers of a pair of absorption lines to analyze.
    index : tuple
        A tuple of strings containing columns names for the pandas DataFrame
        object returned by this function.
    plot : bool, Default: False
        A flag to pass to further functions to tell them whether to create
        plots during the analysis or not.
    save_arrays : bool, Default: False
        A flag to pass to further functions to tell them whether to save and
        write out data related to the measured lines during analysis or not.
    """

    filename = FITSfile.stem

    data = readHARPSfile(str(FITSfile), radvel=True, date_obs=True,
                         hdnum=True, med_snr=True)
    # Convert from Angstroms to nm AFTER converting to vacuum wavelengths
    vac_wl = air2vacESO(data['w']) / 10

    flux = data['f']
    err = data['e']
    radvel = data['radvel']
    date = data['date_obs']
    hdnum = data['hdnum']
    snr = data['med_snr']

    if snr < 200:
        print('SNR less than 200 for {}, not analyzing.'.format(filename))
        return None

    params = (vac_wl, flux, err, radvel, filename, hdnum)

    # Create a list to store the Series objects in
    lines = []

    # Check if a path for graphs exists already, and if not create it one
    graph_path = FITSfile.parent / 'graphs'
    if not graph_path.exists():
        print('No graph directory found, creating one.')
        try:
            graph_path.mkdir()
        except FileExistsError:
            print('Graph directory already exists!')
            raise FileExistsError

    with tqdm(total=len(pairlist)*2, unit='lines',
              desc='Lines analyzed', disable=False) as pbar:
        for linepair in pairlist:
            line = {'date': date, 'object': hdnum,
                    'line1_nom_wl': linepair[0], 'line2_nom_wl': linepair[1]}
            line.update(measurepairsep(linepair, *params, pbar,
                                       plot=plot, plot_dir=graph_path,
                                       date=date, save_arrays=save_arrays))
            lines.append(pd.Series(line, index=index))

#    for line in lines:
#        print("{}, {}: measured separation {:.3f} m/s".format(
#                line['line1_nom_wl'], line['line2_nom_wl'],
#                line['vel_diff_gauss']))

    return lines


def injectGaussianNoise(data, nom_wavelength, num_iter=1000, plot=False):
    """
    Inject Gaussian error into an array of flux values

    Parameters
    ----------
    data : dataFrame
        A pandas DataFrame containing a normalized wavelength, flux,
        and error array for a line core, and the wavelength of the
        lowest pixel, radial velocity of the star, continuum (highest
        measured pixel in a range around the central wavelength), and
        linebottom (flux of the lowest pixel).
    nom_wavelength : float
        The nominal wavelength of the absorption line.
    num_iter : int, Default: 1000.
        Number of times to inject noise and fit the resulting array to
        measure the wavelength of the absorption line.
    plot : bool. Default: False
        If *True*, the function will create a series of histograms showing
        the range of wavelengths measured for each line.

    Returns
    -------
    dict
        A dictionary containing:
            fit_wavelength : float
                The wavelength originally measured from the fit to the
                observational data.
            measured_wavelengths : list
                A list of floats of len(`num_iter`) of all the wavelengths
                measured during the simulations.
            vel_offsets : list
                A list of floats of len(`num_iter`) of all the velocity
                separations between `fit_wavelength` and the wavelengths in
                `measured_wavelength`.
    """
    if not type(num_iter) is int:
        raise TypeError("num_iter must be an integer.")
    xnorm = data['xnorm']
    ynorm = data['ynorm']
    enorm = data['enorm']
    centralwl = data['centralwl'][0]
    radvel = data['radvel'][0]
    continuum = data['continuum'][0]
    linebottom = data['linebottom'][0]
    fluxrange = data['fluxrange'][0]
    e_orig = enorm * fluxrange

    gauss_measured_data = fitGaussian(xnorm, ynorm, enorm, centralwl, radvel,
                                      continuum, linebottom, fluxrange,
                                      verbose=False)
    fit_wavelength = gauss_measured_data['restframe_line_gauss']
    popt_gauss = gauss_measured_data['popt_gauss']
    yfitted = gaussian(xnorm, *popt_gauss)

    measured_wavelengths = []
    vel_offsets = []

    for i in range(0, num_iter, 1):
        noise = np.random.normal(loc=0, scale=e_orig)
        noise /= fluxrange
        ynoisy = yfitted + noise
        gauss_sim_data = fitGaussian(xnorm, ynoisy, enorm, centralwl, radvel,
                                     continuum, linebottom, fluxrange,
                                     verbose=False)
        measured_wl = gauss_sim_data['restframe_line_gauss']
        measured_wavelengths.append(measured_wl)
        vel_sep = getvelseparation(fit_wavelength*1e-9,
                                   measured_wl*1e-9)
        vel_offsets.append(vel_sep)

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        stddev = np.std(vel_offsets)
        ax.hist(vel_offsets - np.median(vel_offsets),
                bins=15, edgecolor='Black', label='std = {}'.format(stddev))

        ax.axvspan(xmin=-1*stddev,
                   xmax=stddev,
                   color='Gray', alpha=0.3)
        outfile = Path('/Users/dberke/Pictures/sims/sim_{0}.png'.format(
                nom_wavelength))
        ax.legend()
        fig.savefig(str(outfile))
        plt.close(fig)

    return {'fit_wavelength': fit_wavelength,
            'measured_wavelengths': measured_wavelengths,
            'vel_offsets': vel_offsets}


def pix_order_to_wavelength(pixel, order, coeffs_dict):
    """
    Returns the wavelength measured on the given pixel in the given order.

    Parameters
    ----------
    pixel : int, Range: 0 to 4095
        The pixel in the dispersion direction where the wavelength will be
        measured.
    order : int, Range: 0 to 71
        The spectral order to measure the wavelength in.
    coeff_dict: dict
        A dictionary containing wavelength solution coefficients in the form
        *ESO DRS CAL TH COEFF LLX*, where *X* ranges from 0 to 287.

    Returns
    -------
    float
        The wavelength observed at the given pixel and order in nanometers.

    Notes
    -----
    The algorithm used is derived from Dumusque 2018 [1]_.

    References
    ----------
    [1] Dumusque, X. "Measuring precise radial velocities on individual
    spectral lines I. Validation of the method and application to mitigate
    stellar activity", Astronomy & Astrophysics, 2018
    """
    if not (0 <= pixel <= 4095):
        print('pixel = {}, must be between 0 and 4095.'.format(pixel))
        raise ValueError
    if not (0 <= order <= 71):
        print('order = {}, must be between 0 and 71.'.format(order))
        raise ValueError

    wavelength = 0.
    for k in range(0, 4, 1):
        dict_key = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + k)
        wavelength += coeffs_dict[dict_key] * (pixel ** k)

    return wavelength / 10.


def wavelength_to_pix(wavelength, coeffs_dict, red_spec_form,
                      blue_spec_form):
    """
    """
    formats = (blue_spec_form, red_spec_form)
    orders = []
    for spec_form in formats:
        matched = 0
        for minwl, maxwl, order in zip(spec_form['startwl'],
                                       spec_form['endwl'], spec_form['order']):
            if minwl <= wavelength <= maxwl:
                orders.append(order)
                matched += 1
            if matched == 2:
                break
    pixels = []
    for order in orders:
        # Convert nanometers to Angstroms here.
        pixels.append(invertWavelengthMap(wavelength * 10, order, coeffs_dict))
    return pixels


def invertWavelengthMap(wavelength, order, coeffs_dict):
    """
    Returns the x-pixel of the CCD where the given wavelength is observed.

    Parameters
    ----------
    wavelength : float
        The wavelength to find the pixel of observation for.
    order : int
        The spectral order to search for the wavelength in.
    coeff_dict : dict
        A dictionary containing the coefficients for the wavelength solutions
        from an observation.

    Returns
    -------
    int
        The pixel in the x-direction (along the dispersion) where the given
        wavelength was measured.
    """
    oldwl = 0.
    for k in range(0, 4096, 1):
        newwl = pix_order_to_wavelength(k, order, coeffs_dict)
        if newwl > wavelength:
            if newwl - wavelength > oldwl - wavelength:
                return k - 1
            else:
                return k
