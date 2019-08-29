#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke
"""

# Module to contain functions potentially useful across multiple programs


import configparser
import datetime as dt
from math import sqrt, log, tau
from pathlib import Path

from bidict import bidict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from tqdm import tqdm
import unyt as u

from conversions import air2vacESO
from obs1d import readHARPSfile1D


config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)
data_dir = Path(config['PATHS']['data_dir'])

# Spectral format files for HARPS blue and red CCDs.
blueCCDpath = data_dir / 'HARPS_CCD_blue.csv'
redCCDpath = data_dir / 'HARPS_CCD_red.csv'

# A bidict of the HARPS order numbers between the spectral order numbers and
# a simple numbering system.
order_numbers = bidict({1: 161, 2: 160, 3: 159, 4: 158, 5: 157, 6: 156,
                        7: 155, 8: 154, 9: 153, 10: 152, 11: 151, 12: 150,
                        13: 149, 14: 148, 15: 147, 16: 146, 17: 145, 18: 144,
                        19: 143, 20: 142, 21: 141, 22: 140, 23: 139, 24: 138,
                        25: 137, 26: 136, 27: 135, 28: 134, 29: 133, 30: 132,
                        31: 131, 32: 130, 33: 129, 34: 128, 35: 127, 36: 126,
                        37: 125, 38: 124, 39: 123, 40: 122, 41: 121, 42: 120,
                        43: 119, 44: 118, 45: 117, 46: 116,
                        47: 114, 48: 113, 49: 112, 50: 111, 51: 110, 52: 109,
                        53: 108, 54: 107, 55: 106, 56: 105, 57: 104, 58: 103,
                        59: 102, 60: 101, 61: 100, 62: 99, 63: 98, 64: 97,
                        65: 96, 66: 95, 67: 94, 68: 93, 69: 92, 70: 91,
                        71: 90, 72: 89})


# Functions
def map_spectral_order(order):
    # TODO: Convert this to a bidict. Also found in maskSpectralRegions.py
    """
    Converts from HARPS' internal 0-71 order numbers to those in the HARPS
    spectral format (89-114, 116-161).

    Parameters
    ----------
    order : int
        An order number in the range [0, 71]

    Returns
    -------
    int
        The number of the dispersed order.
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


def index2wavelength(index, step, min_wl):
    """Return the wavelength associated with an index.

    index: index position of the spectrum list
    step: the step in wavelength per index, in nm
    min_wl: the minimum wavelength of the spectrum, in nm
    """
    return round((step * index + min_wl), 2)


def wavelength2index(wavelength, wavelength_array, reverse=False):
    """Find the index in an iterable associated with a given wavelength.

    Parameters
    ----------
     wavelength : float or unyt_quantity
        The wavelength to find the associated index for. If given as a
        unyt_quantity, the wavelength_array given should be a unyt_array with
        the same dimensions of length.

    wavelength_array : array-like, or unyt_array
        An iterable object containing a sequence of wavelengths, by default *in
        increasing order*. If a unyt_array is given, it should have dimensions
        of length and be one-dimensional.

    Optional
    --------
    reverse : bool, Default: False
        Reverses the given wavelength sequence before evaluating it, in case
        the sequence is given in longer to shorter order.

    Returns
    -------
    int
        The index for which the associated wavelength is closest to the given
        wavelength.

    """

    if reverse:
        wavelength_array.reverse()

    for i in range(0, len(wavelength_array), 1):
        # First find the index for which the value is greater than the given
        # wavelength:
        try:
            if wavelength_array[i] >= wavelength:
                # Then check if it's closest to this index or the previous one.
                # The way it's set up it should always be
                # wl_arr[i-1] <= wl <= wl_arr[i] assuming monotonic increase of
                # wavelengths.
                if abs(wavelength_array[i] - wavelength) >\
                   abs(wavelength - wavelength_array[i - 1]):
                    return i - 1
                else:
                    return i
        except ValueError:
            print(wavelength_array)
            print(wavelength_array[i])
            print(wavelength)
            raise

    # If it reaches the end without finding a matching wavelength, raise an
    # error.
    raise RuntimeError("Couldn't find the given wavelength: {}".
                       format(wavelength))


def date2index(given_date, date_list):
    """Find the closest index prior to a given date in a list of dates.

    Note that that works like a "floor" function, finding the closest timestamp
    that occurs *before* the given date, not necessarily the closest one
    overall.

    Parameters
    ----------
    date : `datetime.date` or `datetime.datetime`
        The given date to search for.
    date_date : iterable collection of `datetime.datetime`s
        A list of timestamps in chronological order.

    Returns
    -------
    int
        The index of the closest timestamp prior to the given date.

    """

    if isinstance(given_date, dt.date):
        date_to_find = dt.datetime(year=given_date.year,
                                   month=given_date.month,
                                   day=given_date.day,
                                   hour=0, minute=0, second=0)
    elif isinstance(given_date, dt.datetime):
        date_to_find = given_date
    else:
        raise RuntimeError('given_date not date or datetime!')

    if (date_to_find <= date_list[0]) or (date_to_find >= date_list[-1]):
        return None

    for i in range(0, len(date_list), 1):
        if date_to_find < date_list[i]:
            return i - 1


def shift_wavelength(wavelength, shift_velocity):
    """Find the new wavelength of a wavelength (single or an array) given a
    velocity to shift it by. Returns in the units given.

    Parameters
    ----------
    wavelength : Unyt unyt_array or unyt_quantity
        Wavelength(s) to be shifted. Will be converted to meters internally.
    radial_velocity : float or int
        Radial velocity to shift the wavelength(s) by. Will be converted to
        meters/second internally.

    Returns
    -------
        float
        The new wavelength, in the original units, shifted according to the
        given radial velocity.

    """

    original_units = wavelength.units

    # Convert the wavelength and radial velocity to base units.
    wavelength.convert_to_units(u.m)
    shift_velocity.convert_to_units(u.m/u.s)

    # Make sure we're not using unphysical velocities!
    assert abs(shift_velocity < u.c), 'Given velocity exceeds speed of light!'

    result = ((shift_velocity / u.c) * wavelength) + wavelength

    return result.to(original_units)


def velocity2wavelength(velocity_offset, wavelength):
    """Return the wavelength separation for a given velocity separation at the
    given wavelength.

    Parameters
    ----------
    velocity_offset : unyt_quantity
        The velocity separation. Can be in any valid units with dimensions of
        length / time.
    wavelength : unyt_quantity
        The wavelength at which the function should be evaluated, since
        it's a function of wavelength. Returned in whatever units it's
        given in (should have dimensions of length).

    Returns
    -------
    unyt_quantity
        The separation in wavelength space for the given velocity offset at
        the given wavelength.

    """

    original_units = wavelength.units
    result = (velocity_offset * wavelength) / u.c
    return result.to(original_units)


def wavelength2velocity(wavelength1, wavelength2):
    """Return velocity separation of a pair of wavelengths.

    Parameters
    ----------
    wavelengthl1 & wavelength2 : unyt_quantity or unyt_array
        The two wavelengths to get the velocity separation between.

    Returns
    -------
    unyt_quantity
        The velocity separation between the given wavelengths in m/s.

    """

    result = (wavelength2 - wavelength1) * u.c /\
             ((wavelength1 + wavelength2) / 2)
    return result.to(u.m/u.s)


def q_alpha_shift(omega, q_coefficient, delta_alpha):
    """Return the velocity change in a transition with a given q-coefficient
    for a given fractional change in alpha.

    Parameters
    ----------
    omega : unyt_quantity with dimensions length, 1/(length), or energy
        The wavenumber, wavelength, or energy of the transition to calculate
        the shift for. Assuming the units are correct this parameter will be
        converted to a wavenumber if necessary internally.
    q_coefficient : float
        The *q*-coefficient for the transition. This is a float, in units of
        reciprocal centimeters.
    delta_alpha : float
        A fractional change in the value of alpha to use for the calculation.

    Returns
    -------
    unyt_quantity with dimensions (length)/(time)
        The velocity separation between the original wavenumber/wavelength/
        energy and the new value.

    Notes
    -----
    The calculation of the shift in a transition's wavenumber is given by the
    formula:
    .. math:: \omega = \omega_0 + q \left((\frac{\alpha}{\alpha_0})^2 -1\right)

    """

    original_value = omega.to_equivalent(u.cm ** -1, equivalence='spectral')

    new_value = original_value + q_coefficient * (delta_alpha ** 2 - 1)

    return wavelength2velocity(original_value, new_value)


def parse_spectral_mask_file(file):
    """Parses a spectral mask file from maskSpectralRegions.py

    Parameters
    ----------
    file : str or Path object
        A path to a text file to parse. Normally this would come from
        maskSpectralRegions.py, but the general format is a series of
        comma-separated floats, two per line, that each define a 'bad'
        region of the spectrum.

    Returns
    -------
    list
        A list of tuples parsed from the file, each one delimiting the
        boundaries of a 'bad' spectral region.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    masked_regions = []
    for line in lines:
        if '#' in line:
            continue
        start, end = line.rstrip('\n').split(',')
        masked_regions.append((float(start), float(end)))

    return masked_regions


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


def gaussian(x, a, b, c, d=0):
    """Return the value of a Gaussian function with the given parameters.


    The parameter `d` controls the baseline of the Gaussian; if it is not
    present, the function will default to a baseline of zero.

    Parameters
    ----------
    x : float
        The value of the independent variable to evaluate the function at.
    a : float
        The amplitude of the Gaussian. Must be Real.
    b : float
        The median (also the center) of the Gaussian. Must be Real.
    c : float
        The standard deviation of the Gaussian. Must be non-zero.

    Optional
    --------
    d : float
        The baseline of the Gaussian. If not given, will default to zero.

    Returns
    -------
    float
        The value of a Gaussian with the given parameters at the given `x`
        position.

    Notes
    -----
    A Gaussian function is given by:
    .. math::    f(x) = D + A e^{- \frac{(x - B)^2}{2C^2}}

    """

    return d + a * np.exp(-1 * ((x - b) ** 2 / (2 * c * c)))


def integrated_gaussian(pixel, amplitude, mu, sigma, baseline):
    """Return the value of a Gaussian integrated between two points (given as a
    tuple in `pixel`).

    The function is given by
    .. math::    f(x_1, x_2) = \sqrt{\frac{\tau}{4}} A\
                 \sigma\left[\erf\left(\frac{x_{2}-\mu}{\sqrt{2}\sigma}\right)\
                 -\erf\left(\frac{x_{1}-\mu}{\sqrt{2}\sigma}\right)\right]\
                 -D x_{1}+D x_{2}


    Parameters
    ----------
    pixel : tuple containing two floats
        A tuple containing the two points to integrate the Gaussian between.
    amplitude : float
        The amplitude of the Gaussian. Must be Real.
    mu : float
        The median (also the center) of the Gaussian. Must be Real.
    sigma : float
        The standard deviation of the Gaussian. Must be non-zero.
    baseline : float
        The baseline of the Gaussian. Must be Real.

    Returns
    -------
    float
        The integrated value under a Gaussian with the given parameters between
        the two values supplied.

    """

    return (sqrt(tau / 4) * sigma * amplitude *
            (erf((pixel[1] - mu) / (sigma * sqrt(2))) -
            erf((pixel[0] - mu) / (sigma * sqrt(2)))) -
            (baseline * pixel[0]) + (baseline * pixel[1])) / (pixel[1] -
                                                              pixel[0])


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
    neg_linedepth = -linedepth
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
#        ax.errorbar(xnorm, ynorm-continuum+linebottom, yerr=enorm,
#                    color='blue', marker='o', linestyle='')
#        ax.plot(xnorm, (gaussian(xnorm, *gauss_params)), color='Black')
#        outfile = Path('/Users/dberke/Pictures/debug_norm.png')
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

    # Multiply by 1e-9 to get nm to m for wavelength2velocity which
    # requires m
    vel_err_gauss = wavelength2velocity(gausscenterwl*1e-9,
                                        (gausscenterwl+wl_err_gauss)*1e-9)
    # Shift line to stellar rest frame
    gaussrestframeline = shift_wavelength(gausscenterwl, -radvel)

    # Get the width (sigma) of the Gaussian
    gauss_sigma = abs(popt_gauss[2] / 1000)
    gauss_sigma_err = perr_gauss[2] / 1000

    # Get the full width at half maximum (approximately 2.355 * sigma)
    gauss_fwhm = 2 * sqrt(2 * log(2)) * gauss_sigma
    gauss_fwhm_err = 2 * sqrt(2 * log(2)) * gauss_sigma_err

    # Convert sigma and FWHM to velocity space
    sigma_vel = wavelength2velocity(gausscenterwl * 1e-9,
                                    (gausscenterwl + gauss_sigma) * 1e-9)
    sigma_vel_err = wavelength2velocity(gausscenterwl * 1e-9,
                                        (gausscenterwl + gauss_sigma_err) *
                                        1e-9)

    fwhm_vel = wavelength2velocity(gausscenterwl * 1e-9,
                                   (gausscenterwl + gauss_fwhm) * 1e-9)
    fwhm_vel_err = wavelength2velocity(gausscenterwl * 1e-9,
                                       (gausscenterwl + gauss_fwhm_err) *
                                       1e-9)

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
    vel_err_par = wavelength2velocity(parcenterwl * 1e-9,
                                      (parcenterwl + wl_err_par) * 1e-9)

    # Shift to stellar rest frame by correcting radial velocity.
    parrestframeline = shift_wavelength(parcenterwl, -radvel)

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

    vel_err_spar = wavelength2velocity(sparcenterwl*1e-9,
                                       (sparcenterwl+wl_err_spar)*1e-9)
    # Convert to restframe of star
    sparrestframeline = shift_wavelength(sparcenterwl, -radvel)

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
    line : str
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

    if type(plot_dir) == str:
        plot_dir = Path(plot_dir)
    # Create a dictionary to store the results
    results = {}

    radvel = float(radvel)
    # Figure out location of line given radial velocity of the star (in km/s)
    shiftedlinewl = shift_wavelength(float(line), radvel)  # In nm here.
#    print('Given radial velocity {} km/s, line {} should be at {:.4f}'.
#          format(radvel, line, shiftedlinewl))
    # 5 km/s by default
    wlrange = velocity2wavelength(velsep, shiftedlinewl)
    # +25 km/s
    continuumrange = velocity2wavelength(velsep+2e4, shiftedlinewl)

    upperwllim = shiftedlinewl + wlrange
    lowerwllim = shiftedlinewl - wlrange
    upperguess = wavelength2index(upperwllim, vac_wl)
    lowerguess = wavelength2index(lowerwllim, vac_wl)
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
        try:
            gaussData = fitGaussian(xnorm, ynorm, enorm, centralwl, radvel,
                                    continuum, linebottom, fluxrange,
                                    verbose=False)
        except RuntimeError:
            fig_err = plt.figure(figsize=(8, 8))
            ax_err = fig_err.add_subplot(1, 1, 1)
            ax_err.set_xlim(left=lowerwllim-0.01, right=upperwllim+0.01)
            ax_err.vlines(shiftedlinewl, color='crimson',
                          ymin=flux[lowerguess:upperguess].min(),
                          ymax=continuum,
                          linestyle='--')
            ax_err.axvspan(xmin=vac_wl[lowerlim], xmax=vac_wl[upperlim],
                           color='red', alpha=0.2)
            ax_err.axvspan(xmin=lowerwllim, xmax=upperwllim,
                           color='grey', alpha=0.25)
            ax_err.errorbar(vac_wl, flux, yerr=err,
                            color='blue', marker='o', linestyle='')
            outfile = Path('/Users/dberke/Pictures/debug.png')
            fig_err.savefig(str(outfile))
            plt.close(fig_err)
            raise
        results['restframe_line_gauss'] = gaussData['restframe_line_gauss']
        results['vel_err_gauss'] = gaussData['vel_err_gauss']
        results['amplitude_gauss'] = gaussData['amplitude_gauss']
        results['amplitude_err_gauss'] = gaussData['amplitude_err_gauss']
        results['width_gauss'] = gaussData['width_gauss']
        results['width_err_gauss'] = gaussData['width_err_gauss']
        results['fwhm_gauss'] = gaussData['fwhm_gauss']
        results['fwhm_gauss_err'] = gaussData['fwhm_gauss_err']
        results['chisq_nu_gauss'] = gaussData['chisq_nu_gauss']
        results['gauss_vel_offset'] = wavelength2velocity(shiftedlinewl * 1e-9,
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
        ax.set_title('{:.4f} nm'.format(line), fontsize=14)
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
            filepath1 = line_dir / '{}_{}_{:.4f}nm.png'.format(
                                                    plot_dir.parent.stem,
                                                    datestring, float(line))
            filepath2 = line_dir / '{}_{}_norm_{:.4f}nm.png'.format(
                                                    plot_dir.parent.stem,
                                                    datestring, float(line))
            results['gauss_graph_path'] = filepath1
            results['gauss_norm_graph_path'] = filepath2
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
                     plot=plot, velsep=2600, pixrange=3,
                     par_fit=False, gauss_fit=True, spar_fit=False,
                     plot_dir=plot_dir, date_obs=date,
                     save_arrays=save_arrays)
    progressbar.update(1)
    line2 = linefind(linepair[1], *params,
                     plot=plot, velsep=2600, pixrange=3,
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
            parveldiff = abs(wavelength2velocity(line1['line_par'],
                                                 line2['line_par']))
            err_par = np.sqrt((line1['err_par'])**2 +
                              (line2['err_par'])**2)
            results['parveldiff'] = parveldiff
            results['pardifferr'] = err_par

        if 'restframe_line_gauss' in (line1 and line2):
            gaussveldiff = abs(wavelength2velocity(
                               line1['restframe_line_gauss'],
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
            sparveldiff = abs(wavelength2velocity(line1['line_spar'],
                                                  line2['line_spar']))
            err_spar = np.sqrt((line1['err_spar'])**2 +
                               (line2['err_spar'])**2)
            results['sparveldiff'] = sparveldiff
            results['spardifferr'] = err_spar

        return results
    else:
        return None


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
        vel_sep = wavelength2velocity(fit_wavelength * 1e-9,
                                      measured_wl * 1e-9)
        vel_offsets.append(vel_sep)

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        stddev = np.std(vel_offsets)
        ax.hist(vel_offsets - np.median(vel_offsets),
                bins=15, edgecolor='Black', label='std = {}'.format(stddev))

        ax.axvspan(xmin=-stddev,
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
