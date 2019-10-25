#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke

A module meant as a catch-all for certain functions and data potentially useful
across ultiple scripts, but which don't really have any commonality.
"""


import configparser
import datetime as dt
from math import sqrt, tau
from pathlib import Path

from bidict import bidict
import numpy as np
from scipy.special import erf
import unyt
import unyt as u

import varconlib as vcl

# Need to find the base directory path dynamically on import so it works when
# testing with Travis.
base_path = Path(__file__).parent
config_file = base_path / '../config/variables.cfg'
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# Spectral format files for HARPS blue and red CCDs.
blueCCDpath = vcl.data_dir / 'HARPS_CCD_blue.csv'
redCCDpath = vcl.data_dir / 'HARPS_CCD_red.csv'

# A bidict of the HARPS order numbers between the spectral order numbers and
# an ordinal numbering system.
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

    assert isinstance(wavelength, unyt.array.unyt_quantity),\
        f'Given wavelength "{wavelength}" is not a unyt_quantity.'

    # For cases where the wavelength is given in descending wavelength order,
    # we can flip it.
    if reverse:
        wavelength_array.reverse()

    # If the given wavelength is not within the limits of the array, raise an
    # error.
    if not wavelength_array[0] < wavelength < wavelength_array[-1]:
        raise RuntimeError("Couldn't find the given wavelength: {}".
                           format(wavelength))

    for i in range(0, len(wavelength_array), 1):
        # First find the index for which the value is greater than the given
        # wavelength:
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


def date2index(given_date, date_list):
    """Find the closest index prior to a given date in a list of dates.

    Note that this works like a "floor" function, finding the closest timestamp
    that occurs *before* the given date, not necessarily the closest one
    overall.

    Parameters
    ----------
    date : `datetime.date` or `datetime.datetime`
        The given date to search for.
    date_list : iterable collection of `datetime.datetime`s
        A list of timestamps in chronological order.

    Returns
    -------
    int or None
        The index of the closest timestamp prior to the given `date`. If all
        dates in the list are past or before the given `date` *None* will be
        returned.

    """

    if isinstance(given_date, dt.datetime):
        date_to_find = given_date

    elif isinstance(given_date, dt.date):
        date_to_find = dt.datetime(year=given_date.year,
                                   month=given_date.month,
                                   day=given_date.day,
                                   hour=0, minute=0, second=0)
    else:
        raise RuntimeError('given_date not date or datetime!')

    if (date_to_find <= date_list[0]) or (date_to_find >= date_list[-1]):
        return None

    for i in range(0, len(date_list), 1):
        if date_to_find < date_list[i]:
            return i


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
    assert abs(shift_velocity) < u.c, 'Given velocity exceeds speed of light!'

    result = ((shift_velocity / u.c) * wavelength) + wavelength

    return result.to(original_units)


def velocity2wavelength(velocity_offset, wavelength, unit=None):
    """Return the wavelength separation for a given velocity separation at the
    given wavelength.

    Parameters
    ----------
    velocity_offset : `unyt.unyt_quantity`
        The velocity separation. Can be in any valid units with dimensions of
        length / time.
    wavelength : `unyt.unyt_quantity`
        The wavelength at which the function should be evaluated, since
        it's a function of wavelength. Returned in whatever units it's
        given in (should have dimensions of length).

    Optional
    --------
    unit : `unyt.unyt_object.Unit`
        A valid unit with dimension of length, such as `unyt.angstrom` or
        `unyt.nm`. This is the units the returned value will be converted to.
        If not given, the returned value will be in the same units as the input
        `wavelength`.

    Returns
    -------
    unyt_quantity
        The separation in wavelength space for the given velocity offset at
        the given wavelength. Units will be the ones given in `units`, or the
        same as `wavelength` if `units` is not given.

    """

    original_units = wavelength.units
    result = (velocity_offset * wavelength) / u.c
    if not unit:
        return result.to(original_units)
    else:
        return result.to(unit)


def wavelength2velocity(wavelength1, wavelength2):
    """Return velocity separation of a pair of wavelengths.

    In terms of order, the second wavelength parameter is subtracted from the
    first; or, the returned velocity is the velocity of the second wavelength
    compared to the first one.

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
    .. math:: \\omega = \\omega_0 + q \\left((\\frac{\\alpha}{\\alpha_0})^2 -1
    \\right)

    """

    original_value = omega.to_equivalent(u.cm ** -1, equivalence='spectral')

    new_value = original_value + q_coefficient * (delta_alpha ** 2 - 1)

    return wavelength2velocity(original_value.to(u.angstrom,
                                                 equivalence='spectral'),
                               new_value.to(u.angstrom,
                                            equivalence='spectral'))


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
    .. math::    f(x_1, x_2) = \\sqrt{\\frac{\\tau}{4}} A
                 \\sigma\\left[\\erf\\left(\\frac{x_{2}-
                 \\mu}{\\sqrt{2}\\sigma}\\right)
                 -\\erf\\left(\\frac{x_{1}-
                 \\mu}{\\sqrt{2}\\sigma}\\right)\\right]
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
