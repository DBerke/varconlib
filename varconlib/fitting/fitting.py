#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:14:08 2020

@author: dberke

This module contains functions and decorators related to generalized fitting of
data to a model, as well as creating synthetic data.

"""

import functools

import numpy as np
from scipy.optimize import curve_fit


def constant_model(data, a):

    return a


def linear_model(data, a, b, c, d):

    return a + b * data[0] + c * data[1] + d * data[2]


def quadratic_model(data, a, b, c, d, e, f, g):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2


def cubic_model(data, a, b, c, d, e, f, g, h, i, j):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2 +\
           h * data[0] ** 3 + i * data[1] ** 3 + j * data[2] ** 3


def quartic_model(data, a, b, c, d, e, f, g, h, i, j, k, l, m):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2 +\
           h * data[0] ** 3 + i * data[1] ** 3 + j * data[2] ** 3 +\
           k * data[0] ** 4 + l * data[1] ** 4 + m * data[2] ** 4


def quintic_model(data, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2 +\
           h * data[0] ** 3 + i * data[1] ** 3 + j * data[2] ** 3 +\
           k * data[0] ** 4 + l * data[1] ** 4 + m * data[2] ** 4 +\
           n * data[0] ** 5 + o * data[1] ** 5 + p * data[2] ** 5


def cross_term_model(data, a, b, c, d, e):

    return a + b * data[0] + c * data[1] + d * data[2] + e * data[1] / data[0]


def quadratic_cross_term_model(data, a, b, c, d, e, f, g, h, i):

    return a + b * data[0] + c * data[1] + d * data[2] + e * data[1]/data[0] +\
           f * data[0] ** 2 + g * data[1] ** 2 + h * data[2] ** 2 +\
           i * (data[1]/data[0]) ** 2


def quadratic_mag_model(data, a, b, c, d, e, f):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[1] / data[0] + f * data[2] ** 2


def quad_full_cross_terms_model(data, a, b, c, d, e, f, g, h, i, j):

    return a + b * data[0] + c * data[1] + d * data[2] +\
           e * data[0] * data[1] + f * data[0] * data[2] +\
           g * data[1] * data[2] +\
           h * data[0] ** 2 + i * data[1] ** 2 + j * data[2] ** 2


def gaussian_noise(ydata, sigma=None):
    """Return an array of normally-distributed noise values.

    Parameters
    ----------
    ydata : array_like
        An array of data values.

    Optional
    --------
    sigma : float
        The standard deviation of the Gaussian to sample from. The default is
        *None*. If not given, the scale will be the square root of each
        individual data point.

    Returns
    -------
    `np.ndarray`
        An array of noise values of the same shape as the input `ydata`, to be
        added to a data set.

    """

    if sigma is not None:
        assert isinstance(sigma, (float, int, np.ndarray)), 'sigma is not' +\
            f' an appropriate type! type: {type(sigma)}'
        scale = sigma
    else:
        scale = np.sqrt(np.abs(ydata))

    return np.random.normal(loc=0., scale=scale, size=ydata.size)


def add_noise(noise_func, *noise_args, **noise_kwargs):
    """Add noise from a given function to the output of the decorated function.

    Parameters
    ----------
    noise_func : callable
        A function to add noise to a data set. Should take as input a 1-D array
        (and optionally additional parameters) and return the same.
    args, kwargs
        Additional arguments passed to this decorator will be passed on through
        to `noise_func`.

    Returns
    -------
    callable
        A decorated function which adds noise from the given `noise_func` to
        its own output.

    """

    def decorator_noise(func):

        @functools.wraps(func)
        def wrapper_noise(*args, **kwargs):

            # Create the values from the function wrapped:
            y = func(*args, **kwargs)
            # Now generate noise to add to those values from the noise function
            # provided to the decorator:
            noise = noise_func(y, *noise_args, **noise_kwargs)

            return y + noise

        return wrapper_noise

    return decorator_noise


def generate_data(function, xdata, func_params):
    """Create a data set using a given function and independent variable(s).

    Parameters
    ----------
    function : callable
        A function to use to generate data points from input values.
    xdata : array_like
        An array of values of shape (N, M) where *N* is the number of input
        parameters needed by `function`. `function` should act on this array
        to produce an array of length M values.
    func_params : iterable
        An iterable collection of parameter values to pass to `function`. They
        will be passed using argument unpacking, so they should be in the order
        expected by the function used for `function`.

    Returns
    -------
    `np.ndarray`
        A array of length M (where M is dependent on the data passed to
        `xdata` of values produced by running `function` on `xdata`.)

    """

    return function(xdata, *func_params)


def curve_fit_data(function, xdata, ydata, func_params, sigma=None):
    """Fit a function to data and return the parameters and covariance matrix.

    Parameters
    ----------
    function : callable
        A function to use to generate data points from input values.
    xdata : array_like
        An array of values of shape (N, M) where *N* is the number of input
        parameters needed by `function`. `function` should act on this array
        to produce an array of length M values.
    ydata : array_like
        A array of values of length M (i.e., the same length as the x-values
        passed to the function) of data points to be used to fit the function
        to.
    func_params : iterable
        An iterable collection of parameter values to pass to `function`. They
        will be passed using argument unpacking, so they should be in the order
        expected by the function used for `function`.

    Optional
    --------
    sigma : array_like
        An array of length M (i.e., the same length as the x-values
        passed to the function) of the standard deviations of the y-values.
        If not passed, all points will be assumed to have the same relative
        error.

    Returns
    -------
    tuple of (`np.ndarray`, `np.ndarray`)
        A tuple containing two arrays, one holding the optimized parameters
        found by the fit, the other the covariance matrix for the parameters.

    """

    # Use "abosolute_sigma" only if actual sigma values are passed, otherwise
    # just use relative values.
    sigma_type = False if sigma is None else True

    popt, pcov = curve_fit(function, xdata, ydata, sigma=sigma,
                           p0=func_params, absolute_sigma=sigma_type,
                           method='lm', maxfev=10000)

    return (popt, pcov)


def check_fit(function, xdata, ydata, func_params):
    """Check the difference between data and a fit to that data.

    Parameters
    ----------
    function : callable
        A function to use to generate data points from input values.
    xdata : array_like
        An array of values of shape (N, M) where *N* is the number of input
        parameters needed by `function`. `function` should act on this array
        to produce an array of length M values.
    ydata : array_like
        A array of values of length M (i.e., the same length as the x-values
        passed to the function) of data points to be used to fit the function
        to.
    func_params : iterable
        An iterable collection of parameter values to pass to `function`. They
        will be passed using argument unpacking, so they should be in the order
        expected by the function used for `function`.

    Returns
    -------
    `np.ndarray`
        A 1-D array of differences between data points and the function.

    """

    return ydata - function(xdata, *func_params)
