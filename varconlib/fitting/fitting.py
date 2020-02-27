#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:14:08 2020

@author: dberke

This module contains functions and decorators related to generalized fitting of
data to a model, as well as creating synthetic data.

"""

import functools

from scipy.optimize import curve_fit


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
