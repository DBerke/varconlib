#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:14:08 2020

@author: dberke

This module contains functions and decorators related to generalized fitting of
data to a model, as well as creating synthetic data.

"""

import functools
from math import sqrt, tau

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import unyt as u


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


def gaussian(x, a, b, c, d=0):
    r"""Return the value of a Gaussian function with the given parameters.


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
    r"""Return the value of a Gaussian integrated between two points (given as a
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


def find_sigma_sys(model_func, x_data, y_data, err_array, beta0,
                   tolerance=0.001):
    """Find the systematic scatter in a dataset with a given model.

    Takes a model function `model_func`, and arrays of x, y, and uncertainties
    (which must have the same length) and an initial guess to the parameters of
    the function, and fits the model to the data. It then checks the reduced
    chi-squared value, and if it is greater than 1 (with a tolerance of 1e-3),
    it adds an additional amount in quadrature to the error array and refits the
    data, continuing until the chi-squared value is within the tolerance.

    Parameters
    ----------
    model_func : callable
        The function to fit the data with.
    x_data : array-like
        The array of x-values to fit.
    y_data : array-like
        The array of y-values to fit. Must have same length as `x_data`.
    err_array : array-like
        The error array for the y-values. Must have same length as `x_data` and
        `y_data`.
    beta0 : tuple
        A tuple of values to use as the initial guesses for the paremeters in
        the function given by `model_func`.
    tolerance : float, Default : 0.001
        The distance from one within which the chi-squared per degree of freedom
        must fall for the iteration to exit. (Note that if the chi-squared value
        is naturally less than one on the first iteration, the iteration with
        end even if the value is not closer to one than the tolerance.)

    Returns
    -------
    dict
        A dictionary containing the following keys:
            popt : tuple
                A tuple of the optimized values found by the fitting routine
                for the parameters.
            pcov : `np.array`
                The covariance matrix for the fit.
            residuals : `np.array`
                The value of `y_data` minus the model values at all given
                independent variables.
            sys_err_list : list of floats
                A list conainting the values of the systematic error at each
                iteration. The last value is the values which brings the
                chi-squared per degree of freedom for the data within the
                tolerance to one.
            chi_squared_list : list of floats
                A list containing the calculated chi-squared per degree of
                freedom for each step of the iteration.

    """

    # Iterate to find what additional systematic error is needed
    # to get a chi^2 of ~1.
    chi_tol = tolerance
    diff = 1
    sys_err = 0
    num_iters = 0
    sigma_sys_change_amount = 0.25  # Range (0, 1)

    chi_squared_list = []
    sigma_sys_list = []

    while diff > chi_tol:

        num_iters += 1

        iter_err_array = np.sqrt(np.square(err_array) +
                                 np.square(sys_err))

        popt, pcov = curve_fit(model_func, x_data, y_data,
                               sigma=iter_err_array,
                               p0=beta0,
                               absolute_sigma=True,
                               method='lm', maxfev=10000)

        model_values = model_func(x_data, *popt)

        residuals = y_data - model_values

        # Find the chi^2 value for this distribution:
        chi_squared = np.sum((residuals / iter_err_array) ** 2)
        dof = len(y_data) - len(popt)
        chi_squared_nu = chi_squared / dof

        sigma_sys_list.append(sys_err)
        chi_squared_list.append(chi_squared_nu)

        diff = abs(chi_squared_nu - 1)
        if diff > 2:
            sigma_sys_change_amount = 0.75
        elif diff > 1:
            sigma_sys_change_amount = 0.5
        else:
            sigma_sys_change_amount = 0.25

        if chi_squared_nu > 1:
            if sys_err == 0:
                sys_err = np.sqrt(chi_squared_nu)
            else:
                sys_err *= (1 + sigma_sys_change_amount)
        elif chi_squared_nu < 1:
            if sys_err == 0:
                # If the chi-squared value is naturally lower
                # than 1, don't change anything, just exit.
                break
            else:
                sys_err *= (1 - sigma_sys_change_amount)

    results_dict = {'popt': popt, 'pcov': pcov,
                    'residuals': residuals,
                    'sys_err_list': sigma_sys_list,
                    'chi_squared_list': chi_squared_list}

    return results_dict
