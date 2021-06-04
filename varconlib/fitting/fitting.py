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
from time import sleep

import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.special import erf

import varconlib as vcl


def constant_model(data, a):
    """
    Return a constant value.

    Parameters
    ----------
    data : array-like
        The independent variable data.
    a : float or int
        A constant value to return.

    Returns
    -------
    a : float or int
        The value input as `a`.

    """

    return a


def linear_model(data, a, b, c, d):
    """
    Return the value of a three-dimensional linear model

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a, b, c, d : float or int
        Values of the coefficient for the model. `a` is the zeroth-order
        (constant) value, while `b`, `c`, and `d` represent the first-order
        (linear) coefficients for the three dimensions.

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2]


def quadratic_model(data, a, b, c, d, e, f, g):
    """
    Return the value of a three-dimensional function of second order.

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : float or int
        The constant term for the function.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e, f, g : float or int
        The values of the coefficients for the quadratic terms of the function.

    Returns
    -------
     float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] +\
        e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2


def cross_term_model(data, a, b, c, d, e):
    """
    Return the value of a linear model with cross-term between metallicity and
    temperature

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : TYPE
        DESCRIPTION.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e : float or int
        The value of the coefficient for the cross-term between temperature and
        metallicity.

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] + e * data[1] / data[0]


def quadratic_mag_model(data, a, b, c, d, e, f):
    """
    Return the value of a three-dimenstional function linear in temperature and
    metallicity (with cross-term) and quadratic in the third term (nominally
    magnitude).

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : float or int
        The value of the constant coefficient.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e : float or int
        The value of the coefficient for the cross-term between temperature and
        metallicity.
    f : float or int
        The coefficient for the quadratic term of the third term (nominally
        absolute magnitude).

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] +\
        e * data[1] / data[0] + f * data[2] ** 2


def quad_cross_term_model(data, a, b, c, d, e, f, g, h):
    """

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : float or int
        The constant term for the function.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e, f, g : float or int
        The values of the coefficients for the quadratic terms of the function.
    h : float or int
        The value of the coefficients for the linear cross-term between
        the [Fe/H]/Teff.

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] +\
        e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2 +\
        h * data[1] / data[0]


def cubic_model(data, a, b, c, d, e, f, g, h, i, j):
    """

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : float or int
        The constant term for the function.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e, f, g : float or int
        The values of the coefficients for the quadratic terms of the function.
    h, i, j : float or int
        The values of the coefficients for the cubic terms of the function.

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] +\
        e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2 +\
        h * data[0] ** 3 + i * data[1] ** 3 + j * data[2] ** 3


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
    r"""Return the value of a Gaussian integrated between two points (given as
    a tuple in `pixel`).

    The function is given by
    .. math::    f(x_1, x_2) = \\frac{\\sqrt{\\frac{\\tau}{4}} A
                 \\sigma\\left[\\mathrm{erf}\\left(\\frac{x_{2}-
                 \\mu}{\\sqrt{2}\\sigma}\\right)
                 -\\mathrm{erf}\\left(\\frac{x_{1}-
                 \\mu}{\\sqrt{2}\\sigma}\\right)\\right]
                 -D x_{1}+D x_{2}}{x_2-x_1}


    Parameters
    ----------
    pixel : tuple containing two floats
        A tuple containing the two points to integrate the Gaussian between.
    amplitude : float
        The amplitude of the Gaussian. Must be Real.
    mu : float
        The mean (also the center) of the Gaussian. Must be Real.
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


def calc_chi_squared_nu(residuals, errors, n_params):
    """
    Return the value of the reduced chi-squared statistic for the given data.

    Parameters
    ----------
    residuals : array-like of floats or ints
        An array of values representing a set of measured deviations from a
        fitted model.
    errors : array-like of floats or ints
        An array of variances for each point, of the same length as `residuals`.
    n_params : int
        The number of fitted parameters in the model.

    Returns
    -------
    float
        The value of chi-squared per degree-of-freedom for the given data and
        number of fitted parameters.

    """

    chi_squared = np.sum(np.square(residuals / errors))
    dof = len(residuals) - n_params
    if dof <= 0:
        return np.nan
    else:
        return chi_squared / dof


def find_sys_scatter(model_func, x_data, y_data, err_array, beta0,
                     n_sigma=2.5, tolerance=0.001, verbose=False):
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
    n_sigma : float
        The number of sigma outside of which a data point is considered an
        outlier.
    tolerance : float, Default : 0.001
        The distance from one within which the chi-squared per degree of freedom
        must fall for the iteration to exit. (Note that if the chi-squared value
        is naturally less than one on the first iteration, the iteration with
        end even if the value is not closer to one than the tolerance.)
    verbose : bool, Default : False
        Whether to print out more diagnostic information on the process.

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
            mask_list : list of lists
                A list containing the mask applied to the data at each
                iteration. Each entry will be a list of 1s and 0s.

    """

    vprint = vcl.verbose_print(verbose)

    # Iterate to find what additional systematic error is needed
    # to get a chi^2 of ~1.
    chi_tol = tolerance
    diff = 1
    sys_err = 0
    iter_err_array = np.sqrt(np.square(err_array) +
                             np.square(sys_err))

    chi_squared_list = []
    sigma_sys_list = []
    mask_list = []
    sigma_sys_change_list = []

    x_data.mask = False
    y_data.mask = False
    err_array.mask = False

    orig_x_data = ma.copy(x_data)
    orig_y_data = ma.copy(y_data)
    orig_errs = ma.copy(err_array)

    last_mask = np.zeros_like(y_data)
    new_mask = np.ones_like(y_data)

    iterations = 0
    chi_squared_flips = 0

    vprint('  #   sigma_sys      diff     chi^2      SSCA   #*   flips')
    while True:
        iterations += 1
        popt, pcov = curve_fit(model_func, x_data, y_data,
                               sigma=iter_err_array,
                               p0=beta0,
                               absolute_sigma=True,
                               method='lm', maxfev=10000)

        iter_model_values = model_func(x_data, *popt)

        iter_residuals = y_data - iter_model_values

        # Find the chi^2 value for this fit:
        chi_squared_nu = calc_chi_squared_nu(iter_residuals, iter_err_array,
                                             len(popt))

        try:
            last_chi_squared = chi_squared_list[-1]
        except IndexError:  # On the first iteration
            pass
        else:
            if chi_squared_nu > 1 and last_chi_squared < 1:
                chi_squared_flips += 1
            elif chi_squared_nu < 1 and last_chi_squared > 1:
                chi_squared_flips += 1
            else:
                pass

        sigma_sys_list.append(sys_err)
        chi_squared_list.append(chi_squared_nu)
        mask_list.append(last_mask)

        diff = abs(chi_squared_nu - 1)

        # Set the amount to change by using the latest chi^2 value.
        sigma_sys_change_amount = np.power(chi_squared_nu, 2/3)
        sigma_sys_change_list.append(sigma_sys_change_amount)

        vprint(f'{iterations:>3}, '
               f'{sys_err:>10.6f}, {diff:>8.4f}, {chi_squared_nu:>8.4f},'
               f' {sigma_sys_change_amount:>8.4f},'
               f' {iter_residuals.count():>3},  {chi_squared_flips}')
        if verbose:
            sleep_length = 0 if chi_squared_flips < 3 else 0.1
            sleep(sleep_length)

        if chi_squared_nu > 1:
            if sys_err == 0:
                sys_err = np.sqrt(chi_squared_nu - 1) * np.nanmedian(err_array)
                # sys_err = np.sqrt(chi_squared_nu)
            else:
                sys_err = sys_err * sigma_sys_change_amount
        elif chi_squared_nu < 1:
            if sys_err == 0:
                # If the chi-squared value is naturally lower
                # than 1, don't change anything, just exit.
                break
            else:
                sys_err = sys_err * sigma_sys_change_amount

        # Construct new error array using all errors.
        iter_err_array = np.sqrt(np.square(orig_errs) +
                                 np.square(sys_err))
        new_mask = np.zeros_like(y_data)

        # Find residuals for all data, including that masked this iteration:
        full_model_values = model_func(orig_x_data, *popt)
        full_residuals = orig_y_data - full_model_values

        # Check for outliers at each point, and mark the mask appropriately.
        for i in range(len(iter_err_array)):
            if abs(full_residuals[i]) > (n_sigma * iter_err_array[i]):
                new_mask[i] = 1

        # Set up the mask on the x and y data and errors for the next iteration.
        for array in (x_data, y_data, iter_err_array):
            if chi_squared_flips < 5:
                array.mask = new_mask
                last_mask = new_mask
            # If chi^2 flips between less than and greater than one too many
            # times, the routine is probably stuck in a loop adding and removing
            # a borderline point, so simply stop re-evaluating points for
            # inclusion.
            else:
                array.mask = last_mask

        # If chi^2 gets within the tolerance and the mask hasn't changed in the
        # last iteration, end the loop.
        if ((diff < chi_tol) and (np.all(last_mask == new_mask))):
            break

        # If the mask is still changing, but the sigma_sys value has
        # clearly converged to a value (by both the 10th and 100th most
        # recent entries being within the given tolerance of the most recent
        # entry), end the loop. Most runs terminate well under 100 steps so
        # this should only catch the problem cases.
        elif ((iterations > 100) and (diff < chi_tol) and
              ((abs(sigma_sys_list[-1] - sigma_sys_list[-10])) < chi_tol) and
              ((abs(sigma_sys_list[-1] - sigma_sys_list[-100])) < chi_tol)):
            break

        # If the chi^2 value is approaching 1 from the bottom, it may be the
        # case that it can never reach 1, even if sigma_sys goes to 0 (but it
        # will take forever to get there). In the case that chi^2_nu < 1,
        # and the values have clearly converged to a value within the tolerance
        # over the last 100 iterations, break the loop. This does leave the
        # derived sigma_sys value somewhat meaningless, but it should be small
        # enough in these cases as to be basically negligible.

        elif ((iterations > 100) and (chi_squared_nu < 1.) and
              ((abs(chi_squared_list[-1]) -
                chi_squared_list[-10]) < chi_tol) and
              ((abs(chi_squared_list[-1]) -
                chi_squared_list[-100]) < chi_tol)):
            # If sigma_sys is less than a millimeter per second, just set it
            # to zero.
            if sys_err < 0.0011:
                sigma_sys_list[-1] = 0
            break

        # If the iterations go on too long, it may be because it's converging
        # very slowly to zero sigma_sys, so give it a nudge if it's still large.
        elif iterations == 500:
            if (sys_err > 0.001) and (sys_err < 0.01) and\
                    (sigma_sys_list[-1] < sigma_sys_list[-2]):
                sys_err = 0.001

        # If it's taking a really long time to converge, but sigma_sys is less
        # than a millimeter per second, just set it to zero and end the loop.
        elif iterations == 999:
            if sys_err < 0.0011:
                sigma_sys_list[-1] = 0
                break
            else:
                print(f'Final sys_err = {sys_err}')
                print(f'Final chi^2 = {chi_squared_nu}')
                print(f'diff = {diff}')
                print(np.all(last_mask == new_mask))
                for i, j in zip(last_mask, new_mask):
                    print(f'{i}  {j}')
                raise RuntimeError("Process didn't converge.")

    # ---------

    results_dict = {'popt': popt, 'pcov': pcov,
                    'residuals': iter_residuals,
                    'sys_err_list': sigma_sys_list,
                    'chi_squared_list': chi_squared_list,
                    'mask_list': mask_list}

    return results_dict
