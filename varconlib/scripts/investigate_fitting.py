#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:26:06 2020

@author: dberke

This script investigates fitting generated data sets and recovering the
parameters of the function used

"""

import argparse

import numpy as np

import varconlib.fitting.fitting as fit


def constant(x, const):
    """Return a constant value.

    Parameters
    ----------
    x : float or iterable of floats
        The x-value to calculate the value of the function at.
    const : float
        A constant value to be returned.

    """

    return const


def line_1d(x, slope, offset):
    """Return the value of a line with given slope and offset.

    Parameters
    ----------
    x : float or iterable of floats
        The x-value to calculate the value of the line at.
    slope : float
        The slope of the line. Must be finite.
    offset : float
        The y-offset of the line.

    Returns
    -------
    float or iterable of floats
        The value at the given `x`, or values if len(x) > 1, in which case it
        will be an array with length = len(x).

    """

    return slope * x + offset


def temp_metal_linear(x, a, b, c):
    """Return the value of a plane with the given parameters.

    Parameters
    ----------
    x : tuple of floats, length-2
        A tuple containing pairs of floats representing temperature and
        metallicity of a star.
    a, b, c : float
        Parameters for the equation. `a` is the offset, `b` is the constant
        for the linear term in temperature, and `c` the constant for the linear
        term in metallicity.

    Returns
    -------
    float
        The value of the function at the given independent variable values with
        the given parameters.

    """

    temp, mtl = x

    return a + b * temp + c * mtl


def plot_residuals(axis, model_func, func_params, xdata, ydata, errors):
    """Plot of the residuals between data and a fit on a given axis.

    Parameters
    ----------
    axis : `matplotlib.axis.Axes` instance
        A `matplotlib` axis to plot the results on.
    model_func: callable
        The model function to use.
    func_params : tuple or list
        A collection of parameter values to pass to the model function being
        evaluated.
    xdata : array_like
        An array of values of shape (N, M) where *N* is the number of input
        parameters needed by `model_func`. `model_func` should act on this
        array to produce an array of length M values.
    ydata : array_like
        An array of values of length M (i.e., the same length as the x-values
        passed to the function) of data points to be used to measure the values
        of the function againts.
    errors : array_like
        An array of values of length M (i.e., the same length as the x- and y-
        values pased to the function) representing the errors on the y-values.

    """

    axis.errorbar()


def main():
    """The main routine for the script."""

    linear_model = fit.add_noise(fit.gaussian_noise)(line_1d)

    x_values = np.linspace(5400, 6100, 15)

    linear_data = fit.generate_data(linear_model, x_values, (0.25, -5800))

    p0 = [np.median(linear_data)]
    print(np.mean(linear_data))

    popt, pcov = fit.curve_fit_data(constant, x_values, linear_data,
                                    p0)

    print(popt)

    residuals = fit.check_fit(constant, x_values, linear_data, p0)
    print(residuals)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main()
