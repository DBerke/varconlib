#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:21:45 2020

@author: dberke

Tests for varconlib.fitting.fitting functions.
"""

from hypothesis import assume, given, note, settings, strategies as st
import numpy as np
import pytest

import varconlib.fitting.fitting as fit


def line_1d(x, slope, offset):

    return slope * x + offset

def constant(x, const):

    return const


class TestDataGeneration(object):
    @given(lower_lim=st.integers(),
           upper_lim=st.integers(),
           num_points=st.integers(2, 10000),
           slope=st.floats(min_value=-1e6, max_value=1e6,
                           allow_nan=False, allow_infinity=False),
           intercept=st.floats(min_value=-1e6, max_value=1e6,
                               allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def testLinearModelGaussianNoise(self, lower_lim, upper_lim, num_points,
                                     slope, intercept):
        assume(upper_lim > lower_lim)

        note(f'lower_lim, upper_lim, num_points = {lower_lim},'
             ' {upper_lim}, {num_points}')
        linear_model = fit.add_noise(fit.gaussian_noise)(line_1d)

        x_values = np.linspace(lower_lim, upper_lim, num_points)

        note(f'm, b = {slope}, {intercept}')
        linear_data = fit.generate_data(linear_model, x_values,
                                        (slope, intercept))

        p0 = [np.median(linear_data)]

        popt, pcov = fit.curve_fit_data(constant, x_values, linear_data,
                                        p0)
        assert np.mean(linear_data) == pytest.approx(popt[0], rel=1e-2)
