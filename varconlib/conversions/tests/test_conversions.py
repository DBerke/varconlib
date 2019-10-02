#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:05:15 2019

@author: dberke

Test script for conversions.py and the associated functions.
"""

import numpy as np
import pytest

import unyt as u

from varconlib.conversions import (air_indexEdlen53, vac2airESO, air2vacESO,
                                   vac2airMorton00, air2vacMortonIAU)


@pytest.fixture(scope='module')
def vacuum_array():
    return u.unyt_array(range(3800, 6900, 10), units=u.angstrom)


@pytest.fixture(scope='module')
def air_array_2D():
    return u.unyt_array(range(5000, 6000, 10), units=u.angstrom).reshape(10,
                                                                         10)


@pytest.fixture(scope='module')
def air_array():
    return u.unyt_array(range(3800, 6900, 10), units=u.angstrom)


@pytest.fixture(scope='module')
def unitless_array():
    return np.array([5000, 5001, 5002, 5003, 5004])


class TestEdlen1953(object):

    def testAirIndex(self):
        assert air_indexEdlen53(5000) == pytest.approx(1.0002789636500335)

    def testAirUnits(self, vacuum_array):
        assert vac2airESO(vacuum_array).units == u.angstrom

    def test2DVacuumUnits(self, air_array_2D):
        assert air2vacESO(air_array_2D).units == u.angstrom

    def testReshapeArrays(self, air_array_2D, verbose=True):
        assert air_array_2D.ndim == 2
        assert np.shape(air2vacESO(air_array_2D)) == (10, 10)

    def testVacuumUnits(self, air_array):
        assert air2vacESO(air_array, verbose=True).units == u.angstrom

    def testNoUnits(self, unitless_array):

        assert vac2airESO(unitless_array) == pytest.approx([4998.60557074,
                                                            4999.60530505,
                                                            5000.60503934,
                                                            5001.60477364,
                                                            5002.60450793])


class TestMorton2000(object):

    def testVac2Air(self, unitless_array):
        assert vac2airMorton00(unitless_array) == pytest.approx([4998.605522,
                                                                 4999.605256,
                                                                 5000.604990,
                                                                 5001.604724,
                                                                 5002.604459])

    def testAir2Vac(self, unitless_array):
        assert air2vacMortonIAU(unitless_array) == pytest.approx([5001.394848,
                                                                  5002.395114,
                                                                  5003.395380,
                                                                  5004.395646,
                                                                  5005.395911])
