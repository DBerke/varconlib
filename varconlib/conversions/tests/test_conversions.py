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

from varconlib.conversions import (air_indexEdlen53, vac2airESO, air2vacESO)


@pytest.fixture(scope='module')
def vacuum_array():
    return u.unyt_array(range(380, 690, 1), units=u.nm)


@pytest.fixture(scope='module')
def air_array_2D():
    return u.unyt_array(range(500, 600, 1), units=u.nm).reshape(10, 10)


@pytest.fixture(scope='module')
def air_array():
    return u.unyt_array(range(380, 690, 1), units=u.nm)


class TestEdlen1953(object):
    def testAirIndex(self):
        assert air_indexEdlen53(500) == pytest.approx(0.99994748)

    def testAirUnits(self, vacuum_array):
        assert vac2airESO(vacuum_array).units == u.nm

    def test2DVacuumUnits(self, air_array_2D):
        assert air2vacESO(air_array_2D).units == u.nm

    def testReshapeArrays(self, air_array_2D):
        assert np.shape(air2vacESO(air_array_2D)) == (10, 10)

    def testVacuumUnits(self, air_array):
        assert air2vacESO(air_array).units == u.nm
