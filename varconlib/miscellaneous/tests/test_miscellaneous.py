#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:28:53 2019

@author: dberke

Test script for functions in varconlib

"""

import datetime as dt

import pytest
import unyt as u

import varconlib.miscellaneous as vcl


@pytest.fixture(scope='class')
def wavelength_array():
    return [4000, 4500, 5000, 5500, 6000, 6500] * u.angstrom


class TestWavelength2Velocity(object):

    @pytest.fixture(scope='class')
    def redder_wavelength(self):
        return 510 * u.nm

    @pytest.fixture(scope='class')
    def bluer_wavelength(self):
        return 490 * u.nm

    def testWavelengths(self, redder_wavelength, bluer_wavelength):
        assert vcl.wavelength2velocity(redder_wavelength, bluer_wavelength) ==\
            pytest.approx(-11991698.32 * u.m / u.s)
        assert vcl.wavelength2velocity(bluer_wavelength, redder_wavelength) ==\
            pytest.approx(11991698.32 * u.m / u.s)


class TestVelocity2Wavelenth(object):

    def testVelocities(self):
        assert vcl.velocity2wavelength(10 * u.km / u.s, 500 * u.nm) ==\
            pytest.approx(0.0166782 * u.nm)
        assert vcl.velocity2wavelength(10 * u.km / u.s, 500 * u.nm,
                                       unit=u.angstrom) == pytest.approx(
                                               0.16678205 * u.angstrom)
        assert vcl.velocity2wavelength(10 * u.km / u.s, 500 * u.nm,
                                       unit=u.pm) == pytest.approx(
                                               16.67820476 * u.pm)


class TestDate2Index(object):

    @pytest.fixture(scope='class')
    def date_list(self):
        years = [2003, 2004, 2006, 2008, 2010, 2013, 2015, 2017]
        months = [2, 5, 8, 3, 11, 8, 9, 1]
        days = [12, 18, 21, 27, 1, 8, 30, 14]
        hours = [2, 18, 6, 4, 15, 22, 21, 13]
        minutes = [23, 17, 55, 45, 38, 19, 2, 37]
        seconds = [12, 14, 57, 44, 24, 28, 19, 7]

        dt_list = []
        for year, month, day, hour, minute, second in zip(years, months, days,
                                                          hours, minutes,
                                                          seconds):
            dt_list.append(dt.datetime(year=year, month=month, day=day,
                                       hour=hour, minute=minute,
                                       second=second))

        return dt_list

    def testNonDateInputValue(self, date_list):
        mock_date = 'Non-date value'
        with pytest.raises(RuntimeError):
            vcl.date2index(mock_date, date_list)

    def testDatetimeValue(self, date_list):
        mock_date = dt.datetime(year=2005, month=2, day=15,
                                hour=0, minute=0, second=0)
        assert vcl.date2index(mock_date, date_list) == 2

    def testDateValue(self, date_list):
        mock_date = dt.date(year=2011, month=2, day=15)
        assert vcl.date2index(mock_date, date_list) == 5

    def testOutOfRangeDates(self, date_list):
        mock_date = dt.datetime(year=2001, month=1, day=1,
                                hour=0, minute=0, second=0)
        assert vcl.date2index(mock_date, date_list) is None
        mock_date = dt.datetime(year=2021, month=1, day=1,
                                hour=0, minute=0, second=0)
        assert vcl.date2index(mock_date, date_list) is None

    def testIndexing(self, date_list):
        mock_date = dt.datetime(year=2005, month=6, day=1,
                                hour=0, minute=0, second=0)
        assert vcl.date2index(mock_date, date_list) == 2
        mock_date = dt.date(year=2014, month=1, day=1)
        assert vcl.date2index(mock_date, date_list) == 6


class TestQCoefficientShifts(object):

    @pytest.fixture(scope='class')
    def transition(self):
        return 20000 * u.cm ** -1

    def testWavenumber(self, transition):
        assert vcl.q_alpha_shift(transition, -500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)
        assert vcl.q_alpha_shift(transition, 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(-14.98963 * u.m / u.s)

    def testWavelength(self, transition):
        assert vcl.q_alpha_shift(transition.to(u.angstrom,
                                               equivalence='spectral'),
                                 -500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)
        assert vcl.q_alpha_shift(transition.to(u.angstrom,
                                               equivalence='spectral'),
                                 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(-14.98963 * u.m / u.s)

    def testEnergy(self, transition):
        assert vcl.q_alpha_shift(transition.to(u.eV,
                                               equivalence='spectral'),
                                 -500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)
        assert vcl.q_alpha_shift(transition.to(u.eV,
                                               equivalence='spectral'),
                                 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(-14.98963 * u.m / u.s)


class TestShiftWavelength(object):

    @pytest.fixture(scope='class')
    def wavelength(self):
        return 5000 * u.angstrom

    @pytest.mark.parametrize(
            'shift_velocity',
            [1e8, -1e8] * u.km / u.s)
    def testUnphysicalVelocity(self, wavelength, shift_velocity):
        with pytest.raises(AssertionError):
            vcl.shift_wavelength(wavelength, shift_velocity)
        with pytest.raises(AssertionError):
            vcl.shift_wavelength(wavelength, shift_velocity)

    def testShiftSingleWavelength(self, wavelength):
        assert pytest.approx(vcl.shift_wavelength(wavelength,
                                                  500 * u.km / u.s),
                             5008.3991 * u.angstrom)

    def testShiftMultipleWavelengths(self, wavelength_array):
        assert pytest.approx(vcl.shift_wavelength(wavelength_array[:2],
                                                  100 * u.km / u.s),
                             [4001.33425638, 4501.50103843] * u.angstrom)


class TestWavelength2Index(object):

    def testBadWavelengthValue(self, wavelength_array):
        with pytest.raises(AssertionError):
            vcl.wavelength2index('6521', wavelength_array,
                                 reverse=False)
        with pytest.raises(AssertionError):
            vcl.wavelength2index(6521, wavelength_array,
                                 reverse=False)
        with pytest.raises(AssertionError):
            vcl.wavelength2index(6521.0, wavelength_array,
                                 reverse=False)

    def testWavelengthOutOfRange(self, wavelength_array):
        with pytest.raises(RuntimeError):
            vcl.wavelength2index(3000 * u.angstrom, wavelength_array,
                                 reverse=False)
            vcl.wavelength2index(8000 * u.angstrom, wavelength_array,
                                 reverse=False)

    def testWavelengthUnyt(self, wavelength_array):
        assert vcl.wavelength2index(4001 * u.angstrom, wavelength_array,
                                    reverse=False) == 0
        assert vcl.wavelength2index(4499 * u.angstrom, wavelength_array,
                                    reverse=False) == 1
        assert vcl.wavelength2index(6400 * u.angstrom, wavelength_array,
                                    reverse=False) == 5

    def testReversedWavelengthArray(self, wavelength_array):
        reversed_wavelengths = [x for x in reversed(wavelength_array)]
        assert vcl.wavelength2index(4001 * u.angstrom, reversed_wavelengths,
                                    reverse=True) == 0
