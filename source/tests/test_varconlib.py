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

import varconlib as vcl


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

    def test_out_of_range_dates(self, date_list):
        mock_date = dt.datetime(year=2001, month=1, day=1, hour=0, minute=0,
                                second=0)
        assert vcl.date2index(mock_date, date_list) is None
        mock_date = dt.datetime(year=2021, month=1, day=1, hour=0, minute=0,
                                second=0)
        assert vcl.date2index(mock_date, date_list) is None

    def test_indexing(self, date_list):
        mock_date = dt.datetime(year=2005, month=6, day=1, hour=0, minute=0,
                                second=0)
        assert vcl.date2index(mock_date, date_list) == 1


class TestQCoefficientShifts(object):

    @pytest.fixture(scope='class')
    def transition(self):
        return 20000 * u.cm ** -1

    def test_wavenumber(self, transition):
        assert vcl.q_alpha_shift(transition, 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)

    def test_wavelength(self, transition):
        assert vcl.q_alpha_shift(transition.to(u.angstrom,
                                               equivalence='spectral'),
                                 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)

    def test_energy(self, transition):
        assert vcl.q_alpha_shift(transition.to(u.eV,
                                               equivalence='spectral'),
                                 500 * u.cm ** -1, 1.000001) ==\
            pytest.approx(14.98963 * u.m / u.s)


