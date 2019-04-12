#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:55:59 2019

@author: dberke

Test suite for the transition_line.Transition object.
"""

import pytest
from transition_line import Transition
import unyt as u
# TODO: possibly add a fixture for test cases?

class TestTransition(object):
    def test_no_units(self):
        with pytest.raises(AttributeError):
            Transition(500, 26, 1)

    def test_unphysical_atomic_number(self):
        with pytest.raises(AssertionError):
            Transition(500 * u.nm, 0, 1)

    def test_atomic_symbol_string(self):
        a = Transition(500 * u.nm, 'fe', 1)
        assert a.atomicNumber == 26

    def test_atomic_number_string(self):
        a = Transition(500 * u.nm, '77', 1)
        assert a.atomicSymbol == 'Ir'

    def test_atomic_number_int(self):
        a = Transition(500 * u.nm, 54, 1)
        assert a.atomicSymbol == 'Xe'

    def test_wrong_atomic_symbol_too_long(self):
        with pytest.raises(TypeError):
            Transition(500 * u.nm, 'Ah!', 1)  # Element of surprise

    def test_unphysical_atomic_symbol(self):
        with pytest.raises(KeyError):
            Transition(500 * u.nm, 'Uo', 1)  # Unobtanium

    def test_wavelength_sorting(self):
        a = Transition(500 * u.nm, 26, 1)
        b = Transition(501 * u.nm, 26, 1)
        assert a < b
        assert b > a

    def test_atomic_species(self):
        a = Transition(500 * u.nm, 26, 1)
        assert a.atomicSpecies == 'Fe I'
        b = Transition(500 * u.nm, 31, 2)
        assert b.atomicSpecies == 'Ga II'

    def test_string_roman_numeral_ionization_state(self):
        a = Transition(500 * u.nm, 26, 'I')
        assert a.ionizationState == 1
        assert a.atomicSpecies == 'Fe I'
        assert a.atomicSymbol == 'Fe'

    def test_wavenumber_conversion(self):
        a = Transition(500 * u.nm, 26, 1)
        assert a.wavenumber.value == pytest.approx(20000)
        a.wavenumber = 25000
        assert a.wavelength.value == pytest.approx(4e-5)
        a.wavenumber = 20000 * u.cm ** -1
        assert a.wavelength.value == pytest.approx(5e-5)

    def test_wavenumber_assignment(self):
        a = Transition(500 * u.nm, 26, 1)
        with pytest.raises(ValueError):
            a.wavenumber = 25000 * u.nm ** -1
        a.wavenumber = 25000
        assert a.wavelength.value == pytest.approx(4e-5)
        assert a.wavelength.units == u.cm
        assert a.wavenumber.units == u.cm ** -1
