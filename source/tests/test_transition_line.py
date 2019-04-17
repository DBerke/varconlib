#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:55:59 2019

@author: dberke

Test suite for the transition_line.Transition object.
"""

from fractions import Fraction
import pytest
import unyt as u
from transition_line import Transition

# TODO: possibly add a fixture for test cases?


class TestTransition(object):
    def test_no_units(self):
        with pytest.raises(AttributeError):
            Transition(500, 26, 1)

    def test_unphysical_atomic_number(self):
        with pytest.raises(AssertionError):
            Transition(500 * u.nm, 0, 1)
        with pytest.raises(AssertionError):
            Transition(500 * u.nm, -1, 1)

    def test_negative_ionization_state(self):
        with pytest.raises(AssertionError):
            Transition(500 * u.nm, 0, -1)

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
            Transition(500 * u.nm, 'X', 1)

    def test_wavelength_sorting(self):
        a = Transition(500 * u.nm, 26, 1)
        b = Transition(501 * u.nm, 26, 1)
        assert a < b
        assert b > a

    def test_equality(self):
        a = Transition(500.0 * u.nm, 26, 1)
        b = Transition(500.0 * u.nm, 26, 1)
        assert a == b
        a.lowerEnergy = b.lowerEnergy = 1.3 * u.eV
        assert a == b
        a.higherEnergy = b.higherEnergy = 5.6 * u.eV
        assert a == b
        a.lowerJ = b.lowerJ = 1.5
        assert a == b
        a.higherJ = b.higherJ = 0.5
        assert a == b
        a.lowerOrbital = b.lowerOrbital = '3d8.(3F).4s.4p.(3P*)'
        assert a == b
        a.higherOrbital = b.higherOrbital = '3d8.4s.(2F).5s '
        assert a == b
        # Now test for inequality.
        b.higherOrbital = 'non-existent'
        assert a != b
        b.lowerOrbital = "it's gone too"
        assert a != b
        b.higherJ = 3
        assert a != b
        b.lowerJ = 0
        assert a != b
        b.higherEnergy = 4 * u.eV
        assert a != b
        b.lowerEnergy = 3 * u.eV
        assert a != b
        b.ionizationState = 2
        assert a != b
        b.ionizationState = 1
        b.atomicNumber = 27
        assert a != b
        b.atomicNumber = 26
        b.wavelength = 501 * u.nm
        assert a != b

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

    def test_fractional_J_conversion(self):
        a = Transition(500 * u.nm, 26, 1)
        a.lowerJ = 1.5
        assert a.lowerJ == Fraction(3, 2)
        a.higherJ = 2
        assert a.higherJ == Fraction(2, 1)
