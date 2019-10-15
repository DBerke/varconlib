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

from varconlib.exceptions import (AtomicNumberError,
                                  IonizationStateError,
                                  BadElementInputError)
from varconlib.transition_line import Transition


class TestTransitionCreation(object):

    def testNoUnits(self):
        with pytest.raises(AttributeError):
            Transition(500, 26, 1)

    def testZeroAtomicNumber(self):
        with pytest.raises(AtomicNumberError):
            Transition(500 * u.nm, 0, 1)

    def testNegativeAtomicNumber(self):
        with pytest.raises(AtomicNumberError):
            Transition(500 * u.nm, -1, 1)

    def testZeroIonizationState(self):
        with pytest.raises(IonizationStateError):
            Transition(500 * u.nm, 26, 0)

    def testNegativeIonizationState(self):
        with pytest.raises(IonizationStateError):
            Transition(500 * u.nm, 26, -1)

    def testUnsupportedRomanNumeral(self):
        with pytest.raises(IonizationStateError):
            Transition(500 * u.nm, 26, 'XL')

    def testAtomicSymbolString(self):
        a = Transition(500 * u.nm, 'fe', 1)
        assert a.atomicNumber == 26

    def testAtomicNumberString(self):
        a = Transition(500 * u.nm, '77', 1)
        assert a.atomicSymbol == 'Ir'

    def testAtomicNumberInt(self):
        a = Transition(500 * u.nm, 54, 1)
        assert a.atomicSymbol == 'Xe'

    def testWrongAtomicSymbolTooLong(self):
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, 'Ah!', 1)  # Element of surprise

    def testNonexistentAtomicSymbol(self):
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, 'X', 1)

    def testNonsensicalAtomicSymbol(self):
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, True, 1)
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, (True,), 1)
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, False, 1)
        with pytest.raises(BadElementInputError):
            Transition(500 * u.nm, [False], 1)

    def testAtomicSpecies(self):
        a = Transition(500 * u.nm, 26, 1)
        assert a.atomicSpecies == 'Fe I'
        b = Transition(500 * u.nm, 31, 2)
        assert b.atomicSpecies == 'Ga II'

    def testStringRomanNumeralIonizationState(self):
        a = Transition(500 * u.nm, 26, 'I')
        assert a.ionizationState == 1
        assert a.atomicSpecies == 'Fe I'
        assert a.atomicSymbol == 'Fe'

    def testWavenumberAssigment(self):
        a = Transition(500 * u.nm, 26, 1)
        assert a.wavenumber.value == pytest.approx(20000)
        a.wavenumber = 20000 * u.cm ** -1
        assert a.wavelength.value == pytest.approx(5e-5)

    def testWavenumberConversion(self):
        a = Transition(500 * u.nm, 26, 1)
        a.wavenumber = 2500000 * 1 / u.m
        assert a.wavelength.value == pytest.approx(4e-5)
        assert a.wavelength.units == u.cm
        assert a.wavenumber.units == u.cm ** -1

    def testFractionalJConversion(self):
        a = Transition(500 * u.nm, 26, 1)
        a.lowerJ = 1.5
        assert a.lowerJ == Fraction(3, 2)
        a.lowerJ = None
        assert a.lowerJ is None
        a.higherJ = 2
        assert a.higherJ == Fraction(2, 1)
        a.higherJ = None
        assert a.higherJ is None


class TestTransitionComparison(object):

    def testWavelengthSorting(self):
        a = Transition(500 * u.nm, 26, 1)
        b = Transition(501 * u.nm, 26, 1)
        assert a < b
        assert not a > b
        assert b > a
        assert not b < a

    def testEquality(self):
        a = Transition(500.0 * u.nm, 26, 1)
        b = Transition(500.0 * u.nm, 26, 1)

        with pytest.raises(AssertionError):
            assert a == 'A non-transition'

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


class TestTransitionRepresentation(object):

    @pytest.fixture(scope='function')
    def mock_transition(self):
        a = Transition(558.0345 * u.nm, 54, 1)
        a.lowerEnergy = 17.012 * u.cm ** -1
        a.lowerOrbital = '(4F)4s a5F'
        a.lowerJ = 1.5

        a.higherEnergy = 12051.823 * u.cm ** -1
        a.higherOrbital = '3Gsp3P x3H'
        a.higherJ = 0

        a.normalizedDepth = 0.234
        return a

    def testRepr(self, mock_transition):

        assert repr(mock_transition) == 'Transition(5580.345 Å, 54, 1)'

    def testStr(self, mock_transition):

        assert str(mock_transition) ==\
               '5580.345 Å Xe I'

        mock_transition.lowerEnergy = None
        mock_transition.higherEnergy = None

        assert str(mock_transition) == '5580.345 Å Xe I'

    def testLabel(self, mock_transition):

        assert mock_transition.label == '5580.345Xe1'

    def testNistFormatting(self, mock_transition):
        expected_string1 = '5580.345 | 17920.039 | Xe I  ' +\
                           '|    17.012 - 12051.823 | (4F)4s a5F ' +\
                           '                         | 3/2  ' +\
                           '| 3Gsp3P x3H                          ' +\
                           '| 0    | 0.234 | \n'
        expected_string2 = '5580.345 | 17920.039 | Xe I  ' +\
                           '|    17.012 - 12051.823 | (4F)4s a5F ' +\
                           '                         | 3/2  ' +\
                           '| 3Gsp3P x3H                          ' +\
                           '| 0    | 0.234 | 0 \n'

        assert mock_transition.formatInNistStyle() == expected_string1

        mock_transition.blendedness = 0
        assert mock_transition.formatInNistStyle() == expected_string2
