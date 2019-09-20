#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:22:24 2019

@author: dberke

Tests for the transition_pair.TransitionPair object.
"""

import pytest

import unyt as u

from varconlib.exceptions import SameWavelengthsError
from varconlib.transition_line import Transition
from varconlib.transition_pair import TransitionPair


class TestTransitionPair(object):

    # Create some dummy transitions to use for testing.
    @pytest.fixture(scope='class')
    def transition_1(self):
        transition_obj1 = Transition(500 * u.nm, 26, 1)
        return transition_obj1

    @pytest.fixture(scope='class')
    def transition_2(self):
        transition_obj2 = Transition(600 * u.nm, 26, 1)
        return transition_obj2

    @pytest.fixture(scope='class')
    def transition_3(self):
        transition_obj3 = Transition(650 * u.nm, 26, 1)
        return transition_obj3

    def test_same_wavelengths_pair(self, transition_1):
        with pytest.raises(SameWavelengthsError):
            TransitionPair(transition_1, transition_1)

    def test_velocity_separation(self, transition_1, transition_2):
        t = TransitionPair(transition_1, transition_2)
        s = t.velocitySeparation
        assert s.value == pytest.approx(54507719.63)

    def test_pair_equality(self, transition_1, transition_2):
        p1 = TransitionPair(transition_1, transition_2)
        p2 = TransitionPair(transition_1, transition_2)
        assert p1 == p2

    def test_comparison(self, transition_1, transition_2, transition_3):
        p1 = TransitionPair(transition_1, transition_2)
        p2 = TransitionPair(transition_2, transition_3)
        p3 = TransitionPair(transition_1, transition_3)
        assert p1 < p2
        assert p2 > p1
        assert not p1 < p1
        assert not p1 > p1
        assert not p1 == p2
        assert p1 < p3
        assert p3 > p1
        assert not p1 > p3
        assert not p3 < p1

    def test_automatic_energy_ordering(self, transition_1, transition_2):
        p = TransitionPair(transition_2, transition_1)
        assert p._lowerEnergyTransition == transition_2
        assert p._higherEnergyTransition == transition_1

    def test_representations(self, transition_1, transition_2):
        p = TransitionPair(transition_1, transition_2)
        assert p.label == '5000.000Fe1_6000.000Fe1'
        assert repr(p) == 'TransitionPair(5000.000 Å Fe I, 6000.000 Å Fe I)'
        assert str(p) == 'Pair: Fe 1 5000.000 Å, Fe 1 6000.000 Å'
