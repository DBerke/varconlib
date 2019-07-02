#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:22:24 2019

@author: dberke

Tests for the transition_pair.TransitionPair object.
"""

import pytest

import unyt as u

from exceptions import SameWavelengthsError
from transition_line import Transition
from transition_pair import TransitionPair


class TestTransitionPair(object):

    @pytest.fixture(scope='class')
    def transition_1(self):
        transition_obj1 = Transition(500 * u.nm, 26, 1)
        return transition_obj1

    @pytest.fixture(scope='class')
    def transition_2(self):
        transition_obj2 = Transition(600 * u.nm, 26, 1)
        return transition_obj2

    def test_same_wavelengths_pair(self, transition_1):
        with pytest.raises(SameWavelengthsError):
            TransitionPair(transition_1, transition_1)

    def test_nominal_separation(self, transition_1, transition_2):
        t = TransitionPair(transition_1, transition_2)
        s = t.nominalSeparation
        assert s.value == pytest.approx(100)

    def test_pair_equality(self, transition_1, transition_2):
        p1 = TransitionPair(transition_1, transition_2)
        p2 = TransitionPair(transition_1, transition_2)
        assert p1 == p2
