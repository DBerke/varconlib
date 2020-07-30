#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:22:24 2019

@author: dberke

Tests for the transition_pair.TransitionPair object.
"""

import pytest

import unyt as u

from varconlib.exceptions import SameWavelengthsError, WrongOrdersNumberError
from varconlib.transition_line import Transition
from varconlib.transition_pair import TransitionPair


class TestTransitionPair(object):

    # Create some dummy transitions to use for testing.
    @pytest.fixture(scope='class')
    def transition_1(self):
        return Transition(500 * u.nm, 26, 1)

    @pytest.fixture(scope='class')
    def transition_2(self):
        return Transition(600 * u.nm, 26, 1)

    @pytest.fixture(scope='class')
    def transition_3(self):
        return Transition(650 * u.nm, 26, 1)

    def testSameWavelengthsPair(self, transition_1):
        with pytest.raises(SameWavelengthsError):
            TransitionPair(transition_1, transition_1)

    def testVelocitySeparation(self, transition_1, transition_2):
        t = TransitionPair(transition_1, transition_2)
        s = t.velocitySeparation
        assert s.value == pytest.approx(54507719.63)

    def testPairEquality(self, transition_1, transition_2):
        p1 = TransitionPair(transition_1, transition_2)
        p2 = TransitionPair(transition_1, transition_2)
        assert p1 == p2
        with pytest.raises(ValueError):
            assert p1 == "A non-pair"

    def testComparison(self, transition_1, transition_2, transition_3):
        p1 = TransitionPair(transition_1, transition_2)
        p2 = TransitionPair(transition_2, transition_3)
        p3 = TransitionPair(transition_1, transition_3)
        assert p1.__lt__(p2)
        assert p2.__gt__(p1)
        assert not p1.__lt__(p1)
        assert not p1.__gt__(p1)
        assert not p1 == p2
        assert not p1 == p3
        assert p1.__lt__(p3)
        assert p3.__gt__(p1)
        assert p3.__lt__(p2)
        assert p2.__gt__(p3)
        assert not p1.__gt__(p3)
        assert not p1.__gt__(p2)
        assert not p2.__lt__(p1)
        assert not p2.__lt__(p3)
        assert not p3.__lt__(p1)
        with pytest.raises(ValueError):
            assert p1.__lt__('A non-pair')
        with pytest.raises(ValueError):
            assert p1.__gt__('A non-pair')

    def testAutomaticEnergyOrdering(self, transition_1, transition_2):
        p = TransitionPair(transition_2, transition_1)
        assert p._lowerEnergyTransition == transition_2
        assert p._higherEnergyTransition == transition_1

    def testRepresentations(self, transition_1, transition_2):
        p = TransitionPair(transition_1, transition_2)
        assert p.label == '5000.000Fe1_6000.000Fe1'
        assert repr(p) == 'TransitionPair(5000.000 Å Fe I, 6000.000 Å Fe I)'
        assert str(p) == 'Pair: Fe 1 5000.000 Å, Fe 1 6000.000 Å'

    def testIter(self, transition_1, transition_2):
        p = TransitionPair(transition_1, transition_2)
        assert transition_1 in p
        assert transition_2 in p

    def testBlendTuple(self, transition_1, transition_2):
        p1 = TransitionPair(transition_1, transition_2)
        with pytest.raises(AttributeError):
            assert p1.blendTuple == (0, 0)
        transition_1.blendedness = 1
        transition_2.blendedness = 2
        p2 = TransitionPair(transition_1, transition_2)
        assert p2.blendTuple == (1, 2)

    def testOrdersToMeasureIn(self, transition_1, transition_2):
        p1 = TransitionPair(transition_1, transition_2)
        assert p1.ordersToMeasureIn is None
        p1.ordersToMeasureIn = [29, 30]
        assert p1.ordersToMeasureIn == (29, 30)
        with pytest.raises(ValueError):
            p1.ordersToMeasureIn = "(29, 30"
        with pytest.raises(WrongOrdersNumberError):
            p1.ordersToMeasureIn = (29, 30, 31)
            p1.ordersToMeasureIn = []
