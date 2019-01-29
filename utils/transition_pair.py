#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:52:11 2019

@author: dberke

The TransitionPair class contains information about a single pair of atomic
transitions.
"""

import unyt as u

class TransitionPair(object):
    """Holds information relating to a single pair of transition lines, in the
    form of two Transition objects.

    """

    def __init__(self, transition1, transition2):
        """Create an instance of TransitionPair.

        Parameters
        ----------
        transition1, transition2: transition_line.Transition objects
        The transitions should be two Transition objects, each representing a
        sincle atomic transition.

        """

        if transition1 > transition2:
            self._transitionLowerEnergy = transition1
            self._transitionHigherEnergy = transition2
        elif transition1 < transition2:
            self._transitionLowerEnergy = transition2
            self._transitionHigherEnergy = transition1
        else:
            raise TransitionPairSameWavelengthError

        self._nominalSeparation = self.getNominalSeparation()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self._transitionHigherEnergy,
                                   self._transitionLowerEnergy)

    def __str__(self):
        return '{} ({} {}): {:.4f}, {:.4f}'.format(self.__class__.__name__,
                self._transitionHigherEnergy.atomicSymbol,
                self._transitionHigherEnergy.ionizationState,
                self._transitionHigherEnergy.wavelength,
                self._transitionLowerEnergy.wavelength)

    def __eq__(self, other):
        if (self._transitionLowerEnergy == other._transitionLowerEnergy)\
          and (self._transitionHigherEnergy == other._transitionHigherEnergy):
            return True
        else:
            return False

    def __gt__(self, other):
        if self._transitionLowerEnergy > other._transitionLowerEnergy:
            return True
        else:
            return False

    def __lt__(self, other):
        if self._transitionLowerEnergy < other._transitionLowerEnergy:
            return True
        else:
            return False

    def getNominalSeparation(self):
        """Return the nominal separation between the wavelengths of the two
        transitions in the pair.

        """

        return self._transitionHigherEnergy.wavelength -\
            self._transitionLowerEnergy.wavelength


class TransitionPairError(Exception):
    """Base class for exceptions in this module."""
    pass


class TransitionPairSameWavelengthError(TransitionPairError):
    """Exception to raise if TransitionPair is given two transitions with the
    same wavelength.

    """

    pass
