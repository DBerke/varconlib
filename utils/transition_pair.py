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

        # Note that comparison of transitions compares their wavelength, not
        # their energy. So the "lower energy transition" is the one with the
        # longer wavelength.
        if transition1 > transition2:
            self._lowerEnergyTransition = transition1
            self._higherEnergyTransition = transition2
        elif transition1 < transition2:
            self._lowerEnergyTransition = transition2
            self._higherEnergyTransition = transition1
        else:
            raise TransitionPairSameWavelengthError

        self._nominalSeparation = self.getNominalSeparation()

    def __iter__(self):
        return iter([self._lowerEnergyTransition,
                     self._higherEnergyTransition])

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self._higherEnergyTransition,
                                   self._lowerEnergyTransition)

    def __str__(self):
        return '{} ({} {}): {:.4f}, {:.4f}'.format(self.__class__.__name__,
                self._higherEnergyTransition.atomicSymbol,
                self._higherEnergyTransition.ionizationState,
                self._higherEnergyTransition.wavelength,
                self._lowerEnergyTransition.wavelength)

    def __eq__(self, other):
        if (self._lowerEnergyTransition == other._lowerEnergyTransition)\
          and (self._higherEnergyTransition == other._higherEnergyTransition):
            return True
        else:
            return False

    def __gt__(self, other):
        if self == other:
            return False
        elif self._lowerEnergyTransition > other._lowerEnergyTransition:
            return True
        elif self._lowerEnergyTransition == other._lowerEnergyTransition:
            if self._higherEnergyTransition > other._higherEnergyTransition:
                return True
            else:
                return False
        else:
            return False

    def __lt__(self, other):
        if self == other:
            return False
        elif self._lowerEnergyTransition < other._lowerEnergyTransition:
            return True
        elif self._lowerEnergyTransition == other._lowerEnergyTransition:
            if self._higherEnergyTransition < other._higherEnergyTransition:
                return True
            else:
                return False
        else:
            return False

    def getNominalSeparation(self):
        """Return the nominal separation between the wavelengths of the two
        transitions in the pair.

        """

        return self._higherEnergyTransition.wavelength -\
            self._lowerEnergyTransition.wavelength


class TransitionPairError(Exception):
    """Base class for exceptions in this module."""
    pass


class TransitionPairSameWavelengthError(TransitionPairError):
    """Exception to raise if TransitionPair is given two transitions with the
    same wavelength.

    """

    pass
