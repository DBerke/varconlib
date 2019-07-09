#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:15:44 2018

@author: dberke

The Transition class contains information about a single atomic transition.
"""

from fractions import Fraction
from math import isclose

from bidict import bidict
import unyt as u

elements = bidict({1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
                   8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg",  13: "Al",
                   14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
                   20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
                   26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga",
                   32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
                   38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
                   44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
                   50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
                   56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm",
                   62: "Sm", 63:  "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho",
                   68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
                   74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
                   80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
                   86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa",
                   92: "U"})

roman_numerals = bidict({1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                         6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'})


class Transition(object):
    """Class to hold information about a single atomic transition.

    """

    def __init__(self, wavelength, element, ionizationState):
        """
        Parameters
        ----------
        wavelength : unyt quantity with dimensions length
            The wavelength of the transition. Should be a unyt quantity, though
            it can be any unit of length and will be converted to nm
            internally. Should be in a vacuum wavelength scale.
        element : int or str
            The atomic number of the element the transition arises from, or the
            standard chemical symbol (e.g., 'He', or 'N'). Either can be given;
            on initialization it will use the given one to find the other from
            a bijective dictionary and set both for the instance. The number
            can also be given as a string; it will first check if the string
            can be converted to an integer before parsing it as an atomic
            symbol.
        ionizationState : int or str
            The ionization state of the atom the transition arises from, after
            the astronomical convention where 1 is un-ionized, 2 is singly-
            ionized, 3 is doubly-ionized, etc. It can also be given as a Roman
            numeral up to 10 (X).

        """

        # See if the wavelength has units already:
        try:
            self.wavelength = wavelength.to(u.nm)
        except AttributeError:
            # If not, raise an error.
            print('Given wavelength has no units!')
            raise
        # Check if the element is given as a number in a string.
        if (type(element) is str) and (len(element) < 3):
            try:
                self.atomicNumber = int(element)
                self.atomicSymbol = elements[self.atomicNumber]
            # If not, see if it's a correct atomic symbol.
            except ValueError:
                cap_string = element.capitalize()
                try:
                    self.atomicNumber = elements.inv[cap_string]
                except KeyError:
                    print('Given atomic symbol not in elements dictionary!')
                    print('Atomic symbol given was "{}".'.format(element))
                    raise
                self.atomicSymbol = cap_string
        elif type(element) is int:
            assert 0 < element < 119, 'Element number not in range [1, 118]!'
            self.atomicNumber = element
            self.atomicSymbol = elements[self.atomicNumber]

        elif (type(element) is int) and (len(element) > 2):
            print('Given string for parameter "element" is too long!')
        else:
            try:
                self.atomicNumber = int(element)
                self.atomicSymbol = elements[self.atomicNumber]
            except ValueError:
                raise TypeError("'element' parameter must be a valid "
                                "integer atomic number or atomic symbol " +
                                "(e.g., 'Fe').")

        # Next check the given ionization state.
        try:
            ionizationState = int(ionizationState)
            assert ionizationState >= 0, 'Ionization state cannot be negative.'
        except ValueError:
            # If it's a string, see if it's a Roman numeral.
            try:
                ionizationState = roman_numerals.inv[ionizationState]
            except KeyError:
                raise ValueError('Ionization state "{}" invalid!'.format(
                                 ionizationState))
        self.ionizationState = ionizationState

        self.lowerEnergy = None
        self._lowerJ = None
        self.lowerOrbital = None
        self.higherEnergy = None
        self._higherJ = None
        self.higherOrbital = None

    @property
    def lowerJ(self):
        return self._lowerJ

    @lowerJ.setter
    def lowerJ(self, new_lowerJ):
        if new_lowerJ is None:
            self._lowerJ = None
        elif type(new_lowerJ) is not Fraction:
            new_lowerJ = Fraction(new_lowerJ)
        self._lowerJ = new_lowerJ

    @property
    def higherJ(self):
        return self._higherJ

    @higherJ.setter
    def higherJ(self, new_higherJ):
        if new_higherJ is None:
            self._higherJ = None
        elif type(new_higherJ) is not Fraction:
            new_higherJ = Fraction(new_higherJ)
        self._higherJ = new_higherJ

    @property
    def atomicSpecies(self):
        return(f'{self.atomicSymbol} {roman_numerals[self.ionizationState]}')

    @property
    def wavenumber(self):
        return 1 / self.wavelength.to(u.cm)

    @wavenumber.setter
    def wavenumber(self, new_wavenumber):
        if type(new_wavenumber) is u.unyt_quantity:
            # TODO: Change to a try new_wavenumber.to(u.cm ** -1)...except
            if new_wavenumber.units == u.cm ** -1:
                self.wavelength = 1 / new_wavenumber
            else:
                raise ValueError('Units for given wavenumber are {}!'.format(
                        new_wavenumber.units))
        else:
            self.wavelength = 1 / (new_wavenumber * u.cm ** -1)

    @property
    def label(self):
        if (not hasattr(self, '_label')) or self._label is None:
            self._label = '{:.4f}{}{}'.format(self.wavelength.to(u.nm).value,
                                              self.atomicSymbol,
                                              self.ionizationState)
        return self._label

    def __repr__(self):
        return "{}({:.4f}, {}, {})".format(self.__class__.__name__,
                                           self.wavelength.to(u.nm),
                                           self.atomicNumber,
                                           self.ionizationState)

    def __str__(self):
        if (self.lowerEnergy is not None) and\
          (self.higherEnergy is not None):
            return "{:.4f} {} ({:.4f}, {:.4f})".format(self.wavelength.
                                                       to(u.nm),
                                                       self.atomicSpecies,
                                                       self.lowerEnergy,
                                                       self.higherEnergy)
        else:
            return "{:.4f} {}".format(self.wavelength,
                                      self.atomicSpecies)

    def __lt__(self, other):
        if self.wavelength.value < other.wavelength.value:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.wavelength.value > other.wavelength.value:
            return True
        else:
            return False

    def __eq__(self, other):
        if type(other) is Transition:
            # If the other thing to be compared is a transition, we need to
            # check all of its attributes and whether they A) exist and B) are
            # equal. Basically this function checks multiple ways they could be
            # NOT equal, and only if none of them trigger does it return that
            # they are.
            # It's still not perfect, as additional information could be
            # attached, but it should cover general use cases.
            if not (isclose(self.wavelength, other.wavelength,
                            rel_tol=1e-5) and
                    self.atomicNumber == other.atomicNumber and
                    self.ionizationState == other.ionizationState):
                return False
            if (self.lowerEnergy is not None) and\
                    (other.lowerEnergy is not None):
                if not isclose(self.lowerEnergy, other.lowerEnergy,
                               rel_tol=1e-5):
                    return False
            if (self.higherEnergy is not None) and\
                    (other.higherEnergy is not None):
                if not isclose(self.higherEnergy, other.higherEnergy,
                               rel_tol=1e-5):
                    return False
            if (self._lowerJ is not None) and\
                    (other._lowerJ is not None):
                if self._lowerJ != other._lowerJ:
                    return False
            if (self._higherJ is not None) and\
                    (other._higherJ is not None):
                if self._higherJ != other._higherJ:
                    return False
            if (self.lowerOrbital is not None) and\
                    (other.lowerOrbital is not None):
                if self.lowerOrbital != other.lowerOrbital:
                    return False
            if (self.higherOrbital is not None) and\
                    (other.higherOrbital is not None):
                if self.higherOrbital != other.higherOrbital:
                    return False
            return True
        else:
            return False
