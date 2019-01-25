#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:15:44 2018

@author: dberke
"""

import unyt as u
from bidict import bidict

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
            internally.

        element : int or str
            The atomic number of the element the transition arises from, or the
            standard chemical symbol (e.g., 'He', or 'N').

        ionizationState : int
            The ionization state of the atom the transition arises from, after
            the astronomical convention where 1 is un-ionized, 2 is singly-
            ionized, 3 is doubly-ionized, etc.

        """

        self.wavelength = wavelength.to(u.nm)
        if type(element) is int:
            self.atomicNumber = element
            self.atomicSymbol = elements[self.atomicNumber]
        elif type(element) is str:
            cap_string = element.capitalize()
            try:
                self.atomicNumber = elements.inv[cap_string]
            except KeyError:
                print('Given atomic symbol not found in elements dictionary!')
                print('Atomic symbol given was "{}".'.format(element))
                raise
            self.atomicSymbol = cap_string

        self.ionizationState = ionizationState
        self.lowerEnergy = None
        self.lowerJ = None
        self.lowerOrbital = None
        self.higherEnergy = None
        self.higherJ = None
        self.higherOrbital = None

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__,
                self.wavelength.value, self.atomicNumber, self.ionzationState)

    def __str__(self):
        return "{} {} {} {:.4f}".format(self.wavelength, self.atomicSymbol,
                                        self.ionizationState, self.lowerEnergy)
