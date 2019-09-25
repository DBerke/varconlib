#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:27:00 2019

@author: dberke

Module to contain custom exceptions.
"""


class Error(Exception):
    """Base class for exceptions for this module."""

    pass


# Errors relating to observation files.
class ObservationError(Error):
    """Errors relating to importing or working with observation files."""

    pass


class BadRadialVelocityError(ObservationError):
    """Error to raise when radial velocity in a file is 'bad'.

    Some HARPS spectra can have non-useful radial velocities, such as -99999.

    """

    def __init__(self, message=None):
        self.message = message


class NewCoefficientsNotFoundError(ObservationError):
    """Error to raise when new calibration coefficients can't be found for an
    observation.

    Slightly more specific than a FileNotFoundError, since if use of the new
    calibration is requested we should probably not use observations for which
    we don't have the new coefficients.

    """

    def __init__(self, message=None):
        self.message = message


class BlazeFileNotFoundError(ObservationError):
    """Error to raise when the blaze file for an observation isn't available.

    """

    def __init__(self, message=None):
        self.message = message


class WavelengthNotFoundInArrayError(ObservationError):
    """Error to raise when a requested wavelength is not found in the given
    wavelength array of a observation.

    """

    def __init__(self, message=None):
        self.message = message


# Errors relating to attempting to fit absorption features.
class FeatureFitError(Error):
    """Errors relating to fitting absorption features."""

    pass


class PositiveAmplitudeError(FeatureFitError):
    """Error for when fitting an absorption feature returns a positive
    amplitude.

    By definition an absorption feature should have a negative amplitude
    compared to the continuum.

    """

    def __init__(self, message=None):
        self.message = message


# Errors relating to Transition objects.
class TransitionLineError(Error):
    """Errors relating to Transition objects."""

    pass


class AtomicNumberError(TransitionLineError):
    """Exception to raise if an unphsyical atomic number is given.

    """

    def __init__(self, message=None):
        self.message = message


class IonizationStateError(TransitionLineError):
    """Exception to raise if an unphysical ionization state is given.

    """

    def __init__(self, message=None):
        self.message = message


class BadElementInputError(TransitionLineError):
    """Exception to raise if a bad element parameter (too long, not a real
    element, not a string or integer, etc.) is given.

    """

    def __init__(self, message=None):
        self.message = message


# Errors relating to TransitionPair objects.
class TransitionPairError(Error):
    """Errors relating to TransitionPair objects."""

    pass


class SameWavelengthsError(TransitionPairError):
    """Exception to raise if TransitionPair is given two transitions with the
    same wavelength.

    """

    def __init__(self, message=None):
        self.message = message
