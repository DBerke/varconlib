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


class TransitionPairError(Error):
    """Errors relating to TransitionPair objects."""

    pass


class SameWavelengthsError(TransitionPairError):
    """Exception to raise if TransitionPair is given two transitions with the
    same wavelength.

    """

    def __init__(self, message=None):
        self.message = message
