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


class BadRadialVelocityError(Error):
    """Error to raise when radial velocity in a file is 'bad'.

    Some HARPS spectra can have non-useful radial velocities, such as -99999.

    """

    def __init__(self, message):
        self.message = message
