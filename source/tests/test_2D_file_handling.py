#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:40:58 2018

@author: dberke

Test script for the HARPSFile2D class and subclasses
"""

import numpy as np
from shutil import copyfile
from pathlib import Path
from astropy.io import fits
from file_io import HARPSFile2D, HARPSFile2DScience

base_test_file = Path('../../data/HARPS.2012-02-26T04:02:48.797_e2ds_A.fits')

test_file = 'test_fits_file.fits'

# First create a copy of this file to test on, since we need to append
# additional HDUs to it:

copyfile(str(base_test_file), test_file)


def test_raw_file_read():
    s = HARPSFile2D(test_file)
    assert s.getHeaderCard('INSTRUME') == 'HARPS', "Couldn't read header card."


def test_obs_file_read():
    s = HARPSFile2DScience(test_file)
    assert s.getHeaderCard('INSTRUME') == 'HARPS', "Couldn't read header card."


def test_arrays_present():
    s = HARPSFile2DScience(test_file)
    assert np.shape(s._wavelengthArray) == (72, 4096)
    assert np.shape(s._photonFluxArray) == (72, 4096)
    assert np.shape(s._errorArray) == (72, 4096)

#with fits.open(test_file) as hdulist:
#    assert len(hdulist) == 4, "Wrong number of HDUs present."
