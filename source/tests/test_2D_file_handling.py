#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:40:58 2018

@author: dberke

Test script for the HARPSFile2D class and subclasses
"""

import pytest
import configparser
import numpy as np
import shutil
from pathlib import Path
import unyt as u
from varconlib import wavelength2index
from obs2d import HARPSFile2D, HARPSFile2DScience


@pytest.fixture(scope='module')
def generic_test_file(tmpdir_factory):
    config = configparser.ConfigParser(interpolation=configparser.
                                       ExtendedInterpolation())
    config.read('/Users/dberke/code/config/variables.cfg')

    data_dir = Path(config['PATHS']['data_dir'])
    base_test_file = data_dir / 'HARPS.2012-02-26T04:02:48.797_e2ds_A.fits'

    tmp_dir = Path(tmpdir_factory.mktemp('test2d').strpath)

    test_file = tmp_dir / 'test_fits_file.fits'

    shutil.copy(str(base_test_file), str(test_file))

    return test_file


class TestGeneric2DFile(object):

    @pytest.fixture(scope='class')
    def s(self, generic_test_file):

        return HARPSFile2D(generic_test_file)

    def testRawFileRead(self, s):
        assert s.getHeaderCard('INSTRUME') == 'HARPS'

    def testHasAttributes(self, s):
        assert hasattr(s, '_header')
        assert hasattr(s, '_rawData')
        assert hasattr(s, 'dateObs')


class TestScience2DFile(object):

    @pytest.fixture(scope='class')
    def s(self, generic_test_file):

        return HARPSFile2DScience(generic_test_file, use_new_coefficients=True,
                                  use_pixel_positions=True)

    def testObsFileRead(self, s):
        assert s.getHeaderCard('INSTRUME') == 'HARPS'

    def testHasGenericAttributes(self, s):
        assert hasattr(s, '_header')
        assert hasattr(s, '_rawData')

    def testHasSpecificAttributes(self, s):
        assert hasattr(s, 'BERV')
        assert hasattr(s, 'radialVelocity')
        assert hasattr(s, 'dateObs')

    def testArrayProperties(self, s):
        assert hasattr(s, 'wavelengthArray')
        assert hasattr(s, 'barycentricArray')
        assert hasattr(s, 'photonFluxArray')
        assert hasattr(s, 'errorArray')
        assert hasattr(s, 'blazeArray')
        # Test for created-on-the-fly arrays:
        assert hasattr(s, 'vacuumArray')
        assert hasattr(s, 'rvCorrectedArray')
        assert hasattr(s, 'pixelLowerArray')
        assert hasattr(s, 'pixelUpperArray')

    def testArraysShapes(self, s):
        assert np.shape(s.wavelengthArray) == (72, 4096)
        assert np.shape(s.barycentricArray) == (72, 4096)
        assert np.shape(s.photonFluxArray) == (72, 4096)
        assert np.shape(s.errorArray) == (72, 4096)
        assert np.shape(s.blazeArray) == (72, 4096)
        assert np.shape(s.pixelLowerArray) == (72, 4096)
        assert np.shape(s.pixelUpperArray) == (72, 4096)

    def testFindWavelength(self, s):
        assert s.findWavelength(5039 * u.angstrom, s.barycentricArray,
                                mid_most=True) == 40
        assert s.findWavelength(5039 * u.angstrom, s.barycentricArray,
                                mid_most=False) == (39, 40)
        index1 = wavelength2index(5039 * u.angstrom, s.barycentricArray[39])
        index2 = wavelength2index(5039 * u.angstrom, s.barycentricArray[40])
        assert abs(index1 - 2047.5) > abs(index2 - 2047.5)
