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
    def testRawFileRead(self, generic_test_file):
        s = HARPSFile2D(generic_test_file)
        assert s.getHeaderCard('INSTRUME') == 'HARPS',\
            "Couldn't read header card."


class TestScience2DFile(object):
    def testObsFileRead(self, generic_test_file):
        s = HARPSFile2DScience(generic_test_file)
        assert s.getHeaderCard('INSTRUME') == 'HARPS',\
            "Couldn't read header card."

    def testArraysPresent(self, generic_test_file):
        s = HARPSFile2DScience(generic_test_file)
        assert np.shape(s._wavelengthArray) == (72, 4096),\
            f'Wavelength array wrong shape! {np.shape(s._wavelengthArray)}'
        assert np.shape(s._photonFluxArray) == (72, 4096),\
            f'Flux array wrong shape! {np.shape(s._photonFluxArray)}'
        assert np.shape(s._errorArray) == (72, 4096),\
            f'Error array wrong shape! {np.shape(s._errorArray)}'
