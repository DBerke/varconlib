#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:40:58 2018

@author: dberke

Test script for the HARPSFile2D class and subclasses
"""

from pathlib import Path
import shutil

import numpy as np
import pytest
import unyt as u

import varconlib
from varconlib.miscellaneous import wavelength2index
from varconlib.obs2d import HARPSFile2D, HARPSFile2DScience

base_test_file = varconlib.data_dir /\
                 'HARPS.2012-02-26T04:02:48.797_e2ds_A.fits'

if not base_test_file.exists():
    pytest.skip('Test file not available.', allow_module_level=True)


@pytest.fixture(scope='module')
def generic_test_file(tmpdir_factory):
    # Find the pristine test file to clone for use in the tests.

    tmp_dir = Path(tmpdir_factory.mktemp('test2d').strpath)

    test_file = tmp_dir / 'test_fits_file.fits'

    shutil.copy(str(base_test_file), str(test_file))

    return test_file


class TestGeneric2DFile(object):

    @pytest.fixture(scope='class')
    def s(self, generic_test_file):

        return HARPSFile2D(generic_test_file)

    def testBadFilename(self):
        with pytest.raises(RuntimeError):
            HARPSFile2D(1)

    def testNonExistentFilename(self):
        with pytest.raises(FileNotFoundError):
            HARPSFile2D('nonexistent/filename')

    def testRawFileRead(self, s):
        assert s.getHeaderCard('INSTRUME') == 'HARPS'

    def testHasAttributes(self, s):
        assert hasattr(s, '_header')
        assert hasattr(s, '_rawData')
        assert hasattr(s, 'dateObs')

#    def testRepr(self, s):
#        assert repr(s) == 'HARPSFile2D({})'.format(generic_test_file
#
#    def testStr(sel, s):
#        assert str(s) == 'HIP-40133, {}'.format(generic_test_file.stem)


class TestScience2DFile(object):

    @pytest.fixture(scope='class')
    def s(self, generic_test_file):

        return HARPSFile2DScience(generic_test_file,
                                  use_new_coefficients=False,
                                  use_pixel_positions=False)

    def testNonExistentFilename(self):
        with pytest.raises(FileNotFoundError):
            HARPSFile2DScience('nonexistent/filename')

    def testObsFileRead(self, s):
        assert s.getHeaderCard('INSTRUME') == 'HARPS'

    def testHasGenericAttributes(self, s):
        assert hasattr(s, '_header')
        assert hasattr(s, '_rawData')

    def testHasPropertyBERV(self, s):
        assert hasattr(s, 'BERV')

    def testHasPropertyRadialVelocity(self, s):
        assert hasattr(s, 'radialVelocity')

    def testHasPropertyDateObs(self, s):
        assert hasattr(s, 'dateObs')

    def testHasPropertyAirmassStart(self, s):
        assert hasattr(s, 'airmassStart')
        assert isinstance(s.airmassStart, float)

    def testHasPropertyAirmassEnd(self, s):
        assert hasattr(s, 'airmassEnd')
        assert isinstance(s.airmassEnd, float)

    def testHasPropertyAirmass(self, s):
        assert hasattr(s, 'airmass')
        assert isinstance(s.airmass, float)

    def testHasWavelengthArray(self, s):
        assert hasattr(s, 'wavelengthArray')

    def testHasBarycentricArray(self, s):
        assert hasattr(s, 'barycentricArray')

    def testHasPhotonFluxArray(self, s):
        assert hasattr(s, 'photonFluxArray')

    def testHasErrorArray(self, s):
        assert hasattr(s, 'errorArray')

    def testHasBlazeArray(self, s):
        assert hasattr(s, 'blazeArray')

    def testHasVacuumArray(self, s):
        assert hasattr(s, 'vacuumArray')

    def testHasRVCorrectedArray(self, s):
        assert hasattr(s, 'rvCorrectedArray')

    def testArraysShapes(self, s):
        assert np.shape(s.wavelengthArray) == (72, 4096)
        assert np.shape(s.barycentricArray) == (72, 4096)
        assert np.shape(s.photonFluxArray) == (72, 4096)
        assert np.shape(s.errorArray) == (72, 4096)
        assert np.shape(s.blazeArray) == (72, 4096)
        if hasattr(s, '_pixelLowerArray'):
            assert np.shape(s.pixelLowerArray) == (72, 4096)
            assert np.shape(s.pixelUpperArray) == (72, 4096)

    def testFindWavelengthInOneOrder(self, s):
        assert s.findWavelength(5039 * u.angstrom, s.barycentricArray,
                                mid_most=True) == 40

    def testFineWavelengthInTwoOrders(self, s):
        assert s.findWavelength(5039 * u.angstrom, s.barycentricArray,
                                mid_most=False) == (39, 40)
        index1 = wavelength2index(5039 * u.angstrom, s.barycentricArray[39])
        index2 = wavelength2index(5039 * u.angstrom, s.barycentricArray[40])
        assert abs(index1 - 2047.5) > abs(index2 - 2047.5)

    def testUpdateFile(self, generic_test_file):
        a = HARPSFile2DScience(generic_test_file)
        a = HARPSFile2DScience(generic_test_file, update=['ALL'])
        assert a.getHeaderCard('INSTRUME') == 'HARPS'
