#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:50:42 2019

@author: dberke

Tests for star.py.

"""

import datetime as dt
from pathlib import Path

import pytest
import unyt as u

import varconlib as vcl
from varconlib.star import Star

# Tell pytest to ignore SerializedWarnings that pop up from storing bidicts
# in an HDF5 file.
pytestmark = pytest.mark.filterwarnings("ignore::hickle.hickle."
                                        "SerializedWarning")


base_test_dir = vcl.data_dir / f'spectra/HD117618'

if not base_test_dir.exists():
    pytest.skip('Test directory not available.', allow_module_level=True)


@pytest.fixture(scope='module')
def test_dir():

    return base_test_dir


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):

    tmpdir = Path(tmpdir_factory.mktemp('test_star').strpath)

    return tmpdir


class TestStar(object):

    @pytest.fixture(scope='class')
    def test_star(self, test_dir):
        return Star('HD117618', star_dir=test_dir, suffix='int',
                    load_data=False)

    def testNonExistentDir(self):
        with pytest.raises(RuntimeError):
            Star('HD117618', star_dir='/nonsensical_dir_that_should_not_exist',
                 suffix='int')

    def testIndexMethods(self):
        s = Star('HD1111')
        with pytest.raises(KeyError):
            s.p_index('')
        with pytest.raises(KeyError):
            s.t_index('')
        with pytest.raises(KeyError):
            s.od_index('')

    def testLabelMethods(self):
        s = Star('HD1111')
        with pytest.raises(KeyError):
            s.p_label('')
        with pytest.raises(KeyError):
            s.t_label('')
        with pytest.raises(KeyError):
            s.od_date('')

    def testName(self, test_star):
        assert test_star.name == 'HD117618'

    def testFiberSplitIndex(self, test_star):
        assert test_star.fiberSplitIndex is None

    def testArrayShapes(self, test_star):
        assert test_star.fitMeansArray.shape == (3, 184)
        assert test_star.fitErrorsArray.shape == (3, 184)
        assert test_star.pairSeparationsArray.shape == (3, 284)
        assert test_star.pairSepErrorsArray.shape == (3, 284)

    def testPairBidict(self, test_star):
        assert test_star.p_index('4217.791Fe1_4219.893V1_16') == 0
        assert test_star.p_label(283) == '6774.190Ni1_6788.733Fe1_70'

    def testTransitionBidict(self, test_star):
        assert test_star.t_index('4217.791Fe1_16') == 0
        assert test_star.t_label(183) == '6788.733Fe1_70'

    def testObsevationDateBidict(self, test_star):
        assert test_star.od_index('2005-05-02T03:49:08.735') == 0
        assert test_star.od_index(dt.datetime(year=2005, month=5,
                                              day=2, hour=3,
                                              minute=49, second=8,
                                              microsecond=735000)) == 0
        with pytest.raises(KeyError):
            test_star.od_index(dt.datetime(year=2000, month=1,
                                           day=1, hour=0,
                                           minute=0, second=0))
        assert test_star.od_date(2) == '2010-04-21T03:55:19.107'
        with pytest.raises(KeyError):
            test_star.od_date(3)

    def testRadialVelocity(self, test_star):
        assert pytest.approx(test_star.radialVelocity, 1.1 * u.m / u.s)

    def testDumpAndRestoreData(self, test_star, tmp_dir):

        star_name = test_star.name
        tmp_file_path = tmp_dir / f'{star_name}_data.hdf5'
        test_star.dumpDataToDisk(tmp_file_path)
        new_star = Star(star_name, tmp_dir)

        for name in test_star.unyt_arrays.values():
            assert u.array.allclose_units(getattr(new_star, name),
                                          getattr(test_star, name))
            assert getattr(new_star, name).units == getattr(test_star,
                                                            name).units
        assert new_star._obs_date_bidict == test_star._obs_date_bidict
        assert new_star._transition_bidict == test_star.\
            _transition_bidict
        assert new_star._pair_bidict == test_star._pair_bidict
