#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:50:42 2019

@author: dberke

Tests for star.py.

"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import unyt as u

import varconlib as vcl
from varconlib.exceptions import StarDirectoryNotFoundError
from varconlib.star import Star


pytestmark = pytest.mark.filterwarnings(("ignore::DeprecationWarning"))

base_test_dir = vcl.data_dir / 'spectra/HD117618'

if not base_test_dir.exists():
    pytest.skip('Test directory not available.', allow_module_level=True)


@pytest.fixture(scope='module')
def test_dir():

    return base_test_dir


@pytest.fixture(scope='module')
def tmp_dir(tmp_path_factory):

    tmpdir = Path(tmp_path_factory.mktemp('test_star'))

    return tmpdir


class TestStar(object):

    @pytest.fixture(scope='class')
    def test_star(self, test_dir):
        return Star('HD117618', star_dir=test_dir,
                    load_data=False, init_params="Casagrande2011",
                    perform_model_correction=True)

    def testNonExistentDir(self):
        with pytest.raises(StarDirectoryNotFoundError):
            Star('HD117618', star_dir='/nonsensical_dir_that_should_not_exist')

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

    def testNumObs(self, test_star):
        assert test_star.getNumObs(slice(None, None)) == 3

    @pytest.mark.parametrize("norm", [True, False])
    def testTransitionOffsetPattern(self, test_star, norm):
        assert len(test_star.getTransitionOffsetPattern(slice(None, None),
                                                        normalized=norm)[0])\
            == len(test_star._transition_bidict.keys())

    @pytest.mark.parametrize('obs_num,expected',
                             [(0, -0.13005375),
                              (1, 24.92201306),
                              (2, 4.58199186)])
    def testBERV(self, test_star, obs_num, expected):
        assert test_star.bervArray[obs_num] ==\
            pytest.approx(expected * u.km / u.s)

    @pytest.mark.skip('Test needs revision.')
    def testSaveAndRestoreData(self, test_star, tmp_dir):
        star_name = test_star.name
        tmp_file_path = tmp_dir
        test_star.saveDataToDisk(tmp_file_path)
        new_star = Star(star_name, star_dir=tmp_dir,
                        init_params='Nordstrom2004',
                        load_data=True)

        for name in test_star.unyt_arrays.values():
            if np.any(np.isnan(getattr(test_star, name))):
                assert np.all(np.isnan(getattr(test_star, name)) ==
                              np.isnan(getattr(new_star, name)))
            else:
                assert u.array.allclose_units(getattr(new_star, name),
                                              getattr(test_star, name))
            assert getattr(new_star, name).units == getattr(test_star,
                                                            name).units
        assert new_star._obs_date_bidict == test_star._obs_date_bidict
        assert new_star._transition_bidict == test_star.\
            _transition_bidict
        assert new_star._pair_bidict == test_star._pair_bidict
