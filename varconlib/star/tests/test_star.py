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
                    dump_data=False)

    def testNonExistentDir(self):
        with pytest.raises(RuntimeError):
            Star('HD117618', star_dir='/nonsensical_dir_that_should_not_exist',
                 suffix='int')

    def testLabelMethods(self):
        s = Star('HD1111')
        with pytest.raises(AttributeError):
            s._p_label('')
        with pytest.raises(AttributeError):
            s._t_label('')
        with pytest.raises(KeyError):
            s._o_label('')

    def testName(self, test_star):

        assert test_star.name == 'HD117618'

    def testArrayShapes(self, test_star):

        assert test_star.fitMeansArray.shape == (3, 184)
        assert test_star.fitErrorsArray.shape == (3, 184)
        assert test_star.pairSeparationsArray.shape == (3, 284)
        assert test_star.pairSepErrorsArray.shape == (3, 284)

    def testTransitionDict(self, test_star):

        assert test_star._t_label('4217.791Fe1_16') == 0
        assert test_star._o_label('2005-05-02T03:49:08.735') == 0
        assert test_star._o_label(dt.datetime(year=2005, month=5,
                                              day=2, hour=3,
                                              minute=49, second=8,
                                              microsecond=735000)) == 0
        with pytest.raises(KeyError):
            test_star._o_label(dt.datetime(year=2000, month=1,
                                           day=1, hour=0,
                                           minute=0, second=0))
        assert test_star._p_label('4217.791Fe1_4219.893V1_16') == 0

    def testDumpData(self, test_star, tmp_dir):

        star_name = test_star.name
        tmp_file_path = tmp_dir / f'{star_name}_data.hdf5'
        test_star.dumpDataToDisk(tmp_file_path)
        new_star = Star(star_name, tmp_dir)

        attr_names = ('fitMeansArray', 'fitErrorsArray',
                      'pairSeparationsArray', 'pairSepErrorsArray')

        for name in attr_names:
            assert u.array.allclose_units(getattr(new_star, name),
                                          getattr(test_star, name))
            assert getattr(new_star, name).units == getattr(test_star,
                                                            name).units
        assert new_star._obs_date_dict == test_star._obs_date_dict
        assert new_star._transition_label_dict == test_star.\
            _transition_label_dict
        assert new_star._pair_label_dict == test_star._pair_label_dict
