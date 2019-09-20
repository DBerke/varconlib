#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:16:18 2019

@author: dberke

VarConLib -- the Varying Constants Library

"""

from pathlib import Path


__all__ = ['base_dir', 'data_dir', 'masks_dir', 'pickles_dir']


# Define some important paths to be available globally relative to the
# absolute path of the parent directory.

base_dir = Path(__file__).parent

data_dir = base_dir / 'data'

masks_dir = data_dir / 'masks'

pickles_dir = data_dir / 'pickles'
