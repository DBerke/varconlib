#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:41:45 2019

@author: dberke

setup.py file.
"""

from setuptools import setup, find_packages

setup(name='varconlib',
      author='Daniel Berke',
      maintainer='DBerke',
      packages=find_packages(),
      package_dir={'varconlib': "varconlib"})
