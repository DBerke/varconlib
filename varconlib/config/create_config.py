#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:24:07 2018

@author: dberke

Script to create a config file for use with varconlib.
"""

import configparser
from pathlib import Path

config_path = Path(__file__).parent / 'variables.cfg'

config = configparser.ConfigParser()

config['PATHS'] = {'HARPS_dir': '/Volumes/External Storage/HARPS',
                   'blaze_file_dir': '${HARPS_dir}/blaze_files',
                   'wavelength_cal_dir': '${HARPS_dir}/wavelength_calibration',
                   'pictures_dir': '/Users/dberke/Pictures',
                   'stars_dir': '${pictures_dir}/Stars',
                   'output_dir': '/Users/dberke/data_output',
                   'temp_dir': '/Users/dberke/tmp'}

with open(config_path, 'w') as configfile:
    config.write(configfile)
    print('Created config file at path:')
    print(str(config_path))
