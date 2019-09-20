#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:56:34 2019

@author: dberke

Script to generate synthetic e2ds spectra.
"""

import os
import datetime
import numpy as np
from numpy.random import normal
import configparser
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm, trange
from obs2d import HARPSFile2D

config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read('/Users/dberke/code/config/variables.cfg')
data_dir = Path(config['PATHS']['data_dir'])
base_file = data_dir / 'HARPS.2012-02-26T04:02:48.797_e2ds_A.fits'

synth_dir = data_dir / 'syntheticSpectra'

if not synth_dir.exists():
    os.mkdir(synth_dir)


def create_synthetic_spectrum(flux_array):
    """Create a synthetic e2ds spectrum and save it to a file.

    Parameters
    ----------
    flux_array : array_like of floats
        An array representing the original flux.

    Returns
    -------
    `np.ndaray`
        A 72 by 4096 array of gaussian noise added to the original.
    """

    return np.array(normal(loc=0, scale=np.sqrt(flux_array),
                           size=flux_array.shape), dtype=float)


base_obs = HARPSFile2D(base_file)
blaze_file = base_obs.getHeaderCard('HIERARCH ESO DRS BLAZE FILE')

file_date = blaze_file[6:16]

blaze_file_dir = Path(config['PATHS']['blaze_file_dir'])
blaze_file_path = blaze_file_dir / 'data/reduced/{}'.format(file_date)\
                    / blaze_file

blaze_file_path = Path('/Users/dberke/code/data/'
                       'HARPS.2012-02-25T22:07:11.413_blaze_A.fits')

if not blaze_file_path.exists():
    print(blaze_file_path)
    raise RuntimeError("Blaze file path doesn't exist!")

blaze_pattern = HARPSFile2D(blaze_file_path)._rawData

for snr in trange(0, 220, 20):
    new_file_name = 'synthetic_SNR{}_e2ds_A.fits'.format(snr)
    new_file_path = synth_dir / new_file_name
    tqdm.write('Creating file {}'.format(new_file_name))

    blaze_function = np.ones([72, 4096])
    for row in range(blaze_pattern.shape[0]):
        blaze_row = np.array([x if x >= 0 else 0 for x in blaze_pattern[row]])
        blaze_function[row] = blaze_row

    blaze_function *= snr ** 2
    errors_array = create_synthetic_spectrum(blaze_function)

    flux_array = blaze_function + errors_array
    print(flux_array[0, :10])
    print(blaze_function[0, :10])

    new_hdu = fits.PrimaryHDU(data=flux_array, header=base_obs._header)
    new_hdu.header['comment'] = 'Synthetic spectrum created {}'.format(
            datetime.date.today().isoformat())

    new_hdulist = fits.HDUList([new_hdu])
    new_hdulist.writeto(new_file_path, overwrite=True)
