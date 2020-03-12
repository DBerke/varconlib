#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:13:37 2019

@author: dberke

A script to measure the angular distance bewteen a star observation and the
position of the Moon, to see if the Moon was avoided during observations.

"""

import argparse
from glob import glob
from pathlib import Path

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import unyt as u

import varconlib


parser = argparse.ArgumentParser(description="A script to investigate the"
                                 "influence of scattered light on our spectra")

parser.add_argument('--plot-moon-separation', action='store_true',
                    help="Create a histogram of the separation between the"
                    " target and the Moon in all our observations.")

parser.add_argument('--plot-light-contamination', action='store_true',
                    help="Create a plot of the various sources of light"
                    " contamination present calculated by SkyCalc.")

args = parser.parse_args()

if args.plot_moon_separation:

    harps_dir = Path(varconlib.config['PATHS']['harps_dir'])

    files = glob(str(harps_dir) + '/HD*/*.fits')

    obs_ra, obs_dec = [], []
    moon_ra, moon_dec = [], []
    obs_date = []

    for file in tqdm(files[:]):
        with fits.open(file) as hdulist:
            obs_ra.append(hdulist[0].header['RA'])
            obs_dec.append(hdulist[0].header['DEC'])
            obs_date.append(hdulist[0].header['DATE-OBS'])
            moon_ra.append(hdulist[0].header['HIERARCH ESO TEL MOON RA'])
            moon_dec.append(hdulist[0].header['HIERARCH ESO TEL MOON DEC'])

    obs_pos = SkyCoord(ra=obs_ra * units.degree,
                       dec=obs_dec * units.degree,
                       frame='fk5')
    moon_pos = SkyCoord(ra=moon_ra * units.degree,
                        dec=moon_dec * units.degree,
                        frame='fk5')

    angular_distances = obs_pos.separation(moon_pos)

    tqdm.write(f'Minimum separation is {np.min(np.array(angular_distances))}')

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Angular distance between star and Moon (degrees)')

    ax.hist(angular_distances.value, bins=18, color='HoneyDew',
            edgecolor='Indigo')
    plt.show()


if args.plot_light_contamination:

    output_file = varconlib.spectra_dir / "output_HD146233.fits"
    with fits.open(output_file) as hdul:
        data = hdul[1].data

#    area_conversion = (15 * u.um) ** 2 * 5
#    area_conversion.convert_to_units(u.m ** 2)
#    print(area_conversion)
    area_conversion = 1

    # Data records in the data:
    # 'lam', 'flux', 'dflux1', 'dflux2', 'trans', 'dtrans1', dtrans2,
    # 'flux_sml', 'flux_ssl', 'flux_zl', 'flux_tie', 'flux_tme', 'flux_ael',
    # 'flux_arc', 'trans_ma', 'trans_o3', 'trans_rs', 'trans_ms'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1, top=1e6)
#    ax.set_ylim(bottom=1e-7, top=1e-5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (photons/s/m$^2$/$\\mu$m/arcsec$^2$)')

    ax.plot(data['lam'], data['flux']*area_conversion,
            color='Black',
            label='Total flux')
    ax.plot(data['lam'], data['flux_sml']*area_conversion,
            color='DeepSkyBlue',
            label='Scattered moonlight', linestyle='--')
    ax.plot(data['lam'], data['flux_ssl']*area_conversion,
            color='Aqua',
            label='Scattered starlight', linestyle='--')
    ax.plot(data['lam'], data['flux_zl']*area_conversion,
            color='CornflowerBlue',
            label='Zodiacal light', linestyle='--')
    ax.plot(data['lam'], data['flux_tme']*area_conversion,
            color='Tomato',
            label='Emission lower atmosphere', linestyle='-.')
    ax.plot(data['lam'], data['flux_ael']*area_conversion,
            color='Orange',
            label='Emission upper atmosphere', linestyle='-.')
    ax.plot(data['lam'], data['flux_arc']*area_conversion,
            color='OrangeRed',
            label='Airglow', linestyle='-.')

    ax.legend(loc='upper left')
    plt.show()
