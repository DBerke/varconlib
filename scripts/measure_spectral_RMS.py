#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:42:51 2018

@author: dberke

Script to check whether RMS of a flat continuum region of a given HARPS
spectrum matches the error array, to see if the error array is well-estimated.
"""

from pathlib import Path
from glob import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import unyt as u
import varconlib as vcl
import conversions
import obs1d
import obs2d

parser = argparse.ArgumentParser()

parser.add_argument('-1d', '--ADP', action='store_true',
                    help='Check using 1D ADP files.')
parser.add_argument('-2d', '--e2ds', action='store_true',
                    help='Check using 2D e2sd files.')
parser.add_argument('-u', '--update', action='store_true', default=False,
                    help='Specifically update the e2ds files.')

args = parser.parse_args()

baseDir = Path('/Users/dberke/HD68168/data/reduced/')
baseDir = Path('/Users/dberke/HD78660/data/reduced/')


limits = ((6198.4, 6198.65) * u.angstrom, (6199.2, 6199.5) * u.angstrom,
          (6201.55, 6201.8) * u.angstrom, (6202.25, 6202.6) * u.angstrom,
          (6224.2, 6224.5) * u.angstrom, (6236.4, 6236.55) * u.angstrom)

update_status = args.update

for lims in tqdm(limits):
    left_lim, right_lim = lims

    shifted_left_lim = conversions.vac2airESO(left_lim)
    shifted_right_lim = conversions.vac2airESO(right_lim)
    print('Original limits (vacuum): {}, {}'.format(left_lim, right_lim))
    print('Air limits: {:.4f}, {:.4f}'.format(shifted_left_lim,
                                              shifted_right_lim))

    if args.ADP:

        title_name = 'ADP files, flux RMS vs. median error'
        plot_type = 'ADP'

        baseDir = Path('/Volumes/External Storage/HARPS/HD68168/')
        files = glob(str(baseDir / 'ADP*.fits'))

        total_fluxes_stddev = []
        total_median_errors = []

        for file in tqdm(files):

            filename = Path(file).stem
            tqdm.write('For {}:'.format(filename))
            data1d = obs1d.readHARPSfile1d(file, radvel=True)

            radvel = float(data1d['radvel']) * u.km / u.s
            rv_left_lim = vcl.shift_wavelength(shifted_left_lim, radvel).\
                                               to(u.angstrom)
            rv_right_lim = vcl.shift_wavelength(shifted_right_lim, radvel).\
                                                to(u.angstrom)

            tqdm.write('RV shifted limits: {:.4f}, {:.4f}'.format(
                       rv_left_lim.to(u.angstrom),
                       rv_right_lim.to(u.angstrom)))
            wavelengths = []
            errors = []
            fluxes = []

            for wl, flux, error in zip(data1d['w'], data1d['f'], data1d['e']):
                if rv_left_lim <= wl <= rv_right_lim:
                    wavelengths.append(wl)
                    errors.append(error)
                    fluxes.append(flux)

            flux_rms = np.std(fluxes)
            med_err = np.median(error)

            tqdm.write('Flux RMS: {:.4f}, median error: {:.4f}'.format(
                       flux_rms,
                       med_err))
            total_fluxes_stddev.append(flux_rms)
            total_median_errors.append(med_err)
#            fig = plt.figure(figsize=(8, 8))
#            ax = fig.add_subplot(1, 1, 1)
#            ax.errorbar(wavelengths, fluxes, yerr=errors, marker='+',
#                        markeredgecolor='Black', color='CornflowerBlue',
#                        ecolor='Indigo', linestyle='')
#            plt.show()
#            plt.close(fig)

    if args.e2ds:

        title_name = r'e2ds files ({}, {}) $\AA$'.format(left_lim, right_lim)
        plot_type = 'e2ds'
        files = glob(str(baseDir / '*/*e2ds_A.fits'))

        total_fluxes_stddev = []
        total_median_errors = []
        total_mean_fluxes = []

        for file in tqdm(files, unit='files'):
            if update_status:
                # We only need it to update the files on the first iteration,
                # after that it can use the updated files without redoing it.
                data2d = obs2d.HARPSFile2DScience(file, update=True)
            else:
                data2d = obs2d.HARPSFile2DScience(file, update=False)
            filename = Path(file).stem
            tqdm.write('For {}:'.format(filename))
#            tqdm.write('BERV = {}'.format(data2d._BERV))
            data2d.BERV_shifted_array = data2d.shiftWavelengthArray(
                                        data2d._wavelengthArray,
                                        data2d._BERV)

            radvel = data2d._radialVelocity
            rv_left_lim = vcl.shift_wavelength(shifted_left_lim, radvel)
            rv_right_lim = vcl.shift_wavelength(shifted_right_lim, radvel)
            tqdm.write('RV shifted limits: {:.4f}, {:.4f}'.format(
                       rv_left_lim.to(u.angstrom),
                       rv_right_lim.to(u.angstrom)))

            wavelengths = []
            errors = []
            fluxes = []

            order = data2d.findWavelength(shifted_left_lim)
            for wl, flux, error in zip(data2d.BERV_shifted_array[order],
                                       data2d._photonFluxArray[order],
                                       data2d._errorArray[order]):
                if rv_left_lim <= wl <= rv_right_lim:
                    wavelengths.append(wl)
                    errors.append(error)
                    fluxes.append(flux)

            flux_rms = np.std(fluxes)
            mean_flux = np.mean(fluxes)
            med_err = np.median(error)
            tqdm.write('Flux RMS: {:.4f}, median error: {:.4f}'.
                       format(flux_rms, med_err))
            total_fluxes_stddev.append(flux_rms)
            total_median_errors.append(med_err)
            total_mean_fluxes.append(mean_flux)
#            fig = plt.figure(figsize=(8, 8))
#            ax = fig.add_subplot(1, 1, 1)
#            ax.errorbar(wavelengths, fluxes, yerr=errors, marker='+',
#                        markeredgecolor='Black', color='CornflowerBlue',
#                        ecolor='Indigo', linestyle='')
#            plt.show()
#            plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title_name)
    ax = fig.add_subplot(1, 1, 1)

    total_fluxes_stddev = np.array(total_fluxes_stddev)
    total_median_errors = np.array(total_median_errors)
    total_mean_fluxes = np.array(total_mean_fluxes)

    x = np.linspace(0, 650, 2)
#    x = np.linspace(0, 0.2, 2)
    ax.plot(x, x, linestyle='-', color='Black')
    ax.scatter(total_fluxes_stddev,
               total_median_errors / np.sqrt(1.6977))
    ax.set_xlabel('Flux RMS')
    ax.set_ylabel('Median error')
#    plt.show()

    save_path = Path('/Users/dberke/Pictures/error_checking/')
    save_file = save_path / 'RMS_vs_error_{}_({}-{})_1gain_div.png'.\
                            format(plot_type, left_lim.value, right_lim.value)
    tqdm.write(f'Saving file as {save_file.name}')
    fig.savefig(str(save_file))
    plt.close(fig)

    update_status = False
