#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:42:51 2018

@author: dberke

Script to check whether RMS of a flat continuum region of a given HARPS
spectrum matches the error array, to see if the error array is well-estimated.
"""

from pathlib import Path
from os import mkdir
import re
from glob import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import unyt as u
import varconlib as vcl
import obs2d

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--update', action='append', default=[],
                    help='Specifically update the e2ds files.')
parser.add_argument('-n', '--names', action='append', default=[],
                    help='Name of star to analyze (or "Synthetic").')
args = parser.parse_args()
print(args)

#star_name = 'HD68168'
#star_name = 'HD78660'
#star_name = 'HD138573'
#star_name = 'HD183658'
#star_name = 'Synthetic'



name_re = re.compile('HD[0-9]{5,6}')
star_names = args.names
print(star_names)

limits = ((6198.4, 6198.65) * u.angstrom, (6199.2, 6199.5) * u.angstrom,
          (6201.55, 6201.8) * u.angstrom, (6202.25, 6202.6) * u.angstrom,
          (6224.2, 6224.5) * u.angstrom, (6236.4, 6236.65) * u.angstrom)
# H-alpha
h_alpha = 6564.614 * u.angstrom
#limits = ((6562, 6568) * u.angstrom,)

update_status = bool(args.update)

observations_dict = {}

for star_name in tqdm(star_names) if len(star_names) > 1 else star_names:

    real_base_dir = Path(f'/Users/dberke/{star_name}/data/reduced/')
    synth_base_dir = Path('/Users/dberke/code/data/')
    if star_name == 'Synthetic':
        base_dir = synth_base_dir
    else:
        if not name_re.match(star_name):
            print('Incorrect star name: {}'.format(star_name))
            exit()
        else:
            base_dir = real_base_dir
    for lims in tqdm(limits):
        left_lim, right_lim = lims

        tqdm.write('Original limits (vacuum): {}, {}'.format(left_lim,
                   right_lim))

        title_name = r'e2ds files ({}, {}) $\AA$'.format(left_lim, right_lim)
        plot_type = 'e2ds'
        files = glob(str(base_dir / '*/*e2ds_A.fits'))

        total_fluxes_stddev = []
        total_median_errors = []
        total_mean_fluxes = []

        for file in tqdm(files, unit='files'):
            try:
                data2d = observations_dict[file]
    #            print(f'Found {file} in dictionary.')
            except KeyError:
                if update_status:
                    # We only need it to update the files on the first
                    # iteration, after that it can use the updated files
                    # without redoing it.
                    data2d = obs2d.HARPSFile2DScience(file, update=args.update)
                    observations_dict[file] = data2d
                else:
                    data2d = obs2d.HARPSFile2DScience(file, update=[])
                    observations_dict[file] = data2d
            filename = Path(file).stem
            tqdm.write('For {}:'.format(filename))
    #            tqdm.write('BERV = {}'.format(data2d._BERV))

            radvel = data2d._radialVelocity
            rv_left_lim = vcl.shift_wavelength(left_lim, radvel)
            rv_right_lim = vcl.shift_wavelength(right_lim, radvel)
            tqdm.write('RV shifted limits: {:.4f}, {:.4f}'.format(
                       rv_left_lim.to(u.angstrom),
                       rv_right_lim.to(u.angstrom)))

            rv_h_alpha = vcl.shift_wavelength(h_alpha, radvel)

            wavelengths = []
            errors = []
            fluxes = []
            blazes = []

            order = data2d.findWavelength(rv_left_lim)
            for wl, flux, error, blaze in zip(data2d.barycentricArray[order],
                                              data2d.photonFluxArray[order],
                                              data2d.errorArray[order],
                                              data2d.blazeArray[order]):
                if rv_left_lim <= wl <= rv_right_lim:
                    wavelengths.append(wl)
                    errors.append(error)
                    fluxes.append(flux)
                    blazes.append(blaze)

            flux_rms = np.std(fluxes)
            mean_flux = np.mean(fluxes)
    #        print(fluxes)
            med_err = np.median(errors)
    #        print(errors)
            if flux_rms / mean_flux > 0.08:
                print(file)
                print('Prolematic file!')
                raise
            tqdm.write('Flux RMS: {:.4f}, median error: {:.4f}'.
                       format(flux_rms, med_err))
            total_fluxes_stddev.append(flux_rms)
            total_median_errors.append(med_err)
            total_mean_fluxes.append(mean_flux)

            save_path = Path('/Users/dberke/Pictures/error_checking/'
                             f'{star_name}/'
                             f'{left_lim.to(u.angstrom).value:.2f}_'
                             f'{right_lim.to(u.angstrom).value:.2f}')
            if not save_path.exists():
                try:
                    mkdir(save_path)
                except FileNotFoundError:
                    try:
                        mkdir(save_path.parent)
                        mkdir(save_path)
                    except FileNotFoundError:
                        raise
            save_file = save_path / f'{filename}.png'

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(wavelengths, fluxes, yerr=errors, marker='+',
                        markeredgecolor='Black', color='CornflowerBlue',
                        ecolor='Indigo', linestyle='')
    #            ax.axvline(h_alpha.to(u.angstrom).value,
    #                       color='Red', label='vac H alpha')
    #            ax.axvline(rv_h_alpha.to(u.angstrom).value,
    #                       color='Green', label='rv H alpha')
    #            plt.show()
            fig.savefig(str(save_file))
            plt.close(fig)

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(title_name)
        ax = fig.add_subplot(1, 1, 1)

        total_fluxes_stddev = np.array(total_fluxes_stddev)
        total_median_errors = np.array(total_median_errors)
        total_mean_fluxes = np.array(total_mean_fluxes)

        normalized_fluxes = total_fluxes_stddev / total_mean_fluxes
        normalized_errors = total_median_errors / total_mean_fluxes

        x_vals = total_fluxes_stddev
        y_vals = total_median_errors

        min_x, max_x = x_vals.max(), x_vals.min()

        x = np.linspace(min_x, max_x, 2)
        ax.plot(x, x, linestyle='-', color='Black')
        ax.scatter(x_vals, y_vals)
        ax.set_xlabel('Flux RMS')
        ax.set_ylabel('Median error')
    #    plt.show()

        save_path = Path(f'/Users/dberke/Pictures/error_checking/{star_name}/')
        save_file = save_path / 'RMS_error_{}_({:.2f}-{:.2f})_oldmaster.png'.\
                                format(star_name,
                                       left_lim.to(u.angstrom).value,
                                       right_lim.to(u.angstrom).value)
        tqdm.write(f'Saving file as {save_file.name}')
        fig.savefig(str(save_file))
        plt.close(fig)

        update_status = False
