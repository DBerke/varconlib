#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:47:55 2018

@author: dberke

A script to empirically measure the depths of all lines from the combined red +
blue line lists from BRASS.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange
import unyt as u

from varconlib.fitting import GaussianFit
from varconlib.miscellaneous import (parse_spectral_mask_file,
                                     velocity2wavelength)
from varconlib.transition_line import Transition
import varconlib as vcl
import varconlib.conversions
import varconlib.obs2d


desc = 'Measure the depths of absorption features in a spectrum.'
parser = argparse.ArgumentParser(description=desc)
target = parser.add_mutually_exclusive_group(required=True)
target.add_argument('--vesta', action='store_true',
                    help='Use an observation of Vesta.')
target.add_argument('--hd146233', action='store_true',
                    help='Use an observation of HD 146233.')
parser.add_argument('-rv', action='store', default=-0.57, type=float,
                    help='Radial velocity for Vesta measurement. (km/s)')

args = parser.parse_args()

data_dir = vcl.data_dir
masks_dir = vcl.masks_dir
harps_dir = vcl.harps_dir
output_dir = vcl.output_dir

purpleLineFile = data_dir / 'BRASS_Combined_List.csv'

raw_line_data = np.genfromtxt(str(purpleLineFile), delimiter=",",
                              skip_header=1,
                              dtype=(float, "U2", int, float, float))

line_data = []
for line in tqdm(raw_line_data):
    wavelength = varconlib.conversions.air2vacMortonIAU(line[0]) * u.angstrom
    element = str(line[1])
    ionization_state = line[2]
    low_energy = line[3]
    depth = line[4]
    transition = Transition(wavelength, element, ionization_state)
    transition.lowerEnergy = low_energy
    transition.depth = depth
    line_data.append(transition)

if args.hd146233:
    # Use for our spectrum a high-SNR (447.6) observation of a solar twin.
    spectrum_file = Path(str(harps_dir) + '/HD146233/data/reduced/2016-03-29/'
                         'HARPS.2016-03-30T07:25:38.139_e2ds_A.fits')
    object_dir = output_dir / 'HD146233/{}'.format(spectrum_file.stem)

if args.vesta:
    # Use a high-SNR (316.8) observation of Vesta. Need to supply RV.
#    spectrum_file = Path(str(harps_dir) + '/Vesta/data/reduced/2014-04-15/'
#                         'HARPS.2014-04-16T05:45:10.757_e2ds_A.fits')
    spectrum_file = Path(str(harps_dir) + '/Vesta/data/reduced/2014-04-15/'
                         'HARPS.2014-04-16T05:45:10.757_e2ds_A.fits')
    object_dir = output_dir / 'Vesta/{}'.format(spectrum_file.stem)

    # The FITS file unfortunately doesn't include a real radial velocity for
    # the observation, so we have to specify it manually. From the NASA/JPL
    # ephemerides, it's -0.57 km/s for this observation, which is what it will
    # default to if no value is given on the command line.
    obs_radial_velocity = args.rv * u.km / u.s

print(f'Analyzing file at {spectrum_file}')

if not object_dir.exists():
    os.mkdir(object_dir)
# Define directory for output pickle files:
output_pickle_dir = object_dir / 'pickles'
if not output_pickle_dir.exists():
    os.mkdir(output_pickle_dir)
# Define paths for plots to go in:
output_plots_dir = object_dir / 'plots'
if not output_plots_dir.exists():
    os.mkdir(output_plots_dir)
closeup_dir = output_plots_dir / 'close_up'
if not closeup_dir.exists():
    os.mkdir(closeup_dir)
context_dir = output_plots_dir / 'context'
if not context_dir.exists():
    os.mkdir(context_dir)

# Masked regions file
no_CCD_bounds_file = masks_dir / 'unusable_spectrum_noCCDbounds.txt'

mask_no_CCD_bounds = parse_spectral_mask_file(no_CCD_bounds_file)

obs = varconlib.obs2d.HARPSFile2DScience(spectrum_file)

# Define the region in which to look for other lines (indicative of blending)
# and how close a line's nominal wavelength must be to the fitted wavelength
# found for it.
blending_limit = 9100 * u.m / u.s  # 3.5*HARPS' 2.6 km/s resolution element
offset_limit = 1500 * u.m / u.s  # based on histogram of offsets


lines_to_write = []
log_lines = []
badlines = 0
goodlines = 0
maskedlines = 0
too_far_lines = 0
blended_lines = 0
for i in trange(len(line_data)):
    line = line_data[i]
    blended = False
    masked = False

    for region in mask_no_CCD_bounds:  # These regions are in vacuum.
        if region[0] <= line.wavelength <= region[1]:
            mask_str = '{:.4f} {} was in a masked region.\n'.format(
                    line.wavelength, line.atomicSpecies)
            tqdm.write(mask_str, end='')
            log_lines.append(mask_str)
            masked = True
            maskedlines += 1
            break
    if masked:
        continue

    # Check for blended lines within 3.5 * HARPS' 2.6 km/s resolution element.
    delta_lambda = velocity2wavelength(blending_limit, line.wavelength)
    for j in range(len(line_data)):
        if j == i:
            continue
        else:
            line2 = line_data[j]

        if abs(line.wavelength - line2.wavelength) <= delta_lambda:
            blended = True
            blend_str = '{:.4f} {} was likely blended.\n'.format(
                    line.wavelength.to(u.nm), line.atomicSpecies)
            tqdm.write(blend_str, end='')
            log_lines.append(blend_str)
            blended_lines += 1
            break
    if blended:
        continue

    plot_closeup = closeup_dir / 'Transition_{:.4f}_{}{}.png'.format(
                line.wavelength.to(u.angstrom).value,
                line.atomicSymbol, line.ionizationState)
    plot_context = context_dir / 'Transition_{:.4f}_{}{}.png'.format(
                line.wavelength.to(u.angstrom).value,
                line.atomicSymbol, line.ionizationState)
    try:
        if args.hd146233:
            fit = GaussianFit(line, obs, close_up_plot_path=plot_closeup,
                              context_plot_path=plot_context)
        elif args.vesta:
            fit = GaussianFit(line, obs, close_up_plot_path=plot_closeup,
                              context_plot_path=plot_context,
                              radial_velocity=obs_radial_velocity)
#        data_dict = vcl.linefind(linewl, vac_wl, flux, err, radvel,
#                                 spectrumFile.stem, 'HD146233',
#                                  gauss_fit=True,
#                                 plot=True, plot_dir=plot_dir,
#                                 velsep=5000 * u.m / u.s,
#                                 date_obs=date_obs)
        fit.plotFit(plot_closeup, plot_context)

    except RuntimeError:
        unfit_str = '{:.4f} {} was unable to be fit.\n'.format(
                line.wavelength.to(u.nm), line.atomicSpecies)
        tqdm.write(unfit_str, end='')
        log_lines.append(unfit_str)
#        args = ['/Users/dberke/code/plotSpec.py',
#                'HD146233/ADP.2016-03-31T01:04:03.410.fits',
#                '-o', 'badlines/Bad_line_{:.4f}.png'.format(linewl), '-v',
#                '-n', '{:.4f}'.format(linewl-0.06),
#                '-m', '{:.4f}'.format(linewl+0.06),
#                '-r', str(radvel), '-l', str(round(linewl, 4))]
#        subprocess.run(args)
        badlines += 1
        continue

    if abs(fit.velocityOffset) < offset_limit:
        line_list = [str(round(float(line.wavelength.to(u.nm).value), 4))]
        line_list.extend([line.atomicSymbol, str(line.ionizationState),
                          str(line.lowerEnergy)])
        line_list.append('{:.4f}'.format(line.depth))  # Calculated depth
        line_list.append('{:.4f}'.format(fit.normalizedLineDepth))
        line_list.append('{:.1f}\n'.format(fit.velocityOffset.
                         to(u.m/u.s).value))
        newline = ','.join(line_list)
        lines_to_write.append(newline)
        success_str = '{:.4f} {} was fit (offset: {:.2f})\n'.\
            format(line.wavelength.to(u.nm), line.atomicSpecies,
                   fit.velocityOffset.to(u.m/u.s))
        tqdm.write(success_str, end='')
        log_lines.append(success_str)
        goodlines += 1
    else:
        offset_str = '{:.4f} {} too far away ({:.1f})\n'.format(
                line.wavelength.to(u.nm), line.atomicSpecies,
                fit.velocityOffset.to(u.m/u.s))
        tqdm.write(offset_str, end='')
        log_lines.append(offset_str)
        too_far_lines += 1

outfile = data_dir / 'BRASS_Vac_Line_Depths_All.csv'
with open(outfile, 'w') as f:
    tqdm.write('Writing results to {}'.format(outfile))
    f.write('Wavl[vac;nm],Elm,Ion,Elow[eV],UnBrCalDep,'
            'MeasNormDepth,GaussVelOffset[m/s]\n')
    f.writelines(lines_to_write)

log_file = data_dir / 'logfiles/measureLines_log.txt'
with open(log_file, 'w') as g:
    g.writelines(log_lines)

# Output some stats
print('\n{}/{} lines were fit ({:.2%})'.format(goodlines,
      len(line_data), goodlines/len(line_data)), end='')
print('\n{}/{} lines were fit too far away ({:.2%})'.format(too_far_lines,
      len(line_data), too_far_lines/len(line_data)), end='')
print('\n{}/{} lines were unable to be fit ({:.2%})'.format(badlines,
      len(line_data), badlines/len(line_data)), end='')
print('\n{}/{} lines were in masked regions ({:.2%})'.format(maskedlines,
      len(line_data), maskedlines/len(line_data)), end='')
print('\n{}/{} lines were likely blended ({:.2%})'.format(blended_lines,
      len(line_data), blended_lines/len(line_data)))
