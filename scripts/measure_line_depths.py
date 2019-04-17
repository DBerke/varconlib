#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:47:55 2018

@author: dberke

A script to empirically measure the depths of all lines from the combined red +
blue line lists from BRASS.
"""

import configparser
from pathlib import Path
import subprocess
import numpy as np
import unyt as u
from tqdm import tqdm, trange
import varconlib as vcl
import obs2d
import conversions
from transition_line import Transition
from fitting import GaussianFit


config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read('/Users/dberke/code/config/variables.cfg')
data_dir = Path(config['PATHS']['data_dir'])
masks_dir = Path(config['PATHS']['masks_dir'])

purpleLineFile = data_dir / 'BRASS_Combined_List.csv'

raw_line_data = np.genfromtxt(str(purpleLineFile), delimiter=",",
                              skip_header=1,
                              dtype=(float, "U2", int, float, float))

line_data = []
for line in raw_line_data:
    wavelength = conversions.air2vacMortonIAU(line[0]) * u.angstrom
    element = str(line[1])
    ionization_state = line[2]
    low_energy = line[3]
    depth = line[4]
    transition = Transition(wavelength, element, ionization_state)
    transition.lowerEnergy = low_energy
    transition.depth = depth
    line_data.append(transition)


# Use for our spectrum a high-SNR (283.3) observation of a solar twin.
spectrum_file = Path('/Users/dberke/HD146233/data/reduced/2016-03-29/'
                     'HARPS.2016-03-30T07:25:38.139_e2ds_A.fits')
#baseDir = Path('/Volumes/External Storage/HARPS/HD146233')
#spectrumFile = baseDir / 'ADP.2016-03-31T01:04:03.410.fits'  # Highest SNR
#radvel = 11.7 * u.km / u.s  # for HD146233
#plot_dir = '/Users/dberke/Pictures/fitlines'

# Masked regions file
no_CCD_bounds_file = masks_dir / 'unusable_spectrum_noCCDbounds.txt'

mask_no_CCD_bounds = vcl.parse_spectral_mask_file(no_CCD_bounds_file)

obs = obs2d.HARPSFile2DScience(spectrum_file)

# Define the region in which to look for other lines (indicative of blending)
# and how close a line's nominal wavelength must be to the fitted wavelength
# found for it.
blending_limit = 9100 * u.m / u.s  # 3.5*HARPS' 2.6 km/s resolution element
offset_limit = 1500 * u.m / u.s  # based on histogram of offsets

#vac_wl = conversions.air2vacESO(data['w'] * u.angstrom).to(u.nm)
#flux = data['f']
#err = data['e']
#date_obs = data['date_obs']

lines_to_write = []
log_lines = []
badlines = 0
goodlines = 0
maskedlines = 0
for i in trange(len(line_data)):
    line = line_data[i]
    blended = False
    masked = False
    # Convert to nanometers for linefind, after converting to vacuum wls
#    linewl = conversions.air2vacMortonIAU(line[0]) * u.angstrom
#    linewl.convert_to_units(u.nm)

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
    delta_lambda = vcl.velocity2wavelength(blending_limit, line.wavelength)
    for j in range(len(line_data)):
        if j == i:
            continue
        else:
            line2 = line_data[j]
#        linewl2 = conversions.air2vacMortonIAU(line2[0]) * u.angstrom
#        linewl2.convert_to_units(u.nm)
        if abs(line.wavelength - line2.wavelength) <= delta_lambda:
            blended = True
            blend_str = '{:.4f} {} was likely blended.\n'.format(
                    line.wavelength.to(u.nm), line.atomicSpecies)
            tqdm.write(blend_str, end='')
            log_lines.append(blend_str)
            break
    if blended:
        continue

    try:
        fit = GaussianFit(line, obs)
#        data_dict = vcl.linefind(linewl, vac_wl, flux, err, radvel,
#                                 spectrumFile.stem, 'HD146233', gauss_fit=True,
#                                 plot=True, plot_dir=plot_dir,
#                                 velsep=5000 * u.m / u.s,
#                                 date_obs=date_obs)
#        norm_depth = data_dict['norm_depth']
#        offset = data_dict['gauss_vel_offset']
        if abs(fit.velocityOffset) < offset_limit:
#            line_list = [str(x) for x in line]
#            line_list.pop(0)
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
                format(line.wavelength.to(u.nm),
                       fit.velocityOffset.to(u.m/u.s), line.atomicSpecies)
            tqdm.write(success_str, end='')
            log_lines.append(success_str)
            goodlines += 1
        else:
            offset_str = '{:.4f} {} too far away ({:.1f})\n'.format(
                    line.wavelength.to(u.nm), fit.velocityOffset.to(u.m/u.s),
                    line.atomicSpecies)
            tqdm.write(offset_str, end='')
            log_lines.append(offset_str)
            badlines += 1
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


outfile = data_dir / 'BRASS_Vac_Line_Depths_All.csv'
with open(outfile, 'w') as f:
    tqdm.write('Writing results to {}'.format(outfile))
    f.write('Wavl[vac;nm],Elm,Ion,Elow[eV],UnBrCalDep,'
            'MeasNormDepth,GaussVelOffset[m/s]\n')
#    f.write('Wavl[vac;nm],Elm,Ion,Elow[eV],BrCalDep,CalBlending[%],'
#            'MeasNormDepth\n')
    f.writelines(lines_to_write)

log_file = data_dir / 'logfiles/measureLines_log.txt'
with open(log_file, 'w') as g:
    g.writelines(log_lines)


print('\n{}/{} lines unable to be fit ({:.2%})'.format(badlines,
      len(line_data), badlines/len(line_data)))

print('\n{}/{} lines were fit ({:.2%})'.format(goodlines,
      len(line_data), goodlines/len(line_data)))

print('\n{}/{} lines were in masked regions ({:.2%})'.format(maskedlines,
      len(line_data), maskedlines/len(line_data)))
