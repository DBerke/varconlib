#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:47:55 2018

@author: dberke

A script to empirically measure the depths of all lines from the combined red +
blue line lists from BRASS.
"""

import numpy as np
import varconlib as vcl
from tqdm import tqdm
from pathlib import Path
import subprocess

redLineFile = Path('data/BRASS2018_Sun_PrelimGraded_Lobel.csv')
purpleLineFile = Path('data/BRASS_Combined_List.csv')

lineData = np.genfromtxt(str(purpleLineFile), delimiter=",", skip_header=1,
                         dtype=(float, "U2", int, float, float))

#lineData = np.genfromtxt(str(redLineFile), delimiter=",", skip_header=1,
#                         dtype=(float, "U2", int, float, float, float))

#baseDir = Path('/Volumes/External Storage/HARPS/1Ceres')
baseDir = Path('/Volumes/External Storage/HARPS/HD146233')
#spectrumFile = baseDir / 'ADP.2014-09-16T11:03:54.137.fits'  # Ceres
spectrumFile = baseDir / 'ADP.2016-03-31T01:04:03.410.fits'  # Highest SNR
radvel = 11.7 # km/s for HD146233
plot_dir = '/Users/dberke/Pictures/fitlines'

# Masked regions file
no_CCD_bounds_file = Path('data/unusable_spectrum_noCCDbounds.txt')

mask_no_CCD_bounds = vcl.parse_spectral_mask_file(no_CCD_bounds_file)

data = vcl.readHARPSfile(str(spectrumFile), date_obs=True)

# Define the region in which to look for other lines (indicative of blending)
# and how close a line's nominal wavelength must be to the fitted wavelength
# found for it.
blending_limit = 9100  # m/s, 3.5*HARPS' 2.6 km/s resolution element
offset_limit = 1500  # m/s, based on histogram of offsets

vac_wl = vcl.air2vacESO(data['w']) / 10  # Convert to nanometers
flux = data['f']
err = data['e']
date_obs = data['date_obs']

lines_to_write = []
log_lines = []
badlines = 0
goodlines = 0
for line in tqdm(lineData):
    blended = False
    masked = False
    # Convert to nanometers for linefind, after converting to vacuum wls
    linewl = vcl.air2vacMortonIAU(line[0]) / 10

    for region in mask_no_CCD_bounds:
        if region[0] <= linewl <= region[1]:
            mask_str = '{:.4f} was in a masked region.\n'.format(linewl)
            tqdm.write(mask_str, end='')
            log_lines.append(mask_str)
            masked = True
            break
    if masked:
        continue

    # Check for blended lines within 3.5 * HARPS' 2.6 km/s resolution element.
    delta_lambda = vcl.getwlseparation(blending_limit, linewl)
    for line2 in lineData:
        linewl2 = vcl.air2vacMortonIAU(line2[0]) / 10
        if 0. < abs(linewl - linewl2) <= delta_lambda:
            blended = True
            blend_str = '{:.4f} was likely blended.\n'.format(linewl)
            tqdm.write(blend_str, end='')
            log_lines.append(blend_str)
            break
    if blended:
        continue

    try:
#        data_dict = vcl.linefind(linewl, vac_wl, flux, err, 3,
#                                 spectrumFile.stem, 'Ceres', gauss_fit=True,
#                                 plot=True, plot_dir=baseDir, velsep=5000,
#                                 date_obs=date_obs)
        data_dict = vcl.linefind(linewl, vac_wl, flux, err, radvel,
                                 spectrumFile.stem, 'HD146233', gauss_fit=True,
                                 plot=True, plot_dir=plot_dir, velsep=5000,
                                 date_obs=date_obs)
        norm_depth = data_dict['norm_depth']
        offset = data_dict['gauss_vel_offset']
        if abs(offset) < offset_limit:
            line_list = [str(x) for x in line]
            line_list.pop(0)
            line_list.insert(0, str(round(linewl, 4)))
            line_list.append('{:.4f},{:.1f}\n'.format(norm_depth, offset))
            newline = ','.join(line_list)
            lines_to_write.append(newline)
            success_str = '{:.4f} was fit (offset: {:.2f} m/s)\n'.\
                           format(linewl, offset)
            tqdm.write(success_str, end='')
            log_lines.append(success_str)
            goodlines += 1
        else:
            offset_str = '{:.4f} too far away ({:.1f} m/s)\n'.format(linewl,
                                                                     offset)
            tqdm.write(offset_str, end='')
            log_lines.append(offset_str)
            filepath1 = data_dict['gauss_graph_path']
            filepath2 = data_dict['gauss_norm_graph_path']
            filepath1.unlink()
            filepath2.unlink()
            badlines += 1
    except RuntimeError:
        unfit_str = '{:.4f} was unable to be fit.\n'.format(linewl)
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


outfile = Path('data/BRASS_Vac_Line_Depths_All.csv')
with open(outfile, 'w') as f:
    f.write('Wavl[vac;nm],Elm,Ion,Elow[eV],UnBrCalDep,'
            'MeasNormDepth,GaussVelOffset\n')
#    f.write('Wavl[vac;nm],Elm,Ion,Elow[eV],BrCalDep,CalBlending[%],'
#            'MeasNormDepth\n')
    f.writelines(lines_to_write)

log_file = Path('data/logfiles/measureLines_log.txt')
with open(log_file, 'w') as g:
    g.writelines(log_lines)


print('\n{}/{} lines unable to be fit ({:.2f}%)'.format(badlines,
      len(lineData), 100*(badlines/len(lineData))))

print('\n{}/{} lines were fit ({:.2f}%)'.format(goodlines,
      len(lineData), 100*(goodlines/len(lineData))))