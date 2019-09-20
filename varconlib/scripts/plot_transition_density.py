#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:00:45 2019

@author: dberke

Script to create a plot showing the density of transition lines from the Kurucz
line list.
"""

import configparser
from copy import copy
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import unyt as u
from tqdm import tqdm

from obs2d import HARPSFile2DScience
import varconlib as vcl

matplotlib.rcParams['axes.formatter.useoffset'] = False

config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read('/Users/dberke/code/config/variables.cfg')
data_dir = Path(config['PATHS']['data_dir'])

# Define useful values relating to the Kurucz line list.
KuruczFile = data_dir / "gfallvac08oct17.dat"

colWidths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
             5, 5, 3, 3, 4, 5, 5, 6)
colNames = ("wavelength", "log gf", "elem", "energy1", "J1", "label1",
            "energy2", "J2", "label2", "gammaRad", "gammaStark", "vanderWaals",
            "ref", "nlte1",  "nlte2", "isotope1", "hyperf1", "isotope2",
            "logIsotope", "hyperfshift1", "hyperfshift2", "hyperF1", "hyperF2",
            "code", "landeGeven", "landeGodd", "isotopeShift")
colDtypes = (float, float, "U6", float, float, "U11", float, float, "U11",
             float, float, float, "U4", int, int, int, float, int, float,
             int, int, "U3", "U3", "U4", int, int, float)

print('Reading Kurucz line list...')
KuruczData = np.genfromtxt(KuruczFile, delimiter=colWidths, autostrip=True,
                           skip_header=842959, skip_footer=987892,
                           names=colNames, dtype=colDtypes,
                           usecols=(0, 2, 3, 4, 5, 6, 7, 8, 18))


min_wl = 5200 * u.angstrom
max_wl = 5201 * u.angstrom

test_old_file = data_dir / 'test_old.fits'
test_old = HARPSFile2DScience(test_old_file, use_new_coefficients=False,
                              use_pixel_positions=False)

order = test_old.findWavelength(min_wl, test_old.barycentricArray)
RV = test_old.radialVelocity
BERV = test_old.BERV

shifted_min = vcl.shift_wavelength(copy(min_wl), -1 * RV)
shifted_max = vcl.shift_wavelength(copy(max_wl), -1 * RV)

#shifted_min = vcl.shift_wavelength(shifted_min, -1 * BERV)
#shifted_max = vcl.shift_wavelength(shifted_max, -1 * BERV)

print('shifted_min is {}'.format(shifted_min))
print('shifted_max is {}'.format(shifted_max))


k_transition_lines = []
k_iron_I_lines = []
tqdm.write('Parsing Kurucz line list...')
for k_transition in tqdm(KuruczData, unit='transitions'):
    wl = k_transition['wavelength'] * u.nm
    if shifted_min <= wl <= shifted_max:
        k_transition_lines.append(wl.to(u.angstrom))
        if k_transition['elem'] == '26.00':
            k_iron_I_lines.append(wl.to(u.angstrom))
tqdm.write('Found {} transitions within wavelength range.'.format(
        len(k_transition_lines)))

print(k_transition_lines[0], k_transition_lines[-1])

shifted_transitions = vcl.shift_wavelength(u.unyt_array(k_transition_lines),
                                           1 * RV)
shifted_iron_I = vcl.shift_wavelength(u.unyt_array(k_iron_I_lines), 1 * RV)

print(shifted_transitions[0], shifted_transitions[-1])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel(r'Wavelength ($\AA$)')
ax.set_ylabel('Photons')

ax.set_xlim(left=min_wl, right=max_wl)

min_index = vcl.wavelength2index(min_wl, test_old.barycentricArray[order])
max_index = vcl.wavelength2index(max_wl, test_old.barycentricArray[order])


for wl in shifted_transitions:
    ax.axvline(wl.to(u.angstrom).value, ymin=0, ymax=1,
               color='Silver', alpha=0.7, zorder=1, linestyle='--')
#for wl in shifted_iron_I:
#    ax.axvline(wl.to(u.angstrom).value, ymin=0, ymax=1,
#               color='FireBrick', alpha=0.7, zorder=2)

test_old.plotErrorbar(order, ax, min_index=min_index, max_index=max_index,
                      color='SandyBrown', ecolor='Sienna',
                      label='HD 68168', barsabove=True)#, zorder=1000)

plt.show()
