#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:54:36 2019

@author: dberke

Script to plot all the transitions from a given list in a wider context in a
given e2ds file.

"""

import argparse
import configparser
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unyt as u
from tqdm import tqdm
import obs2d
import varconlib as vcl

desc = 'Create plots of all features corresponding to given transitions.'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('obs_file', action='store',
                    help='The file to plot from.')

args = parser.parse_args()

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

pickle_dir = Path(config['PATHS']['pickle_dir'])
nist_matched_pickle_file = pickle_dir / 'transitions_NIST_matched.pkl'
tqdm.write('Unpickling transitions dictionary...')
with open(nist_matched_pickle_file, 'r+b') as f:
    nist_dict = pickle.load(f)

all_transitions = []
for value in nist_dict.values():
    all_transitions.extend(value)

all_transitions.sort()

obs_path = Path(args.obs_file)
assert obs_path.exists(), 'Given obs file path does not exist!'

obs = obs2d.HARPSFile2DScience(obs_path)

radvel = obs.radialVelocity
obs_file_name = obs._filename.stem

for transition in tqdm(all_transitions):

    corrected_wavelength = vcl.shift_wavelength(transition.wavelength, radvel)
    order = obs.findWavelength(corrected_wavelength, obs.barycentricArray)

    vel_wl_offset = vcl.velocity2wavelength(25 * u.km / u.s,
                                            corrected_wavelength)

    min_wl = corrected_wavelength - vel_wl_offset
    max_wl = corrected_wavelength + vel_wl_offset

    min_index = vcl.wavelength2index(min_wl, obs.barycentricArray[order])
    max_index = vcl.wavelength2index(max_wl, obs.barycentricArray[order])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('{:.3f} ({})'.format(transition.wavelength.to(u.angstrom),
                 transition.atomicSpecies))

    ax.set_xlabel('Wavelength (angstroms)')
    ax.set_ylabel('Flux (photo-electrons)')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))

    obs.plotErrorbar(order, ax, min_index=min_index, max_index=max_index,
                     color='SandyBrown', ecolor='Sienna', label='Flux',
                     barsabove=True)
    ax.axvline(corrected_wavelength.to(u.angstrom).value,
               color='RosyBrown', label='RV-corrected wavelength')

    ax.legend()

    plot_dir = Path('/Users/dberke/Pictures/Stars/Transition_plots')
    file_path = plot_dir / '{}_{:.3f}_{}{}.png'.format(obs_file_name,
                                                       transition.wavelength.
                                                       to(u.angstrom).value,
                                                       transition.atomicSymbol,
                                                       transition.
                                                       ionizationState)

    fig.savefig(str(file_path))
    plt.close(fig)
