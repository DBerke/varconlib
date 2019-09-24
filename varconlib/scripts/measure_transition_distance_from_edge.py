#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:28:40 2019

@author: dberke


A script to measure how far from the edges of the CCD all the transitions in
our sample are. To do this we use the wavelength array from a star with very
little relative velocity, and adjust it to be (hopefully) zero.

"""

from pathlib import Path
import pickle

from tqdm import tqdm
import unyt as u

from varconlib.fitting.model_fits import GaussianFit
from varconlib.obs2d import HARPSFile2DScience
from varconlib.transition_line import Transition
from varconlib.miscellaneous import wavelength2velocity as wave2vel
import varconlib as vcl


# Use globally-defined pickle_dir:
transitions_file = vcl.pickle_dir / 'final_transitions_selection.pkl'

with open(transitions_file, 'rb') as f:
    transitions = pickle.load(f)

base_file = vcl.data_dir / 'HARPS.2005-05-02T03:49:08.735_e2ds_A.fits'

obs = HARPSFile2DScience(base_file)

wavelength_scale = obs.rvCorrectedArray

# Create some paths for plots to check that the radial velocity corrections are
# working.

temp_dir = Path(vcl.config['PATHS']['temp_dir'])
close_up_path = temp_dir / 'Close_up_test_H1.png'
context_path = temp_dir / 'Context_test_H1.png'


# Test that H-alpha falls where it should.
#H_alpha = Transition(6564.5844 * u.angstrom, 'H', 1)
#
#tqdm.write('Fitting H-alpha to check if radial velocity worked.')
#fit = GaussianFit(H_alpha, obs, radial_velocity=0 * u.m / u.s,
#                  close_up_plot_path=close_up_path,
#                  context_plot_path=context_path,
#                  integrated=True, verbose=True)
#
#tqdm.write('Creating plots at:')
#tqdm.write(str(close_up_path))
#tqdm.write(str(context_path))
#fit.plotFit()

for num, order in enumerate(wavelength_scale):
    tqdm.write('\nOrder: {}'.format(num+1))
    tqdm.write('transition       lower dist.     right dist.')
    for transition in transitions:
        if order[0] < transition.wavelength < order[-1]:
            left_dist = wave2vel(transition.wavelength, order[0])
            right_dist = wave2vel(transition.wavelength, order[-1])
            tqdm.write('{}:   {:.2f}     {:.2f}'.format(transition.label,
                       left_dist.to(u.km/u.s), right_dist.to(u.km/u.s)))
