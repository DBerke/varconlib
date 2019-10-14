#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:28:40 2019

@author: dberke


A script to measure how far from the edges of the CCD all the transitions in
our sample are. To do this we use the wavelength array from a star with very
little relative velocity, and adjust it to be (hopefully) zero.

"""

import argparse
from pathlib import Path
import pickle

from tqdm import tqdm
import unyt as u

from varconlib.fitting.model_fits import GaussianFit
from varconlib.obs2d import HARPSFile2DScience
from varconlib.transition_line import Transition
from varconlib.miscellaneous import wavelength2velocity as wave2vel
import varconlib as vcl


desc = 'A script to find pairs crossing orders and fix them if possible.'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-p', '--print-results', action='store_true',
                    default=False,
                    help='Print out distances for transitions to edge of CCD.')
parser.add_argument('-t', '--test-transition', action='store_true',
                    default=False,
                    help='Fit a known transition to check the RV correction.')
parser.add_argument('-v', '--verbose', action='store_true',
                    default=False,
                    help='Print out more diagnostic information.')

args = parser.parse_args()

# Use globally-defined pickle_dir:
transitions_file = vcl.pickle_dir / 'final_transitions_selection.pkl'
final_pair_selection_file = vcl.pickle_dir / 'final_pairs_selection.pkl'

with open(transitions_file, 'rb') as f:
    transitions = pickle.load(f)

with open(final_pair_selection_file, 'rb') as g:
    pairs = pickle.load(g)

print(f'Found {len(pairs)} total pairs.')

base_file = vcl.data_dir / 'HARPS.2005-05-02T03:49:08.735_e2ds_A.fits'

obs = HARPSFile2DScience(base_file)

wavelength_scale = obs.rvCorrectedArray

# Create some paths for plots to check that the radial velocity corrections are
# working.

temp_dir = Path(vcl.config['PATHS']['temp_dir'])
close_up_path = temp_dir / 'Close_up_test_transition.png'
context_path = temp_dir / 'Context_test_transition.png'


if args.test_transition:
    # Test that a known transition falls where it should.
    transition_check = Transition(4217.791 * u.angstrom, 'Fe', 1)

    tqdm.write('Fitting test transition to check if radial velocity worked.')
    fit = GaussianFit(transition_check, obs, radial_velocity=0 * u.m / u.s,
                      close_up_plot_path=close_up_path,
                      context_plot_path=context_path,
                      integrated=True, verbose=True)

    tqdm.write('Creating plots at:')
    tqdm.write(str(close_up_path))
    tqdm.write(str(context_path))
    fit.plotFit()

for transition in transitions:
    transition.orders_found_in = []


for num, order in enumerate(wavelength_scale):
    if args.verbose:
        tqdm.write('\nOrder: {}'.format(num + 1))  # Off by one
        tqdm.write('transition       lower dist.     right dist.')
    for transition in transitions:
        if order[0] < transition.wavelength < order[-1]:
            transition.orders_found_in.append(num + 1)
            left_dist = wave2vel(transition.wavelength, order[0])
            right_dist = wave2vel(transition.wavelength, order[-1])
            if not hasattr(transition, 'first_order'):
                transition.first_order = {}
                transition.first_order['left_dist'] = left_dist.to(u.km/u.s)
                transition.first_order['right_dist'] = right_dist.to(u.km/u.s)
            elif hasattr(transition, 'first_order'):
                transition.second_order = {}
                transition.second_order['left_dist'] = left_dist.to(u.km/u.s)
                transition.second_order['right_dist'] = right_dist.to(u.km/u.s)
            if args.verbose:
                if hasattr(transition, 'second_order'):
                    tqdm.write('{}:   {:.2f}     {:.2f}'.format(
                                transition.label,
                                transition.second_order['left_dist'],
                                transition.second_order['right_dist']))
                else:
                    tqdm.write('{}:   {:.2f}     {:.2f}'.format(
                                transition.label,
                                transition.first_order['left_dist'],
                                transition.first_order['right_dist']))


num_multi_order_transitions = 0
for transition in transitions:
    if args.verbose:
        print(transition, transition.orders_found_in)
    if len(transition.orders_found_in) == 2:
        num_multi_order_transitions += 1

print(f'Found {num_multi_order_transitions} transitions in two orders.')

transition_dict = {transition.label: transition for transition in transitions}


for pair in pairs:
    for transition in pair:
        transition.orders_found_in = transition_dict[transition.
                                                     label].orders_found_in
        transition.first_order = transition_dict[transition.
                                                 label].first_order
        if hasattr(transition_dict[transition.label], 'second_order'):
            transition.second_order = transition_dict[transition.
                                                      label].second_order

cross_order_pairs = []
for pair in pairs:
    orders1 = pair._higherEnergyTransition.orders_found_in
    orders2 = pair._lowerEnergyTransition.orders_found_in

    # Now, to check various cases. The simplest case is if both transitions are
    # only found in a single order.
    if (len(orders1) == 1) and (len(orders2) == 1):
        # If both are found in the same order, great!
        if orders1[0] == orders2[0]:
            pair.status = [True]
        # If both are in separate orders, we have a genuine cross-order pair.
        else:
            cross_order_pairs.append(pair)
            pair.status = [False]

    # Now, assume the first transition is found in one order and the second in
    # two:
    elif (len(orders1) == 1) and (len(orders2) == 2):
        if orders1[0] == orders2[0]:
            if not (pair._lowerEnergyTransition.
                    first_order['right_dist'] > 100 * u.km / u.s):
                if args.verbose:
                    print(f'{pair} too close on right side!')
                pair.status = [False, None]
            else:
                pair.status = [True, None]
        else:
            print(f'Something went wrong with {pair}')
            print(orders1, orders2)

    # Now, assume the second transition is found in one order and the first in
    # two:
    elif (len(orders1) == 2) and (len(orders2) == 1):
        if orders1[1] == orders2[0]:
            if not (abs(pair._higherEnergyTransition.
                        second_order['left_dist']) > 100 * u.km / u.s):
                if args.verbose:
                    print(f'{pair} too close on left side!')
                pair.status = [None, False]
            else:
                pair.status = [None, True]
        else:
            print(f'Something went wrong with {pair}')
            print(orders1, orders2)

    # Now, to check if both transitions involved show up on two orders:
    # (We can combine the checks from before.)
    elif (len(orders1) == 2) and (len(orders2) == 2):
        pair.status = [None, None]
        if not (pair._lowerEnergyTransition.
                first_order['right_dist'] > 100 * u.km / u.s):
            if args.verbose:
                print(f'Double {pair} too close on right side!')
            pair.status[0] = False
        else:
            pair.status[0] = True
        if not (abs(pair._higherEnergyTransition.
                    second_order['left_dist']) > 100 * u.km / u.s):
            if args.verbose:
                print(f'Double {pair} too close on left side!')
            pair.status[1] = False
        else:
            pair.status[1] = True

    else:
        print(f'Situation not caught by previous checks! {pair}')


if args.verbose:
    affected_transitions = []
    print(f'Out of {len(pairs)} pairs:')
    print('--- Pairs entirely on one order:')
    for pair in pairs:
        if pair.status == [True]:
            print(pair.label)
    print('--- Pairs safe on first order:')
    for pair in pairs:
        if pair.status == [True, None]:
            print(pair.label)
    print('--- Pairs safe on second order:')
    for pair in pairs:
        if pair.status == [None, True]:
            print(pair.label)
    print('--- Pairs safe on both orders:')
    for pair in pairs:
        if pair.status == [True, True]:
            print(pair.label)
    print('--- Pairs safe on first of two orders:')
    for pair in pairs:
        if pair.status == [True, False]:
            print(pair.label)
    print('--- Pairs safe on second of two orders:')
    for pair in pairs:
        if pair.status == [False, True]:
            print(pair.label)
    print('--- Pairs not safe on either order:')
    for pair in pairs:
        if pair.status == [False, False]:
            print(pair.label)
    print('--- Pairs cross-order due to being too close on first order:')
    for pair in pairs:
        if pair.status == [False, None]:
            print(pair.label)
    print('--- Pairs cross-order due to being too close on second order:')
    for pair in pairs:
        if pair.status == [None, False]:
            print(pair.label)
    print('--- Entirely cross-order pairs:')
    for pair in pairs:
        if pair.status == [False]:
            print(pair.label)
            for transition in pair:
                if transition not in affected_transitions:
                    affected_transitions.append(transition)
    print('Affected transitions:')
    affected_transitions.sort()
    for transition in affected_transitions:
        print(transition)
