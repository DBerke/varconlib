#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:41:58 2018

@author: dberke

This script takes a dictionary containing lists of `Transition` objects from
the script select_line_pairs, and an input list of spectral observations and
attempts to fit each of the transitions listed in the dictionary.
"""

import argparse
import configparser
from glob import glob
import os
from pathlib import Path
import pickle

from tqdm import tqdm
import unyt as u

from exceptions import (PositiveAmplitudeError, BlazeFileNotFoundError,
                        NewCoefficientsNotFoundError)
from fitting import GaussianFit
import obs2d

desc = 'Fit absorption features in spectra.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('object_dir', action='store',
                    help='Directory in which to find e2ds sub-folders.')
parser.add_argument('object_name', action='store',
                    help='Name of object to use for storing output.')

parser.add_argument('--start', type=int, action='store', default=0,
                    help='Start position in the list of observations.')
parser.add_argument('--end', type=int, action='store', default=-1,
                    help='End position in the list of observations.')

parser.add_argument('--pixel-positions', action='store_true',
                    default=False,
                    help='Use new pixel positions.')
parser.add_argument('--new-coefficients', action='store_true',
                    default=False,
                    help='Use new calibration coefficients.')
parser.add_argument('--integrated-gaussian', action='store_true',
                    default=False,
                    help='Fit using an integrated Gaussian.')
parser.add_argument('--update', action='store', metavar='HDU-name',
                    nargs='+', default=[],
                    help='Which HDUs to update (WAVE, BARY, FLUX, ERR, BLAZE,'
                    ' or ALL)')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Print out additional information while running.')

args = parser.parse_args()

observations_dir = Path(args.object_dir)
# Check that the path given exists:
if not observations_dir.exists():
    print(observations_dir)
    raise RuntimeError('The given directory does not exist.')

# Check if the given path ends in data/reduced:
if observations_dir.match('*/data/reduced'):
    pass
else:
    observations_dir = observations_dir / 'data/reduced'
    if not observations_dir.exists():
        print(observations_dir)
        raise RuntimeError('The directory does not contain "data/reduced".')

# Currently using /Users/dberke/HD146233/data/reduced/ for a specific file.

# Search through subdirectories for e2ds files:
glob_search_string = str(observations_dir) + '/*/*e2ds_A.fits'
# Get a list of all the data files in the data directory:
data_files = [Path(string) for string in sorted(glob(glob_search_string))]

tqdm.write('Found {} observations in the directory.'.format(len(data_files)))

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

pickle_dir = Path(config['PATHS']['pickle_dir'])
# All unique transitions found within pairs found:
pickle_pairs_transitions_file = pickle_dir / 'pair_transitions.pkl'

# Final selection of 145 transitions
final_selection_file = pickle_dir / 'final_transitions_selection.pkl'

# output_dir = /Users/dberke/data_output
output_dir = Path(config['PATHS']['output_dir'])

# Read the pickled list of transitions
with open(final_selection_file, 'r+b') as f:
    tqdm.write('Unpickling list of transitions...')
    transitions_list = pickle.load(f)

tqdm.write(f'Found {len(transitions_list)} transitions.')

# Set variables for using new calibration methods.
pix_pos = True if args.pixel_positions else False
if pix_pos:
    tqdm.write('Using new pixel positions.')
new_coeffs = True if args.new_coefficients else False
if new_coeffs:
    tqdm.write('Using new wavelength calibration coefficients.')

for obs_path in tqdm(data_files[args.start:args.end]) if\
  len(data_files) > 1 else data_files:
    tqdm.write('Fitting {}...'.format(obs_path.name))
    try:
        obs = obs2d.HARPSFile2DScience(obs_path,
                                       use_pixel_positions=pix_pos,
                                       use_new_coefficients=new_coeffs,
                                       update=args.update)
        # We need to test if new calibration coefficients are available or not,
        # but if the wavelenth array isn't updated it won't call the function
        # that checks for them, so call it manually in that case.
        if set(['ALL', 'WAVE', 'BARY']).isdisjoint(set(args.update)):
            obs.getWavelengthCalibrationFile()
    except BlazeFileNotFoundError:
        tqdm.write('Blaze file not found, continuing.')
        continue
    except NewCoefficientsNotFoundError:
        tqdm.write('New coefficients not found, continuing.')
        continue

    object_dir = output_dir / args.object_name / obs_path.stem

    if not object_dir.parent.exists():
        os.mkdir(object_dir.parent)

    if not object_dir.exists():
        os.mkdir(object_dir)

    # Define directory suffixes based on arguments:
    if (not args.pixel_positions) and (not args.new_coefficients):
        suffix = 'old'
    elif args.pixel_positions and (not args.new_coefficients):
        suffix = 'pix'
    elif (not args.pixel_positions) and args.new_coefficients:
        suffix = 'coeffs'
    elif args.pixel_positions and args.new_coefficients:
        if args.integrated_gaussian:
            suffix = 'int'
        else:
            suffix = 'new'

    # Define directory for output pickle files:
    output_pickle_dir = object_dir / '_'.join(['pickles', suffix])
    if not output_pickle_dir.exists():
        os.mkdir(output_pickle_dir)

    # Define paths for plots to go in:
    output_plots_dir = object_dir / '_'.join(['plots', suffix])

    if not output_plots_dir.exists():
        os.mkdir(output_plots_dir)

    # Create the plot sub-directories.
    closeup_dir = output_plots_dir / 'close_up'
    if not closeup_dir.exists():
        os.mkdir(closeup_dir)

    context_dir = output_plots_dir / 'context'
    if not context_dir.exists():
        os.mkdir(context_dir)

    fits_list = []
    for transition in tqdm(transitions_list):
        tqdm.write('Attempting fit of {}'.format(transition))
        plot_closeup = closeup_dir / 'Transition_{:.4f}_{}{}.png'.format(
                transition.wavelength.to(u.angstrom).value,
                transition.atomicSymbol, transition.ionizationState)
        plot_context = context_dir / 'Transition_{:.4f}_{}{}.png'.format(
                transition.wavelength.to(u.angstrom).value,
                transition.atomicSymbol, transition.ionizationState)
        try:
            fit = GaussianFit(transition, obs, verbose=args.verbose,
                              integrated=args.integrated_gaussian)
            fit.plotFit(plot_closeup, plot_context)
            fits_list.append(fit)
        except (RuntimeError, PositiveAmplitudeError):
            tqdm.write('Warning! Unable to fit {}!'.format(transition))

    tqdm.write('Fit {}/{} transitions in {}.'.format(len(fits_list),
               len(transitions_list), obs_path.name))
    outfile = output_pickle_dir / '{}_gaussian_fits.pkl'.format(
            obs._filename.stem)
    if not outfile.parent.exists():
        os.mkdir(outfile.parent)
    with open(outfile, 'w+b') as f:
        tqdm.write(f'Pickling list of fits at {outfile}')
        pickle.dump(fits_list, f)
