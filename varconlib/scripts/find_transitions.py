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
from glob import glob
import lzma
import os
from pathlib import Path
import pickle

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import (PositiveAmplitudeError,
                                  BlazeFileNotFoundError,
                                  NewCoefficientsNotFoundError)
from varconlib.fitting import GaussianFit
import varconlib.obs2d as obs2d
from varconlib.miscellaneous import (wavelength2index, blueCCDpath,
                                     redCCDpath)

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
                    help='Which HDUs to update (WAVE, BARY, PIXLOWER, PIXUPPER'
                    ' FLUX, ERR, BLAZE, or ALL)')
parser.add_argument('--create-plots', action='store_true', default=False,
                    help='Create plots for each observation and transition.')

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

# Search through subdirectories for e2ds files:
glob_search_string = str(observations_dir) + '/*/*e2ds_A.fits'
# Get a list of all the data files in the data directory:
data_files = [Path(string) for string in sorted(glob(glob_search_string))]

tqdm.write('Found {} observations in the directory.'.format(len(data_files)))

# output_dir = /Users/dberke/data_output
output_dir = Path(vcl.config['PATHS']['output_dir'])
data_dir = output_dir / args.object_name
if not data_dir.exists():
    os.mkdir(data_dir)

# Define edges between pixels to plot to see if transitions overlap them.
edges = (509.5, 1021.5, 1533.5, 2045.5, 2557.5, 3069.5, 3581.5)

# Read the red and blue spectral format files for HARPS.
blue_spec_format = np.loadtxt(blueCCDpath, skiprows=1, delimiter=',',
                              usecols=(0, 5, 6))
red_spec_format = np.loadtxt(redCCDpath, skiprows=1, delimiter=',',
                             usecols=(0, 5, 6))

# Read the pickled list of transitions
with open(vcl.final_selection_file, 'rb') as f:
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

for obs_path in tqdm(data_files[args.start:args.end]) if\
  len(data_files) > 1 else data_files:
    tqdm.write('-' * 40)
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

    obs_dir = data_dir / obs_path.stem

    if not obs_dir.exists():
        os.mkdir(obs_dir)

    ccd_positions_dir = obs_dir.parent / 'ccd_positions'

    if not ccd_positions_dir.exists():
        os.mkdir(ccd_positions_dir)

    # Define directory for output pickle files:
    output_pickle_dir = obs_dir / '_'.join(['pickles', suffix])
    if not output_pickle_dir.exists():
        os.mkdir(output_pickle_dir)

    # Define paths for plots to go in:
    output_plots_dir = obs_dir / '_'.join(['plots', suffix])

    if not output_plots_dir.exists():
        os.mkdir(output_plots_dir)

    # Create the plot sub-directories.
    closeup_dir = output_plots_dir / 'close_up'
    if not closeup_dir.exists():
        os.mkdir(closeup_dir)

    context_dir = output_plots_dir / 'context'
    if not context_dir.exists():
        os.mkdir(context_dir)

    # Create some lists to hold x,y coordinates for the CCD position plot.
    transitions_x, transitions_y, labels = [], [], []

    fits_list = []
    for transition in tqdm(transitions_list):
        tqdm.write('Attempting fit of {}'.format(transition))
        plot_closeup = closeup_dir / '{}_{}_close.png'.format(
                obs_path.stem, transition.label)
        plot_context = context_dir / '{}_{}_context.png'.format(
                obs_path.stem, transition.label)
        try:
            fit = GaussianFit(transition, obs, verbose=args.verbose,
                              integrated=args.integrated_gaussian,
                              close_up_plot_path=plot_closeup,
                              context_plot_path=plot_context)
        except (RuntimeError, PositiveAmplitudeError):
            tqdm.write('Warning! Unable to fit {}!'.format(transition))
            continue

        # Assuming the fit didn't fail, continue on:
        fits_list.append(fit)
        if args.create_plots:
            fit.plotFit(plot_closeup, plot_context)
            y = obs.findWavelength(fit.correctedWavelength,
                                   obs.barycentricArray,
                                   mid_most=True, verbose=False)
            x = wavelength2index(fit.correctedWavelength,
                                 obs.barycentricArray[y])
            transitions_y.append(y + 1)
            transitions_x.append(x)
            labels.append(transition.label)

    # Pickle the list of fits, then compress them to save space before writing
    # them out.
    tqdm.write('Fit {}/{} transitions in {}.'.format(len(fits_list),
               len(transitions_list), obs_path.name))
    outfile = output_pickle_dir / '{}_gaussian_fits.lzma'.format(
            obs._filename.stem)
    if not outfile.parent.exists():
        os.mkdir(outfile.parent)
    with lzma.open(outfile, 'wb') as f:
        tqdm.write(f'Pickling and compressing list of fits at {outfile}')
        f.write(pickle.dumps(fits_list))

    # Create a plot to show locations of transitions on the CCD for this
    # observation.
    if args.create_plots:
        tqdm.write('Creating plot of transition CCD locations...')
        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(left=0, right=4097)
        ax.set_ylim(bottom=16, top=73)

        for i in range(17, 73, 1):
            ax.axhline(i, linestyle='--', color='Gray', alpha=0.7)

        for j in edges:
            ax.axvline(j, linestyle='-', color='SlateGray', alpha=0.8)

        ax.axhline(46.5, linestyle='-.', color='Peru', alpha=0.6)

    #    for row in blue_spec_format:
    #        order = order_numbers.inverse[row[0]]
    #        wls = [shift_wavelength(conversions.air2vacESO(row[i] * u.nm),
    #                                obs.radialVelocity)
    #               for i in (1, 2)]
    #        for wl in wls:
    #            ax.plot(wavelength2index(wl, obs.wavelengthArray[order-1]),
    #                    order, marker='|', color='Blue')
    #            ax.plot(wavelength2index(wl, obs.barycentricArray[order-1]),
    #                    order, marker='|', color='SlateBlue')
    #    for row in red_spec_format:
    #        order = order_numbers.inverse[row[0]]
    #        wls = (shift_wavelength(conversions.air2vacESO(row[i] * u.nm),
    #                                obs.radialVelocity)
    #               for i in (1, 2))
    #        for wl in wls:
    #            ax.plot(wavelength2index(wl, obs.wavelengthArray[order-1]),
    #                    order, marker='|', color='Red')
    #            ax.plot(wavelength2index(wl, obs.barycentricArray[order-1]),
    #                    order, marker='|', color='FireBrick')

        ax.plot(transitions_x, transitions_y, marker='+', color='Sienna',
                linestyle='')
        texts = [plt.text(transitions_x[i], transitions_y[i], labels[i],
                          ha='center', va='center')
                 for i in range(len(labels))]
        tqdm.write('Adjusting label text positions...')
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='OliveDrab'))

        ccd_position_filename = ccd_positions_dir /\
            '{}_CCD_positions.png'.format(obs_path.stem)
        fig.savefig(str(ccd_position_filename))
        plt.close(fig)


# Create hard links to all the fit plots by transition (across star) in
# their own directories, to make it easier to compare transitions
# across observations.
transition_plots_dir = data_dir / 'plots_by_transition'

if not transition_plots_dir.exists():
    os.mkdir(transition_plots_dir)

tqdm.write('Linking fit plots to cross-observation directories.')
for transition in tqdm(transitions_list):

    wavelength_str = '{:.3f}'.format(transition.wavelength.to(u.angstrom)
                                     .value)
    transition_dir = transition_plots_dir / wavelength_str
    close_up_dir = transition_dir / 'close_up'
    context_dir = transition_dir / 'context'

    # Create the required directories if they don't exist.
    for directory in (transition_dir, close_up_dir, context_dir):
        if not directory.exists():
            os.mkdir(directory)

    for plot_type, directory in zip(('close_up', 'context'),
                                    (close_up_dir, context_dir)):

        search_str = str(data_dir) +\
                      '/HARPS*/plots_{}/{}/*{}*.png'.format(suffix,
                                                            plot_type,
                                                            wavelength_str)

        files_to_link = [Path(path) for path in glob(search_str)]
        for file_to_link in files_to_link:
            dest_name = directory / file_to_link.name
            # If the file already exists, linking will fail with an error.
            if not dest_name.exists():
                os.link(file_to_link, dest_name)
