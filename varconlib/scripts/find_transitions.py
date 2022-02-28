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
import logging
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import numpy as np
from p_tqdm import p_map
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import (PositiveAmplitudeError,
                                  BlazeFileNotFoundError,
                                  NewCoefficientsNotFoundError)
from varconlib.fitting import GaussianFit
import varconlib.obs2d as obs2d
from varconlib.miscellaneous import (wavelength2index, blueCCDpath,
                                     redCCDpath, shift_wavelength)

desc = ("Fit absorption features in spectra.\n\n"
        "Example usage:\n"
        r"find_transitions.py /Volumes/External\ Storage/HARPS/HD117618/"
        " HD117618 --pixel-positions --new-coefficients --integrated-gaussian"
        " --create-plots")


def find_transitions_in_obs(obs_path):
    """
    Find transitions in a given observation file.

    Parameters
    ----------
    obs_path : `pathlib.Path` or str
        A file path to an observation file.

    Returns
    -------
    None.

    """
    # tqdm.write('-' * 40)
    vprint('Fitting {}...'.format(obs_path.name))
    try:
        obs = obs2d.HARPSFile2DScience(obs_path,
                                       pixel_positions=pix_pos,
                                       new_coefficients=new_coeffs,
                                       update=args.update)
        # We need to test if new calibration coefficients are available or not,
        # but if the wavelenth array isn't updated it won't call the function
        # that checks for them, so call it manually in that case.
        if set(['ALL', 'WAVE', 'BARY']).isdisjoint(set(args.update)):
            obs.getWavelengthCalibrationFile()
    except BlazeFileNotFoundError:
        err_msg = f'Blaze file not found for {obs_path.name}, continuing.'
        vprint(err_msg)
        logger.warning(err_msg)
        return
    except NewCoefficientsNotFoundError:
        err_msg = f'New coefficients not found for {obs_path.name},'\
                  ' continuing.'
        vprint(err_msg)
        logger.warning(err_msg)
        return

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
    # Create a second set for failed measurments.
    bad_x, bad_y, bad_labels = [], [], []

    fits_list = []
    fit_transitions = 0
    # tqdm.write('Fitting transitions...')
    for transition in transitions_list:
        for order_num in transition.ordersToFitIn:
            vprint(f'Attempting fit of {transition} in order'
                   f' {order_num}')
            plot_closeup = closeup_dir / '{}_{}_{}_close.png'.format(
                    obs_path.stem, transition.label, order_num)
            plot_context = context_dir / '{}_{}_{}_context.png'.format(
                    obs_path.stem, transition.label, order_num)
            try:
                fit = GaussianFit(transition, obs, order_num,
                                  radial_velocity=rv,
                                  verbose=args.verbose,
                                  integrated=args.integrated_gaussian,
                                  close_up_plot_path=plot_closeup,
                                  context_plot_path=plot_context)
            except RuntimeError:
                err_msg = ('Unable to fit'
                           f' {transition}_{order_num} for'
                           f' {obs_path.name}!')
                vprint(err_msg)
                logger.warning(err_msg)
                # Append None to fits list to signify that no fit exists for
                # this transition.
                fits_list.append(None)
                # Fit is plotted automatically upon failing, move on to next
                # transition.
                continue
            except PositiveAmplitudeError:
                err_msg = (f'Fit of {transition} {order_num} failed with'
                           ' PositiveAmplitudeError in'
                           f' {obs_path.name}!')
                vprint(err_msg)
                logger.warning(err_msg)
                fits_list.append(None)
                if args.create_ccd_plots:
                    bad_y.append(order_num + 1)
                    if rv is None:
                        corrected_wavelength = shift_wavelength(
                            transition.wavelength, obs.radialVelocity)
                    else:
                        corrected_wavelength = shift_wavelength(
                            transition.wavelength, rv)
                    bad_x.append(wavelength2index(corrected_wavelength,
                                 obs.barycentricArray[order_num]))
                    bad_labels.append('_'.join((transition.label,
                                               str(order_num))))
                continue

            # Assuming the fit didn't fail, continue on:
            fits_list.append(fit)
            fit_transitions += 1
            if args.create_plots:
                # Plot the fit.
                fit.plotFit(plot_closeup, plot_context)
            if args.create_ccd_plots:
                # Save the x and y pixel positions of the transitions on the CCD
                transitions_y.append(fit.order + 1)
                transitions_x.append(wavelength2index(fit.correctedWavelength,
                                     obs.barycentricArray[order_num]))
                labels.append(fit.label)

    # Pickle the list of fits, then compress them to save space before writing
    # them out.
    info_msg = f'Fit {fit_transitions}/{len(fits_list)} transitions' +\
               f' in {obs_path.name}.'
    vprint(info_msg)
    logger.info(info_msg)
    outfile = output_pickle_dir / '{}_gaussian_fits.lzma'.format(
            obs._filename.stem)
    if not outfile.parent.exists():
        os.mkdir(outfile.parent)
    with lzma.open(outfile, 'wb') as f:
        vprint(f'Pickling and compressing list of fits at {outfile}')
        f.write(pickle.dumps(fits_list))

    # Create a plot to show locations of transitions on the CCD for this
    # observation.
    if args.create_ccd_plots:
        tqdm.write('Creating plot of transition CCD locations...')
        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(left=0, right=4097)
        ax.set_ylim(bottom=16, top=73)
        ax.xaxis.set_major_locator(ticks.MultipleLocator(base=512))
        ax.xaxis.set_minor_locator(ticks.MultipleLocator(base=64))

        ax.yaxis.set_major_locator(ticks.MultipleLocator(base=2))

        ax.set_xlabel('Position across CCD (pixels)')
        ax.set_ylabel('Order number')

        ax.grid(which='major', axis='x', color='Gray', alpha=0.8)
        ax.grid(which='minor', axis='x', color='LightGray', alpha=0.9)

        for i in range(17, 73, 1):
            ax.axhline(i, linestyle='--', color='Gray', alpha=0.7)

        ax.axhline(46.5, linestyle='-.', color='Peru', alpha=0.8)

        ax.plot(transitions_x, transitions_y, marker='+', color='RoyalBlue',
                linestyle='', markersize=8)
        ax.plot(bad_x, bad_y, marker='+', color='FireBrick',
                linestyle='', markersize=10)

        texts = [plt.text(transitions_x[i], transitions_y[i], labels[i],
                          ha='center', va='center')
                 for i in range(len(labels))]
        bad_texts = [plt.text(bad_x[i], bad_y[i], bad_labels[i],
                              ha='center', va='center', color='DarkRed')
                     for i in range(len(bad_labels))]
        tqdm.write('Adjusting label text positions...')
        # Add all the text labels together to adjust them.
        texts.extend(bad_texts)
        adjust_text(texts,
                    arrowprops=dict(arrowstyle='-', color='OliveDrab'),
                    lim=1000, fontsize=9)

        ccd_position_filename = ccd_positions_dir /\
            '{}_CCD_positions.png'.format(obs_path.stem)
        fig.savefig(str(ccd_position_filename))
        plt.close(fig)


parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
parser.add_argument('object_dir', action='store',
                    help='Directory in which to find e2ds sub-folders.')
parser.add_argument('object_name', action='store',
                    help='Name of object to use for storing output.')

parser.add_argument('--output_dir', action='store',
                    help='Directory in which to store output. Defaults to'
                    f""" "{vcl.output_dir}" if not"""
                    ' specified.')

parser.add_argument('--start', type=int, action='store', default=0,
                    help='Start position in the list of observations.')
parser.add_argument('--end', type=int, action='store', default=None,
                    help='End position in the list of observations.')

parser.add_argument('-rv', '--radial-velocity', action='store', type=float,
                    default=None,
                    help='Radial velocity to use for the star, in km/s.'
                    ' If not given will use the value from the FITS files.')

parser.add_argument('--integrated-gaussian', action='store_true',
                    default=False,
                    help='Fit using an integrated Gaussian.')
parser.add_argument('--update', action='store', metavar='HDU-name',
                    nargs='+', default=[],
                    help='Which HDUs to update (WAVE, BARY, PIXLOWER, PIXUPPER'
                    ' FLUX, ERR, BLAZE, or ALL)')

parser.add_argument('--create-plots', action='store_true', default=False,
                    help='Create plots of the fit for each transition.')
parser.add_argument('--create-ccd-plots', action='store_true',
                    help='Create plots of the CCD positions of each transition'
                    ' for each observation.')

parser.add_argument('-s', '--single-threaded', action='store_true',
                    help="Don't use multi-processing for debugging.")
parser.add_argument('-p', '--preview', action='store_true',
                    help='Show which observations will be selected to work on,'
                    ' then exit.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Print out additional information while running.')

args = parser.parse_args()

vprint = vcl.verbose_print(args.verbose)

start_time = time.time()

if args.radial_velocity:
    rv = args.radial_velocity * u.km / u.s
    assert abs(rv) < u.c, 'Given RV exceeds speed of light!'
else:
    rv = args.radial_velocity

observations_dir = Path(args.object_dir)
# Check that the path given exists:
if not observations_dir.exists():
    raise FileNotFoundError('The given directory:\n'
                            f'"{observations_dir}"\n'
                            'could not be found.')

# Check if the given path ends in data/reduced:
if observations_dir.match('*/data/reduced'):
    pass
else:
    observations_dir = observations_dir / 'data/reduced'
    if not observations_dir.exists():
        raise FileNotFoundError('The given directory:\n'
                                f'"{observations_dir}"\n'
                                'could not be found.')

# Search through subdirectories for e2ds files:
glob_search_string = str(observations_dir) + '/*/*e2ds_A.fits'
# Get a list of all the data files in the data directory:
data_files = [Path(string) for string in sorted(glob(glob_search_string))]

files_to_work_on = data_files[slice(args.start, args.end)]

tqdm.write('=' * 41)
tqdm.write(f'Found {len(data_files)} observations in the directory'
           f' for {args.object_name},'
           f' working on {len(files_to_work_on)}:')
for file in files_to_work_on:
    vprint(str(file.name))

tqdm.write('=' * 41)
if args.preview:
    sys.exit(0)

# Set up the output directory.
if not args.output_dir:
    output_dir = Path(vcl.config['PATHS']['output_dir'])
else:
    output_dir = Path(args.output_dir)
data_dir = output_dir / args.object_name
if not data_dir.exists():
    tqdm.write(f'Creating output directory {data_dir}...')
    os.mkdir(data_dir)

# Set up logging.
logger = logging.getLogger('find_transitions')
logger.setLevel(logging.INFO)

log_file = data_dir / f'{args.object_name}.log'
file_handler = logging.FileHandler(log_file, mode='w', delay=False)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Define edges between pixels to plot to see if transitions overlap them.
edges = (509.5, 1021.5, 1533.5, 2045.5, 2557.5, 3069.5, 3581.5)

# Read the red and blue spectral format files for HARPS.
blue_spec_format = np.loadtxt(blueCCDpath, skiprows=1, delimiter=',',
                              usecols=(0, 5, 6))
red_spec_format = np.loadtxt(redCCDpath, skiprows=1, delimiter=',',
                             usecols=(0, 5, 6))

# Read the pickled list of the final selection of transitions:
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

total = len(data_files) - args.start

if args.single_threaded:
    for obs_path in tqdm(files_to_work_on) if\
                len(files_to_work_on) > 1 else files_to_work_on:
        find_transitions_in_obs(obs_path)
else:
    # Use multiprocessing to look at multiple obserations in parallel.
    p_map(find_transitions_in_obs, files_to_work_on)

# Create hard links to all the fit plots by transition (across star) in
# their own directories, to make it easier to compare transitions
# across observations.
transition_plots_dir = data_dir / 'plots_by_transition'

if not transition_plots_dir.exists():
    os.mkdir(transition_plots_dir)

tqdm.write('Linking fit plots to cross-observation directories.')
for transition in tqdm(transitions_list):

    wavelength_str = transition.label
    transition_dir = transition_plots_dir / wavelength_str
    close_up_dir = transition_dir / 'close_up'
    context_dir = transition_dir / 'context'

    # Create the required directories if they don't exist.
    for directory in (transition_dir, close_up_dir, context_dir):
        if not directory.exists():
            os.mkdir(directory)

    for plot_type, directory in zip(('close_up', 'context'),
                                    (close_up_dir, context_dir)):

        search_str = str(data_dir) + '/HARPS*/plots_{}/{}/*{}*.png'.format(
                suffix, plot_type, wavelength_str)

        files_to_link = [Path(path) for path in glob(search_str)]
        for file_to_link in files_to_link:
            dest_name = directory / file_to_link.name
            # If the file already exists, linking will fail with an error.
            if not dest_name.exists():
                os.link(file_to_link, dest_name)

tqdm.write('-' * 30)
duration = time.time() - start_time
tqdm.write(f'Finished analyzing {args.object_name} in {duration:.2f} s.')
