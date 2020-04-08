#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:57:24 2020

@author: dberke

This script collects from varconlib.star.Star objects and stores it for easy
retrieval by other scripts.
"""

import argparse
import os
from pathlib import Path
import pickle
import sys

from bidict import bidict
import h5py
import hickle
import numpy as np
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import (HDF5FileNotFoundError,
                                  PickleFilesNotFoundError)
from varconlib.star import Star


def append_dir(dir1, dir2):
    """Append dir2 to dir2.

    Parameters
    ----------
    dir1 : pathlib.Path
        A path.
    dir2 : str or pathlib.Path
        A second path to be appended to `dir1`.

    """

    return dir1 / dir2


def get_star(star_path, verbose=False, recreate=False):
    """Return a varconlib.star.Star object based on its name.

    Parameters
    ----------
    star_path : str
        A string representing the name of the directory where the HDF5 file
        containing a `star.Star`'s data can be found.

    Optional
    --------
    verbose : bool, Default: False
        If *True*, write out additional information.
    recreate : bool, Default: False
        If *True*, first recreate the star from observations before returning
        it. This will only work on stars which already have an HDF5 file saved,
        and will not create new ones.

    Returns
    -------
    `star.Star`
        A Star object from the directory. Note that this will only use already-
        existing stars, it will not create ones which do not already exist from
        their observations.

    """

    assert star_path.exists(), FileNotFoundError('Star directory'
                                                 f' {star_path}'
                                                 ' not found.')
    recreate = not args.recreate_stars
    try:
        return Star(star_path.stem, star_path, load_data=recreate)
    except IndexError:
        vprint(f'Excluded {star_path.stem}.')
        pass
    except HDF5FileNotFoundError:
        vprint(f'No HDF5 file for {star_path.stem}.')
        pass
    except AttributeError:
        vprint(f'Affected star is {star_path.stem}.')
        raise
    except PickleFilesNotFoundError:
        vprint(f'No pickle files found for {star_path.stem}')
        pass


def get_transition_data_point(star, time_slice, col_index):
    """Return the pair separation for a given star and pair.

    The returned values will be the weighted mean value of the pair
    separation, the standard deviation of all the pair separation
    values for that star in the given time period (pre- or post-fiber
    chage), and the error on the weighted mean.

    Parameters
    ----------
    star : `star.Star`
        The star get the data from.
    time_slice : slice
        A slice object specifying the data to use from the star.
    col_index : int
        The index of the columnn to read from.

    Returns
    -------
    tuple
        Returns a 4-tuple of the weighted mean, the error on the
        weighted mean, the error on the mean (standard deviation / sqrt(N))
        and the standard deviation.

        If there is only one observation for the star, the standard deviation
        returned will be zero and the error on the mean will instead be the
        error from the fit itself.

    """

    offsets = star.fitOffsetsNormalizedArray[time_slice, col_index]
    errs = star.fitErrorsArray[time_slice, col_index]
    weighted_mean, weight_sum = np.average(offsets.value,
                                           weights=errs.value**-2,
                                           returned=True)
    weighted_mean *= u.m/u.s
    error_on_weighted_mean = (1 / np.sqrt(weight_sum)) * u.m / u.s
    if len(offsets) > 1:
        stddev = np.std(offsets)
        error_on_mean = stddev / np.sqrt(star.getNumObs(time_slice))
    else:
        stddev = 0
        error_on_mean = errs[0]

    return (weighted_mean, error_on_weighted_mean, error_on_mean, stddev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collate date from stars into'
                                     ' a standard form saved to disk.')
    parser.add_argument('main_dir', action='store', type=str, nargs=1,
                        help='The main directory within which to find'
                        ' additional star directories.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')
    parser.add_argument('--recreate-stars', action='store_true', default=False,
                        help='Recreate all star.Star HDF5 files from'
                        ' observations.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")

    paper = parser.add_mutually_exclusive_group()
    paper.add_argument('--casagrande2011', action='store_true',
                       help='Use values from Casagrande et al. 2011.')
    paper.add_argument('--nordstrom2004', action='store_true',
                       help='Use values from Nordstrom et al. 2004.')

    args = parser.parse_args()

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    main_dir = Path(args.main_dir[0])
    if not main_dir.exists():
        raise FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    star_list = []
    tqdm.write('Collecting stars...')

    for star_dir in tqdm(args.star_names):
        star = get_star(main_dir / star_dir)
        if star is None:
            pass
        else:
            if args.casagrande2011:
                vprint('Applying values from Casagrande et al. 2011.')
                star.getStellarParameters('Casagrande2011')
            elif args.nordstrom2004:
                vprint('Applying values from Nordstrom et al. 2004.')
                star.getStellarParameters('Nordstrom2004')
            star_list.append(star)
            vprint(f'Added {star.name}')

    tqdm.write(f'Found {len(star_list)} usable stars in total.')

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions in the list.')

    transition_labels = []
    for transition in transitions_list:
        for order_num in transition.ordersToFitIn:
            transition_labels.append('_'.join([transition.label,
                                               str(order_num)]))

    column_dict = {label: num for num, label in enumerate(transition_labels)}

    # Define the data structures to fill with results:
    # EotM = error on the mean
    # EotWM = error on the weighted mean

    row_len = len(star_list)
    col_len = len(transition_labels)

    # Use three-dimensional arrays here -- first axis is for pre- and post-
    # fiber change results. 0 = pre, 1 = post.
    star_transition_offsets = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_EotWM = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_EotM = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_stds = np.full([2, row_len, col_len], np.nan)

    star_temperatures = np.full([row_len, 1], np.nan)
    star_metallicities = np.full([row_len, 1], np.nan)
    star_magnitudes = np.full([row_len, 1], np.nan)
    star_gravities = np.full([row_len, 1], np.nan)

    star_names = bidict()

    total_obs = 0

    # Iterate over all the stars collected:
    tqdm.write('Collecting data from stars...')

    for i, star in enumerate(tqdm(star_list)):
        star_num_obs = star.getNumObs()
        vprint(f'Collating data from {star.name:9} with {star_num_obs:4}'
               ' observations.')
        total_obs += star_num_obs
        star_names[star.name] = i

        for j, label in enumerate(transition_labels):
            pre_slice = slice(None, star.fiberSplitIndex)
            post_slice = slice(star.fiberSplitIndex, None)
            col_index = star.t_index(label)

            if star.hasObsPre:
                star_mean, star_eotwm, star_eotm, star_std =\
                    get_transition_data_point(star, pre_slice, col_index)
                star_transition_offsets[0, i, j] = star_mean
                star_transition_offsets_EotWM[0, i, j] = star_eotwm
                star_transition_offsets_EotM[0, i, j] = star_eotm
                star_transition_offsets_stds[0, i, j] = star_std

            if star.hasObsPost:
                star_mean, star_eotwm, star_eotm, star_std =\
                    get_transition_data_point(star, post_slice, col_index)
                star_transition_offsets[1, i, j] = star_mean
                star_transition_offsets_EotWM[1, i, j] = star_eotwm
                star_transition_offsets_EotM[1, i, j] = star_eotm
                star_transition_offsets_stds[1, i, j] = star_std

            star_temperatures[i] = star.temperature
            star_metallicities[i] = star.metallicity
            star_magnitudes[i] = star.absoluteMagnitude
            star_gravities[i] = star.logG

    star_transition_offsets *= star.fitOffsetsArray.units
    star_transition_offsets_EotWM *= star.fitErrorsArray.units
    star_transition_offsets_EotM *= star.fitErrorsArray.units
    star_transition_offsets_stds *= star.fitErrorsArray.units
    star_temperatures *= u.K

    # Save the output to disk.
    unyt_arrays = ('star_transition_offsets', 'star_transition_offsets_EotWM',
                   'star_transition_offsets_EotM', 'star_standard_deviations',
                   'star_temperatures')
    other_arrays = ('star_metallicities', 'star_magnitudes', 'star_gravities')

    db_file = vcl.stellar_results_file
    vprint(f'Writing output to {str(db_file)}')
    if db_file.exists():
        backup_path = db_file.with_name(db_file.stem + '.bak')
        os.replace(db_file, backup_path)
    with h5py.File(db_file, mode='a') as f:
        for name, array in zip(unyt_arrays,
                               (star_transition_offsets,
                                star_transition_offsets_EotWM,
                                star_transition_offsets_EotM,
                                star_transition_offsets_stds,
                                star_temperatures)):
            vprint(f'{name}: {array.shape}')
            array.write_hdf5(db_file, dataset_name=f'/{name}')

        for name, array in zip(other_arrays, (star_metallicities,
                                              star_magnitudes,
                                              star_gravities)):
            hickle.dump(array, f, path=f'/{name}')
            vprint(f'{name}: {array.shape}')
        hickle.dump(column_dict, f, path='/transition_column_index')
        hickle.dump(star_names, f, path='/star_row_index')
    tqdm.write(f'Collected {total_obs} observations in total from'
               f' {len(star_list)} stars.')
