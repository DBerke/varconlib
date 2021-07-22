#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:57:24 2020

@author: dberke

This script collects from varconlib.star.Star objects and stores it for easy
retrieval by other scripts.
"""

import argparse
from json.decoder import JSONDecodeError
import os
from pathlib import Path
import pickle
import time

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


# Define two groups of stars for purposes of injecting a fake signal.
# Group 1 has fewer observations (4480 vs. 6014), but 2/3 of solar twins
# (and ~80% of observations among them.) This will be the group to have the
# fake signal injected.

group1 = set(['HD1581', 'HD65907', 'HD136352', 'HD146233', 'HD1461', 'HD59468',
              'HD45184', 'HD73524', 'HD177565', 'HD43834', 'HD211415',
              'HD1388', 'HD114853', 'HD68978', 'HD39091', 'HD10180', 'HD20407',
              'HD108309', 'HD97343', 'HD150433', 'HD140901', 'Vesta',
              'HD172051', 'HD196761', 'HD10647', 'HD119638', 'HD93385',
              'HD88742', 'HD20782', 'HD106116', 'HD44594', 'HD144585',
              'HD38973', 'HD140538', 'HD193193', 'HD67458', 'HD83529',
              'HD220507', 'HD55693', 'HD111031', 'HD219482', 'HD102117',
              'HD20619', 'HD78558', 'HD117207', 'HD19467', 'HD208487',
              'HD72769', 'HD76151', 'HD128674', 'HD361', 'HD73256', 'HD136894',
              'HD148816', 'HD205536', 'HD105837', 'HD183658', 'HD38277',
              'HD143114', 'HD171665', 'HD203432', 'HD32724', 'HD68168',
              'HD92788', 'HD11505', 'HD148211', 'HD196800', 'HD30495',
              'HD78747', 'HD95521', 'HD108147', 'HD126525', 'HD141937',
              'HD179949', 'HD184768', 'HD210752', 'HD214953', 'HD222669',
              'HD37962', 'HD70642', 'HD78660', 'HD96937'])
group2 = set(['HD190248', 'HD115617', 'HD69830', 'HD20807', 'HD96700',
              'HD189567', 'HD207129', 'HD38858', 'HD82943', 'HD199288',
              'HD210918', 'HD134060', 'HD217014', 'HD78429', 'HD90156',
              'HD102438', 'HD92719', 'HD98281', 'HD17051', 'HD221356',
              'HD203608', 'HD199960', 'HD4915', 'HD157347', 'HD48938',
              'HD69655', 'HD134987', 'HD31527', 'HD125276', 'HD38382',
              'HD44447', 'HD97037', 'HD168871', 'HD45289', 'HD154417',
              'HD23456', 'HD70889', 'HD157338', 'HD47186', 'HD96423',
              'HD117105', 'HD97998', 'HD180409', 'HD208704', 'HD90905',
              'HD177758', 'HD20766', 'HD71479', 'HD7449', 'HD124292',
              'HD161612', 'HD6735', 'HD125881', 'HD147512', 'HD204385',
              'HD9782', 'HD117618', 'HD2071', 'HD7570', 'HD168443',
              'HD189625', 'HD213575', 'HD4391', 'HD75289', 'HD110619',
              'HD12387', 'HD152391', 'HD215257', 'HD44420', 'HD87838',
              'HD104982', 'HD121504', 'HD138573', 'HD177409', 'HD1835',
              'HD197818', 'HD212708', 'HD222582', 'HD28821', 'HD43587',
              'HD7134', 'HD88725'])


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
    # Flip boolean value, since to recreate (True) the star requires setting
    # its load_data argument to False.
    recreate = not recreate
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
    """Return the weighted mean and some statistics for a given star and
    transition.

    The returned values will be the weighted mean of the transition for all
    observations of the star, the error on the weighted mean, the error on the
    mean, and the standard deviation of all the observations.

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

    data_slice = star.fitOffsetsNormalizedArray[time_slice, col_index]
    mask = np.full_like(data_slice, True, dtype=bool)
    mask[np.isnan(data_slice)] = False
    offsets = data_slice[mask]
    errs = star.fitErrorsArray[time_slice, col_index][mask]

    try:
        weighted_mean, weight_sum = np.average(offsets.value,
                                               weights=errs.value**-2,
                                               returned=True)
    except ZeroDivisionError:
        # If all the observations have been masked out, just return a 4-tuple
        # of NaNs.
        return tuple([u.unyt_quantity(np.nan, units=u.m/u.s, dtype=float)] * 4)

    weighted_mean *= u.m/u.s
    error_on_weighted_mean = (1 / np.sqrt(weight_sum)) * u.m / u.s
    if len(offsets) > 1:
        stddev = np.std(offsets)
        error_on_mean = stddev / np.sqrt(star.getNumObs(time_slice))
    else:
        stddev = 0
        error_on_mean = errs[0]  # Because there's only one error in the array.

    return (weighted_mean, error_on_weighted_mean, error_on_mean, stddev)


def get_pair_data_point(star, time_slice, col_index):
    """Return the weighted mean and some statistics for a given star and pair.

    The returned values will be the weighted mean of the pair for all
    observations of the star where it exists, the error on the weighted mean,
    the error on the mean, and the standard deviation of all the observations.

    Parameters
    ----------
    star : `star.Star`
        The star to get the data from.
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

    data_slice = star.pairSeparationsArray[time_slice, col_index]
    mask = np.full_like(data_slice, True, dtype=bool)
    mask[np.isnan(data_slice)] = False
    separations = data_slice[mask]
    errs = star.pairSepErrorsArray[time_slice, col_index][mask]

    try:
        weighted_mean, weight_sum = np.average(separations.to(u.m/u.s).value,
                                               weights=errs.value**-2,
                                               returned=True)
    except ZeroDivisionError:
        # If all the observations have been masked out, just return a 4-tuple
        # of NaNs.
        return tuple([u.unyt_quantity(np.nan, units=u.m/u.s, dtype=float)] * 4)

    weighted_mean *= u.m/u.s
    error_on_weighted_mean = (1 / np.sqrt(weight_sum)) * u.m / u.s
    if len(separations) > 1:
        stddev = np.std(separations)
        error_on_mean = stddev / np.sqrt(star.getNumObs(time_slice))
    else:
        stddev = 0
        error_on_mean = errs[0]  # Because there's only one error in the array.

    return (weighted_mean, error_on_weighted_mean, error_on_mean, stddev)


def main():
    """Run the main function for the script.

    This collects stars given at the command line and finds the weighted mean
    of the measurements of each transition and pair of transitions taken across
    all their observations, before storing it all in an HDF5 file.

    Returns
    -------
    None

    """

    main_dir = Path(args.main_dir[0])
    if not main_dir.exists():
        raise FileNotFoundError(f'{main_dir} does not exist!')

    tqdm.write(f'Looking in main directory {main_dir}')

    star_list = []
    tqdm.write('Collecting stars...')

    if args.casagrande2011:
        vprint('Applying values from Casagrande et al. 2011.')
    elif args.nordstrom2004:
        vprint('Applying values from Nordstrom et al. 2004.')

    excluded_hot_stars = 0
    excluded_metal_poor_stars = 0
    excluded_obs = 0

    for star_dir in tqdm(args.star_names):
        try:
            star = get_star(main_dir / star_dir, recreate=args.recreate_stars)
        except JSONDecodeError:
            print(f'Error reading JSON from {star_dir}.')
            raise
        if star is None:
            continue
        else:
            # Inject fake signal here.
            if args.inject_fake_signal:
                if star_dir in group1:
                    pass

            if args.casagrande2011:
                star.getStellarParameters('Casagrande2011')
            elif args.nordstrom2004:
                star.getStellarParameters('Nordstrom2004')
            if not args.include_hot_stars:
                if star.temperature > 6072 * u.K:
                    # Don't add this star to the database.
                    vprint(f'{star.name} was too hot'
                           f' ({star.temperature}).')
                    excluded_hot_stars += 1
                    excluded_obs += star.getNumObs()
                    continue
            if star.metallicity < -0.45:
                vprint(f'{star.name} was too metal-poor'
                       f' ({star.metallicity}).')
                excluded_metal_poor_stars += 1
                excluded_obs += star.getNumObs()
                continue
            star_list.append(star)
            vprint(f'Added {star.name}')

    tqdm.write(f'Found {len(star_list)} usable stars in total.')
    if not args.include_hot_stars:
        tqdm.write(f'{excluded_hot_stars} stars were too hot'
                   f' ({excluded_obs} observations in total).')

    vprint(f'{excluded_metal_poor_stars} were too metal-poor.')

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions in the list.')

    # Collect all the transition labels.
    transition_labels = []
    for transition in transitions_list:
        for order_num in transition.ordersToFitIn:
            transition_labels.append('_'.join([transition.label,
                                               str(order_num)]))

    tqdm.write('Unpickling pairs list.')
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)
    vprint(f'Found {len(pairs_list)} pairs in the list.')

    # Collect all the pair labels.
    pair_labels = []
    for pair in pairs_list:
        for order_num in pair.ordersToMeasureIn:
            pair_labels.append('_'.join((pair.label, str(order_num))))

    # Create bidicts to map transition and pair labels to column numbers.
    columns = {label: num for num, label in enumerate(transition_labels)}
    transition_column_dict = bidict(columns)

    pair_columns = {label: num for num, label in enumerate(pair_labels)}
    pair_column_dict = bidict(pair_columns)

    # Define the data structures to fill with results:
    # EotM = error on the mean
    # EotWM = error on the weighted mean

    row_len = len(star_list)
    col_len = len(transition_labels)
    pair_col_len = len(pair_labels)

    # Use three-dimensional arrays here -- first axis is for pre- and post-
    # fiber change results. 0 = pre, 1 = post.
    star_transition_offsets = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_EotWM = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_EotM = np.full([2, row_len, col_len], np.nan)
    star_transition_offsets_stds = np.full([2, row_len, col_len], np.nan)

    star_pair_separations = np.full([2, row_len, pair_col_len], np.nan)
    star_pair_separations_EotWM = np.full([2, row_len, pair_col_len], np.nan)
    star_pair_separations_EotM = np.full([2, row_len, pair_col_len], np.nan)

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
        vprint(f'Collating data from  {star.name:9} with {star_num_obs:4}'
               ' observations.')
        total_obs += star_num_obs
        star_names[star.name] = i

        pre_slice = slice(None, star.fiberSplitIndex)
        post_slice = slice(star.fiberSplitIndex, None)

        for j, label in enumerate(tqdm(transition_labels)):
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
        if not args.transitions_only:
            for k, label in enumerate(tqdm(pair_labels)):
                col_index = star.p_index(label)

                if star.hasObsPre:
                    star_mean, star_eotwm, star_eotm, star_std =\
                        get_pair_data_point(star, pre_slice, col_index)
                    star_pair_separations[0, i, k] = star_mean
                    star_pair_separations_EotWM[0, i, k] = star_eotwm
                    star_pair_separations_EotM[0, i, k] = star_eotm

                if star.hasObsPost:
                    star_mean, star_eotwm, star_eotm, star_std =\
                        get_pair_data_point(star, post_slice, col_index)
                    star_pair_separations[1, i, k] = star_mean
                    star_pair_separations_EotWM[1, i, k] = star_eotwm
                    star_pair_separations_EotM[1, i, k] = star_eotm

        star_temperatures[i] = star.temperature
        star_metallicities[i] = star.metallicity
        star_magnitudes[i] = star.absoluteMagnitude
        star_gravities[i] = star.logg

    star_transition_offsets *= star.fitOffsetsArray.units
    star_transition_offsets_EotWM *= star.fitErrorsArray.units
    star_transition_offsets_EotM *= star.fitErrorsArray.units
    star_transition_offsets_stds *= star.fitErrorsArray.units
    star_temperatures *= u.K
    if not args.transitions_only:
        star_pair_separations *= u.m/u.s
        star_pair_separations_EotWM *= star.pairSepErrorsArray.units
        star_pair_separations_EotM *= star.pairSepErrorsArray.units

    # Save the output to disk.
    unyt_arrays = ('star_transition_offsets', 'star_transition_offsets_EotWM',
                   'star_transition_offsets_EotM', 'star_standard_deviations',
                   'star_temperatures')
    pair_arrays = ('star_pair_separations', 'star_pair_separations_EotWM',
                   'star_pair_separations_EotM')
    other_arrays = ('star_metallicities', 'star_magnitudes', 'star_gravities')

    if not vcl.databases_dir.exists():
        os.mkdir(vcl.databases_dir)

    db_file = vcl.databases_dir / 'stellar_db_uncorrected.hdf5'
    if args.include_hot_stars:
        db_file = vcl.databases_dir / 'stellar_db_uncorrected_hot_stars.hdf5'

    tqdm.write(f'Writing output to {str(db_file)}')
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
            try:
                array.write_hdf5(db_file, dataset_name=f'/{name}')
            except AttributeError:
                print(name)
                print(array)
                raise
        if not args.transitions_only:
            for name, array in zip(pair_arrays,
                                   (star_pair_separations,
                                    star_pair_separations_EotWM,
                                    star_pair_separations_EotM)):
                vprint(f'{name}: {array.shape}')
                array.write_hdf5(db_file, dataset_name=f'/{name}')

        for name, array in zip(other_arrays, (star_metallicities,
                                              star_magnitudes,
                                              star_gravities)):
            hickle.dump(array, f, path=f'/{name}')
            vprint(f'{name}: {array.shape}')
        hickle.dump(transition_column_dict, f, path='/transition_column_index')
        hickle.dump(pair_column_dict, f, path='/pair_column_index')
        hickle.dump(star_names, f, path='/star_row_index')
    tqdm.write(f'Collected {total_obs} observations in total from'
               f' {len(star_list)} stars.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collate data from stars on'
                                     ' their transition and pair measurements'
                                     ' into a standard form saved to disk.')
    parser.add_argument('main_dir', action='store', type=str, nargs=1,
                        help='The main directory within which to find'
                        ' additional star directories.')
    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')
    parser.add_argument('--recreate-stars', action='store_true', default=False,
                        help='Recreate all "star.Star" HDF5 files from'
                        ' observations.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")
    parser.add_argument('--include-hot-stars', action='store_true',
                        help="Include stars hotter than solar + 300 K in the"
                        " database.")
    parser.add_argument('--transitions-only', action='store_true',
                        help='Exclude information on pair measurements.')
    parser.add_argument('--inject-fake-signal', action='store_true',
                        help='Inject a fake signal when recreating stars.'
                        ' (Requires --recreate-stars to work.)')

    paper = parser.add_mutually_exclusive_group()
    paper.add_argument('--casagrande2011', action='store_true',
                       help='Use values from Casagrande et al. 2011.')
    paper.add_argument('--nordstrom2004', action='store_true',
                       help='Use values from Nordstrom et al. 2004.')

    args = parser.parse_args()

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    start_time = time.time()

    main()

    duration = time.time() - start_time
    print(f'Finished in {duration:.1f} seconds.')
