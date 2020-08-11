#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:02:48 2020

@author: dberke

A short script to save out the pair separations for a selection of stars.

"""

import argparse
import csv
import datetime as dt
from json import JSONDecodeError
import os
import pickle

from tqdm import tqdm

import varconlib as vcl
from varconlib.star import Star

header = ['#pair_label', 'delta(v)_pair', 'err_delta(v)_pair',
          'transition1', 'transition2', 't_err1', 't_err2',
          'Teff', '[Fe/H]', 'log(g)', 'M_V']


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


def main():
    """Run the main routine for this script."""

    main_dir = vcl.output_dir

    star_list = []
    tqdm.write('Collecting stars...')

    for star_dir in tqdm(args.star_names):
        try:
            star = get_star(main_dir / star_dir)
        except JSONDecodeError:
            print(f'Error reading JSON from {star_dir}.')
            raise
        if star is None:
            continue
        else:
            star_list.append(star)

    # The directory for storing these CSV files.
    output_dir = vcl.output_dir / 'pair_separation_files'
    if not output_dir.exists():
        os.mkdir(output_dir)

    # Generate a file with properties constant for each star
    tqdm.write('Writing out data for each star.')
    properties_header = ['#star_name', 'Teff (K)', '[Fe/H]', 'log(g)',
                         'M_V', '#obs', 'obs_baseline (days)']

    star_properties_file = output_dir / 'star_properties.csv'
    with open(star_properties_file, 'w', newline='') as f:
        datawriter = csv.writer(f)
        datawriter.writerow(properties_header)

        for star in star_list:
            info_list = [star.name, star.temperature, star.metallicity,
                         star.logg, star.absoluteMagnitude, star.getNumObs(),
                         round(star.obsBaseline / dt.timedelta(days=1), 3)]
            datawriter.writerow(info_list)

    # Import the list of pairs to use.
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    tqdm.write('Writing out data for each pair.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')
            csv_file = output_dir / f'{pair_label}_pair_separations.csv'
            info_list = []

            for star in star_list:
                info_list.append(star.formatPairData(pair, order_num))

        with open(csv_file, 'w', newline='') as f:
            datawriter = csv.writer(f)
            datawriter.writerow(star.formatHeader)
            for row in info_list:
                datawriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save out results of the pair'
                                     ' separations for a selection of stars.')

    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print out more information about the script's"
                        " output.")

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
