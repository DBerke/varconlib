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

from astropy.coordinates import SkyCoord
import astropy.units as unit
from astroquery.simbad import Simbad
from tqdm import tqdm

import varconlib as vcl
from varconlib.exceptions import (HDF5FileNotFoundError,
                                  PickleFilesNotFoundError)
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
    star_names = []
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
            # Put updates to stars that don't require rebuilding here, just
            # call saveDataToDisk() afterwards.
            if 'HD' in star.name:
                star_names.append(star.name)

    # Only update star coordinates if specifically requested.
    star_coords_file = vcl.data_dir / 'Star_coords.csv'
    if args.get_star_coords:
        tqdm.write('Querying Simbad for star coordinates.')
        coords_dict = {}
        results_table = Simbad.query_objects(star_names)

        for star_name, row in tqdm(zip(star_names, results_table),
                                   total=len(star_names)):
            c = SkyCoord(ra=row['RA'], dec=row['DEC'],
                         unit=(unit.hourangle, unit.deg))
            coords_dict[star_name] = [star_name,
                                      c.ra.to_string(unit=unit.hour,
                                                     sep=' '),
                                      c.dec.to_string(unit=unit.degree,
                                                      sep=' '),
                                      c.galactic.l.to_string(unit=unit.degree,
                                                             sep=' '),
                                      c.galactic.b.to_string(unit=unit.degree,
                                                             sep=' ')]

        coords_header = ['#star_name', 'RA', 'DEC', 'l', 'b']
        with open(star_coords_file, 'w', newline='') as f:
            datawriter = csv.writer(f)
            datawriter.writerow(coords_header)
            for key, value in coords_dict.items():
                datawriter.writerow(value)
    else:
        if not star_coords_file.exists():
            raise FileNotFoundError('No Star_coords.csv file exists.')
        else:
            tqdm.write(f'Using cached star coordinates at {star_coords_file}.')
            coords_dict = {}
            with open(star_coords_file, 'r', newline='') as f:
                datareader = csv.reader(f, delimiter=',')
                for row in datareader:
                    coords_dict[row[0]] = row

    # The directory for storing these CSV files.
    output_dir = vcl.output_dir / 'pair_separation_files'
    output_pre = output_dir / 'pre'
    output_post = output_dir / 'post'
    if not output_dir.exists():
        os.mkdir(output_dir)
        os.mkdir(output_pre)
        os.mkdir(output_post)

    if args.stars:
        # Generate a file with properties constant for each star
        tqdm.write('Writing out data for each star.')
        properties_header = ['#star_name', 'RA', 'DEC', 'l', 'b',
                             'Teff (K)', '[Fe/H]', 'log(g)',
                             'M_V', '#obs', 'start_date', 'end_date']

        star_properties_file = output_dir / 'star_properties.csv'
        with open(star_properties_file, 'w', newline='') as f:
            datawriter = csv.writer(f)
            datawriter.writerow(properties_header)

            for star in star_list:
                if 'HD' in star.name:
                    info_list = coords_dict[star.name]
                else:
                    info_list = [star.name]
                    info_list.extend(['-']*4)
                obs_dates = [dt.datetime.fromisoformat(obs_date) for obs_date in
                             star._obs_date_bidict.keys()]
                start_date = min(obs_dates)
                end_date = max(obs_dates)
                info_list.extend([star.temperature.value, star.metallicity,
                                  star.logg, star.absoluteMagnitude,
                                  star.getNumObs(),
                                  start_date.isoformat(timespec='milliseconds'),
                                  end_date.isoformat(timespec='milliseconds')])
                datawriter.writerow(info_list)

    if args.pairs:
        # Import the list of pairs to use.
        with open(vcl.final_pair_selection_file, 'r+b') as f:
            pairs_list = pickle.load(f)

        tqdm.write('Writing out data for each pair.')
        for pair in tqdm(pairs_list):
            for order_num in pair.ordersToMeasureIn:
                pair_label = "_".join([pair.label, str(order_num)])
                for era in ('pre', 'post'):
                    vprint(f'Collecting data for {pair_label} in {era}-fiber'
                           ' change era.')
                    csv_file = output_dir /\
                        f'{era}/{pair_label}_pair_separations_{era}.csv'
                    info_list = []

                    for star in star_list:
                        try:
                            info_list.append(star.formatPairData(pair,
                                                                 order_num,
                                                                 era))
                        except ZeroDivisionError:
                            vprint(f'Error with {star.name}, {era}.')
                            raise

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

    parser.add_argument('-S', '--stars', action='store_true',
                        help="Create a file containing static information for"
                        " each star.")
    parser.add_argument('-P', '--pairs', action='store_true',
                        help="Create files for each pair.")

    parser.add_argument('--get-star-coords', action='store_true',
                        help='Query Simbad for coordinates for all the stars'
                        ' given and store them in a file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print out more information about the script's"
                        " output.")

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
