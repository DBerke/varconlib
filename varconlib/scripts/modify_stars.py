#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:58:04 2021

@author: dberke

Make changes to the data in star.Star objects, up to rebuilding them entirely.
"""

import argparse
from glob import glob
from json.decoder import JSONDecodeError
import lzma
from multiprocessing import Pool, RLock
from pathlib import Path
import pickle
import time

import numpy as np
from p_tqdm import p_umap
from tqdm import tqdm

import varconlib as vcl
from varconlib.exceptions import PickleFilesNotFoundError
from varconlib.star import Star

stars_to_use = ('HD1581', 'HD190248', 'HD10180', 'HD102117', 'HD102438',
                'HD104982', 'HD105837', 'HD106116', 'HD108309', 'HD110619',
                'HD111031', 'HD114853', 'HD11505', 'HD115617', 'HD117105',
                'HD117207', 'HD117618', 'HD12387', 'HD124292', 'HD125881',
                'HD126525', 'HD128674', 'HD134060', 'HD134987', 'HD136352',
                'HD136894', 'HD138573', 'HD1388', 'HD140538', 'HD140901',
                'HD141937', 'HD143114', 'HD144585', 'HD1461', 'HD146233',
                'HD147512', 'HD148211', 'HD148816', 'HD150433', 'HD152391',
                'HD154417', 'HD157338', 'HD157347', 'HD161612', 'HD168443',
                'HD168871', 'HD171665', 'HD172051', 'HD177409', 'HD177565',
                'HD177758', 'HD1835', 'HD183658', 'HD184768', 'HD189567',
                'HD189625', 'HD193193', 'HD19467', 'HD196761', 'HD197818',
                'HD199288', 'HD199960', 'HD203432', 'HD20407', 'HD204385',
                'HD205536', 'HD20619', 'HD2071', 'HD207129', 'HD20766',
                'HD20782', 'HD20807', 'HD208704', 'HD210752', 'HD210918',
                'HD211415', 'HD212708', 'HD213575', 'HD214953', 'HD215257',
                'HD217014', 'HD220507', 'HD222582', 'HD222669', 'HD28821',
                'HD30495', 'HD31527', 'HD32724', 'HD361', 'HD37962', 'HD38277',
                'HD38858', 'HD38973', 'HD39091', 'HD43587', 'HD43834', 'HD4391',
                'HD44420', 'HD44447', 'HD44594', 'HD45184', 'HD45289',
                'HD47186', 'HD4915', 'HD55693', 'HD59468', 'HD65907', 'HD6735',
                'HD67458', 'HD68168', 'HD68978', 'HD69655', 'HD69830',
                'HD70642', 'HD70889', 'HD7134', 'HD72769', 'HD73256', 'HD73524',
                'HD7449', 'HD76151', 'HD78429', 'HD78558', 'HD78660', 'HD78747',
                'HD82943', 'HD83529', 'HD88725', 'HD88742', 'HD90156',
                'HD90905', 'HD92719', 'HD92788', 'HD95521', 'HD96423',
                'HD96700', 'HD96937', 'HD97037', 'HD97343', 'HD9782', 'HD97998',
                'HD98281', 'Vesta')


def recreate_star(star_dir):
    """Create a Star from a given directory.


    Parameters
    ----------
    star_dir :`pathlib.Path`
        The directory in which to find the star's files.

    Returns
    -------
    None.

    """

    tqdm.write(f'Creating {star_dir.stem}')
    try:
        Star(star_dir.stem, star_dir, load_data=False)
    except PickleFilesNotFoundError:
        newstar_dir = Path('/Volumes/External Storage/data_output') /\
            star_dir.stem
        tqdm.write('Using external storage files.')
        Star(star_dir.stem, new_star_dir, load_data=False, output_dir=star_dir)


def create_transition_model_corrected_arrays(star_dir):
    """
    Create the transition model-corrected arrays for a Star from a given
    directory.


    Parameters
    ----------
    star_dir : `pathlib.Path`
        The directory in which to find the star's files.

    Returns
    -------
    None.

    """

    tqdm.write(f'Working on {star_dir.stem}')
    star = Star(star_dir.stem, star_dir, load_data=True)
    star.createTransitionModelCorrectedArrays(model_func='quadratic',
                                              n_sigma=2.5)
    star.createPairSeparationArrays()
    star.saveDataToDisk()


def create_pair_model_corrected_arrays(star_dir):
    """
    Create the pair model-corrected array for a Star from a given directory.

    Parameters
    ----------
    star_dir : `pathlib.Path`
        The directory in which to find the star's files.

    Returns
    -------
    None.

    """

    tqdm.write(f'Working on {star_dir.stem}')
    star = Star(star_dir.stem, star_dir, load_data=True)
    star.createPairModelCorrectedArrays(model_func='quadratic',
                                        n_sigma=4.0)
    star.saveDataToDisk()


def add_pixel_data_to_star(star_dir):
    """
    Add information about the pixel each transition was measured at to a star.

    Parameters
    ----------
    star_dir : `pathlib.Path`
        The directory containing the data for the star.

    Returns
    -------
    None.

    """

    tqdm.write(f'Working on {star_dir.stem}')
    star = Star(star_dir.stem, star_dir, load_data=True)

    # Find pickle files in directory
    search_str = str(star_dir) + f'/HARPS*/pickles_int/*fits.lzma'
    pickle_files = [Path(path) for path in sorted(glob(search_str))]

    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)

    num_obs = len(pickle_files)
    num_cols = 0
    for transition in transitions_list:
        num_cols += len(transition.ordersToFitIn)

    star.pixelArray = np.full((num_obs, num_cols), -1, dtype=int)

    for obs_num, pickle_file in enumerate(tqdm(pickle_files)):
        with lzma.open(pickle_file, 'rb') as f:
            fits_list = pickle.loads(f.read())

        for col_num, fit in enumerate(fits_list):
            if fit is not None:
                star.pixelArray[obs_num, col_num] = fit.centralIndex

    star.saveDataToDisk()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automatically recreate all'
                                     ' stars whose names are given.')
    parser.add_argument('star_names', action='store', type=str, nargs='*',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot. If not given will'
                        ' default to using all stars.')
    parser.add_argument('--recreate-stars', action='store_true',
                        help='Trigger a full rebuild of stars from the pickled'
                        ' results files (LENGTHY!).')
    parser.add_argument('--transitions', action='store_true',
                        help='Create the transition model-corrected arrays and'
                        ' pair separation arrays for stars.')
    parser.add_argument('--pairs', action='store_true',
                        help='Create the pair model-corrected arrays for'
                        ' stars.')
    parser.add_argument('--pixel-positions', action='store_true',
                        help='Read pickled fits to add pixel positions to'
                        ' star.')

    args = parser.parse_args()

    start_time = time.time()

    output_dir = vcl.output_dir

    star_dirs = [output_dir / star_name for star_name in args.star_names]

    if star_dirs == []:
        # No stars given, fall back on included list:
        star_dirs = [output_dir / star_name for star_name in stars_to_use]

    if args.recreate_stars:
        p_umap(recreate_star, star_dirs)

    if args.transitions:
        p_umap(create_transition_model_corrected_arrays, star_dirs)

    if args.pairs:
        p_umap(create_pair_model_corrected_arrays, star_dirs)

    if args.pixel_positions:
        p_umap(add_pixel_data_to_star, star_dirs)

    duration = time.time() - start_time
    print(f'Finished in {duration:.2f} seconds.')
