#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:58:04 2021

@author: dberke

Recreate all stars
"""

import argparse
from json.decoder import JSONDecodeError
from multiprocessing import Pool, RLock
from pathlib import Path
import time

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
    star_dir : str or `pathlib.Path`
        The directory in which to find the star's files.

    Returns
    -------
    None.

    """

    if isinstance(star_dir, str):
        star_dir = Path(star_dir)
    else:
        pass
    tqdm.write(f'Creating {star_dir.stem}')
    try:
        Star(star_dir.stem, star_dir, load_data=False)
    except PickleFilesNotFoundError:
        star_dir = Path('/Volumes/External Storage/data_output') / star_dir.stem
        tqdm.write('Using external storage files.')
        Star(star_dir.stem, star_dir, load_data=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automatically recreate all'
                                     ' stars whose names are given.')
    parser.add_argument('star_names', action='store', type=str, nargs='*',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')

    args = parser.parse_args()

    start_time = time.time()

    output_dir = vcl.output_dir

    star_dirs = [output_dir / star_name for star_name in args.star_names]

    if star_dirs == []:
        # No stars given, fall back on included list:
        star_dirs = [output_dir / star_name for star_name in stars_to_use]

    # tqdm.set_lock(RLock)
    # p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

    # with Pool() as pool:
    #     pool.map(recreate_star, star_dirs)
    p_umap(recreate_star, star_dirs)

    duration = time.time() - start_time
    print(f'Finished recreating stars in {duration:.2f} seconds.')
