#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:19:04 2021

@author: dberke

This script creates CSV files of stars in the SP1, 2, and 3 categories,
and of all stars, containing a bunch of useful information.
"""

import csv
from glob import glob
from pathlib import Path

from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.star import Star


def get_star_info(star):
    """
    Return a comma-delimited string of info about the star.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to describe.

    Returns
    -------
    str
        A comma-delimited string to write out as CSV.

    """

    return [star.name, str(star.temperature.value),
            str(star.metallicity), str(star.logg),
            str(star.absoluteMagnitude), str(star.apparentMagnitude),
            str(star.color), str(star.numObsPre), str(star.numObsPost)]


# Get star directories starting with H or V.
star_dirs = [Path(s) for s in glob(str(vcl.output_dir) + '/[HV]*')]

stars = [Star(s.stem, s) for s in tqdm(star_dirs)]

categories = {'SP1': {'temperature': 100 * u.K,
                      'metallicity': 0.1,
                      'logg': 0.2},
              'SP2': {'temperature': 200 * u.K,
                      'metallicity': 0.2,
                      'logg': 0.3},
              'SP3': {'temperature': 300 * u.K,
                      'metallicity': 0.3,
                      'logg': 0.4}}

solar = {'temperature': 5772 * u.K,
         'metallicity': 0.0,
         'logg': 4.44}

header = ['name', 'T_eff', '[Fe/H]', 'logg', 'absMag', 'appMag', '(b-y)',
          'NObsPre', 'NObsPost', 'NObsTot']

for level in ('SP1', 'SP2', 'SP3'):
    lines = []
    for star in tqdm(stars):
        passes = []
        for param in ('temperature', 'metallicity', 'logg'):
            if abs(getattr(star, param)
                   - solar[param]) <= categories[level][param]:
                passes.append(True)
            else:
                passes.append(False)
        if all(passes):
            star_info = get_star_info(star)
            # Add the pre and post to get total number of observations.
            star_info.append(str(int(star_info[-2]) + int(star_info[-1])))
            lines.append(star_info)

    csv_file = vcl.data_dir / f'Stellar_sample_{level}.csv'
    with open(csv_file, 'w', newline='') as f:
        datawriter = csv.writer(f)
        datawriter.writerow(header)
        for line in lines:
            datawriter.writerow(line)

lines = []
for star in tqdm(stars):
    star_info = get_star_info(star)
    # Add the pre and post to get total number of observations.
    star_info.append(str(int(star_info[-2]) + int(star_info[-1])))
    lines.append(star_info)

csv_file = vcl.data_dir / 'Stellar_sample_all.csv'
with open(csv_file, 'w', newline='') as f:
    datawriter = csv.writer(f)
    datawriter.writerow(header)
    for line in lines:
        datawriter.writerow(line)
