#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:46:58 2021

@author: dberke

Script to shift results by 150 ppb after entire modeling process is complete.

"""

import csv
from glob import glob
from pathlib import Path

from unyt import unyt_quantity

q_shifts_dict = {'4652.593Cr1': unyt_quantity(-2.05037415, 'm/s'),
                 '4653.460Cr1': unyt_quantity(-2.05075632, 'm/s'),
                 '4759.449Ti1': unyt_quantity(-1.88343804, 'm/s'),
                 '4760.600Ti1': unyt_quantity(-1.96952508, 'm/s'),
                 '4799.873Ti2': unyt_quantity(-0.38852075, 'm/s'),
                 '4800.072Fe1': unyt_quantity(3.58317352, 'm/s'),
                 '5138.510Ni1': unyt_quantity(-22.8762263, 'm/s'),
                 '5143.171Fe1': unyt_quantity(-13.13685122, 'm/s'),
                 '5187.346Ti2': unyt_quantity(-1.58622986, 'm/s'),
                 '5200.158Fe1': unyt_quantity(7.01535723, 'm/s'),
                 '6123.910Ca1': unyt_quantity(0.38003171, 'm/s'),
                 '6138.313Fe1': unyt_quantity(-14.96098892, 'm/s'),
                 '6139.390Fe1': unyt_quantity(-16.62010256, 'm/s'),
                 '6153.320Fe1': unyt_quantity(-1.88161347, 'm/s'),
                 '6155.928Na1': unyt_quantity(-0.0387555, 'm/s'),
                 '6162.452Na1': unyt_quantity(0.06096611, 'm/s'),
                 '6168.150Ca1': unyt_quantity(2.63506016, 'm/s'),
                 '6175.044Fe1': unyt_quantity(-2.38808884, 'm/s'),
                 '6192.900Ni1': unyt_quantity(-30.63364808, 'm/s'),
                 '6202.028Fe1': unyt_quantity(-16.0087567, 'm/s'),
                 '6242.372Fe1': unyt_quantity(7.29852295, 'm/s'),
                 '6244.834V1': unyt_quantity(3.36987773, 'm/s')}

group1 = frozenset(['HD45184', 'Vesta', 'HD20782', 'HD45289',
                    'HD220507', 'HD76151', 'HD171665', 'HD138573'])

base_dir = Path('/Users/dberke/Dropbox/Daniel-Michael/')
unshifted_linear_dir = base_dir / 'Results_unshifted_linear/'
unshifted_quadratic_dir = base_dir / 'Results_unshifted_quadratic/'
linear_dir = base_dir / 'Results_shifted_after_linear/'
quad_dir = base_dir / 'Results_shifted_after_quadratic/'

for in_folder, out_folder in zip((unshifted_linear_dir,
                                  unshifted_quadratic_dir),
                                 (linear_dir, quad_dir)):
    print(f'In folder: {in_folder}:')
    files = glob(str(in_folder / '[1-9]*.csv'))
    for file in files:
        file = Path(file)
#        print(file.name)
        t1, t2 = str(file.stem).split('_')[0:2]
#        print(t1, t2)
        shift_blue = q_shifts_dict[t1].value
        shift_red = q_shifts_dict[t2].value
#        print(shift_blue, shift_red)
        shift = shift_red - shift_blue
#        print(shift)
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            lines = [row for row in reader]

        for line in lines[1:]:
            star_name = line[0]
            if star_name in group1:
                line[2] = str(float(line[2]) + shift)
            else:
                pass

        new_file = out_folder / file.name
#        print(new_file)
        with open(new_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lines)
