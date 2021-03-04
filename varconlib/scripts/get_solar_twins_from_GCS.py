#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:31:42 2021

@author: dberke

Create a table of solar twins from the Casgrande et al. 2011 updated values for
stars in the Geneva-Copenhagen Survey.

"""

import csv

from astroquery.simbad import Simbad
import numpy as np
from tqdm import tqdm

import varconlib as vcl

data_file = vcl.data_dir / 'Casagrande2011_GCS_updated_values.tsv'

data = np.genfromtxt(data_file, dtype=None, skip_footer=1, autostrip=True,
                     comments='#', delimiter=';', usecols=(1, 2, 3, 5, 6, 7),
                     encoding='UTF-8',
                     converters={2: lambda s: float(s or np.nan),
                                 3: lambda s: int(s or -1),
                                 5: lambda s: float(s or np.nan)})
print(data[0])
solar_twins = []
star_names = []

for line in data:
    if np.nan not in line:
        if (4.04 <= line[1] <= 4.84) and\
           (5472 <= line[2] <= 6072) and\
           (-0.3 <= line[3] <= 0.3):
            solar_twins.append(list(line))
            star_names.append(''.join(str(line[0]).split(' ')))

assert 'HD146233' in star_names, 'HD146233 not present!'

print(len(solar_twins))
print(solar_twins[0])
print(star_names[:3])
print(len(star_names))

sample_file = vcl.data_dir / 'Stellar_sample_all.csv'
observed_stars = set(np.loadtxt(sample_file, skiprows=1, delimiter=',',
                                usecols=0, dtype=str, encoding='UTF-8'))

customSimbad = Simbad()
customSimbad.add_votable_fields('flux(B)')
customSimbad.add_votable_fields('flux(V)')
customSimbad.add_votable_fields('flux(G)')
customSimbad.add_votable_fields('typed_id')
result_table = customSimbad.query_objects(star_names)
print(len(result_table))
# if len(result_table) != len(star_names):
#     raise RuntimeError('Lengths of lists do not match!')

magnitudes_b = {name.decode('utf-8'): mag for
                name, mag in zip(result_table['TYPED_ID'],
                                 result_table['FLUX_B'])}
magnitudes_v = {name.decode('utf-8'): mag for
                name, mag in zip(result_table['TYPED_ID'],
                                 result_table['FLUX_V'])}
magnitudes_g = {name.decode('utf-8'): mag for
                name, mag in zip(result_table['TYPED_ID'],
                                 result_table['FLUX_G'])}

matched_stars = 0
sp1 = 0
sp2 = 0
sp3 = 0
# print(observed_stars)
for line, star_name in zip(solar_twins, star_names):

    for d in (magnitudes_b, magnitudes_v, magnitudes_g):
        line.append(d[star_name])

    if (4.24 <= line[1] <= 4.64) and\
       (5672 <= line[2] <= 5872) and\
       (-0.1 <= line[3] <= 0.1):
        line.append('SP1')
        sp1 += 1
    elif (4.14 <= line[1] <= 4.74) and\
         (5572 <= line[2] <= 5972) and\
         (-0.2 <= line[3] <= 0.2):
        line.append('SP2')
        sp2 += 1
    else:
        line.append('SP3')
        sp3 += 1
    if star_name in observed_stars:
        line.append('True')
        # print(f'Found {star_name}')
        matched_stars += 1
    else:
        line.append('')

print(f'Found {matched_stars} observed stars.')
print(f'SP1 stars: {sp1}')
print(f'SP2 stars: {sp2}')
print(f'SP3 stars: {sp3}')

header = ('HDnum', 'logg', 'T_eff', '[Fe/H]', 'RA', 'DEC',
          'mag_B', 'mag_V', 'mag_G', 'sample', 'observed?')

outfile = vcl.data_dir / 'GCS_solar_twins_list.csv'
with open(outfile, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',')
    datawriter.writerow(header)
    for row in solar_twins:
        datawriter.writerow(row)
