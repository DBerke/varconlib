#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:43:05 2020

@author: dberke
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from varconlib.star import Star

hd_dict = {'ALPCENA': 'HD128620',
           'BETA-HYI': 'HD2151',
           'HIP73241': 'HD68168',
           'HIP62039': 'HD14802',
           'HIP29432': 'HD88072',
           'HIP49756': 'HD13357',
           'HIP42333': 'HD202628',
           'HIP89650': 'HD167060',
           'HIP101905': 'HD44665',
           'HIP8507': 'HD39881',
           'HIP96160': 'HD142331',
           'HIP28066': 'HD11195',
           'HIP10175': 'HD172051',
           'HIP74432': 'HD59711',
           'HR209': 'HD4391',
           'HIP18844': 'HD36152',
           'HR1010': 'HD124523',
           'HIP68468': 'HD114174',
           'HIP105184': 'HD75302',
           'HR695': 'HD25874',
           'HIP67620': 'HD219057',
           'HIP87769': 'HD3821',
           'HIP3203': 'HD63487',
           'HIP43297': 'HD110537',
           'HIP30037': 'HD12264',
           'HIP25670': 'HD183579',
           'HIP77883': 'HD6204',
           'HIP4909': 'HD120690',
           'HIP36512': 'HD73350',
           'HIP69645': 'HD131923',
           'HIP114615': 'HD129814',
           'HIP118115': 'HD138573',
           'HIP40133': 'HD42618',
           'HIP30158': 'HD163441',
           'HIP10303': 'HD223238',
           'HIP76114': 'HD122194',
           'HIP72043': 'HD45021',
           'HIP64713': 'HD224383',
           'HIP114328': 'HD135101',
           'HIP79578': 'HD20807',
           'HIP102040': 'HD30495',
           'HIP117367': 'HD145825',
           'HR6998': 'HD197076',
           'HIP64150': 'HD218544',
           'HIP9349': 'HD9986',
           'HIP38072': 'HD196390',
           'HIP22263': 'HD115169',
           'HIP7585': 'HD13612B',
           'HIP64673': 'HD115031',
           'HIP64150': 'HD114174',
           'HIP81746': 'HD150248',
           'HIP10303': 'HD13612B',
           'HIP18844': 'HD25874',
           'HIP76114': 'HD138573',
           'HIP25670': 'HD36152',
           'HIP64713': 'HD115169',
           'HIP10175': 'HD13357',
           'HIP68468': 'HD122194',
           'HIP79578': 'HD145825',
           'HIP96160': 'HD183579',
           'HIP118115': 'HD224383',
           'HIP117367': 'HD223238',
           'HIP102040': 'HD197076',
           'HIP30037': 'HD45021',
           'HIP30158': 'HD44665',
           'HIP87769': 'HD163441',
           'HIP8507': 'HD11195',
           'HIP49756': 'HD88072',
           'HIP7585': 'HD9986',
           'HIP114328': 'HD218544',
           'HIP89650': 'HD167060',
           'HIP72043': 'HD129814',
           'HIP64673': 'HD115031',
           'HIP65708': 'HD117126',
           'HIP69645': 'HD124523',
           'HIP114615': 'HD219057',
           'HIP43297': 'HD75302',
           'HIP77883': 'HD142331',
           'HIP83276': 'HD153631',
           'HIP6407': 'HD8291',
           'HIP19807': 'HD26956',
           'SW1101-2351': 'HD41291',
           'HIP28641': '41291',
           'KELT-10': 'HD120690',
           'HIP67620': 'HD200565',
           'CL01510': 'HD11131',
           'HIP103983': 'HD176983'}

data_file = Path('/Users/dberke/code/varconlib/data/Casali20_params.csv')

with open(data_file, 'r') as f:
    data = np.loadtxt(data_file, delimiter=',', skiprows=1,
                      usecols=(0, 1, 2, 3, 5, 6, 7), dtype=str)

names = [name for name in data[:, 0]]

teff_list = []
logg_list = []
feh_list = []

casagrande_teff = []
casagrande_logg = []
casagrande_feh = []

num_matched_stars = 0

tqdm.write('Matching stars...')
for name in tqdm(names):
    for row in data:
        if row[0] == name:
            if 'HD' not in name:
                try:
                    name = hd_dict[name]
                except KeyError:
                    continue
            path = Path(f'/Users/dberke/data_output/{name}')
            # tqdm.write(str(path))
            if path.exists():
                num_matched_stars += 1
                star = Star(name, path, load_data=True)
                teff_list.append(float(row[1]))
                logg_list.append(float(row[2]))
                feh_list.append(float(row[3]))
                casagrande_teff.append(star.temperature.value)
                casagrande_logg.append(star.logg)
                casagrande_feh.append(star.metallicity)
            else:
                tqdm.write(f"{name} wasn't matched.")

tqdm.write(f'Matched {num_matched_stars} stars.')

temperatures = np.array(casagrande_teff) - np.array(teff_list)
loggs = np.array(casagrande_logg) - np.array(logg_list)
metallicities = np.array(casagrande_feh) - np.array(feh_list)

t_lim_lower = 5500
t_lim_upper = 6100
x1 = (t_lim_lower, t_lim_upper)
y1 = np.linspace(*x1, 2)


fig = plt.figure(figsize=(14, 8), tight_layout=True)
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_xlabel('Casagrande+2011 Temperature (K)')
ax1.set_ylabel('Spina+2020 Temperature (K)')
ax1.set_xlim(left=t_lim_lower, right=t_lim_upper)
ax1.set_ylim(bottom=t_lim_lower, top=t_lim_upper)

logg_lim_lower = 4
logg_lim_upper = 4.7
x2 = (logg_lim_lower, logg_lim_upper)
y2 = np.linspace(*x2, 2)

ax2 = fig.add_subplot(2, 3, 2)
ax2.set_xlabel('Casagrande+2011 log(g)')
ax2.set_ylabel('Spina+2020 log(g)')
ax2.set_xlim(left=logg_lim_lower, right=logg_lim_upper)
ax2.set_ylim(bottom=logg_lim_lower, top=logg_lim_upper)

feh_lim_lower = -0.34
feh_lim_upper = 0.34
x3 = (feh_lim_lower, feh_lim_upper)
y3 = np.linspace(*x3, 2)

ax3 = fig.add_subplot(2, 3, 3)
ax3.set_xlabel('Casagrande+2011 [Fe/H]')
ax3.set_ylabel('Spina+2020 [Fe/H]')
ax3.set_xlim(left=feh_lim_lower, right=feh_lim_upper)
ax3.set_ylim(bottom=feh_lim_lower, top=feh_lim_upper)

ax1.plot(casagrande_teff, teff_list,
         linestyle='', marker='o')
ax1.plot(x1, y1,
         color='Black')
ax2.plot(casagrande_logg, logg_list,
         linestyle='', marker='o')
ax2.plot(x2, y2,
         color='Black')
ax3.plot(casagrande_feh, feh_list,
         linestyle='', marker='o')
ax3.plot(x3, y3,
         color='Black')

ax4 = fig.add_subplot(2, 3, 4)
ax4.set_xlabel('Casagrande+2011 Temperature (K)')
ax4.set_ylabel('(Casagrande+2011 – Spina+2020) Temperature (K)')
ax4.set_xlim(left=t_lim_lower, right=t_lim_upper)

ax4.plot(casagrande_teff, temperatures,
         linestyle='', marker='o')
ax4.axhline(y=0, color='Black')

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_xlabel('Casagrande+2011 log(g)')
ax5.set_ylabel('(Casagrande+2011 – Spina+2020) log(g)')
ax5.set_xlim(left=logg_lim_lower, right=logg_lim_upper)

ax5.plot(casagrande_logg, loggs,
         linestyle='', marker='o')
ax5.axhline(y=0, color='Black')

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_xlabel('Casagrande+2011 [Fe/H]')
ax6.set_ylabel('(Casagrande+2011 – Spina+2020) [Fe/H]')

ax6.plot(casagrande_feh, metallicities,
         linestyle='', marker='o')


plt.show(fig)
