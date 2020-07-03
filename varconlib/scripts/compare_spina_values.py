#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:43:05 2020

@author: dberke
"""

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import varconlib as vcl
from varconlib.star import Star
from varconlib.exceptions import HDF5FileNotFoundError

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

casali_dict_file = vcl.data_dir / 'Casali_star_dict.pkl'
with open(casali_dict_file, 'rb') as f:
    casali_dict = pickle.load(f)

hd_dict.update(casali_dict)

sp1_stars_file = vcl.data_dir / 'SP1_Sample.csv'

sp1_names = np.loadtxt(sp1_stars_file, delimiter=',', usecols=0, dtype=str)
sp1_names = set([''.join(name.split(' ')) for name in sp1_names])
# print(sp1_names)

data_file = vcl.data_dir / 'Casali20_Solutions_all_nooutliers_good.csv'

with open(data_file, 'r') as f:
    data = np.loadtxt(data_file, delimiter=',', skiprows=1,
                      usecols=(0, 1, 2, 3, 19, 20, 21), dtype=str)

names = [name for name in data[:, 0]]

sp1_teff = []
sp1_logg = []
sp1_feh = []

sp1_casagrande_teff = []
sp1_casagrande_logg = []
sp1_casagrande_feh = []

teff_list = []
logg_list = []
feh_list = []

teff_err_list = []
logg_err_list = []
feh_err_list = []

casagrande_teff = []
casagrande_logg = []
casagrande_feh = []

num_matched_stars = 0
num_sp1_stars = 0

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
                try:
                    star = Star(name, path, load_data=True)
                except HDF5FileNotFoundError:
                    continue
                tqdm.write(f'Matched {name} ({row[0]})')
                num_matched_stars += 1
                teff_list.append(float(row[1]))
                logg_list.append(float(row[2]))
                feh_list.append(float(row[3]))
                teff_err_list.append(float(row[4]))
                logg_err_list.append(float(row[5]))
                feh_err_list.append(float(row[6]))
                casagrande_teff.append(star.temperature.value)
                casagrande_logg.append(star.logg)
                casagrande_feh.append(star.metallicity)

                if name in sp1_names:
                    tqdm.write('Matched SP1 star')
                    sp1_teff.append(float(row[1]))
                    sp1_logg.append(float(row[2]))
                    sp1_feh.append(float(row[3]))
                    sp1_casagrande_teff.append(star.temperature.value)
                    sp1_casagrande_logg.append(star.logg)
                    sp1_casagrande_feh.append(star.metallicity)
                    num_sp1_stars += 1
                    if float(row[3]) > 0.2:
                        print(f'Anomalous star is {name}')


tqdm.write(f'Matched {num_matched_stars} stars.')
tqdm.write(f'Matched {num_sp1_stars} SP1 stars.')

temperatures = np.array(casagrande_teff) - np.array(teff_list)
loggs = np.array(casagrande_logg) - np.array(logg_list)
metallicities = np.array(casagrande_feh) - np.array(feh_list)

sp1_temperatures = np.array(sp1_casagrande_teff) - np.array(sp1_teff)
sp1_loggs = np.array(sp1_casagrande_logg) - np.array(sp1_logg)
sp1_metallicities = np.array(sp1_casagrande_feh) - np.array(sp1_feh)

point_color = 'CornflowerBlue'
sp1_point_color = 'DarkOrange'
point_size = 5

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

logg_lim_lower = 4.1
logg_lim_upper = 4.6
x2 = (logg_lim_lower, logg_lim_upper)
y2 = np.linspace(*x2, 2)

ax1.errorbar(casagrande_teff, teff_list, yerr=teff_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax1.errorbar(sp1_casagrande_teff, sp1_teff,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax1.plot(x1, y1,
         color='Black')
ax1.legend()

ax2 = fig.add_subplot(2, 3, 2)
ax2.set_xlabel('Casagrande+2011 log(g)')
ax2.set_ylabel('Spina+2020 log(g)')
ax2.set_xlim(left=logg_lim_lower, right=logg_lim_upper)
ax2.set_ylim(bottom=logg_lim_lower, top=logg_lim_upper)

ax2.errorbar(casagrande_logg, logg_list, yerr=logg_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax2.errorbar(sp1_casagrande_logg, sp1_logg,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax2.plot(x2, y2,
         color='Black')
ax2.legend()

feh_lim_lower = -0.34
feh_lim_upper = 0.34
x3 = (feh_lim_lower, feh_lim_upper)
y3 = np.linspace(*x3, 2)

ax3 = fig.add_subplot(2, 3, 3)
ax3.set_xlabel('Casagrande+2011 [Fe/H]')
ax3.set_ylabel('Spina+2020 [Fe/H]')
ax3.set_xlim(left=feh_lim_lower, right=feh_lim_upper)
ax3.set_ylim(bottom=feh_lim_lower, top=feh_lim_upper)

ax3.errorbar(casagrande_feh, feh_list, yerr=feh_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax3.errorbar(sp1_casagrande_feh, sp1_feh,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax3.plot(x3, y3,
         color='Black')
ax3.legend()

ax4 = fig.add_subplot(2, 3, 4)
ax4.set_xlabel('Casagrande+2011 Temperature (K)')
ax4.set_ylabel('(Casagrande+2011 – Spina+2020) Temperature (K)')
ax4.set_xlim(left=t_lim_lower, right=t_lim_upper)

ax4.errorbar(casagrande_teff, temperatures, yerr=teff_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax4.errorbar(sp1_casagrande_teff, sp1_temperatures,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax4.axhline(y=0, color='Black')
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_xlabel('Casagrande+2011 log(g)')
ax5.set_ylabel('(Casagrande+2011 – Spina+2020) log(g)')
ax5.set_xlim(left=logg_lim_lower, right=logg_lim_upper)

ax5.errorbar(casagrande_logg, loggs, yerr=logg_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax5.errorbar(sp1_casagrande_logg, sp1_loggs,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax5.axhline(y=0, color='Black')
ax5.legend()

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_xlabel('Casagrande+2011 [Fe/H]')
ax6.set_ylabel('(Casagrande+2011 – Spina+2020) [Fe/H]')

ax6.errorbar(casagrande_feh, metallicities, yerr=feh_err_list,
             linestyle='', marker='o', markersize=point_size,
             color=point_color, markeredgecolor='Black')
ax6.errorbar(sp1_casagrande_feh, sp1_metallicities,
             linestyle='', marker='o', markersize=point_size,
             color=sp1_point_color, markeredgecolor='Black',
             label='SP1 stars')
ax6.axhline(y=0, color='Black')
ax6.legend()


# plt.show(fig)
outfile = '/Users/dberke/Pictures/Spina_Casagrande_comparison.png'
fig.savefig(outfile)
