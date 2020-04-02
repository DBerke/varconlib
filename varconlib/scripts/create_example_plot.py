#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:49:00 2019

@author: dberke

A script to create an example plot showing how some transitions can be very
stable between stars of varying parameters.

"""

import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pprint import pprint
from tqdm import tqdm
import unyt as u

from varconlib.star import Star
from varconlib.miscellaneous import velocity2wavelength as vel2wave

stars_to_use = ('Vesta',
#                'HD2071',
                'HD19467',
                'HD20782',
#                'HD45184', non-solar twin
                'HD45289',
#                'HD59468', non-solar twin
#                'HD68168',
                'HD78429',
#                'HD78660',
                'HD96423',
                'HD97037',
#                'HD126525',
#                'HD138573',
                'HD140538',
#                'HD146233',
#                'HD157347',
#                'HD183658',
                'HD208704',)
#                'HD220507',)
#                'HD222582',)

stars_to_use = tuple(reversed(stars_to_use))

transitions_to_use = ('4490.998Fe1_25',
                      '4742.856Fe1_32',
                      '4971.304Fe1_38',
                      '5577.637Fe1_50',
                      '6175.044Fe1_61')

pairs_to_use = (#'4500.398Fe1_4503.480Mn1_25',
#                '4576.000Fe1_4577.610Fe2_27',
#                '4589.484Cr2_4599.405Fe1_28',
#                '4637.144Fe1_4638.802Fe1_29',
#                '5138.510Ni1_5143.171Fe1_42',
#                '5187.346Ti2_5200.158Fe1_43',
#                '5571.164Fe1_5577.637Fe1_50',
#                '6123.910Ca1_6139.390Fe1_60',
#                '6138.313Fe1_6139.390Fe1_60',)
                '6178.520Ni1_6192.900Ni1_61',)

# This line prevents the wavelength formatting from being in the form of
# scientific notation.
matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['axes.linewidth'] = 2

fig = plt.figure(figsize=(4, 7), tight_layout=True)
width_ratios = [2]
width_ratios.extend([5] * len(pairs_to_use))
gs = GridSpec(nrows=len(stars_to_use),
              ncols=len(pairs_to_use)+1,
              wspace=0,
              width_ratios=width_ratios,
              figure=fig)

ax_dict = {t: fig.add_subplot(gs[:, num+1]) for num, t in
           enumerate(pairs_to_use)}

for axis in ax_dict.values():
    axis.set_yticklabels('')
#    axis.set_xlim(left=-66 * u.m/u.s, right=65 * u.m/u.s)
    axis.set_xlim(left=-20 * u.m/u.s, right=20 * u.m/u.s)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    axis.set_ylim(bottom=-0.5, top=len(stars_to_use)-0.5)
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axis.axvline(x=0, linestyle='--', color='Black', linewidth=3,
                 zorder=100)
    axis.xaxis.set_tick_params(which='major', width=2, length=6)
    axis.xaxis.set_tick_params(which='minor', width=1.5, length=4)

ax_dict[pairs_to_use[0]].set_yticks(range(0, len(stars_to_use)))
labels = []

stars = []
pair_means = {}
pair_stds = {}
pair_mean_errors = {}
for star_label in stars_to_use:
    tqdm.write(f'Analysing {star_label}')
    star = Star(star_label, f'/Users/dberke/data_output/{star_label}')
    stars.append(star)

    pre_slice = slice(None, star.fiberSplitIndex)

    num_obs = star.getNumObs(pre_slice)

    if star.name != "Vesta":
        labels.append(star.name)# + f'\n #obs: {num_obs}')
    else:
        labels.append('Sun\n(Vesta)')

    for label in pairs_to_use:
        col_index = star.p_index(label)
        means = star.pairSeparationsArray[pre_slice, col_index]
        errors = star.pairSepErrorsArray[pre_slice, col_index]

        weighted_mean = np.average(means, weights=1/errors**2).to(u.m/u.s)
#        error = np.sqrt(np.sum(np.square(errors)))
        if len(means) > 1:
            stddev = np.std(means)
            error = np.std(means) / np.sqrt(num_obs)
        else:
            stddev = errors[0]
            error = errors[0]
        try:
            pair_means[label].append(weighted_mean)
            pair_stds[label].append(stddev)
            pair_mean_errors[label].append(error)
        except KeyError:
            pair_means[label] = [weighted_mean]
            pair_stds[label] = [stddev]
            pair_mean_errors[label] = [error]


for label in pairs_to_use:
    values = u.unyt_array(pair_means[label])
    mean = np.mean(values)
    xvalues = (values - mean) * u.m/u.s
    stddevs = u.unyt_array(pair_stds[label])
    errors = u.unyt_array(pair_mean_errors[label])
    yvalues = list(range(0, len(stars_to_use)))

    weighted_mean, weight_sum = np.average(values,
                                           weights=errors**-2,
                                           returned=True)
    error_on_weighted_mean = (1 / np.sqrt(weight_sum))
#    tqdm.write(f'{error_on_weighted_mean:.2f}')
#    ax_dict[label].errorbar(x=xvalues, y=yvalues,
#                            yerr=None, xerr=stddevs,
#                            marker='', capsize=4, linestyle='',
#                            color='DodgerBlue', ecolor='DodgerBlue',
#                            elinewidth=2, capthick=1.5)
    ax_dict[label].errorbar(x=xvalues, y=yvalues,
                            yerr=None, xerr=errors, markersize=12,
                            marker='o', capsize=7, linestyle='',
                            color='DodgerBlue', ecolor='DodgerBlue',
                            elinewidth=6, capthick=2.4,
                            zorder=1000)

    # Add the Sun's point in in a different color.
    ax_dict[label].errorbar(x=xvalues[-1], y=yvalues[-1],
                            yerr=None, xerr=errors[-1], markersize=12,
                            marker='o', capsize=7, linestyle='',
                            color='ForestGreen', ecolor='ForestGreen',
                            elinewidth=6, capthick=2.4,
                            zorder=1001)




ax_dict[pairs_to_use[0]].set_yticklabels(labels,
                                         rotation='horizontal',
                                         fontsize=12,
                                         horizontalalignment='right',
                                         weight='bold')
for key, axis in ax_dict.items():
    string = key.split('_')
    title = string[0] + '\n' + string[1]
#    axis.set_title(title, weight='bold')
    axis.set_xlabel('Velocity separation\noffset (m/s)', weight='bold',
                    fontsize=14)
#    axis.fill_between((-75, 75), 5.5, 8.5, color='Yellow', alpha=0.2)
    for tick in axis.get_xticklabels():
#        tick.set_rotation(27)
#        tick.set_ha('right')
        tick.set_fontsize(13)
        tick.set_weight('bold')

    axis.get_yticklabels()[-1].set_color('ForestGreen')

plt.show(fig)
