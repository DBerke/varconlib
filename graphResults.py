#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:32:43 2018

@author: dberke
"""

# Script to plot cached results from lineFind.py

import varconlib as vcl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import matplotlib.dates as dates
import datetime as dt
from pathlib import Path
from tqdm import tqdm
from time import sleep


def plot_as_func_of_date(data, linepair, filepath, folded=False):
    """Plot separations as a function of date.

    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('HD146233')
    ax.set_ylabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                  format(linepair[0], linepair[1]), fontsize=18)

    if folded:
        xlabel = 'Date of observation (folded by year)'
    else:
        xlabel = 'Date of observation'
    ax.set_xlabel(xlabel, fontsize=18)
#        ax.set_ylim(bottom=-200, top=200)

    # Select the data to plot
    filtdata = data[(data['line1_nom_wl'] == float(linepair[0])) &
                    (data['line2_nom_wl'] == float(linepair[1]))]
    gaussvel = filtdata['vel_diff_gauss']
    gaussvel -= np.median(gaussvel)
    gausserr = filtdata['diff_err_gauss']

    # Find the RMS of the scatter
    gauss_rms = np.sqrt(np.mean(np.square(gaussvel)))
    # Find the mean error
    mean_err = np.mean(gausserr)

    gaussdatetimes = filtdata['date']
    gausspy = []
    # Convert to Python datetimes so matplotlib's errorbar will work --
    # it doesn't like pandas' native Timestamp object apparently
    for date in gaussdatetimes:
        gausspy.append(date.to_pydatetime())

    if folded:
        gaussdates = [date.replace(year=2000) for date in gausspy]
        format_str = '%b'
        ax.set_xlim(left=dt.date(year=2000, month=1, day=1),
                    right=dt.date(year=2000, month=12, day=31))
    else:
        format_str = '%Y%m%d'
        gaussdates = gausspy

    ax.xaxis.set_major_locator(dates.AutoDateLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter(format_str))

    # Plot the RMS
    ax.axhspan(-1*gauss_rms, gauss_rms, color='gray', alpha=0.3)
    # Plot the median-subtracted velocity differences
    ax.errorbar(gaussdates, gaussvel, yerr=gausserr,
                markerfacecolor='Black', markeredgecolor='Black',
                linestyle='', marker='o',
                markersize=5, elinewidth=2, ecolor='Green',
                capsize=2, capthick=2)

    fig.subplots_adjust(bottom=0.16, wspace=0.0, hspace=0.0)
    fig.autofmt_xdate(bottom=0.16, rotation=30, ha='right')

    ax.annotate('RMS: {:.4f}\nmean error: {:.4f}'.format(gauss_rms, mean_err),
                xy=(0.3, 0.08), xycoords='axes fraction',
                size=14, ha='right', va='center',
                bbox=dict(boxstyle='round', fc='w', alpha=0.2))
#    ax.legend(framealpha=0.4, fontsize=18, numpoints=0)
#    plt.show()

    if folded:
        outfile = filepath / 'graphs' / 'Linepair_{}_{}_folded.png'.format(
                                                         linepair[0],
                                                         linepair[1])
    else:
        outfile = filepath / 'graphs' / 'Linepair_{}_{}.png'.format(
                                                            linepair[0],
                                                            linepair[1])
    fig.savefig(str(outfile), format='png')
    plt.close(fig)


#########################

pairlist = [('443.9589', '444.1128'), ('450.0151', '450.3467'),
            ('459.9405', '460.3290'), ('460.5846', '460.6877'),
            ('465.8889', '466.2840'), ('473.3122', '473.3780'),
            ('475.9448', '476.0601'), ('480.0073', '480.0747'),
            ('484.0896', '484.4496'), ('488.6794', '488.7696'),
            ('490.9102', '491.0754'), ('497.1304', '497.4489'),
            ('500.5115', '501.1420'), ('506.8562', '507.4086'),
            ('507.3492', '507.4086'), ('513.2898', '513.8813'),
            ('514.8912', '515.3619'), ('524.8510', '525.1670'),
            ('537.5203', '538.1069'), ('554.4686', '554.5475'),
            ('563.5510', '564.3000'), ('571.3716', '571.9418'),
            ('579.4679', '579.9464'), ('579.5521', '579.9779'),
            ('580.8335', '581.0828'), ('593.1823', '593.6299'),
            ('595.4367', '595.8344'), ('600.4673', '601.0220'),
            ('616.3002', '616.8146'), ('617.2213', '617.5042'),
            ('617.7065', '617.8498'), ('623.9045', '624.6193'),
            ('625.9833', '626.0427'), ('625.9833', '626.2831'),
            ('647.0980', '647.7413')]

baseDir = Path('/Volumes/External Storage/HARPS')
obj = 'HD146233'

filepath = Path('/Users/dberke/HD146233')  # 18 Sco, G2 (7 files)
filepath = Path('/Volumes/External Storage/HARPS/HD146233')
file = filepath / '{}.csv'.format(filepath.stem)
#file = baseDir / obj / '{}.csv'.format(obj)
print('Data file = {}'.format(file))
sleep(0.5)

data = pd.read_csv(file, header=0, parse_dates=[1], engine='c')

with tqdm(total=2*len(pairlist), unit='Plot') as pbar:
    for linepair in pairlist:
        plot_as_func_of_date(data, linepair, filepath, folded=False)
        pbar.update(1)
        plot_as_func_of_date(data, linepair, filepath, folded=True)
        pbar.update(1)
