#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:13:21 2018

@author: dberke
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


def plot_RMS_histograms(binsize=5):
    """Plot RMS scatter in line pair separations split by blended lines
    """

    data_dir = Path('/Users/dberke/')
    HD146233 = pd.read_csv(data_dir / 'HD146233/HD146233blended.txt', header=0)
    HD45184 = pd.read_csv(data_dir / 'HD45184/HD45184blended.txt', header=0)
    HD183658 = pd.read_csv(data_dir / 'HD183658/HD183658blended.txt', header=0)

    fig = plt.figure(figsize=(9, 9), tight_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    axes = (ax1, ax2, ax3)
    titles = ('0 blended lines', '1 blended line', '2 blended lines')
    colors = ('Chocolate', 'DodgerBlue', 'ForestGreen')
    stars = (HD146233, HD45184, HD183658)
    for i, axis, title in zip(range(0, 3, 1), axes, titles):
        hd146233 = HD146233[HD146233['blendedlines'] == i]['RMS']
        hd45184 = HD45184[HD45184['blendedlines'] == i]['RMS']
        hd183658 = HD183658[HD183658['blendedlines'] == i]['RMS']
        axis.hist((hd146233, hd45184, hd183658),
                  color=colors,
                  edgecolor='Black',
                  bins=range(15, 125, binsize), alpha=1,
                  histtype='barstacked',
                  label=('HD146233', 'HD45184', 'HD183658'))
        axis.set_title(title)
        axis.set_xlabel(r'RMS scatter in line pair $\delta v$ [m/s]', fontsize=16)
        axis.set_ylim(bottom=0, top=13)
        axis.set_xlim(left=10, right=135)
        axis.legend()

    outfile = Path('/Users/dberke/Pictures/blendtest_{}mps.png'.
                   format(binsize))
    #plt.show()
    fig.savefig(str(outfile))
    plt.close(fig)


plot_RMS_histograms(binsize=10)
