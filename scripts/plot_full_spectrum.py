#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:54:59 2018

@author: dberke
"""

import matplotlib
import matplotlib.pyplot as plt
import varconlib as vcl
import numpy as np
from pathlib import Path
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)
plt.rcParams['text.usetex'] = True

outPicDir = Path("/Users/dberke/Pictures/Full_spectrum")
infile = Path('/Volumes/External Storage/HARPS/HD146233/'
              'ADP.2016-03-31T01:04:03.410.fits')

spectrum = vcl.readHARPSfile(infile)

w = np.array(spectrum['w'])  # In Angstroms here!
f = np.array(spectrum['f'])
e = spectrum['e']
vac_wl = np.array(vcl.air2vacESO(w))
wl = vac_wl / 10  # Convert to nm.
wl = vcl.lineshift(wl, -11.7)
print('File opened successfully.')

leftwl = 685.7
while leftwl < 691.1:
    rightwl = leftwl + 0.1
    leftpos = vcl.wavelength2index(wl, leftwl)
    rightpos = vcl.wavelength2index(wl, rightwl)
    maxflux = f[leftpos:rightpos].max()
    minflux = f[leftpos:rightpos].min()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Wavelength (nm)", fontsize=20)
    ax.set_ylabel("Intensity", fontsize=20)
    ax.grid(which='major', axis='both')
    ax.set_xlim(left=leftwl, right=rightwl)
    ax.set_ylim(bottom=minflux*0.95, top=maxflux*1.05)
    outfile = outPicDir / 'Spectrum_{:.1f}_{:.1f}nm.png'.format(leftwl,
                                                                rightwl)

    ax.errorbar(wl, f, yerr=e, marker='.', markersize=4,
                linestyle='-', linewidth=1, alpha=1)

    fig.savefig(str(outfile))
    plt.close(fig)
    print('Saved file for region {:.1f}, {:.1f}'.format(leftwl, rightwl))
    leftwl = rightwl
