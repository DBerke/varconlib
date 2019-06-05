#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:35:53 2018

@author: dberke
"""

# Script to write a given value into the header of specified FITS files.

import os.path
from glob import glob
from astropy.io import fits
import varconlib as vcl


def writeHDnames(FITSfile, HDnum):
    """Write HD numbers intos a new header card   
    """
    with fits.open(FITSfile, 'update') as hdul:
        hdul[0].header['HDNUM'] = HDnum

def writeRadVels(FITSfile, HDnum, radVelDict):
    """Write radial velocities from an external file into a new header card
    """
    with fits.open(FITSfile, 'update') as hdul:
        hdul[0].header['RADVEL'] = radVelDict[HDnum]

def write_vac_wlscale(FITSfile):
    """Write an array of calculated vacuum wavelengths into a new array
    """
    with fits.open(FITSfile, 'update') as hdul:
        air_wl = hdul[1].data['WAVE'][0] * 10 # Convert to Angstroms
        print(air_wl[:5])
        vac_wl = vcl.air2vacESO(air_wl) / 10 # Convert back to nm
        print(vac_wl[:5])

def writeHeaders():
    searchStr = os.path.join(baseDir, "HD*")
    dirs = glob(searchStr)
    for dir in dirs:
        HDnum = os.path.split(dir)[1]
        searchFiles = os.path.join(dir, "*.fits")
        files = glob(searchFiles)
        print('Working on {} (found {} files)...'.format(HDnum, len(files)))
        for FITSfile in files:
            #writeHDnames(FITSfile, HDnum)
            writeRadVels(FITSfile, HDnum, radVels)
            #write_vac_wlscale(FITSfile)

baseDir = "/Volumes/External Storage/HARPS/"

radVels = {}
radVelsFile = os.path.join(baseDir, 'radvel_medians_assigned.txt')
with open(radVelsFile, 'r') as f:
    for line in f.readlines():
        ln = line.split()
        radVels[ln[0]] = ln[1]


#writeHeaders()
import numpy as np
testFile = os.path.join(baseDir, 'test/test.fits')
with fits.open(testFile, 'update') as hdul:
    hdul.info()
    hdr = hdul[0].header
    orig_table = hdul[1].data
    orig_cols = orig_table.columns
    print(orig_cols)
    air_wl = hdul[1].data['WAVE'][0]
    vac_wl = vcl.air2vacESO(air_wl)
    print(air_wl[:10])
    print(vac_wl[:10])
#    print(hdr['OBJECT'])
#    hdr.set('HDNUM', 'HD68168')
#    hdr.set('RADVEL', '9.39')
    new_col = fits.Column(name='VACWAVE', format='313158D',
                                       array=vac_wl,
                                       unit='Angstrom')
    print('New column created.')
    print(new_col)
    newcols = [c for c in hdul[1].columns] + [new_col]
    print(newcols)
    hdu = fits.BinTableHDU.from_columns(newcols, nrows=313158)
#    print(hdr['HDNUM'])
#    print(hdr['RADVEL'])
#    print(hdul[1].data['VACWAVE'])
