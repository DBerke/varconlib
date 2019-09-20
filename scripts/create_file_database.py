#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:42:19 2018

@author: dberke

A simple script to scrape all the ADP files in
/Volumes/External Storage/HARPS/* and construct a CSV database of their
filenames, dates of observation, name of origin file, and object.
"""

import pandas as pd
from astropy.io import fits
from pathlib import Path
from tqdm import tqdm

databaseLines = []

baseDir = Path('/Volumes/External Storage/HARPS')

ADPfiles = baseDir.glob('*/ADP*.fits')
filelist = [file for file in ADPfiles]
print(len(filelist))

for file in tqdm(filelist):
    with fits.open(file) as hdul:
        header = hdul[0].header
        date_obs = header['DATE-OBS']
        arcfile = header['ARCFILE']
        origfile = header['ORIGFILE']
        try:
            obj = header['HDNUM']
        except KeyError:
            obj = header['OBJECT']
        databaseLines.append((obj, date_obs, arcfile, origfile))

# Sort the lines by object.
databaseLines.sort(key=lambda tup: tup[0])
databaseLines = pd.DataFrame(databaseLines)
print(databaseLines)

databaseFile = Path('data/ObjectDatabase.csv')
databaseLines.to_csv(databaseFile, index=False)

