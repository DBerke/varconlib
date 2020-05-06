#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:35:53 2018

@author: dberke
"""

# Script to write values into the headers of FITS files.

import argparse
from pathlib import Path

from astropy.io import fits

desc = "Write the given value into the header of the given FITS file."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('FITSfile', action='store', type=str,
                    help='The FITS file to modify the header of.')
parser.add_argument('keyword', action='store', type=str,
                    help='The header card keyword to store the value in.')
parser.add_argument('value', action='store',
                    help='The value to store in the card.')

parser.add_argument('-c', '--comment', type=str, action='store',
                    help='Optional comment for when adding a new card.')
parser.add_argument('-x', '--extension', type=int, action='store',
                    default=0,
                    help='The extension number, if more than one (default: 0)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print out additional information about the process.')

args = parser.parse_args()

filepath = Path(args.FITSfile)
if not filepath.exists():
    raise FileNotFoundError("The given file path doesn't exist:"
                            f"{filepath}")

try:
    value = float(args.value)
except ValueError:
    value = str(args.value)

hdulist = fits.open(filepath, mode='update')
hdulist[args.extension].header.set(args.keyword, value=value,
                                   comment=args.comment)
hdulist.close(output_verify='warn')
if args.verbose:
    print(f'Wrote value "{value}" as a {type(value)} to "{args.keyword}".')
