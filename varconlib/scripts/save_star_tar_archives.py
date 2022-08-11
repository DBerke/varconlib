#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:12:27 2022

@author: dberke

This script creates a TAR archive of the observations that went into a given
`varconlib.Star` object.
"""

import argparse
from datetime import timedelta, date
from glob import glob
import os
from pathlib import Path
import tarfile

from tqdm import tqdm

import varconlib as vcl


def main():

    if args.blaze:

        gather_blaze_files()

    names = args.star_names

    if names:

        for star_name in tqdm(names) if len(names) > 1 else names:

            gather_file_names(star_name)


def gather_blaze_files():

    blaze_path = vcl.harps_blaze_files_dir

    os.chdir(blaze_path)
    blaze_files = glob('data/reduced/*/*blaze_A.fits')
    print(f'Found {len(blaze_files)} blaze files total.')

    tarfilename = str(blaze_path) + '/HARPS_blaze_files.tar'
    with tarfile.open(tarfilename, 'w:gz') as tar:
        for filepath in tqdm(blaze_files):
            tar.add(filepath)


def gather_file_names(star_name):

    star_path = vcl.output_dir / star_name

    if not args.hdd:
        file_names = glob(str(star_path) + '/HARPS*')
    else:
        file_names = glob(f'/Volumes/External Storage/data_output/{star_name}'
                          '/HARPS*')

    os.chdir(vcl.harps_dir)
    num = 100

    if not args.hdd:
        tarfilename = vcl.harps_dir / f'{star_name}/{star_name}.tar'
        tqdm.write(str(tarfilename))
        make_tar_file(star_name, tarfilename, file_names)
    else:
        n = len(file_names)
        tqdm.write(f'n is {n}')
        a, b = 0, num
        t = tqdm(total=len(file_names))
        while a < n:
            tqdm.write(f'a, b = {a}, {b}')
            tarfilename = vcl.harps_dir / f'{star_name}/{star_name}_{a}.tar'
            tqdm.write(str(tarfilename))
            make_tar_file(star_name, tarfilename, file_names[a:b])
            a += num
            b += num
            t.update(num)
        t.close()


def make_tar_file(star_name, tarfilename, file_names):
    one_day = timedelta(days=1)

    spectra = []
    with tarfile.open(tarfilename, 'w:gz') as tar:
        for name in tqdm(file_names):
            name = Path(name).name + '.fits'
            strdate = name[6:16]
            objdate = date(int(strdate[:4]), int(strdate[5:7]),
                           int(strdate[8:]))
            filepath = f'{star_name}/data/reduced/{strdate}/{name}'

            spectra.append(filepath)

            try:
                tar.add(filepath)
            except FileNotFoundError:
                filepath = f'{star_name}/data/reduced/{objdate-one_day}/{name}'
                tar.add(filepath)
            tqdm.write(f'Added {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save a TAR archive of the'
                                     ' files that went into the creation of'
                                     'the given star(s).')

    parser.add_argument('star_names', action='store', type=str, nargs='*',
                        help='The names of stars (directories) containing the'
                        ' stars to be used.')
    parser.add_argument('--hdd', action='store_true',
                        help='For stars whose result files are on the external'
                        ' HDD.')
    parser.add_argument('--blaze', action='store_true',
                        help='Tar up the blaze files.')

    args = parser.parse_args()

    main()
