#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:02:48 2020

@author: dberke

A short script to save out the pair separations for a selection of stars.

"""

import argparse
import csv
import pickle

from tqdm import tqdm

import varconlib as vcl

header = '#pair_label, transition1, transition2, t_err1, t_err2,' +\
         ' Teff, [Fe/H], log(g), M_V'


def main():
    """Run the main routine for this script."""

    main_dir = vcl.output_dir

    star_list = []
    tqdm.write('Collecting stars...')

    for star_dir in tqdm(args.star_names):
        try:
            star = get_star(main_dir / star_dir)
        except JSONDecodeError:
            print(f'Error reading JSON from {star_dir}.')
            raise
        if star is None:
            continue
        else:
            star_list.append(star)

    # Import the list of pairs to use.
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    # The directory for storing these CSV files.
    output_dir = vcl.output_dir / 'pair_separation_files'

    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, order_num])
            tqdm.write(f'Collecting data for {pair_label}.')
            output_file = output_dir / f'{pair_label}_pair_separations.csv'
            info_list = []

            for star in tqdm(star_list):
                info_list.append(star.formatPairData(pair, order_num))

        with open(csv_file, 'w', newline='') as f:
            datawriter = csf.writer(f)
            datawriter.writerow(header)
            for row in info_list:
                datawriter.write(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParse(description='Save out results of the pair'
                                    ' separations for a selection of stars.')

    parser.add_argument('star_names', action='store', type=str, nargs='+',
                        help='The names of stars (directories) containing the'
                        ' stars to be used in the plot.')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
