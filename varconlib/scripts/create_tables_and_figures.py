#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:21:59 2020

@author: dberke

This script creates the necessary figures and tables for my two papers and
thesis.

"""

import argparse
from math import ceil
import os
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.transition_line import roman_numerals


parser = argparse.ArgumentParser(description='Create all the necessary figures'
                                 ' and tables for my two papers and thesis.')

parser.add_argument('--tables', action='store_true',
                    help='Save out tables in LaTeX format to text files.')
parser.add_argument('--figures', action='store_true',
                     help='Create and save plots and figures.')

parser.add_argument('-v', '--verbose', action='store_true',
                    help="Print out more information about the script's"
                    " output.")

args = parser.parse_args()

vprint = vcl.verbose_print(args.verbose)

output_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables')
if not output_dir.exists():
    os.mkdir(output_dir)

if args.tables:

    tqdm.write('Unpickling transitions list.')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions.')

    tqdm.write('Unpickling pairs list.')
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)
    vprint(f'Found {len(pairs_list)} pairs.')

    tables_dir = output_dir / 'tables'
    if not tables_dir.exists():
        os.mkdir(tables_dir)

    pairs_table_file = tables_dir / 'pairs_table.txt'

    transition_headers = [r'Wavelength (\AA, vacuum)',
                          r'Wavenumber (\si{\per\centi\meter})',
                          'Species',
                          r'Energy (\si{\per\centi\meter})',
                          'Orbital configuration',
                          'J',
                          r'Energy (\si{\per\centi\meter})',
                          'Orbital configuration',
                          'J',
                          'Orders to fit in']

    n = 3
    fraction = ceil(len(transitions_list) / n)

    slices = (slice(0, fraction), slice(fraction, 2 * fraction),
              slice(2 * fraction, None))
    for i, s in enumerate(slices):
        transitions_formatted_list = []
        transitions_table_file = tables_dir / f'transitions_table_{i}.txt'
        for transition in tqdm(transitions_list[s]):
            line = [f'{transition.wavelength.to(u.angstrom).value:.3f}',
                    f'{transition.wavenumber.value:.3f}',
                    ''.join((r'\ion{', transition.atomicSymbol,
                             '}{',
                             roman_numerals[transition.ionizationState].lower(),
                             '}')),
                    transition.lowerEnergy.value,
                    transition.lowerOrbital,
                    transition.lowerJ,
                    transition.higherEnergy.value,
                    transition.higherOrbital,
                    transition.higherJ,
                    transition.ordersToFitIn]

            transitions_formatted_list.append(line)

        transitions_output = tabulate(transitions_formatted_list,
                                      headers=transition_headers,
                                      tablefmt='latex_raw',
                                      floatfmt=('.3f', '.3f', '', '.3f', '', '',
                                                '.3f' '', '', ''))

        if transitions_table_file.exists():
            os.unlink(transitions_table_file)
        with open(transitions_table_file, 'w') as f:
            f.write(transitions_output)

