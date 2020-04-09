#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:27:15 2020

@author: dberke


A script for selecting stars for the SP1 sample of solar twins.
"""

import argparse
from glob import glob
from pathlib import Path

from tqdm import tqdm

import varconlib as vcl
from varconlib.exceptions import HDF5FileNotFoundError
from varconlib.star import Star


def get_star(star_path, verbose=False):
    """Return a varconlib.star.Star object based on its name.

    Parameters
    ----------
    star_path : str
        A string representing the name of the directory where the HDF5 file
        containing a `star.Star`'s data can be found.

    Optional
    --------
    verbose : bool, Default: False
        If *True*, write out additional information.

    Returns
    -------
    `star.Star`
        A Star object from the directory. Note that this will only use already-
        existing stars, it will not create ones which do not already exist from
        their observations.

    """

    assert star_path.exists(), FileNotFoundError('Star directory'
                                                 f' {star_path}'
                                                 ' not found.')
    try:
        return Star(star_path.stem, star_path, load_data=True)
    except IndexError:
        if verbose:
            tqdm.write(f'Excluded {star_path.stem}.')
        pass
    except HDF5FileNotFoundError:
        if verbose:
            tqdm.write(f'No HDF5 file for {star_path.stem}.')
        pass
    except AttributeError:
        if verbose:
            tqdm.write(f'Affected star is {star_path.stem}.')
        raise


def main():
    """Run the main script routine."""

    # Define vprint to only print when the verbose flag is given.
    vprint = vcl.verbose_print(args.verbose)

    output_dir = Path(vcl.config['PATHS']['output_dir'])
    search_str = f'{str(output_dir)}/HD*'
    star_dirs = glob(search_str)

    star_list = []
    tqdm.write('Collecting stars...')
    for star_dir in tqdm(star_dirs):
        star = get_star(Path(star_dir))
        if star is None:
            pass
        else:
            if args.casagrande2011:
                vprint('Applying values from Casagrande et al. 2011.')
                star.getStellarParameters('Casagrande2011')
            elif args.nordstrom2004:
                vprint('Applying values from Nordstrom et al. 2004.')
                star.getStellarParameters('Nordstrom2004')
            star_list.append(star)
            vprint(f'Added {star.name}.')

            # Check if the star fits the SP1 sample criteria:
            if (5680 <= star.temperature <= 5880) and\
               (4.23 <= star.logG <= 4.65) and\
               (-0.11 <= star.metallicity <= 0.11):
                # print(type(star.name))
                # print(type(star.temperature.value))
                # print(type(star.metallicity))
                # print(type(star.logG))
                tqdm.write(','.join((star.name,
                                     str(int(star.temperature.value)),
                                     str(star.metallicity), str(star.logG),
                                     str(star.getNumObs()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collate date from stars into'
                                     ' a standard form saved to disk.')
    # parser.add_argument('main_dir', action='store', type=str, nargs=1,
    #                     help='The main directory within which to find'
    #                     ' additional star directories.')
    # parser.add_argument('star_names', action='store', type=str, nargs='+',
    #                     help='The names of stars (directories) containing the'
    #                     ' stars to be used in the plot.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more output about what's happening.")

    paper = parser.add_mutually_exclusive_group()
    paper.add_argument('--casagrande2011', action='store_true',
                       help='Use values from Casagrande et al. 2011.')
    paper.add_argument('--nordstrom2004', action='store_true',
                       help='Use values from Nordstrom et al. 2004.')

    args = parser.parse_args()

    main()
