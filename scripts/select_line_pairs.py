#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:57:52 2018

@author: dberke
"""

# Code to iterate through a given line list to identify pairs given
# various constraints.

import argparse
import configparser
import pickle
import datetime
from math import sqrt
from time import sleep
from pathlib import Path
import numpy as np
import numpy.ma as ma
from astroquery.nist import Nist
from tqdm import tqdm
import varconlib as vcl
import unyt as u
from transition_line import Transition
from transition_pair import TransitionPair

elements = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
            "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
            "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
            "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
            "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
            "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
            "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
            "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
            "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
            "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
            "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
            "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
            "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 70,
            "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
            "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
            "U": 92}


def wn2eV(percm):
    """Return the energy given in cm^-1 in eV.

    Invert cm^-1, divide by 100, divide into c, multiply by h, divide by e.

    """
    if percm == 0.:
        result = 0.
    else:
        percm = percm * u.cm**-1
        E = percm.to(u.m**-1) * u.hmks * u.c
        result = E.to(u.eV)
    return result


def eV2wn(eV):
    """Return energy given in eV in cm^-1.

    """

    if eV == 0.:
        result = 0.
    else:
        eV = eV * u.eV
        nu = eV.to(u.J) / (u.hmks * u.c)
        result = nu.to(u.cm**-1)
    return result


def parse_spectral_mask_file(file):
    """Parses a spectral mask file from maskSpectralRegions.py

    Parameters
    ----------
    file : str or Path object
        A path to a text file to parse. Normally this would come from
        maskSpectralRegions.py, but the general format is a series of
        comma-separated floats, two per line, that each define a 'bad'
        region of the spectrum.

    Returns
    -------
    list
        A list of tuples parsed from the file, each one delimiting the
        boundaries of a 'bad' spectral region.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    masked_regions = []
    for line in lines:
        if '#' in line:
            continue
        start, end = line.rstrip('\n').split(',')
        masked_regions.append((float(start), float(end)))

    return masked_regions


def line_is_masked(line, mask):
    """
    Checks if a given spectral line is in a masked region of the spectrum.

    Parameters
    ----------
    line : float
        The wavelength of the line to check, in nanometers.
    mask : list
        A list of tuples, where each tuple is a two-tuple of floats denoting
        the start and end of a 'bad' spectral range to be avoided.

    Returns
    -------
    bool
        *True* if the line is within one of the masked regions, *False*
        otherwise.

    """

    for region in mask:
        if region[0] < line < region[1]:
            return True
    return False


def harmonize_lists(BRASS_transitions, Kurucz_transitions, spectral_mask,
                    wl_tolerance=1000, energy_tolerance=10000):
    """Iterate over the BRASS list of transtitions and harmonize it with the
    Kurucz list.

    Parameters
    ----------
    BRASS_transitions : list of Transitions
        A list of Transition objects representing transitions from the BRASS
        list. These will contain the vacuum wavelength of the transition, the
        element and ionization state it came from, the lower energy of the
        orbital, and the normalized depth of the absorption line it produces.
    Kurucz_transitions : list of Transitions
        A list of Transition objects representing transitions from the Kurucz
        list. These will contain the vacuum wavelength of the transition, the
        element and ionization state it came from, the lower and higher
        energies, momentums (J), and orbital configurations of the transition,
        and the log of the isotope abundance.
    spectral_mask : list of tuples
        A list of tuples delimiting 'bad' regions of the spectrum that should
        be avoided.
    wl_tolerance : int
        The tolerance in m/s (though the units are added internally) to search
        within around the BRASS list's given wavelength.
    energy_tolerance : int
        The tolerance in m/s (though the units are added internally) to search
        within around the BRASS list's lower energy value.

    """

    # Variables to keep track of how many lines fall into various categories.
    n_masked_lines = 0
    n_multi_isotopes = 0

    wl_tolerance = wl_tolerance * u.m / u.s
    energy_tolerance = energy_tolerance * u.m / u.s

    matched_zero = []
    matched_one = []
    matched_mult = []

    for b_line in tqdm(BRASS_transitions, unit='transitions'):
        # If the line is in a masked region of the spectrum, don't bother with
        # it, just pass.
        if line_is_masked(float(b_line.wavelength.to(u.nm).value),
                          spectral_mask):
            tqdm.write('{} is in a masked region.'.format(str(b_line)))
            n_masked_lines += 1
            continue

        delta_wavelength = vcl.get_wavelength_separation(wl_tolerance,
                                                         b_line.wavelength)
        delta_wavelength = delta_wavelength.to(u.nm)
        delta_wl_energy = vcl.get_wavelength_separation(energy_tolerance,
                                                        b_line.wavelength)

        energy1 = (1 / b_line.wavelength).to(u.cm ** -1)
        energy2 = (1 / (b_line.wavelength + delta_wl_energy)).to(u.cm ** -1)

        delta_energy = abs(energy2 - energy1)

        if args.verbose:
            tqdm.write('For {} (Z = {}), the wavelength tolerance is {:.4f},'.
                       format(str(b_line), b_line.atomicNumber,
                              delta_wavelength))

            tqdm.write('and the lower energy tolerance is {:.4f}.'.format(
                       delta_energy))

        # Set up a list to contain potential matched lines.
        matched_lines = []
        # Set up a list to contain distance info for all lines of the same
        # ionic species.
        same_species_lines = []

        # Go through Kurucz lines and store any that match.
        for k_line in Kurucz_transitions:
            if (k_line.atomicNumber == b_line.atomicNumber)\
              and (k_line.ionizationState == b_line.ionizationState):

                energy_diff = abs(k_line.lowerEnergy - b_line.lowerEnergy)
                wavelength_diff = abs(k_line.wavelength - b_line.wavelength)

                same_species_lines.append((k_line, wavelength_diff,
                                           energy_diff))

                if (energy_diff < delta_energy) and\
                   (wavelength_diff < delta_wavelength):
                    matched_lines.append(k_line)

        # If there's only one match (yay!) just save it.
        if len(matched_lines) == 1:
            matched_lines[0].normalizedDepth = b_line.normalizedDepth
            matched_one.append(matched_lines[0])
            tqdm.write('{} matched with one line.'.format(str(b_line)))

        # If there's no match, list nearest possible matches.
        elif len(matched_lines) == 0:
            matched_zero.append((b_line, delta_wavelength, delta_energy))
            tqdm.write('{} unmatched.'.format(str(b_line)))

            closest_lines = {}
            diff_scores = []
            for item in same_species_lines:
                diff_score = sqrt(item[1].value**2 + item[2].value**2)
                closest_lines[diff_score] = item
                diff_scores.append(diff_score)

            tqdm.write('{}{}:'.format(b_line.atomicSymbol,
                       b_line.ionizationState))
            for score in sorted(diff_scores)[:5]:
                item = closest_lines[score]
                tqdm.write('{:.2f} | {:8.4f} ({:.4f}) | {:.4f} ({:.4f})'
                           .format(score,
                                   item[0].wavelength, item[1],
                                   item[0].lowerEnergy, item[2]))

        # If there's more than one match, it could be due to isotopes, so check
        # that first:
        elif len(matched_lines) > 1:
            atomic_numbers = set()
            ion_states = set()
            lower_energies = set()
            for transition in matched_lines:
                atomic_numbers.add(transition.atomicNumber)
                ion_states.add(transition.ionizationState)
                lower_energies.add(float(transition.lowerEnergy.value))
            if (len(atomic_numbers) == 1)\
                    and (len(ion_states) == 1)\
                    and (len(lower_energies) == 1):
                isotope_dict = {t.isotopeFraction: t for t in matched_lines}
                # Sort the dictionary keys to find the transition with the
                # highest isotope fraction, and use that one.
                line_to_use = isotope_dict[sorted(isotope_dict.keys())[-1]]
                line_to_use.normalizedDepth = b_line.normalizedDepth
                matched_one.append(line_to_use)
                n_multi_isotopes += 1
                tqdm.write('{} matched out of multiple isotope lines.'.format(
                        str(b_line)))
            # If it's not isotopes, then it truly is matched to multiple lines,
            # so note that.
            else:
                matched_lines.insert(0, b_line)
                matched_mult.append(matched_lines)
                tqdm.write('{} matched to multiple lines.'.format(str(b_line)))

    for item in matched_zero:
        print('{} {:.4f} {:.4f}'.format(*item))
    print('wavelength tolerance: {}, energy tolerance: {}'.format(wl_tolerance,
          energy_tolerance))
    tqdm.write('Out of {} lines in the BRASS list:'.format(len(
            BRASS_transitions)))
    tqdm.write('{} were in masked regions.'.format(n_masked_lines))
    tqdm.write('{} were unable to be matched at all.'.format(
            len(matched_zero)))
    tqdm.write('{} were successfully matched with one line.'.format(
            len(matched_one)))
    tqdm.write('(Of those, {} were picked out of multiple isotopes.)'.format(
            n_multi_isotopes))
    tqdm.write('{} were matched with multiple (non-isotope) lines.'.format(
            len(matched_mult)))
    for item in matched_mult:
        for transition in item:
            print(str(transition))

    # Create a pickled file of the transitions that have been found.
    config = configparser.ConfigParser(interpolation=configparser.
                                       ExtendedInterpolation())

    pickle_file = pickle_dir / 'transitions.pickle'
    pickle_file_backup = pickle_dir / 'transitions_{}.pickle'.format(
            datetime.date.today().isoformat())
    tqdm.write('Saving transitions to {}'.format(pickle_file))
    with open(pickle_file, 'w+b') as f:
        pickle.dump(matched_one, f)
    with open(pickle_file_backup, 'w+b') as f:
        pickle.dump(matched_one, f)


def find_line_pairs(transition_list, out_file=None,
                    min_norm_depth=0.3, max_norm_depth=0.7,
                    velocity_separation=400*u.km/u.s,
                    line_depth_difference=0.05):
    """Find pairs of nearby transitions from a list, within constraints.

    Parameters
    ----------
    transition_list : a list of transition_line.Transition objects.

    out_file : Path object
    A Path object pointing to a file where the list of line pairs can be
    pickled to.

    min_norm_depth : float
    A float between 0 and 1 representing the minimum normalized depth of an
    absorption line to consider (0 being continuum, 1 being saturated). Should
    be less than maxNormDepth in absolute value.

    max_norm_depth : float
    A float between 0 and 1 representing the maximum normalized depth of an
    absorption line to consider (0 being continuum, 1 being saturated). Should
    be greater in absolute value than minNormDepth.

    velocity_separation : unyt_quantity, dimensions of velocity
    A velocity representing how far around a transition to search for matching
    transitions. Can use any units of velocity, will be converted to m/s
    interally.

    line_depth_difference : float
    A number representing how much two transitions can vary in normalized depth
    before being considered unfit as a pair.

    """

    velocity_separation.convert_to_units(u.m/u.s)
    transition_pair_list = []

    if args.verbose:
        tqdm.write('Running with depth limits = ({}, {})'.format(
                   min_norm_depth, max_norm_depth) +
                   ', velocity separation = {}, depth diff = {}'.format(
                           velocity_separation, line_depth_difference))

    for transition1 in tqdm(transition_list, unit='transitions'):
        # Check that the transition falls within normalized depth limits.
        if not(min_norm_depth <= transition1.normalizedDepth
               <= max_norm_depth):
            continue

        # If it's fine, figure out how much wavelength space to search around
        # it based on the velocity separation.
        delta_wl = vcl.get_wavelength_separation(velocity_separation,
                                                 transition1.wavelength)
        lowerLim = transition1.wavelength - delta_wl
        upperLim = transition1.wavelength + delta_wl

        # Iterate over the entire list of transitions.
        for transition2 in transition_list:

            # Avoid matching a transition with itself.
            if transition1 == transition2:
                continue

            # Check that the transition is within the velocity separation
            # limits.
            if not(lowerLim <= transition2.wavelength <= upperLim):
                continue

            # Check that the transition falls within normalized depth limits.
            if not(min_norm_depth <= transition2.normalizedDepth
                   <= max_norm_depth):
                continue

            # Check that the two transitions' depths don't exceed the defined
            # maximum depth difference.
            if (abs(transition1.normalizedDepth -
                    transition2.normalizedDepth) > line_depth_difference):
                continue

            # Only bother with transitions from the same element and ionization
            # state.
            if (transition1.atomicNumber !=
                transition2.atomicNumber) or (transition1.ionizationState !=
                                              transition2.ionizationState):
                continue

            # If a line makes it through all the checks, it's considered a
            # match for the initial line. Create a TransitionPair object
            # containing both of them and continue checking.
            pair = TransitionPair(transition1, transition2)
            if pair in transition_pair_list:
                pass
            else:
                transition_pair_list.append(pair)

    if out_file is not None:
        with open(out_file, 'w+b') as f:
            pickle.dump(transition_pair_list, f)

    return transition_pair_list


def query_nist(transition_list, species_set):
    """Query NIST for the given ionic species.

    Since querying NIST for multiple species at once using astroquery doesn't
    seem to work, this function queries on a per-species basis, using a
    wavelength range encompassing all the transitions of the that species. A
    list is created of Transition objects parsed from the returned results,
    which is then stored in a dictionary under the species name (e.g., "Fe II")

    Parameters
    ----------
    transition_list : list of transition_line.Transition objects
        A list of Transition objects representing all the transitions to be
        searched for in NIST.
    species_set : set of strings
        These should be of a form of a valid ionic species that NIST would
        recognize, i.e., "Cr I", "Ti II", "Fe I", etc.

    Returns
    -------
    dict
        A dictionary with species a keys and lists of Transition objects of
        that species as values.
        Example: {"Fe I": [Transition(500 u.nm, 26, 1),...]}

    """

    # Query NIST for information on each transition.
    tqdm.write('Querying NIST for transition information...')
    master_transition_dict = {}
    for species in tqdm(species_set):
        tqdm.write('--------------')
        tqdm.write(f'Querying {species}...')
        species_list = []
        for transition in transition_list:
            if transition.atomicSpecies == species:
                species_list.append(transition)

        min_wavelength = species_list[0].wavelength - 0.5 * u.angstrom
        max_wavelength = species_list[-1].wavelength + 0.5 * u.angstrom

        tqdm.write(f'Min: {min_wavelength},  max: {max_wavelength}')

        sleep(1)
        table = Nist.query(min_wavelength.to_astropy(),
                           max_wavelength.to_astropy(),
                           energy_level_unit='cm-1',
                           output_order='wavelength',
                           wavelength_type='vacuum',
                           linename=species)
        tqdm.write(f'{len(table)} transitions found for {species}.')
        if len(table) != 0:
            table.remove_columns(['Ritz', 'Rel.', 'Aki', 'fik', 'Acc.', 'Type',
                                  'TP', 'Line'])

            nist_transitions = []
            for row in tqdm(table):
                if row[0] is ma.masked:
                    nist_wavenumber = float(row[1]) * u.cm ** -1
                    nist_wavelength = (1 / nist_wavenumber).to(u.nm)
                    tqdm.write(f'{nist_wavenumber} --> {nist_wavelength}')
                elif row[1] is ma.masked:
                    nist_wavelength = float(row[0]) * u.nm
                    nist_wavenumber = (1 / nist_wavelength).to(u.cm ** -1)
                    tqdm.write(f'{nist_wavelength} --> {nist_wavenumber}')
                elif row[2] is ma.masked:
                    tqdm.write(f'No energy information for {row[0]}')
                    continue
                else:
                    nist_wavelength = float(row[0]) * u.nm
                    nist_wavenumber = float(row[1]) * u.cm ** -1
                nist_energy_levels = row[2].split('-')
                strip_chars = ' &d?'
                nist_energy1 = float(nist_energy_levels[0].strip(strip_chars))
                nist_energy2 = float(nist_energy_levels[1].strip(strip_chars))
                nist_lower_orbital = row[3]
                nist_upper_orbital = row[4]
                try:
                    nist_transition = Transition(nist_wavelength,
                                                 *species.split())
                except:
                    print(row)
                    raise
                nist_transition.lowerEnergy = nist_energy1 * u.cm ** -1
                nist_transition.higherEnergy = nist_energy2 * u.cm ** -1
                nist_transition.lowerOrbital = nist_lower_orbital
                nist_transition.higherOrbital = nist_upper_orbital
                nist_transition.wavenumber = nist_wavenumber

                nist_transitions.append(nist_transition)

            master_transition_dict[species] = nist_transitions

    return master_transition_dict


# ----- Main routine of code -----

desc = 'Select line pairs to analyze from the Kurucz and BRASS line lists.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--match_lines', action='store_true',
                    default=False,
                    help='Flag to match transitions between lists.')
parser.add_argument('-dw', '--delta_wavelength', action='store',
                    default=1000, type=int,
                    help='The wavelength tolerance in m/s.')
parser.add_argument('-de', '--delta_energy', action='store',
                    default=10000, type=int,
                    help='The energy tolerance in m/s.')

parser.add_argument('--pair_lines', action='store_true',
                    default=False,
                    help='Find pairs of transition lines from list.')

parser.add_argument('--query_lines', action='store_true',
                    default=False,
                    help='Query transitions from NIST.')

parser.add_argument('--verbose', action='store_true',
                    default=False,
                    help='Print out more information during the process.')

args = parser.parse_args()

global line_offsets
line_offsets = []

config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read('/Users/dberke/code/config/variables.cfg')

# These two files produces wavelengths in air, in Angstroms.
redFile = "../data/BRASS2018_Sun_PrelimGraded_Lobel.csv"
blueFile = "../data/BRASS2018_Sun_PrelimSpectroWeblines_Lobel.csv"

# This file produces wavelengths in vacuum, in nm.
purpleFile = '../data/BRASS_Vac_Line_Depths_All.csv'

# Define useful values relating to the Kurucz line list.
KuruczFile = "../data/gfallvac08oct17.dat"
colWidths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
             5, 5, 3, 3, 4, 5, 5, 6)
colNames = ("wavelength", "log gf", "elem", "energy1", "J1", "label1",
            "energy2", "J2", "label2", "gammaRad", "gammaStark", "vanderWaals",
            "ref", "nlte1",  "nlte2", "isotope1", "hyperf1", "isotope2",
            "logIsotope", "hyperfshift1", "hyperfshift2", "hyperF1", "hyperF2",
            "code", "landeGeven", "landeGodd", "isotopeShift")
colDtypes = (float, float, "U6", float, float, "U11", float, float, "U11",
             float, float, float, "U4", int, int, int, float, int, float,
             int, int, "U3", "U3", "U4", int, int, float)

masks_dir = Path(config['PATHS']['masks_dir'])
pickle_dir = Path(config['PATHS']['pickle_dir'])
pickle_file = pickle_dir / 'transitions.pickle'
pickle_pairs_file = pickle_dir / 'transition_pairs.pickle'

CCD_bounds_file = masks_dir / 'unusable_spectrum_CCDbounds.txt'
# Path('/Users/dberke/code/data/unusable_spectrum_CCDbounds.txt')
no_CCD_bounds_file = masks_dir / 'unusable_spectrum_CCDbounds.txt'
# Path('data/unusable_spectrum_noCCDbounds.txt')

mask_CCD_bounds = vcl.parse_spectral_mask_file(CCD_bounds_file)
mask_no_CCD_bounds = vcl.parse_spectral_mask_file(no_CCD_bounds_file)

#redData = np.genfromtxt(redFile, delimiter=",", skip_header=1,
#                        dtype=(float, "U2", int, float, float, float))
#print("Read red line list.")
#blueData = np.genfromtxt(blueFile, delimiter=",", skip_header=1,
#                     dtype=(float, "U2", int, float, float))
#print("Read blue line list.")

if args.match_lines:
    purpleData = np.genfromtxt(purpleFile, delimiter=",", skip_header=1,
                               dtype=(float, "U2", int, float, float, float))
    tqdm.write("Read purple line list.")

    KuruczData = np.genfromtxt(KuruczFile, delimiter=colWidths, autostrip=True,
                               skip_header=842959, skip_footer=987892,
                               names=colNames, dtype=colDtypes,
                               usecols=(0, 2, 3, 4, 5, 6, 7, 8, 18))
    tqdm.write("Read Kurucz line list.")

    # Create lists of transitions from the BRASS and Kurucz line lists.
    b_transition_lines = []
    for b_transition in tqdm(purpleData, unit='transitions'):
        wl = b_transition[0] * u.nm
        elem = elements[b_transition[1]]
        ion = b_transition[2]
        eLow = b_transition[3] * u.eV
        depth = b_transition[5]

        transition = Transition(wl, elem, ion)
        transition.lowerEnergy = 1/(eLow.to(u.cm, equivalence='spectral'))
        transition.normalizedDepth = depth

        b_transition_lines.append(transition)

    k_transition_lines = []
    for k_transition in tqdm(KuruczData, unit='transitions'):
        wl = k_transition['wavelength'] * u.nm
        # The element and ionionzation state from the Kurucz list is given as a
        # floating point number, e.g., 58.01, where the integer part is the
        # atomic number and the charge is the hundredths part (which is off by
        # one from astronomical usage, e.g. HI would have a hundredths part of
        # '00', while FeII would be '01').
        elem_str = k_transition['elem'].split('.')
        elem_num = int(elem_str[0])
        elem_ion = int(elem_str[1]) + 1
        energy1 = k_transition['energy1']
        energy2 = k_transition['energy2']
        if energy1 < energy2:
            lowE = k_transition['energy1'] * u.cm**-1
            lowOrb = k_transition['label1']
            lowJ = k_transition['J1']
            highE = k_transition['energy2'] * u.cm**-1
            highOrb = k_transition['label2']
            highJ = k_transition['J2']
        else:
            lowE = k_transition['energy2'] * u.cm**-1
            lowOrb = k_transition['label2']
            lowJ = k_transition['J2']
            highE = k_transition['energy1'] * u.cm**-1
            highOrb = k_transition['label1']
            highJ = k_transition['J1']
        isotope_frac = k_transition['logIsotope']

        transition = Transition(wl, elem_num, elem_ion)
        transition.lowerEnergy = lowE
        transition.lowerJ = lowJ
        transition.lowerOrbital = lowOrb
        transition.higherEnergy = highE
        transition.higherJ = highJ
        transition.higherOrbital = highOrb
        transition.isotopeFraction = isotope_frac

        k_transition_lines.append(transition)

    # Now, match between the BRASS list and Kurucz list as best we can.

    harmonize_lists(b_transition_lines, k_transition_lines, mask_no_CCD_bounds,
                    wl_tolerance=args.delta_wavelength,  # TODO: add units here
                    energy_tolerance=args.delta_energy)

if args.pair_lines:
    # Parameters for widest selection of line pairs:
    # minimum depth: 0.15
    # maximum depth: 0.90
    # maximum depth difference: 0.2
    # maximum velocity separation: 800 km/s

    print('Unpickling transition lines...')
    with open(pickle_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    print('Found {} transitions.'.format(len(transitions_list)))

    pairs = find_line_pairs(transitions_list, min_norm_depth=0.15,
                            max_norm_depth=0.9,
                            velocity_separation=800*u.km/u.s,
                            line_depth_difference=0.2)

    print(f'Found {len(pairs)} pairs.')
    with open(pickle_pairs_file, 'w+b') as f:
        pickle.dump(pairs, f)


if args.query_lines:

    with open(pickle_pairs_file, 'r+b') as f:
        pairs = pickle.load(f)
    transition_set = set()
    species_set = set()
    high_energy_pairs = []
    for pair in pairs:
        for transition in pair:
            transition_set.add(transition)
            species_set.add(transition.atomicSpecies)
            if (transition.higherEnergy > 50000 * u.cm ** -1) or\
               (transition.lowerEnergy > 50000 * u.cm ** -1):
                if pair not in high_energy_pairs:
                    high_energy_pairs.append(pair)

    transition_list = [x for x in transition_set]

    transition_list.sort()
#    high_energy_lines = set()
#
#    print(f'Total distinct transitions: {len(transition_list)}')

#    for transition in tqdm(transition_list):
#        if (item.higherEnergy > 50000) or (item.lowerEnergy > 50000):
#            high_energy_lines.add(item)
#
#    print(f'High energy lines (E > 50000 cm^-1): {len(high_energy_lines)}')
#
#    print(f'Total affected pairs: {len(high_energy_pairs)}')

    nist_pickle_file = pickle_dir / 'transitions_NIST_returned.pickle'
    try:
        with open(nist_pickle_file, 'r+b') as f:
            transition_dict = pickle.load(f)
    except FileNotFoundError:
        transition_dict = query_nist(transition_list, species_set)
        tqdm.write('Pickling NIST query...')
        with open(nist_pickle_file, 'w+b') as f:
            pickle.dump(transition_dict, f)

    print(transition_dict.keys())
