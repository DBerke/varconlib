#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:57:52 2018

@author: dberke
"""

# Code to iterate through a given line list to identify pairs given
# various constraints.

import argparse
from math import trunc, sqrt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import varconlib as vcl
import unyt as u
from transition_line import Transition

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
#        wl = (1 / percm) / 100
#        vu = lambda2nu(wl)
#        result = (vu * h) / e
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


def vac2air(wl_vac):
    """Take an input vacuum wavelength in nm and return the air wavelength.

    Formula taken from www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    from Morton (2000, ApJ. Suppl., 130, 403) (IAU standard)
    """
    s = 1e3 / wl_vac
    n = 1 + 0.0000834254 + (0.02406147 / (130 - s**2)) +\
        (0.00015998 / (38.9 - s**2))
    return wl_vac / n


def vac2airPeckReeder(wl_vac):
    """
    Return the air wavelength of a vacuum wavelength in nanometers using
    the formula from Peck & Reeder 1972.

    Formula taken from Peck & Reeder, J. Opt. Soc. Am. 62, 958 (1972).
    https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-62-8-958&id=54743
    """
    s = 1e3 / wl_vac
    n = 1 + ((8060.51 + (2480990 / (132.274 - s**2)) + (17455.7 / (39.32457 -
             s**s))) / 1e8)
    return wl_vac / n


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


def format_line_dict(line_dict):
    """Return a formatted text string of a dictionary representing a line.

    Parameters
    ----------
    line_dict : dict
        A dictionary containing information about a line from the the Kurucz
        line list. Should contain the keys 'wavelength', 'wavenumber',
        'lowE', 'lowJ', 'lowOrb', 'highE', 'highJ', 'highOrb'.

    """

    string1 = '{wavelength} {wavenumber}'.format(**line_dict)
    string2 = ' {lowE} {lowJ} {lowOrb}'.format(**line_dict)
    string3 = ' {highE} {highJ} {highOrb}'.format(**line_dict)

    return string1 + string2 + string3


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

        delta_wavelength = vcl.getwlseparation(wl_tolerance.value,
                                               b_line.wavelength.to(u.m).value)
        delta_wavelength = (delta_wavelength * u.m).to(u.nm)
        delta_wl_energy = vcl.getwlseparation(energy_tolerance.value,
                                              b_line.wavelength.to(u.m).value)
#                                              (1/b_line.lowerEnergy)
#                                              .to(u.m).value)
        delta_wl_energy = delta_wl_energy * u.m

        energy1 = (1 / b_line.wavelength).to(u.cm ** -1)
        energy2 = (1 / (b_line.wavelength + delta_wl_energy)).to(u.cm ** -1)

        delta_energy = abs(energy2 - energy1)

        tqdm.write('For {} (Z = {}), the wavelength tolerance is {:.4f},'.
                   format(str(b_line), b_line.atomicNumber, delta_wavelength))

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

                if (energy_diff < delta_energy)\
                  and (wavelength_diff < delta_wavelength):
                    matched_lines.append(k_line)

        tqdm.write('Total of {} lines of {}{} found.'.format(
                len(same_species_lines),
                b_line.atomicSymbol, b_line.ionizationState))

        if len(matched_lines) == 1:
            matched_lines[0].normalizedDepth = b_line.normalizedDepth
            matched_one.append(matched_lines[0])
            tqdm.write('{} matched with one line.'.format(str(b_line)))

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
            else:
                matched_lines.insert(0, b_line)
                matched_mult.append(matched_lines)
                tqdm.write('{} matched to multiple lines.'.format(str(b_line)))
        tqdm.write('\n')

    for item in matched_zero:
        print('{} {:.4f} {:.4f}'.format(*item))
    print('wavelength tolerance: {}, energy tolerance: {}'.format(wl_tolerance,
          energy_tolerance))
    tqdm.write('Out of {} lines in the BRASS list:'.format(len(
            BRASS_transitions)))
#    tqdm.write('{} were in masked regions.'.format(n_masked_lines))
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


def matchKuruczLines(wavelength, elem, ion, eLow, vacuum_wl=True,
                     wl_tolerance=1000*u.meter/u.second,
                     energy_tolerance=10000*u.meter/u.second):
    """Return the line from Kurucz list matching given parameters.

    Parameters
    ----------
    wavelength : float
        The wavelength of the line to be matched, in vacuum, in nm.
    elem : str
        A string representing the standard two-letter chemical abbreviation
        for the chemical element responsible for the transition being matched.
    ion : int
        An integer representing the ionization state of the the element
        responsible for the transition being matched.
    eLow : float
        The energy of the lower state of the transition being matched, in eV.
    vacuum_wl : bool, Default : True
        If *True*, return the wavelengths in vacuum.

    """

    found_lines = []
    wavelength = wavelength * u.nm
    wl_vel_tolerance = vcl.getwlseparation(wl_tolerance.value, wavelength)
    print('The wavelength tolerance of {} at {} is {:.4f}'.format(
            wl_tolerance, wavelength, wl_vel_tolerance))
    for line in KuruczData:
        # For working with the purple list with its wavelengths in vac, nm.
        wl = line['wavelength'] * u.nm
#        wl = round(10 * vac2air(line['wavelength']), 3)
        # This distance is VERY important: 0.003 for nm, 0.03 for Angstroms
        if (abs(wl - wavelength) < wl_vel_tolerance):
            print(f'Found line with wavelength diff = {wl - wavelength:.4f}')
            line_offsets.append(abs(wl - wavelength))
            elem_num = trunc(line['elem'])
            elem_ion = int((line['elem'] - elem_num) * 100 + 1)
#            print(elem_num, elem_ion)
            if elements[elem] == elem_num and ion == elem_ion:
                energy1 = line['energy1']
                energy2 = line['energy2']
                e_lower = eV2wn(eLow)
                print(f'Lower energy of Kurucz line {wl} is {e_lower}')
                if energy1 < energy2:
                    lowE = line['energy1'] * u.cm**-1
                    lowOrb = line['label1']
                    lowJ = line['J1']
                    highE = line['energy2'] * u.cm**-1
                    highOrb = line['label2']
                    highJ = line['J2']
                else:
                    lowE = line['energy2'] * u.cm**-1
                    lowOrb = line['label2']
                    lowJ = line['J2']
                    highE = line['energy1'] * u.cm**-1
                    highOrb = line['label1']
                    highJ = line['J1']
                energy_diff = abs(e_lower - lowE)
                print(f'The energy difference is {energy_diff}')
                delta_wl = vcl.getwlseparation(energy_tolerance.value,
                                               (1 / e_lower).to(u.m).value)
                delta_wl = delta_wl * u.m
                print(f'delta_wl is {delta_wl}')
                energy1 = (1 / wavelength).to(u.cm ** -1)
                energy2 = (1 / (wavelength + delta_wl)).to(u.cm ** -1)

                en_tolerance = abs(energy2 - energy1)
                print(f'The energy tolerance is {en_tolerance}')
                if energy_diff < en_tolerance:
                    wavenumber = round((1e8 / (line['wavelength'] * 10)), 3)
                    found_lines.append({'wavelength': line['wavelength'],
                                        'wavenumber': wavenumber,
                                        'lowE': lowE,
                                        'lowJ': lowJ,
                                        'lowOrb': lowOrb,
                                        'highE': highE,
                                        'highJ': highJ,
                                        'highOrb': highOrb})
                    print('\nFound a match!')
                    print('wavelength: {} lower energy {}'.format(
                            line['wavelength'], lowE))
                    print(energy_diff, en_tolerance, '\n')
#                    if not vacuum_wl:
#                        PeckReederWL = vac2airPeckReeder(line['wavelength'])
#                        return (round(PeckReederWL, 4), wavenumber),\
#                               (lowE, lowJ, lowOrb, highE, highJ, highOrb)
#                    else:
#                        return (line['wavelength'], wavenumber),\
#                               (lowE, lowJ, lowOrb, highE, highJ, highOrb)

    return found_lines


def matchLines(lines, outFile, minDepth=0.3, maxDepth=0.7,
               velSeparation=400000, lineDepthDiff=0.05, vacuum_wl=True,
               spectralMask=None, CCD_bounds=False):
    """Return a list of line matches given the input parameters

    lines: list of lines to look through
    minDepth: (0, 1) minimum depth of line to consider
    maxDepth: (0, 1) maximum depth of line to consider
    velSeparation: velocity separation in m/s to search in (converted
                   wavelength)
    lineDepthDiff: max difference in line depths to consider

    """

    prematchedLines = set()
    elements_found = set()
    n = 0
    n_iron = 0
    n_masked = 0
    n_out_of_depth = 0

    output_lines = []

    matches_0 = []
    matches_1 = []
    matches_mult = []

    matches_0_file = Path('data/matches0.txt')
    matches_1_file = Path('data/matches1.txt')
    matches_mult_file = Path('data/matchesMult.txt')

    logfiles = (matches_0_file, matches_1_file, matches_mult_file)
    linelogs = (matches_0, matches_1, matches_mult)

    for item in tqdm(lines[:5]):
        line_matched = False
        wl = item[0]
#        tqdm.write('Searching for matches for line {}'.format(wl))
        elem = item[1]
        ion = item[2]
        eLow = item[3]
        depth = item[5]  # Use the measured depth.

        if spectralMask:
            # If the line falls in masked region, skip it.
            if line_is_masked(wl, spectralMask):
                n_masked += 1
                continue

        # Check line depth first
        if not (minDepth <= depth <= maxDepth):
            n_out_of_depth += 1
            continue

        # Figure out the wavelength separation at this line's wavelength
        # for the given velocity separation. Ignore lines outside of this.
        delta_wl = vcl.getwlseparation(velSeparation, wl)

        print('\nSearching line at {}nm with lower energy of {} eV '.format(
                wl, eLow) +
              'or {:.4f}'.format(1/((eLow * u.eV).to(u.cm,
                                 equivalence='spectral'))))

        for line in lines:

            # See if it is in a masked portion of the spectrum,
            # if a mask is given.
            if spectralMask:
                if line_is_masked(line[0], spectralMask):
                    # Reject the line and move on to the next one.
                    continue

            # Check to see line hasn't been matched already.
            if int(line[0] * 10) in prematchedLines:
                continue

            # Check that the lines are within the wavelength
            # separation but not the same.
            if not (0. < abs(line[0] - wl) < delta_wl):
                continue

            # Make sure the second line is within depth limits.
            if not (minDepth <= line[5] <= maxDepth):
                continue

            # Check line depth differential, reject if outside threshold.
            if not abs(depth - line[5]) < lineDepthDiff:
                continue

            # Check to make sure both the element and ionization states match.
            if (elem != line[1]) or (ion != line[2]):
                continue

            # If it makes it through all the checks, get the lines' info.
            if not line_matched:
                # If this is the first match for this line, get its info first.

                found_lines = matchKuruczLines(wl, elem, ion, eLow,
                                               vacuum_wl=vacuum_wl)
                if not found_lines:
                    tqdm.write("\nCouldn't find a match for")
                    tqdm.write("\n{} {}{} {}eV".format(wl, elem, ion, eLow))
                    matches_0.append(line)
                    continue

                print('found_lines =')
                print(found_lines)
                if (len(found_lines) > 1):
                    print('Multiple line matches found!')
                    matches_mult.append(line)
                    print(item)
                    for found_line in found_lines:
                        matches_mult.append(format_line_dict(found_line))
                        print(found_line)
#                    print('vac_wl = {}'.format(vac_wl))
                matches_1.append(line)
                matches_1.append(format_line_dict(found_lines[0]))
                lineStr = "\n{:0<8} {:0<9} {}{} ".format(
                                  found_lines[0]['wavelength'],
                                  found_lines[0]['wavenumber'],
                                  elem, ion) +\
                          "{:0<9} {} {:10} {:0<9} {} {:10}".format(
                                  found_lines[0]['lowE'],
                                  found_lines[0]['lowJ'],
                                  found_lines[0]['lowOrb'],
                                  found_lines[0]['highE'],
                                  found_lines[0]['highJ'],
                                  found_lines[0]['highOrb'])
                output_lines.append(lineStr)
                output_lines.append("\n")
#                    print(lineStr)

                prematchedLines.add(int(wl * 10))
                elements_found.add(elem)
                line_matched = True

            matched_lines = matchKuruczLines(line[0], line[1], line[2],
                                             line[3], vacuum_wl=vacuum_wl)

            if not matched_lines:
                tqdm.write("\nCouldn't find a match for")
                tqdm.write("  {} {}{} {}eV".format(
                  line[0], line[1], line[2], line[3]))
                continue

            if (len(matched_lines) > 1):
                    print('Multiple line matches found!')
                    print(item)
                    for matched_line in matched_lines:
                        print(matched_line)

            matchStr = "{:0<8} {:0<9} {}{} ".\
                       format(matched_lines[0]['wavelength'],
                              matched_lines[0]['wavenumber'],
                              line[1], line[2]) +\
                       "{:0<9} {} {:10} {:0<9} {} {:10}".\
                       format(matched_lines[0]['lowE'],
                              matched_lines[0]['lowJ'],
                              matched_lines[0]['lowOrb'],
                              matched_lines[0]['highE'],
                              matched_lines[0]['highJ'],
                              matched_lines[0]['highOrb'])
            output_lines.append(matchStr)
            output_lines.append("\n")

            n += 1
            if elem == 'Fe' and ion == 1:
                n_iron += 1

    with open(str(outFile), 'w') as f:
        print('Writing linefile {}'.format(outFile))
        f.write("#wl({})   wave#   ion    eL     JL"
                "     orbL       eH     JH    orbH\n".format(
                    'vac' if vacuum_wl else 'air'))
        f.writelines(output_lines)

    for logfile, linelog in zip(logfiles, linelogs):
        with open(logfile, 'w') as g:
            for entry in linelog:
                templist = [str(x) for x in entry]
                string = ' '.join(templist) + '\n'
                g.write(string)

    print("\n{} matches found.".format(n))
    print("{}/{} were FeI".format(n_iron, n))
    print(elements_found)
    print("Min depth = {}, max depth = {}".format(minDepth, maxDepth))
    print("Vel separation = {} [km/s], line depth diff = {}".format(
          velSeparation / 1000, lineDepthDiff))
    print('CCD bounds used: {}'.format('yes' if CCD_bounds else 'no'))

    print('{} lines in masked regions.'.format(n_masked))
    print('{} lines out of depth limits.'.format(n_out_of_depth))
    print('{} lines unmatched.'.format(len(matches_0)))
    print('{} lines matched once.'.format(len(matches_1)))
    print('{} lines matched mutiple times.'.format(len(matches_mult)))

#    logfile = 'data/linelists/line_selection_logfile.txt'
#    with open(logfile, 'a') as g:
#        g.write("{} matches found.\n".format(n))
#        g.write("{}/{} were FeI\n".format(n_iron, n))
#        g.write(str(elements))
#        g.write('\n')
#        g.write("Min depth = {}, max depth = {}\n".format(minDepth, maxDepth))
#        g.write("Vel separation = {} [km/s], line depth diff = {}\n".format(
#              velSeparation / 1000, lineDepthDiff))
#        g.write('CCD bounds used: {}\n\n'.format('yes' if CCD_bounds
#                else 'no'))


##### Main routine of code #####

desc = 'Select line pairs to analyze from the Kurucz and BRASS line lists.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-dw', '--delta_wavelength', action='store',
                    default=1000, type=int,
                    help='The wavelength tolerance in m/s.')
parser.add_argument('-de', '--delta_energy', action='store',
                    default=10000, type=int,
                    help='The energy tolerance in m/s.')

args = parser.parse_args()

global line_offsets
line_offsets = []

# These two files produces wavelengths in air, in Angstroms.
redFile = "data/BRASS2018_Sun_PrelimGraded_Lobel.csv"
blueFile = "data/BRASS2018_Sun_PrelimSpectroWeblines_Lobel.csv"

# This file produces wavelengths in vacuum, in nm.
purpleFile = 'data/BRASS_Vac_Line_Depths_All.csv'

KuruczFile = "data/gfallvac08oct17.dat"
colWidths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
             5, 5, 3, 3, 4, 5, 5, 6)

CCD_bounds_file = Path('data/unusable_spectrum_CCDbounds.txt')
no_CCD_bounds_file = Path('data/unusable_spectrum_noCCDbounds.txt')

mask_CCD_bounds = vcl.parse_spectral_mask_file(CCD_bounds_file)
mask_no_CCD_bounds = vcl.parse_spectral_mask_file(no_CCD_bounds_file)

#redData = np.genfromtxt(redFile, delimiter=",", skip_header=1,
#                        dtype=(float, "U2", int, float, float, float))
#print("Read red line list.")
#blueData = np.genfromtxt(blueFile, delimiter=",", skip_header=1,
#                     dtype=(float, "U2", int, float, float))
#print("Read blue line list.")

purpleData = np.genfromtxt(purpleFile, delimiter=",", skip_header=1,
                     dtype=(float, "U2", int, float, float, float))
print("Read purple line list.")

# Code to match up the red and blue lists.
#num_matched = 0
#unmatched = 0
#for line1 in redData:
#    matched = False
#    wl1 = line1[0]
#    energy1 = line1[3]
#    for line2 in blueData:
#        wl2 = line2[0]
#        energy2 = line2[3]
#        if (abs(wl1 - wl2) <= 0.1) and (abs(energy1 - energy2) <= 0.0015):
##            print('{} in red matches {} in blue'.format(wl1, wl2))
#            num_matched += 1
#            matched = True
#            break
#    if not matched:
#        print('{} not matched'.format(wl1))
#        unmatched += 1
#print('{} total matched'.format(num_matched))
#print('{} not matched'.format(unmatched))


KuruczData = np.genfromtxt(KuruczFile, delimiter=colWidths, autostrip=True,
                           skip_header=842959, skip_footer=987892,
                           names=["wavelength", "log gf", "elem", "energy1",
                                  "J1", "label1", "energy2", "J2", "label2",
                                  "gammaRad", "gammaStark", "vanderWaals",
                                  "ref", "nlte1",  "nlte2", "isotope1",
                                  "hyperf1", "isotope2", "logIsotope",
                                  "hyperfshift1", "hyperfshift2", "hyperF1",
                                  "hyperF2", "code", "landeGeven", "landeGodd"
                                  "isotopeShift"],
                           dtype=[float, float, "U6", float, float,
                                  "U11", float, float, "U11", float,
                                  float, float, "U4", int, int, int,
                                  float, int, float, int, int, "U3",
                                  "U3", "U4", int, int, float],
                           usecols=(0, 2, 3, 4, 5, 6, 7, 8, 18))
print("Read Kurucz line list.")

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
    # floating point number, e.g., 58.01, where the integer part is the atomic
    # number and the charge is the hundredths part (which is off by one from
    # astronomical usage).
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
                wl_tolerance=args.delta_wavelength,
                energy_tolerance=args.delta_energy)

goldStandard = "data/GoldStandardLineList.txt"
testStandard = "data/GoldStandardLineList_test.txt"
outDir = Path('data/linelists')

#depths = ((0.3, 0.7), (0.2, 0.7), (0.3, 0.8),
#          (0.2, 0.8), (0.3, 0.9), (0.2, 0.9))
#seps = (400000, 500000, 600000)
#bounds = (mask_no_CCD_bounds, mask_CCD_bounds)
#diffs = (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
#for depth in depths:
#    for sep in seps:
#        for bound, value in zip(bounds, (False, True)):
#            for diff in diffs:
#                CCD_tag = '_CCD' if value else ''
#                fileName = 'Lines_{0}-{1}_{2}kms_{3}{4}.txt'.format(
#                            depth[0], depth[1], int(sep/1000), diff, CCD_tag)
#                outFile = outDir / fileName
#                matchLines(redData, outFile,
#                           minDepth=depth[0], maxDepth=depth[1],
#                           velSeparation=sep, lineDepthDiff=diff,
#                           spectralMask=bound, CCD_bounds=value)


#filename = outDir / 'Lines_purple_0.15-0.9_800kms_0.2_test.txt'
#matchLines(purpleData, filename, minDepth=0.15, maxDepth=0.9,
#            velSeparation=800000, lineDepthDiff=0.2, vacuum_wl=True,
#            spectralMask=mask_no_CCD_bounds, CCD_bounds=False)
#goldSystematic = "/Users/dberke/Documents/GoldSystematicLineList.txt"
#matchLines(redData, goldSystematic, minDepth=0.2, maxDepth=0.8,
#           velSeparation=800000, lineDepthDiff=0.1)
#matchLines(blueData, minDepth=0.3, maxDepth=0.7, velSeparation=400000,
#               lineDepthDiff=0.05)

#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(1, 1, 1)
#ax.set_xlabel('$\Delta (\lambda - \lambda_0)$ nm')
#ax.hist(line_offsets, bins=20, linewidth=1, edgecolor='black')
#plt.show()
