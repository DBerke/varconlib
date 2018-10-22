#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:57:52 2018

@author: dberke
"""

# Code to interate through BRASS list of "red" lines (specially curated)
# to identify pairs given various constraints.

import numpy as np
from math import trunc
from scipy.constants import lambda2nu, h, e
import varconlib as vcl
from pathlib import Path
from tqdm import tqdm

elements = {"Na": 11,
            "Si": 14,
            "Ca": 20,
            "Sc": 21,
            "Ti": 22,
            "Cr": 24,
            "Mn": 25,
            "Fe": 26,
            "Ni": 28}


def wn2eV(percm):
    """Return the energy given in cm^-1 in eV

    Invert cm^-1, divide by 100, divide into c, multiply by h, divide by e

    """
    if percm == 0.:
        result = 0.
    else:
        wl = (1 / percm) / 100
        vu = lambda2nu(wl)
        result = (vu * h) / e
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
    """Checks if a given spectral line is in a masked region of the spectrum.

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


def matchKuruczLines(wavelength, elem, ion, eLow):
    """Return the line from Kurucz list matching given parameters.

    Parameters
    ----------
    wavelength : float
    elem : float
    ion : int
    eLow : float

    wavelength:
    elem:
    ion
    eLow
    """
    for line in KuruczData:
        wl = round(10 * vac2air(line['wavelength']), 3)
        if abs(wl - wavelength) < 0.03:
            elem_num = trunc(line['elem'])
            elem_ion = int((line['elem'] - elem_num) * 100 + 1)
#            print(elem_num, elem_ion)
            if elements[elem] == elem_num and ion == elem_ion:
                energy1 = round(wn2eV(line['energy1']), 3)
                energy2 = round(wn2eV(line['energy2']), 3)
                if energy1 < energy2:
                    lowE = line['energy1']
                    lowOrb = line['label1']
                    lowJ = line['J1']
                    highE = line['energy2']
                    highOrb = line['label2']
                    highJ = line['J2']
                else:
                    lowE = line['energy2']
                    lowOrb = line['label2']
                    lowJ = line['J2']
                    highE = line['energy1']
                    highOrb = line['label1']
                    highJ = line['J1']
                if abs(eLow - energy1) < 0.03 or abs(eLow - energy2) < 0.03:
                    wavenumber = round((1e8 / (line['wavelength'] * 10)), 3)
                    PeckReederWL = vac2airPeckReeder(line['wavelength'])
                    return (round(PeckReederWL, 4), wavenumber),\
                           (lowE, lowJ, lowOrb, highE, highJ, highOrb)


def matchLines(lines, outFile, minDepth=0.3, maxDepth=0.7,
               velSeparation=400000, lineDepthDiff=0.05,
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
    elements = set()
    n = 0
    n_iron = 0

    output_lines = []

    for item in tqdm(lines):
        line_matched = False
        wl = item[0]
        elem = item[1]
        ion = item[2]
        eLow = item[3]
        depth = item[4]

        if spectralMask:
            # If the line falls in masked region, skip it.
            if line_is_masked(wl / 10, spectralMask):
                continue

        # Check line depth first
        if not (minDepth <= depth <= maxDepth):
            continue

        # Figure out the wavelength separation at this line's wavelength
        # for the given velocity separation. Ignore lines outside of this.
        delta_wl = vcl.getwlseparation(velSeparation, wl)

        for line in lines:

            # See if it is in a masked portion of the spectrum,
            # if a mask is given.
            if spectralMask:
                if line_is_masked(line[0] / 10, spectralMask):
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
            if not (minDepth <= line[4] <= maxDepth):
                continue

            # Check line depth differential, reject if outside threshold.
            if not abs(depth - line[4]) < lineDepthDiff:
                continue

            # Check to make sure both the element and ionization states match.
            if (elem != line[1]) or (ion != line[2]):
                continue



            # If it makes it through all the checks, get the lines' info.
            if not line_matched:
                # If this is the first match for this line, get its info first.
                try:
                    vac_wl, lineInfo = matchKuruczLines(wl, elem, ion, eLow)
                    lineStr = "\n{:0<8} {:0<9} {}{} {:0<9} {} {:10} {:0<9} {} {:10}".\
                              format(*vac_wl, elem, ion, *lineInfo)
                    output_lines.append(lineStr)
                    output_lines.append("\n")
#                    print(lineStr)
                except TypeError:
                    print("\nCouldn't find orbital info for")
                    print("\n{} {}{} {}eV".format(wl, elem, ion, eLow))
                prematchedLines.add(int(wl * 10))
                elements.add(elem)
                line_matched = True
            try:
                vac_wl, lineInfo = matchKuruczLines(line[0], line[1], line[2],
                                                    line[3])
                matchStr = "{:0<8} {:0<9} {}{} {:0<9} {} {:10} {:0<9} {} {:10}".\
                           format(*vac_wl, line[1], line[2], *lineInfo)
                output_lines.append(matchStr)
                output_lines.append("\n")
#                print(matchStr)
            except TypeError:
                print("\nCouldn't find orbital info for")
                print("  {} {}{} {}eV".format(
                  line[0], line[1], line[2], line[3]))
            n += 1
            if elem == 'Fe' and ion == 1:
                n_iron += 1

    with open(str(outFile), 'w') as f:
        print('Writing linefile {}'.format(outFile))
        f.write("#wl(air)   wave#   ion    eL     JL     orbL       eH     JH    orbH\n")
        f.writelines(output_lines)

    print("\n{} matches found.".format(n))
    print("{}/{} were FeI".format(n_iron, n))
    print(elements)
    print("Min depth = {}, max depth = {}".format(minDepth, maxDepth))
    print("Vel separation = {} [km/s], line depth diff = {}".format(
          velSeparation / 1000, lineDepthDiff))
    print('CCD bounds used: {}'.format('yes' if CCD_bounds else 'no'))

    logfile = 'data/linelists/line_selection_logfile.txt'
    with open(logfile, 'a') as g:
        g.write("{} matches found.\n".format(n))
        g.write("{}/{} were FeI\n".format(n_iron, n))
        g.write(str(elements))
        g.write('\n')
        g.write("Min depth = {}, max depth = {}\n".format(minDepth, maxDepth))
        g.write("Vel separation = {} [km/s], line depth diff = {}\n".format(
              velSeparation / 1000, lineDepthDiff))
        g.write('CCD bounds used: {}\n\n'.format('yes' if CCD_bounds
                else 'no'))


# Main body of code

redFile = "data/BRASS2018_Sun_PrelimGraded_Lobel.csv"
blueFile = "data/BRASS2018_Sun_PrelimSpectroWeblines_Lobel.csv"
KuruczFile = "data/gfallvac08oct17.dat"
colWidths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
             5, 5, 3, 3, 4, 5, 5, 6)

CCD_bounds_file = Path('data/unusable_spectrum_CCDbounds.txt')
no_CCD_bounds_file = Path('data/unusable_spectrum_noCCDbounds.txt')

mask_CCD_bounds = parse_spectral_mask_file(CCD_bounds_file)
mask_no_CCD_bounds = parse_spectral_mask_file(no_CCD_bounds_file)

redData = np.genfromtxt(redFile, delimiter=",", skip_header=1,
                        dtype=(float, "U2", int, float, float, float))
print("Read red line list.")
#blueData = np.genfromtxt(blueFile, delimiter=",", skip_header=1,
#                     dtype=(float, "U2", int, float, float))
#print("Read blue line list.")

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
                           dtype=[float, float, float, float, float,
                                  "U11", float, float, "U11", float,
                                  float, float, "U4", int, int, int,
                                  float, int, float, int, int, "U3",
                                  "U3", "U4", int, int, float],
                           usecols=(0, 2, 3, 4, 5, 6, 7, 8))
print("Read Kurucz line list.")
#print(KuruczData[5:10])


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
filename = outDir / 'Lines_0.15-0.9_800kms_0.2.txt'
matchLines(redData, filename, minDepth=0.15, maxDepth=0.9,
            velSeparation=800000, lineDepthDiff=0.2,
            spectralMask=mask_no_CCD_bounds, CCD_bounds=False)
#goldSystematic = "/Users/dberke/Documents/GoldSystematicLineList.txt"
#matchLines(redData, goldSystematic, minDepth=0.2, maxDepth=0.8,
#           velSeparation=800000, lineDepthDiff=0.1)
#matchLines(blueData, minDepth=0.3, maxDepth=0.7, velSeparation=400000,
#               lineDepthDiff=0.05)
