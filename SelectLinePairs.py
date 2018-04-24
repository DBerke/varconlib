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
from scipy.constants import lambda2nu, c, h, e

elements = {"Si": 14,
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


def get_wl_separation(v, wl):
    """Return the absolute wavelength separation for a given velocity separation
    
    Velocity should be m/s, wavelength returned in whatever units it's
    given in.
    
    """
    return (v * wl) /  c


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
    Return the air wavelength of a vacuum wavelength in nm using Peck-Reeder formula.
    
    Forumala taken from Peck & Reeder, J. Opt. Soc. Am. 62, 958 (1972).
    https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-62-8-958&id=54743
    """
    s = 1e3 / wl_vac
    n = 1 + ((8060.51 + (2480990 / (132.274 - s**2)) + (17455.7 / (39.32457 -\
             s**s))) / 1e8)
    return wl_vac / n


def matchKuruczLines(wavelength, elem, ion, eLow):
    """Return the line from Kurucz list matching given parameters

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
            #print(elem_num, elem_ion)
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
                #print("{}eV, {} eV".format(lowE, eLow))
                if abs(eLow - energy1) < 0.03 or abs(eLow - energy2) < 0.03:
                    #print("Match found for line at {}!".format(wavelength))
                    #print(line)
                    wavenumber = round((1e8 / (line['wavelength'] * 10)), 3)
                    PeckReederWL = vac2airPeckReeder(line['wavelength'])
                    return (round(PeckReederWL, 4), wavenumber), (lowE, lowJ,
                           lowOrb, highE, highJ, highOrb)


def matchLines(lines, outFile, minDepth=0.3, maxDepth=0.7,
               velSeparation=400000, lineDepthDiff=0.05):
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
    with open(outFile, 'w') as f:
        f.write("#wl(air)   wave#   ion    eL     JL     orbL       eH     JH    orbH\n")
        for item in lines:
            match = False
            wl = item[0]
            elem = item[1]
            ion = item[2]
            eLow = item[3]
            depth = item[4]
            # Check line depth first
            if minDepth <= depth <= maxDepth:
                for line in lines:
                    # Check to see line hasn't been matched already
                    if int(line[0] * 10) not in prematchedLines:
                        # Figure out the wavelength separation at given wavelength
                        # for a given velocity separation.
                        delta_wl = get_wl_separation(velSeparation, wl)
                        # Check that the lines are within the wavelength
                        # separation but not the same 
                        if 0. < abs(line[0] - wl) < delta_wl:
                            if minDepth <= line[4] <= maxDepth:
                                # Check line depth differential
                                if abs(depth - line[4]) < lineDepthDiff:
                                    if elem == line[1] and ion == line[2]:
                                        if match == False:
                                            try:
                                                vac_wl, lineInfo = matchKuruczLines(wl,
                                                                            elem,
                                                                            ion,
                                                                            eLow)
                                                lineStr = "\n{:0<8} {:0<9} {}{} {:0<9} {} {:10} {:0<9} {} {:10}".format(
                                                      *vac_wl, elem, ion, *lineInfo)
                                                f.write(lineStr)
                                                f.write("\n")
                                                print(lineStr)
                                            except TypeError:
                                                print("\nCouldn't find orbital info for")
                                                print("\n{} {}{} {}eV".format(wl,
                                                  elem, ion, eLow))
                                            prematchedLines.add(int(wl * 10))
                                            elements.add(elem)
                                        match = True
                                        try:                                        
                                            vac_wl, lineInfo = matchKuruczLines(line[0],
                                                                        line[1],
                                                                        line[2],
                                                                        line[3])
                                            matchStr = "{:0<8} {:0<9} {}{} {:0<9} {} {:10} {:0<9} {} {:10}".format(
                                                  *vac_wl, line[1], line[2], *lineInfo)
                                            f.write(matchStr)
                                            f.write("\n")
                                            print(matchStr)
                                        except TypeError:
                                            print("\nCouldn't find orbital info for")
                                            print("  {} {}{} {}eV".format(
                                              line[0], line[1], line[2], line[3]))
                                        n += 1
                                        if elem == 'Fe' and ion == 1:
                                            n_iron += 1

    print("\n{} matches found.".format(n))
    print("{}/{} were FeI".format(n_iron, n))
    
#    print(prematchedLines)
    print(elements)
    print("Min depth = {}, max depth = {}".format(minDepth, maxDepth))
    print("Vel separation = {} [m/s], line depth diff = {}".format(velSeparation,
          lineDepthDiff))


# Main body of code

redFile = "/Users/dberke/Documents/BRASS2018_Sun_PrelimGraded_Lobel.csv"
blueFile = "/Users/dberke/Documents/BRASS2018_Sun_PrelimSpectroWeblines_Lobel.csv"
KuruczFile = "/Users/dberke/Documents/gfallvac08oct17.dat"
colWidths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
             5, 5, 3, 3, 4, 5, 5, 6)

redData = np.genfromtxt(redFile, delimiter=",", skip_header=1,
                     dtype=(float, "U2", int, float, float, float))
print("Read red line list.")
blueData = np.genfromtxt(blueFile, delimiter=",", skip_header=1,
                     dtype=(float, "U2", int, float, float))
print("Read blue line list.")
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


goldStandard = "/Users/dberke/Documents/GoldStandardLineList.txt"
matchLines(redData, goldStandard, minDepth=0.3, maxDepth=0.7,
            velSeparation=400000, lineDepthDiff=0.05)
goldSystematic = "/Users/dberke/Documents/GoldSystematicLineList.txt"
#matchLines(redData, goldSystematic, minDepth=0.2, maxDepth=0.8,
#           velSeparation=800000, lineDepthDiff=0.1)
#matchLines(blueData, minDepth=0.3, maxDepth=0.7, velSeparation=400000,
#               lineDepthDiff=0.05)


