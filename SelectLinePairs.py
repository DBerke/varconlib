#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:57:52 2018

@author: dberke
"""

# Code to interate through BRASS list of "red" lines (specially curated)
# to identify pairs given various constraints.

import numpy as np
from scipy.constants import lambda2nu, c, h, e

infile = "/Users/dberke/BRASS2018_Sun_PrelimGraded_Lobel.csv"

data = np.genfromtxt(infile, delimiter=",", skip_header=1,
                     dtype=(float, "|S2", int, float, float, float))

def wn2eV(percm):
    """Return the energy given in cm^-1 in eV
    
    Invert cm^-1, divide by 100, divide into c, multiply by h, divide by e
    
    """
    wl = (1 / percm) / 100
    vu = lambda2nu(wl)
    return (vu * h) / e

def get_wl_separation(v, wl):
    """Return the absolute wavelength separation for a given velocity separation
    
    Velocity should be m/s, wavelength returned in whatever units it's
    given in.
    
    """
    return (v * wl) /  c


def matchLines(minDepth=0.3, maxDepth=0.7, velSeparation=400000,
               lineDepthDiff=0.05):
    """Return a list of line matches given the input parameters
    
    
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
    for item in data:
        match = False
        wl = item[0]
        elem = item[1]
        ion = item[2]
        eLow = item[3]
        depth = item[4]
        # Check line depth first
        if minDepth <= depth <= maxDepth:
            for line in data:
                # Check to see line hasn't been matched already
                if int(line[0] * 10) not in prematchedLines:
                    # Figure out the wavelength separation at given wavelength
                    # for a given velocity separation.
                    delta_wl = get_wl_separation(velSeparation, wl)
                    # Check that the lines are within the wavelength
                    # separation but not the same 
                    if 0. < abs(wl - line[0]) < delta_wl:
                        if minDepth <= line[4] <= maxDepth:
                            # Check line depth differential
                            if abs(depth - line[4]) < lineDepthDiff:
                                if elem == line[1] and ion == line[2]:
                                    if match == False:
                                        print("{} {} {} {}eV {}".format(wl,
                                              str(elem), ion, eLow, depth))
                                        prematchedLines.add(int(wl * 10))
                                        elements.add(elem)
                                    match = True
                                    print("    {} {} {} {} {}".format(line[0],
                                          str(line[1]), line[2], line[3],
                                          line[4]))
                                    n += 1
                                    if elem == b'Fe' and ion == 1:
                                        n_iron += 1

    print("{} matches found.".format(n))
    print("{}/{} were FeI".format(n_iron, n))
    
    print(prematchedLines)
    print(elements)

matchLines(minDepth=0.3, maxDepth=0.7, velSeparation=400000,
               lineDepthDiff=0.05)