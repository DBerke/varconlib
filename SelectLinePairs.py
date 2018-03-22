#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:57:52 2018

@author: dberke
"""

# Code to interate through BRASS list of "red" lines (specially curated)
# to identify pairs given various constraints.

import numpy as np

infile = "/Users/dberke/BRASS2018_Sun_PrelimGraded_Lobel.csv"

data = np.genfromtxt(infile, delimiter=",", skip_header=1,
                     dtype=(float, "|S2", int, float, float, float))

#print(data)
prematchedLines = set()
elements = set()
n = 0
for item in data:
    match = False
    wl = item[0]
    elem = item[1]
    ion = item[2]
    depth = item[4]
    # Check line depth first
    if 0.3 <= depth <= 0.7:
        for line in data:
            # Check to see line hasn't been matched already
            if int(line[0] * 10) not in prematchedLines:
                # Check that the lines are within 5 Angstroms, but not the same 
                if 0. < abs(wl - line[0]) < 5.:
                    if 0.3 <= line[4] <= 0.7:
                        # Check line depth differential
                        if abs(depth - line[4]) < 0.05:
                            if elem == line[1] and ion == line[2]:
                                if match == False:
                                    print(wl, elem, ion, depth)
                                    prematchedLines.add(int(wl * 10))
                                    elements.add(elem)
                                match = True
                                print("    {} {}".format(line[0], line[4]))
                                n += 1

print("{} matches found.".format(n))

print(prematchedLines)
print(elements)