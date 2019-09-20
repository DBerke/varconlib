#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:54:34 2018

@author: dberke
"""

# Script to sort radial velocities in HARPS data on a per-star basis and
# output the unique results returned, and a median value if there are more
# than one.

import numpy as np

infile = "/Volumes/External Storage/HARPS/radvel_no_header.txt"

data = np.loadtxt(infile, dtype=str)
stars = {}
radvels = {}

for line in data:
    #print(line)
    name = line[0].split("/")[0]
    #print(name)
    radvel = float(line[1])
    try:
        stars[name].add(radvel)
        radvels[name].append(radvel)
    except KeyError:
        stars[name] = set()
        radvels[name] = []
        stars[name].add(radvel)
        radvels[name].append(radvel)
n, m = 0, 0

rvs = []
outfile = "/Volumes/External Storage/HARPS/radvel_medians_measured.txt"
with open(outfile, 'w') as f:
    for k in sorted(radvels.keys()):
        med = np.median(radvels[k])
        if abs(med) > 150 or med == 0.:
            print(k, med)
        rvs.append(med)
        f.write("{} {}\n".format(k, med))
        n += 1
        if len(stars[k]) == 1:
            m += 1

print("{}/{} stars have a single radvel.".format(m, n))
print(stars['HD196800'])


