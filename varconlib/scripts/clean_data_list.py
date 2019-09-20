#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:52:24 2018

@author: dberke
"""

# Script to strip quotation marks and redundant lines from the list of stars
# from the data selection made with Topcat.

infile = "/Users/dberke/Documents/Data_selection_stars_HD_num_nospace.txt"
outfile = "/Users/dberke/Documents/Data_selection_stars_HD_num_cleaned.txt"

with open(infile, 'r') as f:
    lines = f.readlines()

obj_set = set()
line_list = []
for line in lines:
    cleaned = line.strip()#[1:-1] not needed if no space in label
    if not cleaned in obj_set:
        obj_set.add(cleaned)
        line_list.append(cleaned)

print(line_list)
print(len(line_list))
        
with open(outfile, 'w') as g:
    for line in line_list:
        g.write(line + '\n')