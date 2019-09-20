#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:50:33 2018

@author: dberke

Requires the 'convert' command from ImageMagick to be available.
"""

import subprocess
import re
from glob import glob
from pathlib import Path
from tqdm import tqdm
from varconlib import pairlist


def append_graphs(main_dir):
    """Create a graph combining graphs of a pair and delta-v for all linepairs
    """

    pairs_graphs = [file for file in sorted(main_dir.glob('Linepair*.png'))]
    obs_dirs = [directory for directory in sorted(main_dir.glob('ADP*'))]
    #print(line_graphs[0], line_graphs[68])
    for obs_dir in tqdm(obs_dirs, total=len(obs_dirs), unit='Observations'):
        line_graphs = [file for file in sorted(obs_dir.glob('HD*.png'))]
        for pair in pairlist:
            outfile = obs_dir / 'Overview_{}_{}.png'.format(pair[0], pair[1])
            pair1 = []
            pair2 = []
            midpair = []
            for file in line_graphs:
                if pair[0] in str(file):
                    pair1.append(file)
                elif pair[1] in str(file):
                    pair2.append(file)
            for file in pairs_graphs:
                if '{}_{}'.format(pair[0], pair[1]) in str(file):
                    midpair.append(file)

            # Run the 'convert' command from ImageMagick
            params = ['convert', '(']
            params.append(str(pair1[0]))
            params.append(str(pair1[1]))
            params.extend(['-gravity', 'north', '-append', ')', '('])
            params.append(str(midpair[0]))
            params.append(str(midpair[1]))
            params.extend(['-gravity', 'north', '-append', ')', '('])
            params.append(str(pair2[0]))
            params.append(str(pair2[1]))
            params.extend(['-gravity', 'north', '-append', ')', '+append'])
            params.append(outfile)

            subprocess.run(params)


def average_lines(main_dir, style='mean'):
    """Create graphs averaging all observations for each absorption line

    style: can be 'mean' or 'min', argument passed to convert
    """

    #obs_dirs = [directory for directory in sorted(main_dir.glob('ADP*'))]
    all_files = [file for file in sorted(main_dir.glob('ADP*/HD*.png'))]
    filtered_files = [file for file in all_files if 'norm' not in str(file)]
    lines = []
    for pair in pairlist:
        lines.append(pair[0])
        lines.append(pair[1])
    for line in tqdm(lines, total=len(lines), unit='Lines'):
        regex = re.compile('{}nm.png$'.format(line))
        files = [file for file in filtered_files if regex.search(str(file))]
        params = ['convert']
        params.extend(files)
        params.extend(['-evaluate-sequence', style])
        outfile = main_dir / '{}_{}.png'.format(style, line)
        params.append(outfile)

        subprocess.run(params)

    tqdm.write('Created graphs using "{}" style.'.format(style))


main_dir = Path('/Volumes/External Storage/HARPS/HD146233/graphs')
main_dir = Path('/Volumes/External Storage/HARPS/HD78660/graphs')
main_dir = Path('/Volumes/External Storage/HARPS/HD183658/graphs')
main_dir = Path('/Volumes/External Storage/HARPS/HD45184/graphs')
main_dir = Path('/Volumes/External Storage/HARPS/HD138573/graphs')

if not main_dir.name == 'graphs':
    print('Given directory does not appear correct! (Should end in graphs)')
    exit(1)


append_graphs(main_dir)

average_lines(main_dir, 'min')
average_lines(main_dir, 'mean')
