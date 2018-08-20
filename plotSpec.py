#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:42:39 2018

@author: dberke
"""

# CLI script to plot arbitrary spectra

import matplotlib.pyplot as plt
import varconlib as vcl
import argparse
import numpy as np
import matplotlib
import os.path
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)
plt.rcParams['text.usetex'] = True

# The Kit Peak transmission spectrum comes from Hinkle, Wallace, & Livingston,
# 2003.
# The solar spectrum comes from Reiners et al., 2015.

# Start main body.
parser = argparse.ArgumentParser(description='Plot HARPS spectra')
parser.add_argument('filenames', action='store', nargs='+',
                    help='List of filenames to plot.')
parser.add_argument('-o', '--name', action='store',
                    default='Spectrum.png',
                    help='Name of output file (in Pictures).')
parser.add_argument('-n', '--minx', action='store', default=378, type=float,
                    help='The min x value (in nm).')
parser.add_argument('-m', '--maxx', action='store', default=692, type=float,
                    help='The max x value (in nm).')
parser.add_argument('-i', '--miny', action='store', type=float,
                    help='The minimum y value.')
parser.add_argument('-j', '--maxy', action='store', type=float,
                    help='The maximum y value.')
parser.add_argument('-s', '--sun', action='store_true',
                    help='Plot the IAG solar spectrum as well.')
parser.add_argument('-b', '--boost', action='store', type=float, default=1,
                    help='Factor to boost the solar spectrum by. (-1 for auto)')
parser.add_argument('-v', '--vacuum', action='store_true',
                    help='Plot spectra in vacuum wavelengths.')
parser.add_argument('-e', '--espresso', action='store_true',
                    help='Plot an ESPRESSO file.')
parser.add_argument('-u', '--unnormalized', action='store_true',
                    help='Plot the non-normalized solar flux.')
parser.add_argument('-r', '--radvel', action='store', type=float,
                    help='Correct spectrum by given radial velocity. (km/s)')
parser.add_argument('-t', '--transmission', action='store_true',
                    help='Plot the Kitt Peak 2003 transmission spectrum.')
parser.add_argument('-z', '--normalized', action='store_true',
                    help='Plot the spectra normalized by their highest point.')
parser.add_argument('-l', '--lines', nargs='+',
                    help='Plot vertical lines at the given positions.')

args = parser.parse_args()
#args = parser.parse_args('-n 588.1 -m 588.7 -sb 50000\
#                         HD117618/ADP.2014-09-16T11:07:25.643.fits\
#                         HD117618/ADP.2014-09-24T09:42:16.870.fits\
#                         HD117618/ADP.2014-09-24T09:44:01.477.fits\
#                         HD117618/ADP.2014-10-02T10:03:45.660.fits\
#                         -o HD117618_BRASS_air_sun.png'.split())


outPicDir = "/Users/dberke/Pictures/"
FITSfileDir = "/Volumes/External Storage/HARPS/"

solarSpectrum = "/Users/dberke/code/spectra/IAG_solar_atlas_V1_405-1065.txt"
transSpectrum = "/Users/dberke/code/spectra/transdata_0.5_1_mic.txt"
if args.sun:
    print('Importing IAG solar spectrum...', end='')
    wavenumber, sun_norm_flux, sun_flux = np.genfromtxt(solarSpectrum,
                                                        unpack=True)
    print('done.')
    print('Reversing solar wavelength array...', end='')
    wl_sun_vac = np.flipud(np.array((1 / wavenumber) * 1e7))  # In nm here
    print('reversed.')
    print('Reversing solar flux array...', end='')
    sol_norm_flux = np.flipud(np.array(sun_norm_flux))
    sol_flux = np.flipud(np.array(sun_flux))
    print('reversed.')

if args.transmission:
    print('Importing Kitt Peak transmission spectrum...', end='')
    wavenumber_tr, trans_norm_flux = np.genfromtxt(transSpectrum, unpack=True)
    print('done.')
    print('Reversing KP wavelength array...', end='')
    # Wavenumbers are given in vacuum, per the readme for the spectrum
    wl_trans_vac = np.flipud(np.array((1 / wavenumber_tr) * 1e7))  # In nm here
    print('reversed.')
    print('Reversing KP flux array...', end='')
    trans_norm_flux = np.flipud(np.array(trans_norm_flux))
    print('done.')

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Wavelength (nm)", fontsize=20)
ax.set_ylabel("Intensity", fontsize=20)
#fig.suptitle("{}, {} spectra".format(obj, len(files)))

ax.grid(which='major', axis='both')
ax.set_xlim(left=args.minx, right=args.maxx)
if (args.miny and args.maxy) or args.maxy:
    ax.set_ylim(bottom=args.miny, top=args.maxy)
    print('Top and bottom set to {} and {}.'.format(args.miny, args.maxy))
outfile = os.path.join(outPicDir, args.name)

for obj in args.filenames:
    infile = os.path.join(FITSfileDir, obj)
    if not os.path.exists(infile):
        print("File {} does not exist!".format(infile))
        exit(1)
    else:
        print('Parsing {}'.format(obj))
        if args.espresso:
            spectrum = vcl.readESPRESSOfile(infile)
        else:
            spectrum = vcl.readHARPSfile(infile, obj=True)
        w = np.array(spectrum['w']) # In Angstroms here!
        f = np.array(spectrum['f'])
        e = spectrum['e']
        vac_wl = np.array(vcl.air2vacESO(w))
        if args.vacuum:
            wl = vac_wl / 10 # Convert to nm.
        else:
            wl = np.array(vcl.vac2airMorton00(vac_wl))/10 # Convert to nm
        if args.radvel:
            wl = vcl.lineshift(wl, -1*args.radvel)
        leftpos = vcl.wavelength2index(wl, args.minx)
        rightpos = vcl.wavelength2index(wl, args.maxx)
        if args.normalized:
            flux = f / f.max()
            err = e / f.max()
        else:
            flux = f
            err = e
        try:
            maxflux = flux[leftpos:rightpos].max()
        except ValueError:
            print("Couldn't find a maximum flux in the given region.")
            print("Region exceeds the spectral array, try a smaller region?")
            maxflux = 1
        # Plot the spectrum
        ax.errorbar(wl, flux, yerr=err,
                    linestyle='', marker='.', alpha=1,
                    label=spectrum['obj'])

if not (args.miny or args.maxy):
    print('No y-limits givens, set to 0 and {}.'.format(maxflux*1.15))
    ax.set_ylim(bottom=0, top=maxflux*1.15)

if args.sun:
    print(args)
    if args.vacuum == False:
        # Need to convert from nm to Angstroms before converting from
        # vacuum to air, then back.
        wl_sun_air = np.array(vcl.vac2airMorton00(wl_sun_vac*10)/10)
        wl_sun = wl_sun_air
    else:
        wl_sun = wl_sun_vac
    if args.unnormalized:
        solar_flux = sol_flux
    else:
        solar_flux = sol_norm_flux
    if args.boost == -1:
        leftsun = vcl.wavelength2index(wl_sun, args.minx)
        rightsun = vcl.wavelength2index(wl_sun, args.maxx)
        maxsunflux = solar_flux[leftsun:rightsun].max()
        args.boost = maxflux / maxsunflux
        print('Boost automatically set to {}.'.format(maxflux))
    boosted_flux = solar_flux * args.boost
    ax.plot(wl_sun, boosted_flux, color='green',
            linestyle='', marker='.', alpha=1,
            label='IAG Solar Spectrum')

if args.transmission:
    ax.plot(wl_trans_vac, trans_norm_flux, color='DarkCyan',
            linestyle='', marker='.', alpha=1,
            label='Kitt Peak transmission spectrum')

if args.lines:
    xpositions = [float(line) for line in args.lines]
    ax.vlines(xpositions, 0, 1, transform=ax.get_xaxis_transform(),
              color='black', linestyle='--', alpha=0.6)

ax.legend(fontsize=24)
fig.savefig(outfile, format='png')
plt.close(fig)
