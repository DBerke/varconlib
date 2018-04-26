#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke
"""

# Module to contain functions potentially useful across multiple programs

import numpy as np
from astropy.io import fits

def readHARPSfile(FITSfile):
    """Read a HARPS FITS file and return a dictionary of information.
    
    FITSfile: a path to a HARPS FITS file to be read.
    
    output: a dictionary containing the following information:
        obj: the object name from the 'OBJECT' flag
        w: the wavelength array
        f: the flux array
        e: the estimated error array (HARPS has no error array)
        wlmin: the minimum wavelength
        date_obs: the date the file was observed
        spec_bin: the wavelength bin size
        med_snr: the median SNR of the flux array
        hd_num: the HD number of the star
        radvel: the radial velocity of the star
    """

    with fits.open(FITSfile) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data
        obj = header['OBJECT']
        wavelmin = hdulist[0].header['WAVELMIN']
        date_obs = hdulist[0].header['DATE-OBS']
        spec_bin = hdulist[0].header['SPEC_BIN']
        med_SNR = hdulist[0].header['SNR']
        hd_num = hdulist[0].header['HDNUM']
        radvel = hdulist[0].header['RADVEL']
        w = data.WAVE[0]
        f = data.FLUX[0]
        e = 1.e6 * np.absolute(f)
        for i in np.arange(0, len(f), 1):
            if (f[i] > 0.0):
                e[i] = np.sqrt(f[i])
    return {'obj':obj, 'w':w, 'f':f, 'e':e,
            'wlmin':wavelmin, 'date_obs':date_obs,
            'spec_bin':spec_bin, 'med_snr':med_SNR,
            'hd_num':hd_num, 'radvel':radvel}


def air_indexEdlen53(l, t=15., p=760.):
    """Return the index of refraction of air at given temp, pressures, and wl (A)
    
    l: vacuum wavelength in Angstroms
    t: temperature (don't use)
    p: pressure (don't use)
    
    Formula is Edlen 1953, provided directly by ESO
    """
    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t)\
    * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n


def vac2airESO(ll):
    """Return a vacuum wavlength from an air wavelength (A) using Edlen 1953
    
    ll: air wavlength in Angstroms
    
    """
    llair = ll/air_indexEdlen53(ll)
    return llair


def air2vacESO(air_arr):
    """Take an array of air wavelengths (A) and return an array of vacuum wavelengths   
    
    air_arr: an array-like list of wavelengths in air (Angstroms)

    returns: an array of wavelengths in vacuum (Angstroms)
    """
    if not type(air_arr) == np.ndarray:
        air_arr = np.array(air_arr)

    tolerance = 1e-12

    vac = []

    for i in range(0, len(air_arr)):
        newwl = air_arr[i]
        oldwl = 0.
        while abs(oldwl - newwl) > tolerance:
            oldwl = newwl
            n = air_indexEdlen53(newwl)
            newwl = air_arr[i] * n

        vac.append(newwl)
    vac_arr = np.array(vac)

    return vac_arr

def vac2airMorton00(wl_vac):
    """Take an input vacuum wavelength in Angstroms and return the air wavelength.
    
    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    from Morton (2000, ApJ. Suppl., 130, 403) (IAU standard)
    """
    s = 1e4 / wl_vac
    n = 1 + 0.0000834254 + (0.02406147 / (130 - s**2)) +\
        (0.00015998 / (38.9 - s**2))
    return wl_vac / n


def air2vacMortonIAU(wl_air):
    """Take an input air wavelength in Angstroms and return the vacuum wavelength.
    
    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    """
    s = 1e4 / wl_air
    n = 1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s**2))\
        + (0.0001599740894897 / (38.92568793293 - s**2))
    return wl_air * n

def index2wavelength(index, step, min_wl):
    """Return the wavelength associated with an index.
    
    index: index position of the spectrum list
    step: the step in wavelength per index, in nm
    min_wl: the minimum wavelength of the spectrum, in nm
    """
    return round((step * index + min_wl), 2)


#def wavelength2index(wl, step, min_wl):
    """Return the index of the given wavelength.
    
    wl: the wavelength of the position of interest
    step: the step in wavelength, in nm
    min_wl: the minimum wavelength of the spectrum, in nm
    """
    
#    return int((wl - min_wl) / step)

def wavelength2index(wl_arr, wl):
    """Find the index in a list associated with a given wavelength
    
    wl_arr: an iterable object of wavelengths, in increasing order
    wl: the wavelength to search for

    returns: the first index for which the wavelength is larger than the given    
    """
    for i in range(len(wl_arr)):
        if wl_arr[i] >= wl:
            return i
        
    print("Couldn't find the given wavelength: {}".format(wl))
    raise ValueError
