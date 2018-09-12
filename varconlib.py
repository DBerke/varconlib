#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke
"""

# Module to contain functions potentially useful across multiple programs

import numpy as np
import datetime as dt
from astropy.io import fits
from astropy.constants import c

# Some generic information useful in different scripts

pairlist = [('443.9589', '444.1128'), ('450.0151', '450.3467'),
            ('459.9405', '460.3290'), ('460.5846', '460.6877'),
            ('465.8889', '466.2840'), ('473.3122', '473.3780'),
            ('475.9448', '476.0601'), ('480.0073', '480.0747'),
            ('484.0896', '484.4496'), ('488.6794', '488.7696'),
            ('490.9102', '491.0754'), ('497.1304', '497.4489'),
            ('500.5115', '501.1420'), ('506.8562', '507.4086'),
            ('507.3492', '507.4086'), ('513.2898', '513.8813'),
            ('514.8912', '515.3619'), ('524.8510', '525.1670'),
            ('537.5203', '538.1069'), ('554.4686', '554.5475'),
            ('563.5510', '564.3000'), ('571.3716', '571.9418'),
            ('579.4679', '579.9464'), ('579.5521', '579.9779'),
            ('580.8335', '581.0828'), ('593.1823', '593.6299'),
            ('595.4367', '595.8344'), ('600.4673', '601.0220'),
            ('616.3002', '616.8146'), ('617.2213', '617.5042'),
            ('617.7065', '617.8498'), ('623.9045', '624.6193'),
            ('625.9833', '626.0427'), ('625.9833', '626.2831'),
            ('647.0980', '647.7413')]

elemdict = {('443.9589', '444.1128'): 26,
            ('450.0151', '450.3467'): 25,
            ('459.9405', '460.3290'): 26,
            ('460.5846', '460.6877'): 26,
            ('465.8889', '466.2840'): 26,
            ('473.3122', '473.3780'): 28,
            ('475.9448', '476.0601'): 22,
            ('480.0073', '480.0747'): 26,
            ('484.0896', '484.4496'): 26,
            ('488.6794', '488.7696'): 26,
            ('490.9102', '491.0754'): 26,
            ('497.1304', '497.4489'): 26,
            ('500.5115', '501.1420'): 28,
            ('506.8562', '507.4086'): 26,
            ('507.3492', '507.4086'): 26,
            ('513.2898', '513.8813'): 26,
            ('514.8912', '515.3619'): 22,
            ('524.8510', '525.1670'): 26,
            ('537.5203', '538.1069'): 26,
            ('554.4686', '554.5475'): 26,
            ('563.5510', '564.3000'): 26,
            ('571.3716', '571.9418'): 26,
            ('579.4679', '579.9464'): 14,
            ('579.5521', '579.9779'): 26,
            ('580.8335', '581.0828'): 26,
            ('593.1823', '593.6299'): 26,
            ('595.4367', '595.8344'): 26,
            ('600.4673', '601.0220'): 26,
            ('616.3002', '616.8146'): 20,
            ('617.2213', '617.5042'): 26,
            ('617.7065', '617.8498'): 28,
            ('623.9045', '624.6193'): 14,
            ('625.9833', '626.0427'): 22,
            ('625.9833', '626.2831'): 22,
            ('647.0980', '647.7413'): 26}

badlines = frozenset(['506.8562', '507.4086', '593.1823', '593.6299',
                      '595.4367', '595.8344', '600.4673', '601.0220',
                      '647.0980', '647.7413'])


def readHARPSfile(FITSfile, obj=False, wavelenmin=False, date_obs=False,
                  spec_bin=False, med_snr=False, hdnum=False, radvel=False):
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

    result = {}
    with fits.open(FITSfile) as hdulist:
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        data = hdulist[1].data
        w = data.WAVE[0]
        gain = header0['GAIN']
        # Multiply by the gain to convert from ADUs to photoelectrons
        f = data.FLUX[0] * gain
        e = 1.e6 * np.absolute(f)
        # Construct an error array by taking the square root of each flux
        for i in np.arange(0, len(f), 1):
            if (f[i] > 0.0):
                e[i] = np.sqrt(f[i])
        result['w'] = w
        result['f'] = f
        result['e'] = e
        if obj:
            result['obj'] = header1['OBJECT']
        if wavelenmin:
            result['wavelmin'] = header0['WAVELMIN']
        if date_obs:
            result['date_obs'] = dt.datetime.strptime(header0['DATE-OBS'],
                                                      '%Y-%m-%dT%H:%M:%S.%f')
        if spec_bin:
            result['spec_bin'] = header0['SPEC_BIN']
        if med_snr:
            result['med_snr'] = header0['SNR']
        if hdnum:
            result['hdnum'] = header0['HDNUM']
        if radvel:
            result['radvel'] = header0['RADVEL']

    return result


def readESPRESSOfile(ESPfile):
    """Read an ESPRESSO file and return a dictionary of information

    ESPfile: a path to the ESPRESSO file to be read

    output: a dictionary containing the following information:
        obj: the name from the OBJECT card
        w: the wavelength array
        f: the flux array
        e: the error array
    """
    with fits.open(ESPfile) as hdulist:
        data = hdulist[1].data
        obj = hdulist[0].header['OBJECT']
        w = data['wavelength']
        f = data['flux']
        e = data['error']
    return {'obj': obj, 'w': w, 'f': f, 'e': e}


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
    """Take an array of air wls (A) and return an array of vacum wls

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

        vac.append(round(newwl, 2))
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


def wavelength2index(wl_arr, wl, reverse=False):
    """Find the index in a list associated with a given wavelength

    wl_arr: an iterable object of wavelengths, *in increasing order*
    wl: the wavelength to search for
    reverse: a Boolean for if the wavelength array is listed from large to
             small. Will first re

    returns: the index for which the wavelength is closest to the given
    """
    length = len(wl_arr)
    for i in range(0, length, 1):
        # First find the index for which the value is greater than the given wl
        if wl_arr[i] >= wl:
            # Then check if it's closest to this index or the previous one
            # Note that the way it's set up it should always be
            # wl_arr[i-1] <= wl <= wl_arr[i]
            if wl_arr[i]-wl > wl-wl_arr[i-1]:
                return i-1
            else:
                return i

    print("Couldn't find the given wavelength: {}".format(wl))
    raise ValueError


def lineshift(line, radvel):
    """Find the new position of a line given the radial velocity of a star

    line: line position. Can be nm or Angstroms, will return in same units
    radvel: radial velocity in km/s

    returns: the new line position
    """
    return ((radvel * 1000 / c.value) * line) + line


def getwlseparation(v, wl):
    """Return wavelength separation for a given velocity separation at a point

    v: the velocity separation. Should be in m/s
    wl: the wavelength at which the function should be evaluated, since
        it's also a function of wavelength. Returned in whatever units it's
        given in.

    """
    return (v * wl) / c.value


def getvelseparation(wl1, wl2):
    """Return velocity separation for a pair of points in wavelength space

    wl1 & wl2: wavelengths to get the velocity separation between in m/s.
               Should be in meters.
    """
    return (wl2 - wl1) * c.value / ((wl1 + wl2) / 2)
