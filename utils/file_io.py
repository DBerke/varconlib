#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:54:34 2018

@author: dberke

This library contains functions to deal with opening files from HARPS (both
1D ADP files and 2D extracted e2ds files) and ESPRESSO.
"""

import numpy as np
import datetime as dt
import unyt
from tqdm import tqdm, trange
from astropy.io import fits
from pathlib import Path


class HARPSFile2D:
    """
    Class to contain data from a HARPS 2D extracted spectrum file.
    """
    def __init__(self, FITSfile, verbose=False):
        if type(FITSfile) is str:
            self.filename = Path(FITSfile)
        else:
            self.filename = FITSfile
        with fits.open(self.filename) as hdulist:
            self.header = hdulist[0].header
            self.raw_flux_arr = hdulist[0].data

        # Get the gain from the file header:
#        gain = self.header['HIERARCH ESO DRS CCD CONAD']
#        self.photon_flux_arr = self.raw_flux_arr * gain
#        self.error_arr = self.make_error_array(self.photon_flux_arr,
#                                               verbose)
#
#        self.wavelength_arr = self.make_wavelength_array(self.raw_flux_arr)
#        tqdm.write('Finished constructing wavelength array.')

    def __repr__(self):
        return '{}, {}'.format(self.header['OBJECT'], self.filename.stem)


    def get_header_card(self, flag):
        """
        Return the value of the header card with the given flag.
        """
        return self.header[flag]

    def make_error_array(self, array, verbose=False):
        """
        Construct an error array based on the reported flux values for the
        observation.

        Parameters
        ----------
        array : array_like
            The array to be used as the base to construct the error array from.
            By default this should be the instances' `photon_flux_arr`, which
            will require the `flux_calibrate_array` method to be run first.
        verbose : bool, Default: False
            If *True*, the method will print out how many pixels with negative
            flux were found during the process of constructing the error array
            (with the position of the pixel in the array and its flux) and
            in a summary afterwards stating how many were found.

        Returns
        -------
        NumPy array
            An array with the same shape as the input array containing the
            errors, assuming Poisson statistics for the noise. This is simply
            the square root of the flux in each pixel.
        """
        bad_pixels = 0
        error_arr = np.ones(array.shape)
        for m in range(array.shape[0]):
            for n in range(array.shape[1]):
                if array[m, n] < 0:
                    if verbose:
                        tqdm.write(array[m, n], m, n)
                    error_arr[m, n] = 999
                    bad_pixels += 1
                else:
                    error_arr[m, n] = np.sqrt(array[m, n])
        if verbose:
            tqdm.write('{} pixels with negative flux found.'.
                       format(bad_pixels))
        return error_arr

    def make_wavelength_array(self, array):
        """
        Construct a wavelength array (in Angstroms) for the observation.

        Parameters
        ----------
        array : array_like
            The array to be used to construct the wavelength array. Only the
            shape of the array is actually used so it just needs to be an array
            of the proper shape (72, 4096). The default value uses the
            `raw_flux_arr` array which should always be available.

        Returns
        -------
        NumPy array
            An array of the same shape as the input array specifying the
            wavelength of each pixel in Angstroms.

        Notes
        -----
        The algorithm used is derived from Dumusque 2018 [1]_.

        References
        ----------
        [1] Dumusque, X. "Measuring precise radial velocities on individual
        spectral lines I. Validation of the method and application to mitigate
        stellar activity", Astronomy & Astrophysics, 2018
        """
        wavelength_arr = np.zeros(array.shape) * unyt.angstrom
        for order in trange(0, 72):
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = self.header[coeff]
                for pixel in range(0, 4096):
                    wavelength_arr[order, pixel] += coeff_val *\
                                                    (pixel ** i) *\
                                                    unyt.angstrom
        return wavelength_arr

    def flux_calibrate_array(self, array,
                             card_title='HIERARCH ESO DRS CCD CONAD'):
        """
        Calibrate the raw flux array using the gain to recover the
        photoelectron flux.

        Parameters
        ----------
        array : array_like
            The array containing the raw flux to calibrate. By default this
            will be `self.raw_flux_arr`.
        card_title : str
            A string representing the value of the header card to read to find
            the gain. By default set to the value found in HARPS e2ds files.

        Returns
        -------
        NumPy array
            An array created by multiplying the input array by the gain from
            the file header.
        """
        # Get the gain from the file header:
        gain = self.header[card_title]
        photon_flux_arr = self.raw_flux_arr * gain
        return photon_flux_arr



def readHARPSfile1d(FITSfile, obj=False, wavelenmin=False, date_obs=False,
                    spec_bin=False, med_snr=False, hdnum=False, radvel=False,
                    coeffs=False):
    """Read a HARPS ADP FITS file and return a dictionary of information.

    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    obj : bool, Default: False
        If *True*, the output will contain the contents of the OBJECT FITS
        header card.
    wavelenmin : bool, Default: False
        If *True*, the output will contain the contents of the WAVELMIN FITS
        header card.
    date_obs : bool, Default: False
        If *True*, the output will contain the contents of the DATE-OBS FITS
        header card.
    spec_bin : bool, Default: False
        If *True*, the output will contain the contents of the SPEC_BIN FITS
        header card.
    med_snr : bool, Default: False
        If *True*, the output will contain the contents of the SNR FITS header
        card.
    hdnum : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        HDNUM FITS header card. (Added to unify object identifiers across all
        stars, some of which were occasionally identified by things other than
        HD number.)
    radvel : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        RADVEL FITS header card. (Added to unify the radial velocity for each
        star, as a small minority of stars had different radial velocity
        information in their HIERARCH ESO TEL TAFG RADVEL header cards.)
    coeffs : bool, Default: False
        If *True*, the output will contain the contents of the various
        *ESO DRS CAL TH COEFF LLX* header cards, where *X* ranges from 0 to
        287.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:

        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            The estimated error array (HARPS returns no error array by
            default).

        Optionally
        ==========
        obj : str
            The object name from the 'OBJECT' flag.
        wlmin : float
            The minimum wavelength.
        date_obs : datetime object
            The date the file was observed.
        spec_bin : float
            The wavelength bin size.
        med_snr : float
            The median SNR of the flux array.
        hd_num : str
            The HD identifier of the star in the format "HDxxxxxx".
        radvel : float
            The radial velocity of the star in km/s.
        If the `coeffs` keyword argument is *True*, there will be 288 entries
        of the form "ESO DRS CAL TH COEFF LLX": *value*, where X will range
        from 0 to 287.
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
        # Construct an error array by taking the square root of each flux value
        try:
            # First assume no negative flux values and use Numpy array
            # magic to speed up the process.

            # According to Dumusque 2018 HARPS has a dark-current and read-out
            # noise of 12 photo-electrons, so first add the square of that to
            # the flux, then take the square root to add them in quadrature:
            f_plus_err = f + 144
            e = np.sqrt(f_plus_err)
        except ValueError:
            # If that raises an error, do it element-by-element.
            for i in np.arange(0, len(f), 1):
                if (f[i] > 0.0):
                    e[i] = np.sqrt(f[i] + 144)
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

        # If the coeffs keyword is given, returna all 288 wavelength solution
        # coefficients.
        if coeffs:
            for i in range(0, 288, 1):
                key_string = 'ESO DRS CAL TH COEFF LL{0}'.format(str(i))
                result[key_string] = header0[key_string]

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
