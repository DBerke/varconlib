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


class HARPSFile2D(object):
    """
    Class to contain data from a HARPS 2D extracted spectrum file.
    """
    def __init__(self, FITSfile):
        if type(FITSfile) is str:
            self.filename = Path(FITSfile)
        else:
            self.filename = FITSfile
        with fits.open(self.filename) as hdulist:
            self.header = hdulist[0].header
            self.data = hdulist[0].data
        self.header_cards = {}

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.filename)

    def __str__(self):
        return '{}, {}'.format(self.header['OBJECT'], self.filename.stem)

    def get_header_card(self, flag):
        """
        Return the value of the header card with the given flag.
        """
        try:
            value = self.header_cards[flag]
        except KeyError:
            value = self.header[flag]
            self.header_cards[flag] = value
        return value


class HARPSFile2DScience(HARPSFile2D):
    """
    Subclass of HARPSFile2D to handle observations specifically.
    """
    def __init__(self, blazefile=None):
        super().__init__()
        self._raw_flux_array = self.data
        self._wavelength_array = None
        self._photon_flux_array = None
        self._error_array = None
#        if blazefile:
#            self.blaze_correct(self, blazefile)

    def calibrate_self(self, verbose=False):
        """
        Calibrate the data and create error and wavelength arrays for it.

        """

        # Calibrate ADUs to photoelectrons using the gain
        self._photon_flux_array = self.get_photon_flux_array(self.
                                                             _raw_flux_array)

        # Construct an error array using the photon flux in each pixel
        self._error_arr = self.get_error_array(self._photon_flux_array,
                                               verbose=verbose)

        # Create a wavelengtha array using the headers in the file
        self._wavelength_arr = self.get_wavelength_array(self._raw_flux_array)

    def blaze_correct(self, blazefile):
        """
        Blaze-correct an observation using a separate blaze file.

        Parameters
        ----------
        blazefile : HARPSFile2D instance
            A blaze file imported as a HARPSFile2D instance.

        """

        self._photon_flux_arr / blazefile.data
        self._error_arr / blazefile.data

    def make_wavelength_array(self, source_array):
        """
        Construct a wavelength array (in Angstroms) for the observation.

        Parameters
        ----------
        source_array : array_like
            The array to be used to construct the wavelength array. Only the
            shape of the array is actually used so it just needs to be an array
            of the proper shape (72, 4096). The default value uses the
            `_raw_flux_array` array which should always be available.

        Returns
        -------
        NumPy array
            An array of the same shape as the input array specifying the
            wavelength of each pixel (element in the array) in Angstroms.

        Notes
        -----
        The algorithm used is derived from Dumusque 2018 [1]_.

        References
        ----------
        [1] Dumusque, X. "Measuring precise radial velocities on individual
        spectral lines I. Validation of the method and application to mitigate
        stellar activity", Astronomy & Astrophysics, 2018

        """

        wavelength_array = np.zeros(source_array.shape) * unyt.angstrom
        for order in trange(0, 72, total=72, unit='orders'):
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = self.header[coeff]
                for pixel in range(0, 4096):
                    wavelength_array[order, pixel] += coeff_val *\
                                                      (pixel ** i) *\
                                                      unyt.angstrom
        return wavelength_array

    def get_photon_flux_array(self, source_array=None,
                              card_title='HIERARCH ESO DRS CCD CONAD'):
        """
        Calibrate the raw flux array using the gain to recover the
        photoelectron flux.

        Parameters
        ----------
        source_array : array_like
            The array containing the raw flux to calibrate. If none is given
            will default to `self._raw_flux_array`.
        card_title : str
            A string representing the value of the header card to read to find
            the gain. By default set to the value found in HARPS e2ds files,
            'HIERARCH ESO DRS CCD CONAD'.

        Returns
        -------
        NumPy array
            An array created by multiplying the input array by the gain from
            the file header.

        """

        if source_array is None:
            source_array = self._raw_flux_array
        # Get the gain from the file header:
        gain = self.header[card_title]
        assert type(gain) == float, f"Gain value is a {type(gain)}!"
        photon_flux_array = source_array * gain
        return photon_flux_array

    def get_error_array(self, source_array=None, verbose=False):
        """
        Construct an error array based on the reported flux values for the
        observation.

        Parameters
        ----------
        source_array : array_like
            The array to be used as the base to construct the error array from.
            Will default to the instances' `_photon_flux_array`.
        verbose : bool, Default: False
            If *True*, the method will print out how many pixels with negative
            flux were found during the process of constructing the error array
            (with the position of the pixel in the array and its flux) and
            in a summary afterwards state how many were found.

        Returns
        -------
        NumPy array
            An array with the same shape as the input array containing the
            errors, assuming Poisson statistics for the noise. This is simply
            the square root of the flux in each pixel.

        """

        if source_array is None:
            if self.photon_flux_array is None:
                source_array = self.get_photon_flux_array()
            else:
                source_array = self._photon_flux_array
        bad_pixels = 0
        error_array = np.ones(source_array.shape)
        for m in range(source_array.shape[0]):
            for n in range(source_array.shape[1]):
                if source_array[m, n] < 0:
                    if verbose:
                        tqdm.write(source_array[m, n], m, n)
                    error_array[m, n] = 1e5
                    bad_pixels += 1
                else:
                    error_array[m, n] = np.sqrt(source_array[m, n])
        if verbose:
            tqdm.write('{} pixels with negative flux found.'.
                       format(bad_pixels))
        return error_array


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
