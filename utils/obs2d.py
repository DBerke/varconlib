#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:54:34 2018

@author: dberke

This library contains functions to deal with opening 2D HARPS extracted e2ds
files.
"""

import configparser
import numpy as np
import unyt
from tqdm import tqdm, trange
from astropy.io import fits
from pathlib import Path

config_file = Path('config/variables.cfg')


class HARPSFile2D(object):
    """
    Class to contain data from a HARPS 2D extracted spectrum file.

    """

    def __init__(self, FITSfile):
        if type(FITSfile) is str:
            self._filename = Path(FITSfile)
        else:
            self._filename = FITSfile
        with fits.open(self._filename) as hdulist:
            self._header = hdulist[0].header
            self._rawData = hdulist[0].data

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self._filename)

    def __str__(self):
        return '{}, {}'.format(self._header['OBJECT'], self._filename.stem)

    def getHeaderCard(self, flag):
        """
        Return the value of the header card with the given flag.

        """

        return self._header[flag]

    def plotSelf(self):
        """
        Return a plot of the data.
        """
        pass


class HARPSFile2DScience(HARPSFile2D):
    """
    Subclass of HARPSFile2D to handle observations specifically.
    """
    # TODO: Integrate update keyword
    def __init__(self, FITSfile, update=False):
        if type(FITSfile) is str:
            self._filename = Path(FITSfile)
        else:
            self._filename = FITSfile
        if update:
            mode = 'update'
        else:
            mode = 'append'
        with fits.open(self._filename, mode=mode) as hdulist:
            self._header = hdulist[0].header
            self._rawData = hdulist[0].data
            self._rawFluxArray = self._rawData
            self._wavelengthArray = None
            self._photonFluxArray = None
            self._errorArray = None

            # Try to read the wavelength array, or create it if it doesn't
            # exist.
            try:
                self._wavelengthArray = hdulist['WAVE'].data * unyt.angstrom
            except KeyError:
                if update:
                    print("File opened in 'update' mode but no arrays exist!")
                    raise RuntimeError
                tqdm.write('Writing new wavelength HDU.')
                self.writeWavelengthHDU(hdulist)
            # If we're updating the file, overwrite what exists.
            if update:
                print('Overwriting wavelength HDU.')
                self.writeWavelengthHDU(hdulist)

            # Try to read the flux array, or create it if it doesn't exist.
            try:
                self._photonFluxArray = hdulist['FLUX'].data
            except KeyError:
                if update:
                    print("File opened in 'update' mode but no arrays exist!")
                    raise RuntimeError
                self.writePhotonFluxHDU(hdulist)
                tqdm.write('Writing new photon flux HDU.')
            # If we're updating the file, overwrite what exists.
            if update:
                print('Overwriting photon flux HDU.')
                self.writePhotonFluxHDU(hdulist)

            # Try to read the error array, or create it if it doesn't exist.
            try:
                self._errorArray = hdulist['ERR'].data
            except KeyError:
                if update:
                    print("File opened in 'update' mode but no arrays exist!")
                    raise RuntimeError
                self.writeErrorHDU(hdulist)
                tqdm.write('Writing new error HDU.')
            # If we're updating the file, overwrite what exists.
            if update:
                print('Overwriting error array HDU.')
                self.writeErrorHDU(hdulist)

    def calibrateSelf(self, verbose=False):
        """
        Create error and wavelength arrays for the observation and convert from
        ADUs to photons.

        Will only create the arrays and do the calibration if they haven't
        already been done, so calling it multiple times should't cause any
        additional slowdown.

        """

        if self._photonFluxArray is None:
            # Calibrate ADUs to photoelectrons using the gain
            self._photonFluxArray = self.getPhotonFluxArray(self._rawFluxArray)

        if self._errorArray is None:
            # Construct an error array using the photon flux in each pixel
            self._errorArray = self.getErrorArray(self._photonFluxArray,
                                                  verbose=verbose)
        if self._wavelengthArray is None:
            # Create a wavelength array using the headers in the file
            self._wavelengthArray = self.getWavelengthArray(self._rawFluxArray)

    def getBlazeFile(self):
        """Find and return the blaze file associated with this observation.

        Returns
        -------

        """

        blaze_file = self.getHeaderCard('HIERARCH ESO DRS BLAZE FILE')

        config = configparser.ConfigParser(interpolation=configparser.
                                           ExtendedInterpolation())
        config.read(config_file)

        blaze_file_dir = Path(config['PATHS']['blaze_file_dir'])
        blaze_file_path = blaze_file_dir / blaze_file
        print(blaze_file_path)

    def blazeCorrectSelf(self, blaze_file):
        """
        Blaze-correct an observation using a separate blaze file.

        Parameters
        ----------
        blazefile : HARPSFile2D instance
            A blaze file imported as a HARPSFile2D instance.

        """

        assert 'blaze' in str(blaze_file), "'blaze' not found in file name!"

        if (self._photonFluxArray is None) or (self._errorArray is None):
            self.calibrateSelf()

        self._photonFluxArray /= blaze_file._rawData
        self._errorArray /= blaze_file._rawData

    def getWavelengthArray(self):
        """
        Construct a wavelength array (in Angstroms) for the observation.

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

        source_array = self._rawFluxArray
        wavelength_array = np.zeros(source_array.shape) * unyt.angstrom
        for order in trange(0, 72, total=72, unit='orders'):
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = self._header[coeff]
                for pixel in range(0, 4096):
                    wavelength_array[order, pixel] += coeff_val *\
                                                      (pixel ** i) *\
                                                      unyt.angstrom
        return wavelength_array

    def getPhotonFluxArray(self, card_title='HIERARCH ESO DRS CCD CONAD'):
        """
        Calibrate the raw flux array using the gain to recover the
        photoelectron flux.

        Parameters
        ----------
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

        source_array = self._rawFluxArray
        # Get the gain from the file header:
        gain = self._header[card_title]
        assert type(gain) == float, f"Gain value is a {type(gain)}!"
        photon_flux_array = source_array * gain
        return photon_flux_array

    def getErrorArray(self, verbose=False):
        """
        Construct an error array based on the reported flux values for the
        observation.

        Parameters
        ----------
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

        # If the photon flux array doesn't exist yet, create it.
        if self._photonFluxArray is None:
            self._photonFluxArray = self.getPhotonFluxArray()
        source_array = self._photonFluxArray
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

    def writeWavelengthHDU(self, hdulist):
        """
        Write out a wavelength array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        """

        self._wavelengthArray = self.getWavelengthArray()

        # Create an HDU for the wavelength array.
        wavelength_HDU = fits.ImageHDU(data=self._wavelengthArray,
                                       name='WAVE')

        # Add some cards to its header containing the minimum and maximum
        # wavelengths in each order.
        wavelength_cards = []
        for i in range(0, 72):
            for kind, pos in zip(('min', 'max'), (0, -1)):
                keyword = f'ORD{i}{kind.upper()}'
                value = '{:.3f}'.format(self._wavelengthArray[i, pos])
                comment = '{} wavelength of order {}'.format(kind.capitalize(),
                                                             i)
                wavelength_cards.append((keyword, value, comment))
        wavelength_HDU.header.extend(wavelength_cards)
        try:
            hdulist['WAVE'] = wavelength_HDU
        except KeyError:
            hdulist.append(wavelength_HDU)
        hdulist.flush(output_verify="exception", verbose=True)

    def writePhotonFluxHDU(self, hdulist):
        """
        Write out a photon flux array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        """

        self._photonFluxArray = self.getPhotonFluxArray()

        # Create an HDU for the photon flux array.
        photon_flux_HDU = fits.ImageHDU(data=self._photonFluxArray,
                                        name='FLUX')
        try:
            hdulist['FLUX'] = photon_flux_HDU
        except KeyError:
            hdulist.append(photon_flux_HDU)
        hdulist.flush(output_verify="exception", verbose=True)

    def writeErrorHDU(self, hdulist):
        """
        Write out an error array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        """

        self._errorArray = self.getErrorArray()

        # Create an HDU for the error array.
        error_HDU = fits.ImageHDU(data=self._errorArray,
                                  name='ERR')
        try:
            hdulist['ERR'] = error_HDU
        except KeyError:
            hdulist.append(error_HDU)
        hdulist.flush(output_verify="exception", verbose=True)

    def findWavelength(self, wavelength=None, unit=None):
        """
        Find which orders contain a given wavelength.

        This function will return the indices of the wavelength orders that
        contain the given wavelength. The result will be a length-1 or -2 tuple
        containing integers in the range [0, 71].

        Parameters
        ----------
        wavelength : unyt.array.unyt_quantity
            The wavelength to find in the wavelength array. This should be a
            unyt.array.unyt_quantity object of length 1.

        Returns
        -------
        tuple
            A tuple of ints of length 1 or 2, representing the indices of the
            orders in which the input wavelength is found. The integers will
            be in the range [0, 7].

        """

        wavelength_to_find = wavelength.to(unyt.angstrom)

        # Create the wavelength array if it doesn't exist yet for some reason.
        if self._wavelengthArray is None:
            self._wavelengthArray = self.getWavelengthArray()

        # Make sure the wavelength to find is in the array in the first place.
        array_min = self._wavelengthArray[0, 0]
        array_max = self._wavelengthArray[71, -1]
        assert array_min <= wavelength_to_find <= array_max,\
            "Given wavelength not found within array limits! {}, {}".format(
                    array_min, array_max)

        # Set up a list to hold the indices of the orders where the wavelength
        # is found.
        orders_wavelength_found_in = []
        for order in range(0, 72):
            order_min = self._wavelengthArray[order, 0]
            order_max = self._wavelengthArray[order, -1]
            if order_min <= wavelength_to_find <= order_max:
                orders_wavelength_found_in.append(order)
        assert len(orders_wavelength_found_in) < 3,\
            "Wavelength found in more than two orders!"
        return tuple(orders_wavelength_found_in)

    def plotOrder(self, index, passed_axis, **kwargs):
        """
        Plot a single order of the data, given its index.

        Parameters
        ----------
        index : int
            An integer in the range [0, 71] representing the index of the
            order to plot.
        passed_axis : a matplotlib Axes instance
            The order specified will be plotted onto this Axes object.
        **kwargs
            Any additional keyword arguments are passed on to matplotlib's
            `plot` function.

        """

        # Check that the index is correct.
        assert 0 <= index <= 71, "Index is not in [0, 71]!"

        if (self._wavelengthArray is None) or (self._errorArray is None):
            self.calibrateSelf()
        ax = passed_axis

        # Plot onto the given axis.
        ax.plot(self._wavelengthArray[index], self._photonFluxArray[index],
                **kwargs)
