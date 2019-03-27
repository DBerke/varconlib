#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:54:34 2018

@author: dberke

This library contains functions to deal with opening 2D HARPS extracted e2ds
files.
"""

import configparser
from copy import copy
import numpy as np
import unyt as u
import varconlib as vcl
from tqdm import tqdm, trange
from astropy.io import fits
from pathlib import Path
from conversions import air2vacESO

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)


class HARPSFile2D(object):
    """Class to contain data from a HARPS 2D extracted spectrum file.

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
        Get the value of the header card with the given flag.

        Parameters
        ----------
        flag : str
            The key of of the FITS header to get the value of.

        Returns
        -------
        str
            The value of the FITS header associated with the given key.

        """

        return self._header[flag]

    def plotSelf(self):
        """
        Return a plot of the data.
        """
        # TODO: Implement a plot system.
        pass


class HARPSFile2DScience(HARPSFile2D):
    """Subclass of HARPSFile2D to handle observations specifically.

    """

    def __init__(self, FITSfile, update=[]):
        """Parse a given FITS file containing an observation into a usable
        HARPSFile2DScience object.

        Parameters
        ----------
        FITSfile : string or `pathlib.Path` object
            Represents the location of the file to read. If given as a string
            it will be converted to a Path object internally.

        Optional
        --------
        update : collection of strings
            Whether to force writing of the wavelength, flux, and error arrays.
            `update` may be given either as a boolean or a list of strings
            containing some or all of 'WAVE', 'BARY', 'FLUX', or 'ERR'.

            Setting it as *False* will cause the initialization process to use
            the values already present in the file being opened. Setting it to
            *True* will cause all HDUs to be updated.

            Any HDUs listed in the tuple will be updated, while those not
            listed will be read from the file being opened. Updating the
            wavelength solution usually takes the most time, so if it doesn't
            need to be changed passing a tuple of ('FLUX', 'ERR') will update
            the FLUX and ERR HDUs without performing the time-costly updating
            of the wavelength solution.

        """

        if type(FITSfile) is str:
            self._filename = Path(FITSfile)
        else:
            self._filename = FITSfile
        if not self._filename.exists():
            print(self._filename)
            raise FileNotFoundError('The given path does not exist!')
        if update:
            file_open_mode = 'update'
        else:
            file_open_mode = 'append'
        hdulist = fits.open(self._filename, mode=file_open_mode)
        self._header = hdulist[0].header
        self._rawData = hdulist[0].data
        self._rawFluxArray = copy(self._rawData)
        self._blazeFile = None

        # Since we may not have the blaze files on hand, only try to find
        # them if we really need them, i.e. when opening a file for the
        # first time or when explicitly updating it.
        if (len(hdulist) == 1) or file_open_mode == 'update':
            self._blazeFile = self.getBlazeFile()

        # Try to read the wavelength array, or create it if it doesn't
        # exist.
        err_str = "File opened in 'update' mode but no arrays exist!"
        try:
            self._wavelengthArray = hdulist['WAVE'].data * u.angstrom
        except KeyError:
            if ('ALL' in update) or ('WAVE' in update):
                raise RuntimeError(err_str)
            tqdm.write('Writing new wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action='warn')
        # If we're updating the file, overwrite the existing wavelengths.
        if ('ALL' in update) or ('WAVE' in update):
            tqdm.write('Overwriting wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action='warn')

        # Try to read the barycentric-vacuum wavelength array, or create it if
        # if doesn't exist yet.
        try:
            self._barycentricArray = hdulist['BARY'].data * u.angstrom
        except KeyError:
            if ('ALL' in update) or ('BARY' in update):
                raise RuntimeError(err_str)
            tqdm.write('Writing new barycentric wavelength HDU.')
            self.writeBarycentricHDU(hdulist, verify_action='warn')
        if ('ALL' in update) or ('BARY' in update):
            tqdm.write('Overwriting barycentric wavelength HDU.')
            self.writeBarycentricHDU(hdulist, verify_action='warn')

        # Try to read the flux array, or create it if it doesn't exist.
        try:
            self._photonFluxArray = hdulist['FLUX'].data
        except KeyError:
            if ('ALL' in update) or ('FLUX' in update):
                raise RuntimeError(err_str)
            self.writePhotonFluxHDU(hdulist, verify_action='warn')
            tqdm.write('Writing new photon flux HDU.')
        # If we're updating the file, overwrite the existing fluxes.
        if ('ALL' in update) or ('FLUX' in update):
            tqdm.write('Overwriting photon flux HDU.')
            self.writePhotonFluxHDU(hdulist, verify_action='warn')

        # Try to read the error array, or create it if it doesn't exist.
        try:
            self._errorArray = hdulist['ERR'].data
        except KeyError:
            if ('ALL' in update) or ('ERR' in update):
                raise RuntimeError(err_str)
            self.writeErrorHDU(hdulist, verify_action='warn')
            tqdm.write('Writing new error HDU.')
        # If we're updating the file, overwrite the existing uncertainties.
        if ('ALL' in update) or ('ERR' in update):
            tqdm.write('Overwriting error array HDU.')
            self.writeErrorHDU(hdulist, verify_action='warn')

        # Try to read the blaze array, or create it if it doesn't exist.
        try:
            self._blazeArray = hdulist['BLAZE'].data
        except KeyError:
            if ('ALL' in update) or ('BLAZE' in update):
                raise RuntimeError(err_str)
            self.writeBlazeHDU(hdulist, verify_action='warn')
            tqdm.write('Writing new blaze HDU')
        # If we're updating the file, overwrite the existing uncertainties.
        if ('ALL' in update) or ('BLAZE' in update):
            tqdm.write('Overwriting blaze array HDU.')
            self.writeBlazeHDU(hdulist, verify_action='warn')
        hdulist.close(output_verify='warn')

    @property
    def wavelengthArray(self):
        if not hasattr(self, '_wavelengthArray'):
            print('Creating wavelength array.')
            self._wavelengthArray = self.getWavelengthArray()
        return self._wavelengthArray

    @property
    def vacuumArray(self):
        if not hasattr(self, '_vacuumArray'):
            print('Creating vacuum wavelength array.')
            self._vacuumArray = self.getVacuumArray()
        return self._vacuumArray

    @property
    def barycentricArray(self):
        if not hasattr(self, '_barycentricArray'):
            print('Creating barycentric vacuum wavelength array.')
            self._barycentricArray = self.getBarycentricArray()
        return self._barycentricArray

    @property
    def photonFluxArray(self):
        if not hasattr(self, '_photonFluxArray'):
            print('Generating photon flux array.')
            self._photonFluxArray = self.getPhotonFluxArray()
        return self._photonFluxArray

    @property
    def errorArray(self):
        if not hasattr(self, '_errorArray'):
            print('Generating error array.')
            self._errorArray = self.getErrorArray()
        return self._errorArray

    @property
    def blazeArray(self):
        if not hasattr(self, '_blazeArray'):
            print('Generating blaze array.')
            self._blazeArray = self.getBlazeArray()
        return self._blazeArray

    @property
    def radialVelocity(self):
        if not hasattr(self, '_radialVelocity'):
            try:
                rv_card = 'HIERARCH ESO TEL TARG RADVEL'
                self._radialVelocity = u.km / u.s * \
                                       float(self.getHeaderCard(rv_card))
            except KeyError:
                print('No radial velocity card found for this observation!')
                raise
        return self._radialVelocity

    @property
    def BERV(self):
        # BERV = Barycentric Earth Radial Velocity
        if not hasattr(self, '_BERV'):
            try:
                berv_card = 'HIERARCH ESO DRS BERV'
                self._BERV = float(self.getHeaderCard(berv_card)) * u.km / u.s
            except KeyError:
                print('No BERV card found for this observation!')
                raise
        return self._BERV

    def getBlazeFile(self):
        """Find and return the blaze file associated with this observation.

        Returns
        -------
        obs2d.HARPSFile2D object
            A HARPSFile2D object created from the blaze file associated with
            this observation via its header card.

        """

        try:
            blaze_file = self.getHeaderCard('HIERARCH ESO DRS BLAZE FILE')
        except KeyError:
            raise KeyError('No blaze file listed for this observation!')

        file_date = blaze_file[6:16]

        blaze_file_dir = Path(config['PATHS']['blaze_file_dir'])
        blaze_file_path = blaze_file_dir /\
            'data/reduced/{}'.format(file_date) /\
            blaze_file

        if not blaze_file_path.exists():
            tqdm.write(str(blaze_file_path))
            raise RuntimeError("Blaze file path doesn't exist!")

        return HARPSFile2D(blaze_file_path)

    def getWavelengthArray(self):
        """Construct a wavelength array (in Angstroms) for the observation.

        By default, the wavelength array returned using the coefficients in
        the headers are in air wavelengths, and uncorrected for the Earth's
        barycentric motion.

        Returns
        -------
        `numpy.ndarray`
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
        wavelength_array = np.zeros(source_array.shape, dtype=float)
        for order in trange(0, 72, total=72, unit='orders'):
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = self._header[coeff]
                for pixel in range(0, 4096):
                    wavelength_array[order, pixel] += coeff_val * (pixel ** i)

        return wavelength_array * u.angstrom

    def getVacuumArray(self):
        """Correct the calculated air wavelength array back into vacuum.

        Returns
        -------
        `unyt.unyt_array`
            The wavelength array for the observation converted into vacuum
            wavelengths using the Edlen 1953 formula used by the HARPS
            pipeline.

        """

        vacuumArray = air2vacESO(self.getWavelengthArray())

        return vacuumArray

    def getBarycentricArray(self):
        """Correct the vacuum wavelength array by the barycentric Earth radial
        velocity (BERV).

        Returns
        -------
        `unyt.unyt_array`
            The vacuum wavelength array in barycentric coordinates.

        """

        barycentricArray = self.shiftWavelengthArray(self.getVacuumArray(),
                                                     self.BERV)
        return barycentricArray

    def getPhotonFluxArray(self):
        """Calibrate the raw flux array using the gain, then correct it using
        the correct blaze file to recover the photoelectron flux.

        Returns
        -------
        `numpy.ndarray`
            An array created by multiplying the input array by the gain from
            the file header.

        """

        # Blaze-correct the photon flux array:
        photon_flux_array = self._rawFluxArray / self.getBlazeArray()

        return photon_flux_array

    def getErrorArray(self):
        """Construct an error array based on the reported flux values for the
        observation, then blaze-correct it.

        Returns
        -------
        NumPy array
            An array with the same shape as the input array containing the
            errors, assuming Poisson statistics for the noise. This is simply
            the square root of the flux in each pixel.

        """

        # According to Dumusque 2018 HARPS has a dark-current and read-out
        # noise of 12 photo-electrons.
        dark_noise = 12

        photon_flux_array = self._rawFluxArray
        error_array = np.array([[np.sqrt(x + dark_noise ** 2) if x >= 0
                                 else 1e5 for x in row]
                                for row in photon_flux_array], dtype=float)

        # Correct the error array by the blaze function:
        error_array = error_array / self.blazeArray

        return error_array

    def getBlazeArray(self):
        """Return the blaze function array from the blaze file associated with
        the observation.

        Returns
        -------
        `np.ndarray`
            A (72, 4096) array containing the blaze function value at each
            point in the CCD.
        """

        if not hasattr(self, '_blazeFile') or self._blazeFile is None:
            self._blazeFile = self.getBlazeFile()
        blaze_array = self._blazeFile._rawData

        return blaze_array

    def writeWavelengthHDU(self, hdulist, verify_action='exception'):
        """Write out a wavelength array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        Optional
        --------
        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            <http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._wavelengthArray = self.getWavelengthArray()
        # Create an HDU for the wavelength array.
        wavelength_HDU = fits.ImageHDU(data=self.wavelengthArray, name='WAVE')
        try:
            hdulist['WAVE'] = wavelength_HDU
        except KeyError:
            hdulist.append(wavelength_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def writeBarycentricHDU(self, hdulist, verify_action='exception'):
        """Write out an array of barycentric vacuum wavelengths to the
        currently-opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        Optional
        --------
        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            <http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._barycentricArray = self.getBarycentricArray()
        barycentric_HDU = fits.ImageHDU(data=self.barycentricArray,
                                        name='BARY')
        try:
            hdulist['BARY'] = barycentric_HDU
        except KeyError:
            hdulist.append(barycentric_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def writePhotonFluxHDU(self, hdulist, verify_action='exception'):
        """Write out a photon flux array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.


        Optional
        --------
        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            <http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._photonFluxArray = self.getPhotonFluxArray()
        # Create an HDU for the photon flux array.
        photon_flux_HDU = fits.ImageHDU(data=self.photonFluxArray, name='FLUX')
        try:
            hdulist['FLUX'] = photon_flux_HDU
        except KeyError:
            hdulist.append(photon_flux_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def writeErrorHDU(self, hdulist, verify_action='exception'):
        """Write out an error array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            <http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._errorArray = self.getErrorArray()
        # Create an HDU for the error array.
        error_HDU = fits.ImageHDU(data=self.errorArray, name='ERR')
        try:
            hdulist['ERR'] = error_HDU
        except KeyError:
            hdulist.append(error_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def writeBlazeHDU(self, hdulist, verify_action='exception'):
        """Write out a blaze function array to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            <http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._blazeArray = self.getBlazeArray()
        blaze_HDU = fits.ImageHDU(data=self.blazeArray, name='BLAZE')

        try:
            hdulist['BLAZE'] = blaze_HDU
        except KeyError:
            hdulist.append(blaze_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def shiftWavelengthArray(self, wavelength_array, shift_velocity):
        """Doppler shift a wavelength array by an amount equivalent to a given
        velocity.

        Parameters
        ----------
        wavelength_array : unyt_array
            An array containing wavelengths to be Doppler shifted. Needs units
            of dimension length.

        velocity : unyt_quantity
            A Unyt quantity with dimensions length/time to shift the wavelength
            array by.

        Returns
        -------
        Unyt unyt_array
            An array of the same shape as the given array, Doppler shifted by
            the given radial velocity.

        """

        return vcl.shift_wavelength(wavelength=wavelength_array,
                                    shift_velocity=shift_velocity)

    def findWavelength(self, wavelength=None, mid_most=True):
        """Find which orders contain a given wavelength.

        This function will return the indices of the wavelength orders that
        contain the given wavelength. The result will be a length-1 or -2 tuple
        containing integers in the range [0, 71].

        Parameters
        ----------
        wavelength : unyt_quantity
            The wavelength to find in the wavelength array. This should be a
            unyt_quantity object of length 1.

        mid_most : bool, Default : True
            In a 2D extracted echelle spectrograph like HARPS, a wavelength
            near the ends of an order can appear a second time in an adjacent
            order. By default `findWavelength` will return only the single
            order where the wavelength is closest to the geometric center of
            the CCD, which corresponds to the point where the signal-to-noise
            ratio is highest. Setting this to *False* will allow for the
            possibility of a length-2 tuble being returned containing the
            numbers of both orders a wavelength is found in.

        Returns
        -------
        If mid_most is false: tuple
            A tuple of ints of length 1 or 2, representing the indices of
            the orders in which the input wavelength is found.

        If mid_most is true: int
            An int representing the order in which the wavelength found is
            closest to the geometrical center.

            In both cases the integers returned will be in the range [0, 71].

        """

        wavelength_to_find = wavelength.to(u.angstrom)

        # Make sure the wavelength to find is in the array in the first place.
        array_min = self.barycentricArray[0, 0]
        array_max = self.barycentricArray[-1, -1]
        assert array_min <= wavelength_to_find <= array_max,\
            "Given wavelength not in array limits! ({}, {})".format(array_min,
                                                                    array_max)

        # Set up a list to hold the indices of the orders where the wavelength
        # is found.
        orders_wavelength_found_in = []
        for order in range(0, 72):
            order_min = self.wavelengthArray[order, 0]
            order_max = self.wavelengthArray[order, -1]
            if order_min <= wavelength_to_find <= order_max:
                orders_wavelength_found_in.append(order)
                if len(orders_wavelength_found_in) == 1:
                    continue
                elif len(orders_wavelength_found_in) == 2:
                    break

        if mid_most:
            # Only one array: great, return it.
            if len(orders_wavelength_found_in) == 1:
                return orders_wavelength_found_in[0]

            # Found in two arrays: figure out which is closer to the geometric
            # center of the CCD, which conveiently falls around the middle
            # of the 4096-element array.
            elif len(orders_wavelength_found_in) == 2:
                order1, order2 = orders_wavelength_found_in
                index1 = vcl.wavelength2index(wavelength_to_find,
                                              self.barycentricArray[order1])
                index2 = vcl.wavelength2index(wavelength_to_find,
                                              self.barycentricArray[order2])
                # Check which index is closest to the pixel in the geometric
                # center of the 4096-length array, given 0-indexing in Python.
                if abs(index1 - 2047.5) > abs(index2 - 2047.5):
                    return order2
                else:
                    return order1
        else:
            return tuple(orders_wavelength_found_in)

    def plotOrder(self, index, passed_axis, **kwargs):
        """Plot a single order of the data, given its index.

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

        ax = passed_axis

        # Plot onto the given axis.
        ax.plot(self._wavelengthArray[index], self._photonFluxArray[index],
                **kwargs)
