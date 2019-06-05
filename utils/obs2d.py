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
from math import isnan
from pathlib import Path

from astropy.io import fits
import numpy as np
from tqdm import tqdm, trange
import unyt as u

from conversions import air2vacESO
import varconlib as vcl
from exceptions import BadRadialVelocityError


config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# Read some path variables from the config file.
blaze_file_dir = Path(config['PATHS']['blaze_file_dir'])
pixel_geom_files_dir = Path(config['PATHS']['pixel_geom_files_dir'])
wavelength_cals_dir = Path(config['PATHS']['wavelength_cal_dir'])


class HARPSFile2D(object):
    """Class to contain data from a HARPS 2D extracted spectrum file.

    """

    def __init__(self, FITSfile):
        if type(FITSfile) is str:
            self._filename = Path(FITSfile)
        else:
            self._filename = FITSfile
        if not self._filename.exists():
            tqdm.write('Given filename does not exist!')
            tqdm.write(self._filename)
            raise FileNotFoundError
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
            The key of the FITS header to get the value of.

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

    def __init__(self, FITSfile, update=[], use_new_coefficients=True,
                 use_pixel_positions=True):
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
            self._blazeFile = HARPSFile2D(self.getBlazeFile())

        # Define an error string for updating a file that hasn't been opened
        # previously
        err_str = "File opened in 'update' mode but no arrays exist!"

        # Try to read the wavelength array, or create it if it doesn't
        # exist.
        # ??? See if these can be covered with a wrapper.
        try:
            self._wavelengthArray = hdulist['WAVE'].data * u.angstrom
        except KeyError:
            if ('ALL' in update) or ('WAVE' in update):
                raise RuntimeError(err_str)
            tqdm.write('Writing new wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action='warn',
                                    use_new_coefficients=use_new_coefficients,
                                    use_pixel_positions=use_pixel_positions)
        # If we're updating the file, overwrite the existing wavelengths.
        if ('ALL' in update) or ('WAVE' in update):
            tqdm.write('Overwriting wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action='warn',
                                    use_new_coefficients=use_new_coefficients,
                                    use_pixel_positions=use_pixel_positions)

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
            tqdm.write('No wavelength array, something went wrong!')
#            print('Creating wavelength array.')
#            self._wavelengthArray = self.getWavelengthArray()
        return self._wavelengthArray

    @property
    def vacuumArray(self):
        if not hasattr(self, '_vacuumArray'):
            tqdm.write('Creating vacuum wavelength array.')
            self._vacuumArray = self.getVacuumArray()
        return self._vacuumArray

    @property
    def barycentricArray(self):
        if not hasattr(self, '_barycentricArray'):
            tqdm.write('Creating barycentric vacuum wavelength array.')
            self._barycentricArray = self.getBarycentricArray()
        return self._barycentricArray

    @property
    def photonFluxArray(self):
        if not hasattr(self, '_photonFluxArray'):
            tqdm.write('Generating photon flux array.')
            self._photonFluxArray = self.getPhotonFluxArray()
        return self._photonFluxArray

    @property
    def errorArray(self):
        if not hasattr(self, '_errorArray'):
            tqdm.write('Generating error array.')
            self._errorArray = self.getErrorArray()
        return self._errorArray

    @property
    def blazeArray(self):
        if not hasattr(self, '_blazeArray'):
            tqdm.write('Generating blaze array.')
            self._blazeArray = self.getBlazeArray()
        return self._blazeArray

    @property
    def radialVelocity(self):
        if not hasattr(self, '_radialVelocity'):
            try:
                rv_card = 'HIERARCH ESO TEL TARG RADVEL'
                radial_velocity = float(self.getHeaderCard(rv_card)) * \
                    u.km / u.s
            except KeyError:
                print('No radial velocity card found for this observation!')
                raise

            if isnan(radial_velocity):
                raise BadRadialVelocityError('Radial velocity is NaN!')
            if abs(radial_velocity) > 5000:
                print(radial_velocity)
                raise BadRadialVelocityError('Radial velocity is suspiciously'
                                             ' high! {}'.format(
                                                     radial_velocity))
            self._radialVelocity = radial_velocity
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
        `pathlib.Path` object
            A `Path` object containing the path to the blaze file associated
            with this observation via its header card.

        """

        try:
            blaze_file = self.getHeaderCard('HIERARCH ESO DRS BLAZE FILE')
        except KeyError:
            raise KeyError('No blaze file listed for this observation!')

        file_date = blaze_file[6:16]

        blaze_file_path = blaze_file_dir /\
            'data/reduced/{}'.format(file_date) /\
            blaze_file

        if not blaze_file_path.exists():
            tqdm.write(str(blaze_file_path))
            raise FileNotFoundError("Blaze file path doesn't exist!")

        return blaze_file_path

    def getPixelPositionGeomFile(self):
        """Return the path ot the pixel position geometry file.

        The pixel sizes in the HARPS CCDs are not entirely uniform across their
        widths, as shown in Coffinet et al. 2019 [1]_. This leads to systematic
        erros in the wavelength calibration, which by default assumes perfectly
        regular pixel sizes at all locations. This method retrieves information
        on the pixel center positions across the CCDs [2]_ in the form of a
        72x4096 array corresponding to the shape of an exracted 2D spectrum.

        Returns
        -------
        `pathlib.Path` object
            A `Path` object pointing to the pixel position geometry file
            provided by C. Lovis.

        References
        ----------
        [1] A. Coffinet, C. Lovis, X. Dumusque, F. Pepe, "New wavelength
        calibration of the HARPS spectrograph", Astronomy & Astrophysics, 2019

        [2] C. Lovis, private communication.

        """

        pixel_pos_file_path = pixel_geom_files_dir /\
            'pixel_geom_pos_HARPS_2004_A.fits'

        if not pixel_pos_file_path.exists():
            tqdm.write(str(pixel_pos_file_path))
            raise FileNotFoundError("Pixel positions file doesn't exist!")

        return pixel_pos_file_path

    def getPixelSizeGeomFile(self):
        """Return the path to the pixel size geometry file.

        The pixel sizes in the HARPS CCDs are not entirely uniform across their
        widths, as shown in Coffinet et al. 2019 [1]_. This leads to systematic
        erros in the wavelength calibration, which by default assumes perfectly
        regular pixel sizes at all locations. This method retrieves information
        on the pixel sizes across the CCDs [2]_ in the form of a 72x4096 array
        corresponding to the shape of an exracted 2D spectrum.

        Returns
        -------
        `pathlib.Path` object
            A `Path` object pinting to the pixel size geometry file
            provided by C. Lovis.

        References
        ----------
        [1] A. Coffinet, C. Lovis, X. Dumusque, F. Pepe, "New wavelength
        calibration of the HARPS spectrograph", Astronomy & Astrophysics, 2019

        [2] C. Lovis, private communication.

        """

        pixel_size_file_path = pixel_geom_files_dir /\
            'pixel_geom_size_HARPS_2004_A.fits'

        if not pixel_size_file_path.exists():
            tqdm.write(str(pixel_size_file_path))
            raise FileNotFoundError("Pixel size file doesn't exist!")

        return pixel_size_file_path

    def getWavelengthCalibrationFile(self):
        """Return the path to the wavelength calibration file associated with
        this observation from its header keyword.

        Returns
        -------
        `pathlib.Path` object
            A `Path` object pointing to the re-calibrated wavelength
            calibration file associated with this observation, which should
            have the same name as the file mentioned in the 'HIERARCH ESO DRS
            CAL TH FILE' header card.

        """

        cal_file_name = self.getHeaderCard("HIERARCH ESO DRS CAL TH FILE")
        cal_file_path = wavelength_cals_dir / cal_file_name

        if not cal_file_path.exists():
            tqdm.write(str(cal_file_path))
            raise FileNotFoundError("Calibration file doesn't exist!")

        return cal_file_path

    def getWavelengthArray(self, use_new_coefficients=True,
                           use_pixel_positions=True):
        """Construct a wavelength array (in Angstroms) for the observation.

        By default, the wavelength array returned using the coefficients in
        the headers are in air wavelengths, and uncorrected for the Earth's
        barycentric motion.

        Optional
        --------
        use_new_coefficients : bool, Default : True
            Whether or not to attempt to use newly-derived wavelength
            calibration coefficients following Coffinet et al. 2019 [1]_.

        use_pixel_positions : bool, Default : True
            Whether or not to use more accurate positions (in 'pixel space')
            for the centers of pixels when evaluating the wavelength solution,
            as detailed in Coffinet et al. 2019 [1]_.

        Returns
        -------
        `numpy.ndarray`
            An array of the same shape as the input array specifying the
            wavelength of each pixel (element in the array) in Angstroms.

        Notes
        -----
        The algorithm used is derived from Dumusque 2018 [2]_:
        .. math::
            \lambda_{i,j}=P_{4\cdot i}+P_{4\cdot i+1}\times j+\\
            P_{4\cdot i+2}\times j^2+P_{4\cdot i+3}\times j^3

        where :math:`\lambda_{i,j}` is the wavelength at pixel *j* in order
        *i*, in ångströms. The `use_pixel_positions` keyword controls whether
        to assume all pixels have the same size (normalized to integer
        positions) or to use the values derived in Coffinet et al. 2019 [1]_

        References
        ----------
        [1] A. Coffinet, C. Lovis, X. Dumusque, F. Pepe, "New wavelength
        calibration of the HARPS spectrograph", Astronomy & Astrophysics, 2019

        [2] X. Dumusque, "Measuring precise radial velocities on individual
        spectral lines I. Validation of the method and application to mitigate
        stellar activity", Astronomy & Astrophysics, 2018

        """

        if use_new_coefficients:
            # Try to use the new coefficients provided, unless there
            # isn't a file containing them.
            try:
                coeffs_file = HARPSFile2D(self.getWavelengthCalibrationFile())
                tqdm.write('Found new calibration file.')
            except FileNotFoundError:
                coeffs_file = self
                tqdm.write('New calibration file could not be found!')
                tqdm.write('Falling back on old calibration coefficients.')
        else:
            coeffs_file = self

        if use_pixel_positions:
            # Use the new pixel positions file provided.
            pixel_pos_file = HARPSFile2D(self.getPixelPositionGeomFile())
            pixel_positions = pixel_pos_file._rawData
        else:
            pixel_positions = np.array([[x for x in range(0, 4096)]
                                       for row in range(0, 72)])

        source_array = self._rawFluxArray
        wavelength_array = np.zeros(source_array.shape, dtype=float)
        # Step through the 72 spectral orders
        for order in trange(0, 72, total=72, unit='orders'):
            # Iterate through the third-order fit coefficients
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = coeffs_file.getHeaderCard(coeff)
                for pixel, pixel_pos in enumerate(pixel_positions[order, :]):
                    wavelength_array[order, pixel] += coeff_val *\
                                                      (pixel_pos ** i)

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

        vacuumArray = air2vacESO(self.wavelengthArray)

        return vacuumArray

    def getBarycentricArray(self):
        """Correct the vacuum wavelength array by the barycentric Earth radial
        velocity (BERV).

        Returns
        -------
        `unyt.unyt_array`
            The vacuum wavelength array in barycentric coordinates.

        """

        barycentricArray = self.shiftWavelengthArray(self.vacuumArray,
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
        photon_flux_array = self._rawFluxArray / self.blazeArray

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
            self._blazeFile = HARPSFile2D(self.getBlazeFile())
        blaze_array = self._blazeFile._rawData

        return blaze_array

    def writeWavelengthHDU(self, hdulist, verify_action='warn', **kwargs):
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
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        self._wavelengthArray = self.getWavelengthArray(**kwargs)
        # Create an HDU for the wavelength array.
        wavelength_HDU = fits.ImageHDU(data=self.wavelengthArray, name='WAVE')
        try:
            hdulist['WAVE'] = wavelength_HDU
        except KeyError:
            hdulist.append(wavelength_HDU)
        hdulist.flush(output_verify=verify_action, verbose=True)

    def writeBarycentricHDU(self, hdulist, verify_action='warn'):
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
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
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

    def writePhotonFluxHDU(self, hdulist, verify_action='warn'):
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
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
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

    def writeErrorHDU(self, hdulist, verify_action='warn'):
        """Write out an error array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
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

    def writeBlazeHDU(self, hdulist, verify_action='warn'):
        """Write out a blaze function array to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.

        verify_action : str
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
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
        wavelength_array : `unyt.unyt_array`
            An array containing wavelengths to be Doppler shifted. Needs units
            of dimension length.

        velocity : `unyt.unyt_quantity`
            A Unyt quantity with dimensions length/time to shift the wavelength
            array by.

        Returns
        -------
        `unyt.unyt_array`
            An array of the same shape as the given array, Doppler shifted by
            the given radial velocity.

        """

        return vcl.shift_wavelength(wavelength=wavelength_array,
                                    shift_velocity=shift_velocity)

    def findWavelength(self, wavelength, mid_most=True, verbose=False):
        """Find which orders contain a given wavelength.

        This function will return the indices of the wavelength orders that
        contain the given wavelength. The result will be a tuple of length 1 or
        2 containing integers in the range [0, 71].

        Parameters
        ----------
        wavelength : unyt_quantity
            The wavelength to find in the wavelength array. This should be a
            unyt_quantity object of length 1.

        Optional
        --------
        mid_most : bool, Default : True
            In a 2D extracted echelle spectrograph like HARPS, a wavelength
            near the ends of an order can appear a second time in an adjacent
            order. By default `findWavelength` will return only the single
            order where the wavelength is closest to the geometric center of
            the CCD, which corresponds to the point where the signal-to-noise
            ratio is highest. Setting this to *False* will allow for the
            possibility of a length-2 tuble being returned containing the
            numbers of both orders a wavelength is found in.

        verbose : bool, Default : False
            If *True*, the function will print out additional information such
            as the minimum and maximum values of the array and for each order.

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
        if verbose:
            tqdm.write(str(array_min), str(array_max))
        if not (array_min <= wavelength_to_find <= array_max):
            tqdm.write("Given wavelength not in array limits!")
            tqdm.write("Given wavelength: {} ({}, {})".format(
                    wavelength_to_find, array_min, array_max))
            raise RuntimeError

        # Set up a list to hold the indices of the orders where the wavelength
        # is found.
        orders_wavelength_found_in = []
        for order in range(0, 72):
            order_min = self.barycentricArray[order, 0]
            order_max = self.barycentricArray[order, -1]
            if verbose:
                tqdm.write(str(order_min), str(order_max))
            if order_min <= wavelength_to_find <= order_max:
                orders_wavelength_found_in.append(order)
                if len(orders_wavelength_found_in) == 1:
                    continue
                elif len(orders_wavelength_found_in) == 2:
                    break

        if len(orders_wavelength_found_in) == 0:
            raise RuntimeError('Wavelength not found in array!')
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
                raise RuntimeError("Wavelength found in >2 arrays!")
        else:
            return tuple(orders_wavelength_found_in)

    def plotErrorbar(self, order, passed_axis, min_index=None,
                     max_index=None, *args, **kwargs):
        """Create an errorbar plot of a single order of the observation.

        This method will use the barycentric-corrected wavelength array and
        blaze-corrected photon flux array.

        Parameters
        ----------
        order : int
            An integer in the range [0, 71] representing the index of the
            order to plot.
        passed_axis : a matplotlib Axes instance
            The order specified will be plotted onto this Axes object.

        """

        # Check that the index is correct.
        assert 0 <= order <= 71, "Index is not in [0, 71]!"

        ax = passed_axis

        # Plot onto the given axis.
        ax.errorbar(self.barycentricArray[order, min_index:max_index],
                    self.photonFluxArray[order, min_index:max_index],
                    yerr=self.errorArray[order, min_index:max_index],
                    *args, **kwargs)
