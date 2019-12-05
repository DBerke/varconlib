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
import datetime as dt
from math import isnan
from pathlib import Path

from astropy.io import fits
import numpy as np
from tqdm import tqdm, trange
import unyt as u

import varconlib as vcl
from varconlib.conversions import air2vacESO
from varconlib.exceptions import (BadRadialVelocityError,
                                  NewCoefficientsNotFoundError,
                                  BlazeFileNotFoundError,
                                  WavelengthNotFoundInArrayError)
from varconlib.miscellaneous import wavelength2index, shift_wavelength


base_path = Path(__file__).parent
config_file = base_path / '../config/variables.cfg'
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)

# Read some path variables from the config file.
# These are for directories outside the package parent directory.
blaze_file_dir = Path(config['PATHS']['blaze_file_dir'])
wavelength_cals_dir = Path(config['PATHS']['wavelength_cal_dir'])

pixel_geom_files_dir = vcl.data_dir / 'pixel_geom_files'
pixel_size_file = pixel_geom_files_dir / 'pixel_geom_size_HARPS_2004_A.fits'
pixel_pos_file = pixel_geom_files_dir / 'pixel_geom_pos_HARPS_2004_A.fits'

# Define barycenters of each order of fiber A to use with the new calibration
# coefficients. These barycenters come from C. Lovis (private communication)
order_barycenters = {0: 2058.19619875, 1: 2303.03101155, 2: 1714.78184979,
                     3: 2039.10938787, 4: 2073.83335682, 5: 2113.06660854,
                     6: 1967.40201128, 7: 1955.5199909, 8: 2260.21532786,
                     9: 2345.13976825, 10: 1600.67126601, 11: 2219.46465358,
                     12: 1703.17829463, 13: 1830.43072233, 14: 2330.77633219,
                     15: 2242.2323373, 16: 2161.63938098, 17: 2103.83909842,
                     18: 2030.56647448, 19: 2028.10455978, 20: 1987.15700559,
                     21: 2093.29334908, 22: 1503.40223261, 23: 2218.47620643,
                     24: 2074.74966125, 25: 1903.88311521, 26: 1804.5784936,
                     27: 1872.01030628, 28: 1989.48750792, 29: 1808.27607907,
                     30: 2281.79320743, 31: 2108.80991269, 32: 2046.02691096,
                     33: 1829.94534136, 34: 2274.82477587, 35: 2018.88758685,
                     36: 1772.16802274, 37: 1986.35240348, 38: 2357.29421969,
                     39: 2104.22326262, 40: 1831.90393061, 41: 2285.16323923,
                     42: 2056.91007412, 43: 1888.3231257, 44: 1826.74684748,
                     45: 2118.60595923, 46: 2107.16748608, 47: 1788.87547361,
                     48: 1791.29731027, 49: 2000.83192526, 50: 1568.9476432,
                     51: 1712.33483484, 52: 1614.5869981, 53: 2496.96599212,
                     54: 1998.55104358, 55: 2002.86578518, 56: 1597.0016393,
                     57: 1787.78941935, 58: 2141.16935421, 59: 2140.47316633,
                     60: 1529.78026616, 61: 1542.81670425, 62: 1857.79112214,
                     63: 1791.27630583, 64: 1930.19573565, 65: 1937.32176376,
                     66: 1463.07051377, 67: 2091.18218294, 68: 2240.88136565,
                     69: 1762.46301572, 70: 2172.64792852, 71: 1904.59716115}


class HARPSFile2D(object):
    """Class to contain data from a HARPS 2D extracted spectrum file.

    """

    def __init__(self, FITSfile):
        if isinstance(FITSfile, str):
            self._filename = Path(FITSfile)
        elif isinstance(FITSfile, Path):
            self._filename = FITSfile
        else:
            raise RuntimeError('File name not str or Path!')
        if not self._filename.exists():
            raise FileNotFoundError('Filename {} does not exist!'.format(
                    self._filename))
        with fits.open(self._filename, mode='readonly') as hdulist:
            self._header = hdulist[0].header
            data = hdulist[0].data
            self._rawData = self._reshape_if_necessary(data)

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self._filename)

    def __str__(self):
        return '{}, {}'.format(self._header['OBJECT'], self._filename.stem)

    @property
    def dateObs(self):
        if not hasattr(self, '_dateObs'):
            date_string = self.getHeaderCard('DATE-OBS')
            self._dateObs = dt.datetime.strptime(date_string,
                                                 '%Y-%m-%dT%H:%M:%S.%f')
        return self._dateObs

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

    def _reshape_if_necessary(self, array):
        """Reshape the main data array if necessary.

        The pixel geometry files from Lovis give a table in the shape
        (4096, 72) instead of the (72, 4096) that HARPS normally uses. Thus
        data from these files needs to be reshaped to be used. This function
        checks the shape of the array given, and reshapes it if necessary.

        Parameters
        ----------
        array : array-like
            The array to have its shape checked, and reshaped if necessary.

        Returns
        -------
        np.array
            The array, in the shape (72, 4096) (whether originally or after
            having been reshaped).

        """

        if array.shape == (4096, 72):
            array = array.reshape((72, 4096))
            tqdm.write('Data array reshaped from (4096, 72) to (72, 4096).')
        return array


class HARPSFile2DScience(HARPSFile2D):
    """Subclass of HARPSFile2D to handle observations specifically.

    """

    def __init__(self, FITSfile, update=[], new_coefficients=True,
                 pixel_positions=True):
        """Parse a given FITS file containing an observation into a usable
        HARPSFile2DScience object.

        Parameters
        ----------
        FITSfile : string or `pathlib.Path` object
            Represents the location of the file to read. If given as a string
            it will be converted to a Path object internally.

        Optional
        --------
        update : list of strings
            Whether to force writing of the wavelength, flux, and error arrays.
            `update` may be given as a list of strings containing some or all
            of 'WAVE', 'BARY', 'PIXLOWER', 'PIXUPPER', 'FLUX', 'ERR', or
            'BLAZE'. The value 'ALL' may also be given as a shortcut for
            updating all values.

            Any HDUs listed in the tuple will be updated, while those not
            listed will be read from the file being opened. Updating any of the
            wavelength solutions usually takes the most time, so if it doesn't
            need to be changed passing a tuple of ('FLUX', 'ERR') will update
            the FLUX and ERR HDUs without performing the time-costly updating
            of the wavelength solution.
        new_coefficients : bool, Default : True
            Whether to use new coefficients from the HARPS recalibration per-
            formed in Coffinet et al. 2019[1]_.
        pixel_positions : bool, Default : True
            Whether to use pixel positions derived from the HARPS recalibation
            performed in Coffinet et al. 2019[1]_.

        References
        ----------
        [1] A. Coffinet, C. Lovis, X. Dumusque, F. Pepe, "New wavelength
        calibration of the HARPS spectrograph", Astronomy & Astrophysics, 2019

        """

        if isinstance(FITSfile, Path):
            self._filename = FITSfile
        else:
            self._filename = Path(FITSfile)

        if not self._filename.exists():
            tqdm.write(str(self._filename))
            raise FileNotFoundError('The given path does not exist!')

        if update:
            file_open_mode = 'update'
        else:
            file_open_mode = 'append'

#        with fits.open(self._filename, mode=file_open_mode) as hdulist:
        # Note: using a with block doesn't currently work because the __exit__
        # method for fits.open always uses an output_verify value of
        # 'exception', making it impossible to use more lenient options within
        # the block.
        hdulist = fits.open(self._filename, mode=file_open_mode)

        self._header = hdulist[0].header
        data = hdulist[0].data
        self._rawData = self._reshape_if_necessary(data)
        self._rawFluxArray = copy(self._rawData)
        self._blazeFile = None

        # Define an error string for trying to update a file that hasn't
        # been opened previously
        err_str = "File opened in 'update' mode but no arrays exist!"

        # Define the verification level for FITS files not meeting the FITS
        # standard.
        verify_action = 'warn'

        # Try to read the wavelength array, or create it if it doesn't
        # exist.
        try:
            self._wavelengthArray = hdulist['WAVE'].data * u.angstrom
        except KeyError:
            if ('ALL' in update) or ('WAVE' in update):
                raise RuntimeError(err_str)
            tqdm.write('Writing new wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action=verify_action,
                                    use_new_coefficients=new_coefficients,
                                    use_pixel_positions=pixel_positions)
        # If we're updating the file, overwrite the existing wavelengths.
        if ('ALL' in update) or ('WAVE' in update):
            tqdm.write('Overwriting wavelength HDU.')
            self.writeWavelengthHDU(hdulist, verify_action=verify_action,
                                    use_new_coefficients=new_coefficients,
                                    use_pixel_positions=pixel_positions)

        # Try to read the barycentric-vacuum wavelength array or create it
        # if if doesn't exist yet.
        try:
            self._barycentricArray = hdulist['BARY'].data * u.angstrom
        except KeyError:
            if ('ALL' in update) or ('BARY' in update):
                raise RuntimeError(err_str)
            tqdm.write('Writing new barycentric wavelength HDU.')
            self.writeBarycentricHDU(hdulist, self.barycentricArray,
                                     'BARY', verify_action=verify_action)
        if ('ALL' in update) or ('BARY' in update):
            tqdm.write('Overwriting barycentric wavelength HDU.')
            del self._barycentricArray
            self.writeBarycentricHDU(hdulist, self.barycentricArray,
                                     'BARY', verify_action=verify_action)

        # Try to read the barycentric lower pixel edge array, or create it
        # if it doesn't exist yet.
        try:
            self._pixelLowerArray = hdulist['PIXLOWER'].data * u.angstrom
        except KeyError:
            try:
                pixel_array = self.pixelLowerArray
            except NewCoefficientsNotFoundError:
                tqdm.write('No new coefficients file, not writing array.')
            else:
                if ('ALL' in update) or ('PIXLOWER' in update):
                    raise RuntimeError(err_str)
                tqdm.write('Writing new pixel lower edges HDU.')
                self.writeBarycentricHDU(hdulist, self.pixelLowerArray,
                                         'PIXLOWER',
                                         verify_action=verify_action)
        if ('ALL' in update) or ('PIXLOWER' in update):
            tqdm.write('Overwriting lower pixel wavelength HDU.')
            del self._pixelLowerArray
            pixel_array = self.pixelLowerArray
            self.writeBarycentricHDU(hdulist, pixel_array, 'PIXLOWER',
                                     verify_action=verify_action)

        # Try to read the barycentric upper pixel edge array, or create it
        # if it doesn't exist yet.
        try:
            self._pixelUpperArray = hdulist['PIXUPPER'].data * u.angstrom
        except KeyError:
            try:
                pixel_array = self.pixelUpperArray
            except NewCoefficientsNotFoundError:
                tqdm.write('No new coefficients file, not writing array.')
            else:
                if ('ALL' in update) or ('PIXUPPER' in update):
                    raise RuntimeError(err_str)
                tqdm.write('Writing new pixel upper edges HDU.')
                self.writeBarycentricHDU(hdulist, self.pixelUpperArray,
                                         'PIXUPPER',
                                         verify_action=verify_action)
        if ('ALL' in update) or ('PIXUPPER' in update):
            tqdm.write('Overwriting lower pixel wavelength HDU.')
            del self._pixelUpperArray
            pixel_array = self.pixelUpperArray
            self.writeBarycentricHDU(hdulist, pixel_array, 'PIXUPPER',
                                     verify_action=verify_action)

        # Try to read the flux array, or create it if it doesn't exist.
        try:
            self._photonFluxArray = hdulist['FLUX'].data
        except KeyError:
            if ('ALL' in update) or ('FLUX' in update):
                raise RuntimeError(err_str)
            self.writePhotonFluxHDU(hdulist, verify_action=verify_action)
            tqdm.write('Writing new photon flux HDU.')
        # If we're updating the file, overwrite the existing fluxes.
        if ('ALL' in update) or ('FLUX' in update):
            tqdm.write('Overwriting photon flux HDU.')
            self.writePhotonFluxHDU(hdulist, verify_action=verify_action)

        # Try to read the error array, or create it if it doesn't exist.
        try:
            self._errorArray = hdulist['ERR'].data
        except KeyError:
            if ('ALL' in update) or ('ERR' in update):
                raise RuntimeError(err_str)
            self.writeErrorHDU(hdulist, verify_action=verify_action)
            tqdm.write('Writing new error HDU.')
        # If we're updating the file, overwrite the existing uncertainties.
        if ('ALL' in update) or ('ERR' in update):
            tqdm.write('Overwriting error array HDU.')
            self.writeErrorHDU(hdulist, verify_action=verify_action)

        # Try to read the blaze array, or create it if it doesn't exist.
        try:
            self._blazeArray = hdulist['BLAZE'].data
        except KeyError:
            if ('ALL' in update) or ('BLAZE' in update):
                raise RuntimeError(err_str)
            self.writeBlazeHDU(hdulist, verify_action=verify_action)
            tqdm.write('Writing new blaze HDU.')
        # If we're updating the file, overwrite the existing uncertainties.
        if ('ALL' in update) or ('BLAZE' in update):
            tqdm.write('Overwriting blaze array HDU.')
            self.writeBlazeHDU(hdulist, verify_action=verify_action)

        hdulist.close(output_verify='warn')

    @property
    def wavelengthArray(self):
        if not hasattr(self, '_wavelengthArray'):
            tqdm.write('No wavelength array, something went wrong!')
        return self._wavelengthArray

    @property
    def vacuumArray(self):
        if not hasattr(self, '_vacuumArray'):
            tqdm.write('Creating vacuum wavelength array.')
            self._vacuumArray = self.getVacuumArray(self.wavelengthArray)
        return self._vacuumArray

    @property
    def barycentricArray(self):
        if not hasattr(self, '_barycentricArray'):
            tqdm.write('Creating barycentric vacuum wavelength array.')
            self._barycentricArray = self.barycenterCorrect(self.vacuumArray)
        return self._barycentricArray

    @property
    def pixelLowerArray(self):
        if not hasattr(self, '_pixelLowerArray'):
            tqdm.write('Creating lower pixel edge barycentric vacuum array.')
            pixel_lower = self.pixelPosArray - self.pixelSizeArray / 2
            try:
                self.getWavelengthCalibrationFile()
            except NewCoefficientsNotFoundError:
                raise
            else:
                pixel_lower_air = self.getWavelengthArray(
                        use_new_coefficients=True,
                        use_pixel_positions=pixel_lower)
                pixel_lower_vac = self.getVacuumArray(pixel_lower_air)
                self._pixelLowerArray = self.barycenterCorrect(pixel_lower_vac)

        return self._pixelLowerArray

    @property
    def pixelUpperArray(self):
        if not hasattr(self, '_pixelUpperArray'):
            tqdm.write('Creating upper pixel edge barycentric vacuum array.')
            pixel_upper = self.pixelPosArray + self.pixelSizeArray / 2
            try:
                self.getWavelengthCalibrationFile()
            except NewCoefficientsNotFoundError:
                raise
            else:
                pixel_upper_air = self.getWavelengthArray(
                        use_new_coefficients=True,
                        use_pixel_positions=pixel_upper)
                pixel_upper_vac = self.getVacuumArray(pixel_upper_air)
                self._pixelUpperArray = self.barycenterCorrect(pixel_upper_vac)

        return self._pixelUpperArray

    @property
    def pixelSizeArray(self):
        if not hasattr(self, '_pixelSizeArray'):
            self._pixelSizeArray = HARPSFile2D(pixel_size_file)._rawData
        return self._pixelSizeArray

    @property
    def pixelPosArray(self):
        if not hasattr(self, '_pixelPosArray'):
            self._pixelPosArray = HARPSFile2D(pixel_pos_file)._rawData
        return self._pixelPosArray

    @property
    def rvCorrectedArray(self):
        if not hasattr(self, '_rvCorrectedArray'):
            tqdm.write('Creating RV-corrected array.')
            self._rvCorrectedArray = shift_wavelength(self.barycentricArray,
                                                      -self.radialVelocity)
        return self._rvCorrectedArray

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
                tqdm.write('No radial velocity found for this observation!')
                raise

            if isnan(radial_velocity):
                raise BadRadialVelocityError('Radial velocity is NaN!')
            if abs(radial_velocity) > 5000:
                raise BadRadialVelocityError('Radial velocity for'
                                             f' {self._filename}'
                                             ' is suspiciously'
                                             f' high: {radial_velocity}')
            self._radialVelocity = radial_velocity
        return self._radialVelocity

    @property
    def BERV(self):
        # BERV = Barycentric Earth Radial Velocity
        if not hasattr(self, '_BERV'):
            self._BERV = float(self.getHeaderCard('HIERARCH ESO DRS BERV'))\
                               * u.km / u.s
        return self._BERV

    @property
    def airmassStart(self):
        if not hasattr(self, '_airmassStart'):
            self._airmassStart = float(self.getHeaderCard(
                                       'HIERARCH ESO TEL AIRM START'))
        return self._airmassStart

    @property
    def airmassEnd(self):
        if not hasattr(self, '_airmassEnd'):
            self._airmassEnd = float(self.getHeaderCard(
                                     'HIERARCH ESO TEL AIRM END'))
        return self._airmassEnd

    @property
    def airmass(self):
        if not hasattr(self, '_airmass'):
            self._airmass = (self.airmassStart + self.airmassEnd) / 2
        return self._airmass

    @property
    def exptime(self):
        if not hasattr(self, '_exptime'):
            self._exptime = float(self.getHeaderCard('EXPTIME'))
        return self._exptime

    def getBlazeFile(self):
        """Find and return the blaze file associated with this observation.

        Returns
        -------
        `pathlib.Path` object
            A `Path` object containing the path to the blaze file associated
            with this observation via its header card.

        """

        blaze_file = self.getHeaderCard('HIERARCH ESO DRS BLAZE FILE')

        file_date = blaze_file[6:16]

        blaze_file_path = blaze_file_dir /\
            'data/reduced/{}'.format(file_date) /\
            blaze_file

        if not blaze_file_path.exists():
            tqdm.write(str(blaze_file_path))
            raise BlazeFileNotFoundError("Blaze file path doesn't exist!")

        return blaze_file_path

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
            raise NewCoefficientsNotFoundError("Calibration file not found:"
                                               f"{str(cal_file_path)}")

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

        use_pixel_positions : bool or `numpy.ndarray`, Default : True
            This does different things depending on if a boolean or an array is
            passed.

            If a boolean is passed, a value of *True* will use the more
            accurate positions (in 'pixel space', as detailed in Coffinet et
            al. 2019 [1]_) for the centers of pixels when evaluating the
            wavelength solution. If *False*, it will use nominal values for the
            pixel centers assuming a nominal pixel size of 1.

            If an array is passed, it will use the values from that array. It
            must be a (72, 4096) size array of floating point values without
            units attached.

        Returns
        -------
        `numpy.ndarray`
            An array of the same shape as the input array specifying the
            wavelength of each pixel (element in the array) in Angstroms.

        Notes
        -----
        The algorithm used is derived from Dumusque 2018 [2]_:
        .. math::
            \\lambda_{i,j}=P_{4\\cdot i}+P_{4\\cdot i+1}\\times j+\\
            P_{4\\cdot i+2}\\times j^2+P_{4\\cdot i+3}\\times j^3

        where :math:`\\lambda_{i,j}` is the wavelength at pixel *j* in order
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
            except NewCoefficientsNotFoundError:
                coeffs_file = self
                tqdm.write('New calibration file could not be found!')
                tqdm.write('Falling back on old calibration coefficients.')
                raise NewCoefficientsNotFoundError
        else:
            coeffs_file = self

        if isinstance(use_pixel_positions, np.ndarray):
            assert use_pixel_positions.shape == (72, 4096), 'Provided pixel'
            f'array wrong shape: {use_pixel_positions.shape}'
            pixel_positions = use_pixel_positions
            tqdm.write('Using provided pixel positions.')
        elif use_pixel_positions is True:
            # Use the new pixel positions file provided.
            pixel_positions = self.pixelPosArray
        elif use_pixel_positions is False:
            pixel_positions = np.array([[x for x in range(0, 4096)]
                                       for row in range(0, 72)])

        source_array = self._rawFluxArray
        wavelength_array = np.zeros(source_array.shape, dtype=float)
        # Step through the 72 spectral orders
        for order in trange(0, 72, total=72, unit='orders'):
            if use_new_coefficients:
                order_barycenter = order_barycenters[order]
            else:
                order_barycenter = 0
            # Iterate through the third-order polynomial fit coefficients
            for i in range(0, 4, 1):
                coeff = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + i)
                coeff_val = coeffs_file.getHeaderCard(coeff)
                for pixel, pixel_pos in enumerate(pixel_positions[order, :]):
                    wavelength_array[order, pixel] += coeff_val *\
                      ((pixel_pos - order_barycenter) ** i)

        return wavelength_array * u.angstrom

    def getVacuumArray(self, array):
        """Convert an air wavelength array into vacuum.

        Returns
        -------
        `unyt.unyt_array`
            The wavelength array for the observation converted into vacuum
            wavelengths using the Edlen 1953 formula used by the HARPS
            pipeline.

        """

        return air2vacESO(array)

    def barycenterCorrect(self, array):
        """Correct the given wavelength array by the barycentric Earth radial
        velocity (BERV).

        Parameters
        ----------
        array : `unyt.unyt_array`
            A wavelength array (prefarably in vacuum wavelengths, though this
            is not necesary).

        Returns
        -------
        `unyt.unyt_array`
            A wavelength array correct for the Earth's barycentric motion.

        """

        return self.shiftWavelengthArray(array, self.BERV)

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

        return HARPSFile2D(self.getBlazeFile())._rawData

    def writeWavelengthHDU(self, hdulist, verify_action='warn', **kwargs):
        """Write out a wavelength array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.
        verify_action : str, optional
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
        hdulist.flush(output_verify=verify_action, verbose=False)

    def writeBarycentricHDU(self, hdulist, array, array_name,
                            verify_action='warn'):
        """Write out an array of barycentric vacuum wavelengths to the
        currently-opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.
        array : property
            The array to be written; should be one of self.barycentricArray,
            self.pixelLowerArray, self.pixelUpperArray.
        array_name : str
            The name corresponding to the array pased. Should be 'BARY',
            'PIXLOWER', or 'PIXUPPER', respectively.
        verify_action : str, optional
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        hdu = fits.ImageHDU(data=array, name=array_name)
        try:
            hdulist[array_name] = hdu
        except KeyError:
            hdulist.append(hdu)

        hdulist.flush(output_verify=verify_action, verbose=False)

    def writePhotonFluxHDU(self, hdulist, verify_action='warn'):
        """Write out a photon flux array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.
        verify_action : str, optional
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        # Create an HDU for the photon flux array.
        photon_flux_HDU = fits.ImageHDU(data=self.photonFluxArray, name='FLUX')
        try:
            hdulist['FLUX'] = photon_flux_HDU
        except KeyError:
            hdulist.append(photon_flux_HDU)
        hdulist.flush(output_verify=verify_action, verbose=False)

    def writeErrorHDU(self, hdulist, verify_action='warn'):
        """Write out an error array HDU to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.
        verify_action : str, optional
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        # Create an HDU for the error array.
        error_HDU = fits.ImageHDU(data=self.errorArray, name='ERR')
        try:
            hdulist['ERR'] = error_HDU
        except KeyError:
            hdulist.append(error_HDU)
        hdulist.flush(output_verify=verify_action, verbose=False)

    def writeBlazeHDU(self, hdulist, verify_action='warn'):
        """Write out a blaze function array to the currently opened file.

        Parameters
        ----------
        hdulist : an astropy HDUList object
            The HDU list of the file to modify.
        verify_action : str, optional
            One of either ``'exception'``, ``'ignore'``, ``'fix'``,
            ``'silentfix'``, or ``'warn'``.
            More information can be found in the Astropy `documentation.
            `<http://docs.astropy.org/en/stable/io/fits/api/verification.html
            #verify>`_
            The default value is to print a warning upon encountering a
            violation of any FITS standard.

        """

        blaze_HDU = fits.ImageHDU(data=self.blazeArray, name='BLAZE')

        try:
            hdulist['BLAZE'] = blaze_HDU
        except KeyError:
            hdulist.append(blaze_HDU)
        hdulist.flush(output_verify=verify_action, verbose=False)

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

        return shift_wavelength(wavelength_array, shift_velocity)

    def findWavelength(self, wavelength, wavelength_array,
                       mid_most=True, verbose=False):
        """Find which orders contain a given wavelength.

        This function will return the indices of the wavelength orders that
        contain the given wavelength. The result will be a tuple of length 1 or
        2 containing integers in the range [0, 71].

        Parameters
        ----------
        wavelength : unyt_quantity
            The wavelength to find in the wavelength array. This should be a
            unyt_quantity object of length 1.
        wavelength_array : `unyt.unyt_array`
            An array of wavelengths in the shape of a HARPS extracted spectrum
            (72, 4096) to be searched.
        mid_most : bool, Default : *True*, optional
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
        err_str = "Given wavelength not in array limits: {} ({}, {})".format(
                  wavelength_to_find, wavelength_array[0, 0],
                  wavelength_array[-1, -1])
        if not (wavelength_array[0, 0] <= wavelength_to_find
                <= wavelength_array[-1, -1]):
            raise WavelengthNotFoundInArrayError(err_str)

        # Set up a list to hold the indices of the orders where the wavelength
        # is found.
        orders_wavelength_found_in = []
        for order in range(0, 72):
            if (wavelength_array[order, 0] <= wavelength_to_find
                    <= wavelength_array[order, -1]):
                orders_wavelength_found_in.append(order)
                if len(orders_wavelength_found_in) == 1:
                    continue
                elif len(orders_wavelength_found_in) == 2:
                    break

        assert len(orders_wavelength_found_in) > 0, 'Wavelength not found'
        ' in array.'

        if mid_most:
            # If only one array: great, return it.
            if len(orders_wavelength_found_in) == 1:
                return orders_wavelength_found_in[0]

            # Found in two arrays: figure out which is closer to the geometric
            # center of the CCD, which conveiently falls around the middle
            # of the 4096-element array.
            elif len(orders_wavelength_found_in) == 2:
                order1, order2 = orders_wavelength_found_in
                index1 = wavelength2index(wavelength_to_find,
                                          wavelength_array[order1])
                index2 = wavelength2index(wavelength_to_find,
                                          wavelength_array[order2])
                # Check which index is closest to the pixel in the geometric
                # center of the 4096-length array, given 0-indexing in Python.
                if abs(index1 - 2047.5) > abs(index2 - 2047.5):
                    return order2
                else:
                    return order1
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
