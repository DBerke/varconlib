#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:59:22 2019

@author: dberke

A class to hold information about a single star, both metadata relating to the
star itself and arrays of information about absorption features fitted in the
various observations taken of it.

"""

import datetime as dt
from glob import glob
import json
import lzma
import os
from pathlib import Path
import pickle
import re

from bidict import namedbidict
import h5py
import hickle
import numpy as np
import numpy.ma as ma
from tqdm import tqdm, trange
import unyt as u
import unyt.dimensions as dimensions

import varconlib as vcl
from varconlib.exceptions import (HDF5FileNotFoundError,
                                  PickleFilesNotFoundError,
                                  StarDirectoryNotFoundError)
from varconlib.miscellaneous import wavelength2velocity as wave2vel


class Star(object):
    """A class to hold information gathered from analysis of (potentially many)
    observations of a single star.

    The `Star` class is intended to hold information relating to a single
    star: both information intrinsic to the star (absolute magnitude,
    metallicity, color, etc.) and information about the fits of given
    transitions in the spectra of observations of that star.

    Attributes
    ----------
    fitMeansArray : `unyt.unyt_array`
        A two-dimensional array holding the mean of the fit (in wavelength
        space) for each absorption feature in each observation of the star.
        Rows correspond to observations, columns to transitions. Units are Å.
    fitErrorsArray : `unyt.unyt_array`
        A two-diemsional array holding the standard deviation of the measured
        mean of the fit for each absorption feature of each observation of the
        star. Rows correspond to observations, columns to transitions. Units
        are m/s.
    fitOffsetsArray : `unyt.unyt_array`
        A two-dimensional array holding the offset from the expected wavelength
        of the measured mean of each absorption feature in each observation of
        the star. Rows correspond to observations, columns to transitions.
        Units are m/s.
    fitOffsetsCorrectedArray : `unyt.unyt_array`
        A two-dimensional array holding the measured offsets from
        fitOffsetsArray corrected for the systematic calibration errors in
        HARPS.
    ccdCorrectionArray : `unyt.unyt_array`
        A two-dimensional array holding the corrections computed for each
        transition at its location on the CCD, applied to fitOffsetsArray to
        create fitOffsetsCorrectedArray.
    fitOffsetsNormalizedArray : 'unyt.unyt_array'
        A two-dimensional array holding the offset from the expected wavelength
        of the measured mean of each absorption feature in each observations of
        the star, with each row (corresponding to an observation) having been
        corrected by the measured radial velocity offset for that observation,
        determined as the mean value of the offsets of all transitions for that
        observation from fitOffsetsCorrectedArray.
    pairSeparationsArray : `unyt.unyt_array`
        A two-dimensional array holding the velocity separation values for each
        pair of transitions for each observation of the star. Rows correspond
        to observations, columns to pairs. Units are m/s.
    pairSepErrorsArray : `unyt.unyt_array`
        A two-dimensional array holding the standard deviation of the velocity
        separation values for each pair of transitions for each observation of
        the star. Rows correspond to observations, columns to pairs. Units are
        m/s.
    bervArray : `unyt.unyt_array`
        A one-dimensional array holding the barycentric Earth radial velocity
        (BERV) for each observation of the star. Index number corresponds to
        row numbers in the two-dimensional arrays. Units are km/s.
    chiSquaredNuArray : `unyt.unyt_array`
        A two-dimensional array holding the reduced chi-squared value of the
        Gaussian fit to each transition for each observation of the star.
        Rows correspond to observations, columns to transitions.
    airmassArray : `np.ndarray` of floats
        A one-dimensional array holding the airmass for each observation of the
        star.
    _obs_date_bidict : `bidict_named.namedbidict`
        A custom `namedbidict` with attribute 'date_for' and 'index_for' which
        can be used to get the date for a given index, and vice versa.
    _transition_bidict : `bidict_named.namedbidict`
        A custom `namedbidict` with attributes 'label_for' and 'index_for'
        which can be used to get the transition label for a given index, and
        vice versa.
    _pair_bidict : `bidict_named.namedbidict`
        A custom `namedbidict` with attributes 'label_for' and 'index_for'
        which can be used to get the pair label for a given index, and vice
        versa.
    radialVelocity : `unyt.unyt_quantity`
        The intrinsic radial velocity of the star.
    temperature : `unyt.unyt_quantity`
        The effective temperature of the star.
    metallicity : float
        The metallicity of the star.
    absoluteMagnitude : float
        The absolute V-band magnitude of the star.
    apparentMagnitude : float
        The apparent V-band magnitude of the star.
    logg : float
        The logarithm of the surface gravity of the star. Technicallly in units
        of cm / s / s but unitless in code.
    fiberSplitIndex : None or int
        A value representing either the index of the first observation after the
        HARPS fiber change in May 2015, or None if all observations were prior
        to the change.
    numObsPre : int
        The number of observations of this star taken before the fiber change.
    numObsPost : int
        The number of observations of this star taken after the fiber change.
    specialAttributes: dict
        A dictionary (possibly empty) of certain characteristics or attributes
        of some stars which are rare enough to not warrant storing for every
        stars, e.g., whether a star has a known companion (planet or star) or if
        it is know to be variable. Currently has two recognized keywords:
            'is_variable' : the type of variable (BY Draconis, etc.)
            'is_multiple' : the total number of stars in the system
            'has_planets': the number of planets around this star
        This dictionary is constructed from JSON files stored in each star's
        directory as appropriate.

    """

    # Define the version of the format being saved.
    global CURRENT_VERSION
    # TODO: update this
    CURRENT_VERSION = '1.1.0'

    # Define dataset names and corresponding attribute names to be saved
    # and loaded when dumping star data.
    unyt_arrays = {'/arrays/transition_means': 'fitMeansArray',
                   '/arrays/transition_errors': 'fitErrorsArray',
                   '/arrays/offsets': 'fitOffsetsArray',
                   '/arrays/pair_separations': 'pairSeparationsArray',
                   '/arrays/pair_separation_errors': 'pairSepErrorsArray',
                   '/arrays/BERV_array': 'bervArray',
                   '/arrays/observation_rv': 'obsRVOffsetsArray',
                   '/arrays/normalized_offsets': 'fitOffsetsNormalizedArray',
                   '/arrays/corrected_offsets': 'fitOffsetsCorrectedArray',
                   '/arrays/systematic_corrections': 'ccdCorrectionArray'}

    other_attributes = {'/metadata/version': 'version',
                        '/arrays/reduced_chi_squareds': 'chiSquaredNuArray',
                        '/arrays/airmasses': 'airmassArray',
                        '/bidicts/obs_date_bidict': '_obs_date_bidict',
                        '/bidicts/transition_bidict': '_transition_bidict',
                        '/bidicts/pair_bidict': '_pair_bidict',
                        '/data/radial_velocity': 'radialVelocity',
                        '/data/temperature': 'temperature',
                        '/data/metallicity': 'metallicity',
                        '/data/absolute_magnitude': 'absoluteMagnitude',
                        '/data/apparent_magnitude': 'apparentMagnitude',
                        '/data/logg': 'logg'}

    # Define some custom namedbidict objects.
    DateMap = namedbidict('ObservationDateMap', 'date', 'index')
    TransitionMap = namedbidict('TransitionMap', 'label', 'index')
    PairMap = namedbidict('PairMap', 'label', 'index')

    # Date of fiber change in HARPS:
    fiber_change_date = dt.datetime(year=2015, month=6, day=1,
                                    hour=0, minute=0, second=0)

    def __init__(self, name, star_dir=None, suffix='int',
                 transitions_list=None, pairs_list=None,
                 load_data=None, init_params="Casagrande2011"):
        """Instantiate a `star.Star` object.

        Parameters
        ----------
        name : str
            A name for the star to identify it.

        Optional
        --------
        star_dir : `pathlib.Path`
            A Path object specifying the root directory to look in for fits of
            the star's spectra. If given, will be passed to the
            `initializeFromFits` method.
        suffix : str, Default: 'int'
            A string to be added to the subdirectory names to distinguish
            between different reduction methods. Defaults to 'int' for
            'integrated Gaussian' fits.
        transitions_list : list
            A list of `transition_line.Transition` objects. If `star_dir` is
            given, will be passed to `initializeFromFits`, otherwise no effect.
        pairs_list : list
            A list of `transition_pair.TransitionPair` objects. If `star_dir`
            is given, will be passed to `initializeFromFits`, otherwise no
            effect.
        load_data : bool or None, Default : None
            Controls whether to attempt to read a file containing data for the
            star or create it from scratch from observations in the `star_dir`.
            If *False*, the Star will be created from observations, regardless
            of whether an HDF5 file already exists.
            If *True*, the Star will *only* be created from an HDF5 file if one
            exists, and will fail if one does not (even if it could have been
            created from the observations in the directory).
            If *None*, then the code will first attempt to create a Star using
            a previously-created HDF5 file, and if one does not exist will
            attempt to create one using observations. This is essentially the
            'pragmatic' option, and is also the default behavior.
        init_params : str, ['Nordstrom2004', 'Casagrande2011'], Default :
                    'Casagrande2011'
            Which paper's derivation of the stellar parameters for this star to
            use.

        """

        self.name = str(name)
        self.version = CURRENT_VERSION

        # Initialize some attributes to be filled later.
        self._obs_date_bidict = self.DateMap()
        self._transition_bidict = self.TransitionMap()
        self._pair_bidict = self.PairMap()
        self.bervArray = None
        self.airmassArray = None
        self.fitMeansArray = None
        self.fitErrorsArray = None
        self.fitOffsetsArray = None
        self.pairSeparationsArray = None
        self.pairSepErrorsArray = None

        self.hasObsPre = False
        self.hasObsPost = False

        self._radialVelocity = None
        self._temperature = None
        self._metallicity = None
        self._absoluteMagnitude = None
        self._apparentMagnitude = None
        self._logg = None
        self.specialAttributes = {}

        if transitions_list:
            self.transitionsList = transitions_list
        if pairs_list:
            self.pairsList = pairs_list

        if (star_dir is not None):
            star_dir = Path(star_dir)
            self.hdf5file = star_dir / f'{name}_data.hdf5'
            if load_data is False or\
                    (load_data is None and not self.hdf5file.exists()):
                self.constructFromDir(star_dir, suffix,
                                      pairs_list=pairs_list,
                                      transitions_list=transitions_list)
                self.getPairSeparations()
                self.saveDataToDisk(self.hdf5file)
            elif (load_data is True or load_data is None)\
                    and self.hdf5file.exists():
                self.constructFromHDF5(self.hdf5file)
            else:
                raise HDF5FileNotFoundError('No HDF5 file found for'
                                            f' {self.hdf5file}.')
            self.specialAttributes.update(self.getSpecialAttributes(star_dir))

        # Figure out when the observations for the star were taken, and set the
        # appropriate flags.
        if self.fiberSplitIndex is None:
            self.hasObsPre = True
        elif self.fiberSplitIndex == 0:
            self.hasObsPost = True
        else:
            self.hasObsPre = True
            self.hasObsPost = True

        assert init_params in ('Nordstrom2004', 'Casagrande2011'),\
            f'{init_params} is not a valid paper name.'
        self.getStellarParameters(init_params)

    def constructFromDir(self, star_dir, suffix='int', transitions_list=None,
                         pairs_list=None):
        """
        Collect information on fits in observations of the star, and organize
        it.

        Parameters
        ----------
        star_dir : `pathlib.Path` or str
            A path object representing the root directory to look in for fits
            to the star's spectra.

        Optional
        --------
        suffix : str, default "int"
            The suffix to affix to end of the sub-directories under the main
            star directory. The default stands for "integrated Gaussian".
        transitions_list : list
            A list of `transition_line.Transition` objects. If this is omitted
            the default list of transitions selected for use will be read, but
            this will be slower.
        pairs_list : list
            A list of `transition_pair.TransitionPair` objects. If this is
            omitted the default list of pairs selected for use will be read,
            but this will be slower.

        """

        if isinstance(star_dir, str):
            star_dir = Path(star_dir)

        # Check that the given directory exists.
        if not star_dir.exists():
            raise StarDirectoryNotFoundError('The given directory does not'
                                             f' exist: {star_dir}')

        # Get a list of pickled fit results in the given directory.
        search_str = str(star_dir) + f'/HARPS*/pickles_{suffix}/*fits.lzma'
        pickle_files = [Path(path) for path in sorted(glob(search_str))]

        if len(pickle_files) == 0:
            raise PickleFilesNotFoundError('No pickled fits found'
                                           f' in {star_dir}.')

        means_list = []
        errors_list = []
        offsets_list = []
        chi_squared_list = []
        ccd_corrections_list = []
        offsets_corrected_list = []

        total_obs = len(pickle_files)
        self.bervArray = np.empty(total_obs)
        self.airmassArray = np.empty(total_obs)
        self.obsRVOffsetsArray = np.empty((total_obs, 1))

        # For each pickle file:
        for obs_num, pickle_file in enumerate(tqdm(pickle_files)):

            with lzma.open(pickle_file, 'rb') as f:
                fits_list = pickle.loads(f.read())
            # Save the observation date.
            # ??? Maybe save as datetime objects rather than strings?
            for fit in fits_list:
                if fit is not None:
                    self._obs_date_bidict[fit.dateObs.isoformat(
                                          timespec='milliseconds')] = obs_num
                    # Save the BERV and airmass.
                    self.bervArray[obs_num] = fit.BERV.to(u.km/u.s).value
                    self.airmassArray[obs_num] = fit.airmass
                    break

            # Iterate through all the fits in the pickled list and save their
            # values only if the fit was 'good' (i.e., a mean value exists and
            # the amplitude of the fitted Gaussian is negative).
            obs_means = []
            obs_errors = []
            obs_offsets = []
            obs_chi_squareds = []
            obs_ccd_corrections = []
            obs_offsets_corrected = []
            for fit in fits_list:
                # Check that a fit 1) exists, 2) has a negative amplitude (since
                # amplitude is unconstrained, a positive amplitude is a failed
                # fit because it's ended up fitting a peak), and 3) within
                # 5 km/s of its expected wavelength (since the fit only looks
                # within that range, if it's outside it MUST be wrong).
                if (fit is not None) and (fit.amplitude < 0) and\
                    abs(wave2vel(fit.mean,
                                 fit.correctedWavelength)) < 5 * u.km / u.s:
                    fit_mean = fit.mean.to(u.angstrom).value
                    fit_error = fit.meanErrVel.to(u.m/u.s).value
                    fit_offset = fit.velocityOffset.to(u.m/u.s).value
                    fit_chi_squared = fit.chiSquaredNu
                    fit_ccd_correction = self._correctCCDSystematic(
                        fit.order, fit.centralIndex)
                    fit_offset_corrected = fit_offset - fit_ccd_correction
                else:
                    fit_mean = float('nan')
                    fit_error = float('nan')
                    fit_offset = float('nan')
                    fit_chi_squared = float('nan')
                    fit_ccd_correction = float('nan')
                    fit_offset_corrected = float('nan')

                obs_means.append(fit_mean)
                obs_errors.append(fit_error)
                obs_offsets.append(fit_offset)
                obs_chi_squareds.append(fit_chi_squared)
                obs_ccd_corrections.append(fit_ccd_correction)
                obs_offsets_corrected.append(fit_offset_corrected)

            # Create a list of each type of data for each observation.
            means_list.append(obs_means)
            errors_list.append(obs_errors)
            offsets_list.append(obs_offsets)
            chi_squared_list.append(obs_chi_squareds)
            ccd_corrections_list.append(obs_ccd_corrections)
            offsets_corrected_list.append(obs_offsets_corrected)
            self.obsRVOffsetsArray[obs_num] = np.nanmedian(
                obs_offsets_corrected)

        # Collate the above lists into arrays containing the results of all
        # observations of the given star.
        self.fitMeansArray = u.unyt_array(np.asarray(means_list),
                                          u.angstrom)
        self.fitErrorsArray = u.unyt_array(np.asarray(errors_list),
                                           u.m/u.s)
        self.fitOffsetsArray = u.unyt_array(np.asarray(offsets_list),
                                            u.m/u.s)
        self.obsRVOffsetsArray *= u.m / u.s
        self.ccdCorrectionArray = u.unyt_array(ccd_corrections_list,
                                               u.m/u.s)
        self.fitOffsetsCorrectedArray = u.unyt_array(offsets_corrected_list,
                                                     u.m/u.s)
        self.fitOffsetsNormalizedArray = self.fitOffsetsCorrectedArray -\
            self.obsRVOffsetsArray
        self.bervArray *= u.km / u.s
        self.chiSquaredNuArray = np.array(chi_squared_list)

        transition_labels = []
        for transition in self.transitionsList:
            for order_num in transition.ordersToFitIn:
                transition_labels.append('_'.join((transition.label,
                                                   str(order_num))))

        pair_labels = []
        for pair in self.pairsList:
            for order_num in pair.ordersToMeasureIn:
                pair_labels.append('_'.join((pair.label, str(order_num))))

        self._pair_bidict = self.PairMap({pair_label: num for num,
                                          pair_label in
                                          enumerate(pair_labels)})
        self._transition_bidict = self.TransitionMap({transition_label: num
                                                      for num, transition_label
                                                      in enumerate(
                                                          transition_labels)})

    def getPairSeparations(self):
        """Create attributes containing pair separations and associated errors.

        This method creates attributes called pairSeparationsArray and
        pairSepErrorsArray containing lists of pair separations and associated
        errors in each row corresponding to an observation of this star.
        """

        # Set up the arrays for pair separations and errors
        pairSeparationsArray = np.empty([len(self._obs_date_bidict),
                                         len(self._pair_bidict)])
        pairSepErrorsArray = np.empty([len(self._obs_date_bidict),
                                       len(self._pair_bidict)])

        for pair in self.pairsList:
            for order_num in pair.ordersToMeasureIn:
                pair_label = '_'.join((pair.label, str(order_num)))
                label1 = '_'.join((pair._higherEnergyTransition.label,
                                   str(order_num)))
                label2 = '_'.join((pair._lowerEnergyTransition.label,
                                   str(order_num)))

                pairSeparationsArray[:, self.p_index(pair_label)]\
                    = wave2vel(self.fitMeansArray[:, self.t_index(label1)],
                               self.fitMeansArray[:, self.t_index(label2)])

                self.pairSeparationsArray = u.unyt_array(pairSeparationsArray,
                                                         units='m/s')

                pairSepErrorsArray[:, self.p_index(pair_label)]\
                    = np.sqrt(self.fitErrorsArray[:, self.t_index(label1)]**2,
                              self.fitErrorsArray[:, self.t_index(label2)]**2)

                self.pairSepErrorsArray = u.unyt_array(pairSepErrorsArray,
                                                       units='m/s')

    def getOutliersMask(self, fit_results_dict, n_sigma=2.5,
                        dump_cache=False):
        """Return a 2D mask for values in this star's transition measurements.

        This method takes a function of three stellar parameters (temperature,
        metallicity, and absolute magnitude) and a variable number of
        coefficients. These coefficients are provided in a dictionary for each
        transition, for pre- and post-fiber change instances. It then calculates
        a correction for each observation's fitted wavelength and checks if the
        resultant position is more than `n_sigma` times the standard deviation
        for that transition given in `sigmas_dict` from zero.

        fit_results_dict : dict
            A dictionary containing results from a call to
            varconlib.miscellaneous.get_params_file(), with information from a
            fitting model.

        Optional
        --------
        n_sigma : float, Default : 2.5
            The number of standard deviations a point must be away from the
            value for this transition found by correcting using the fitting
            function to be considered an outlier (and thus masked).
        dump_cache : bool, Default : False
            If true, any cached values will be cleared, allowing the calculation
            and creation of a new model-corrected array and outliers mask. Only
            necessary if fitting using different masks on the same `Star` in the
            same session, as the cached values are not saved and do not persist.

        Returns
        -------
        length-2 tuple
            A tuple containing two 2D-arrays: 'corrected_array' which holds the
            values of `self.fitOffsetsNormalizedArray` corrected by the fitting
            model provided, and 'mask_array' which contains a sequence of zeros
            and ones which can be used as a mask in a NumPy `MaskedArray`.
            Measurements greater than `n_sigma` away from zero are considered to
            be outliers and masked accordingly.

        """

        if dump_cache:
            try:
                delattr(self, '_cachedOutliers')
                delattr(self, '_cachedMask')
            # If someone tries to use dump_cache before a cache exists:
            except AttributeError:
                pass

        if hasattr(self, '_cachedOutliers') and hasattr(self, '_cachedMask'):
            return (self._cachedOutliers, self._cachedMask)

        function = fit_results_dict['model_func']
        coeffs_dict = fit_results_dict['coeffs']

        stellar_params = np.stack((self.temperature, self.metallicity,
                                   self.absoluteMagnitude), axis=0)

        corrected_array = np.zeros_like(self.fitOffsetsArray, dtype=float)
        # Initialize the mask to False (not masked) with the same shape
        # as the data for this star:
        mask_array = np.full_like(self.fitOffsetsNormalizedArray,
                                  fill_value=0, dtype=int)

        for key in tqdm(self._transition_bidict.keys()):
            col_num = self._transition_bidict.index_for[key]

            if self.hasObsPre:
                label = key + '_pre'
                pre_slice = slice(None, self.fiberSplitIndex)

                correction = u.unyt_array(function(stellar_params,
                                                   *coeffs_dict[label]),
                                          units=u.m/u.s)
                corrected_array[pre_slice, col_num] =\
                    self.fitOffsetsNormalizedArray[pre_slice, col_num] -\
                    correction
                data_slice = corrected_array[pre_slice, col_num]
                error_slice = self.fitErrorsArray[pre_slice, col_num]

                for i in range(len(data_slice)):
                    if abs(data_slice[i]) > n_sigma * error_slice[i]:
                        mask_array[i, col_num] = 1

            if self.hasObsPost:
                label = key + '_post'
                post_slice = slice(self.fiberSplitIndex, None)

                correction = u.unyt_array(function(stellar_params,
                                                   *coeffs_dict[label]),
                                          units=u.m/u.s)
                corrected_array[post_slice, col_num] =\
                    self.fitOffsetsNormalizedArray[post_slice, col_num] -\
                    correction
                data_slice = corrected_array[post_slice, col_num]
                error_slice = self.fitErrorsArray[post_slice, col_num]
                for i in range(len(data_slice)):
                    if abs(data_slice[i]) > n_sigma * error_slice[i]:
                        mask_array[i+self.fiberSplitIndex, col_num] = 1

        self._cachedOutliers = corrected_array
        self._cachedMask = mask_array

        return (corrected_array, mask_array)

    def saveDataToDisk(self, file_path=None):
        """Save important data arrays to disk in HDF5 format.

        Saves various arrays which are time-consuming to create to disk in HDF5
        format for easy retrieval later.

        Parameters
        ----------
        file_path : `pathlib.Path` or `str`
            The file name to save the data to. If `str`, will be converted to
            a `Path` object.

        """

        if file_path is None:
            file_path = self.hdf5file
        else:
            file_path = Path(file_path)

        if file_path.exists():
            # Save the previously existing file as a backup.
            backup_path = file_path.with_suffix(".bak")
            os.replace(file_path, backup_path)

        for dataset_name, attr_name in self.unyt_arrays.items():
            getattr(self, attr_name).write_hdf5(file_path,
                                                dataset_name=dataset_name)

        with h5py.File(file_path, mode='a') as f:

            for path_name, attr_name in self.other_attributes.items():
                hickle.dump(getattr(self, attr_name), f, path=path_name)

    def constructFromHDF5(self, filename):
        """Retrieve datasets from HDF5 file.

        Loads data previously saved to disc into an initialized `Star` object,
        ready for use without needed to create or collate it again.

        Paramters
        ---------
        filename : `pathlib.Path` or `str`
            An HDF5 file name to retrieve previously-saved data from.

        """

        for dataset_name, attr_name in self.unyt_arrays.items():
            dataset = u.unyt_array.from_hdf5(filename,
                                             dataset_name=dataset_name)
            setattr(self, attr_name, dataset)

        with h5py.File(filename, mode='r') as f:

            for path_name, attr_name in self.other_attributes.items():
                try:
                    setattr(self, attr_name, hickle.load(f, path=path_name))
                except AttributeError:
                    print(f'Attribute "{attr_name}" with path "{path_name}"'
                          f' was not found in saved data for {self.name}.')

    def _correctCCDSystematic(self, order, pixel):
        """Return the velocity correction for a given pixel and order number.

        This function corrects measurements of velocity offsets from their
        expected position using the binned residuals found by Milankovic et al.
        2020 using HARPS' laser frequency comb.

        Parameters
        ----------
        order : int
            The HARPS order on which the observation was made. This should be an
            integer in the range [0, 71].
        pixel : int
            The horizontal pixel position on the HARPS CCD where the center of
            the observed feature was found. Should be an integer in the range
            [0, 4096].

        Returns
        -------
        float
            A quantity representing the amount to correct the given feature
            measurement by in order to correct for CCD systematics.
            (Numerically it will be a value in m/s, though without units.)

        """

        try:
            order = int(order)
        except ValueError:
            print(f'Given "order" can not be cast to an integer: {order}')
        try:
            pixel = int(pixel)
        except ValueError:
            print(f'Given "pixel" can not be cast to an integer: {pixel}')

        assert 0 <= order <= 72, 'Given "order" not in [0, 71]!'
        assert 0 <= pixel <= 4096, 'Given "pixel" not in [0, 4096]!'

        if 61 <= order <= 71:
            block = 'block1'
        elif 46 <= order <= 60:
            block = 'block2'
        else:
            block = 'block3'

        shift_dict = self.ccdSystematicsDict[block]

        key_pos = sorted(shift_dict.keys())
        if pixel < key_pos[0]:
            return shift_dict[key_pos[0]]
        elif pixel > key_pos[-1]:
            return shift_dict[key_pos[-1]]
        else:
            for pos in key_pos:
                if pixel == pos:
                    return shift_dict[pos]
                elif pixel < pos:
                    # Find the equation of the line made by interpolating
                    # between the two surrounding points so we can find the
                    # value to return.
                    x1, x2 = pos - 64, pos
                    y1, y2 = shift_dict[x1], shift_dict[x2]
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - (m * x1)
                    result = m * pixel + b
                    assert -15 < result < 15,\
                        f'{pixel}, {order}: ({x1}, {y1}), ({x2}, {y2}),' +\
                        ' {m}, {b}, {result}'
                    return result
            # If for some reason it runs through everything and still doesn't
            # find an answer, raise an error:
            raise RuntimeError('Unable to find correction!')

    @property
    def ccdSystematicsDict(self):
        """Return the CCD systematic offset data.

        The order used to refer to HARPS orders is a contunous zero-based index
        counting the orders present on the HARPS detector from 0 to 72. It does
        not skip echelle order 115 which is physically not present as it falls
        between the two CCDs. The relationship between echelle order and this
        numbering scheme is seen in the following table:

        Block | Echelle orders | 0-based continuous indexing
        ----------------------------------------------------
        |  1  |    89 - 99     |   61 - 71                 |
        |  2  |   100 - 114    |   46 - 60                 |
        |  3  |   116 - 134    |   26 - 45                 |
        |  4  |   135 - 161    |    0 - 25                 |
        ----------------------------------------------------

        Note that block 4 doesn't have its own residual values, so it uses those
        from block 3.

        Returns
        -------
        dict
            A dictionary containing as keys a list of pixel positions along the
            CCD, and as values the measured smoothed velocity offsets from zero
            in m/s.

        """

        if not hasattr(self, '_ccdSystematicDict'):
            self._ccdSystematicDict = {}

            data_dir = vcl.data_dir / 'residual_data'

            for num, block_dict in enumerate(('block1', 'block2', 'block3')):
                data_file = data_dir /\
                    f'residuals_block_{num+1}_forDB_64bins.txt'
                data = np.loadtxt(data_file, skiprows=2, dtype=float)
                pos_dict = {int(key): value for key, value in zip(data[:, 0],
                                                                  data[:, 1])}
                self._ccdSystematicDict[block_dict] = pos_dict

        return self._ccdSystematicDict

    @property
    def transitionsList(self):
        """Return the list of transitions used by this `Star`."""
        if not hasattr(self, '_transitions_list'):
            # Read the default list of chosen transitions.
            with open(vcl.final_selection_file, 'r+b') as f:
                self._transitions_list = pickle.load(f)
        return self._transitions_list

    @transitionsList.setter
    def transitionsList(self, transitions_list):
        if isinstance(transitions_list, list):
            if isinstance(transitions_list[0], vcl.transition_line.Transition):
                self._transitions_list = transitions_list

    @property
    def pairsList(self):
        """Return the list of transition pairs used by this `Star`."""
        if not hasattr(self, '_pairs_list'):
            # Read the default list of chosen pairs.
            with open(vcl.final_pair_selection_file, 'r+b') as f:
                self._pairs_list = pickle.load(f)
        return self._pairs_list

    @pairsList.setter
    def pairsList(self, pairs_list):
        if pairs_list is not None:
            if isinstance(pairs_list, list):
                if isinstance(pairs_list[0],
                              vcl.transition_pair.TransitionPair):
                    self._pairs_list = pairs_list

    @property
    def radialVelocity(self):
        """Return the radial velocity of this star."""
        if self._radialVelocity is None:
            rv_file = vcl.data_dir / 'StellarRadialVelocities.txt'
            rv_dict = {}
            with open(rv_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    name, rv = line.rstrip().split(' ')
                    rv_dict[name] = float(rv) * u.km / u.s
                try:
                    self._radialVelocity = rv_dict[self.name]
                except KeyError:
                    print(f"{self.name} not found in rv_dict")
                    return None
        return self._radialVelocity

    @radialVelocity.setter
    def radialVelocity(self, new_RV):
        assert new_RV.units.dimensions == dimensions.length / dimensions.time,\
            f'New radial velocity has units {new_RV.units.dimensions}!'
        self._radialVelocity = new_RV

    @property
    def temperature(self):
        """Return the temperature of this star."""
        if self._temperature is None:
            self._temperature = self._getStellarProperty('temperature')
        return self._temperature

    @temperature.setter
    def temperature(self, new_T):
        assert new_T.units.dimensions == dimensions.temperature,\
            f'New temperature has units {new_T.units}!'
        self._temperature = new_T

    @property
    def metallicity(self):
        """Return the metallicity of this star."""
        if self._metallicity is None:
            self._metallicity = self._getStellarProperty('metallicity')
        return self._metallicity

    @metallicity.setter
    def metallicity(self, new_mtl):
        if not isinstance(new_mtl, (float, int)):
            new_mtl = float(new_mtl)
        assert -5. < new_mtl < 2., f'Metallicity value seems odd: {new_mtl}.'
        self._metallicity = new_mtl

    @property
    def absoluteMagnitude(self):
        """Return the absolute magnitude of this star."""
        if self._absoluteMagnitude is None:
            self._absoluteMagnitude = self._getStellarProperty(
                    'absoluteMagnitude')
        return self._absoluteMagnitude

    @absoluteMagnitude.setter
    def absoluteMagnitude(self, new_mag):
        if not isinstance(new_mag, (float)):
            new_mag = float(new_mag)
        self._absoluteMagnitude = new_mag

    @property
    def apparentMagnitude(self):
        """Return the apparent magnitude of this star."""
        if self._apparentMagnitude is None:
            self._apparentMagnitude = self._getStellarProperty(
                    'apparentMagnitude')
        return self._apparentMagnitude

    @apparentMagnitude.setter
    def apparentMagnitude(self, new_mag):
        if not isinstance(new_mag, (float)):
            new_mag = float(new_mag)
        self._apparentMagnitude = new_mag

    @property
    def logg(self):
        """Return the logarithm of the surface gravity of this star."""
        if self._logg is None:
            self._logg = self._getStellarProperty('logg')
        return self._logg

    @logg.setter
    def logg(self, new_log_g):
        if not isinstance(new_log_g, (float, int)):
            new_log_g = float(new_log_g)
        self._logg = new_log_g

    # TODO: Add an exposure time array?
    @property
    def fiberSplitIndex(self):
        """Find the point to split between pre- and post-fiber change dates."""
        if not hasattr(self, '_fiberSplitIndex'):
            self._fiberSplitIndex = self._getFiberSplitIndex()

        return self._fiberSplitIndex

    @property
    def numObsPre(self):
        """Return the number of observations pre-fiber-change for this star."""
        if self.fiberSplitIndex == 0:
            return 0
        elif self.fiberSplitIndex is None:
            return self.getNumObs()
        else:
            return self.getNumObs(slice(None, self.fiberSplitIndex))

    @property
    def numObsPost(self):
        """Return the number of observations post-fiber-change for this star."""
        if self.fiberSplitIndex == 0:
            return self.getNumObs()
        elif self.fiberSplitIndex is None:
            return 0
        else:
            return self.getNumObs(slice(self.fiberSplitIndex, None))

    def getSpecialAttributes(self, star_dir):
        """
        Read and save any special attributes for the star from an external file.

        In order to capture certain factor which apply only to some stars, this
        function will read an optional JSON file in the star's directory and
        save the results to a dictionary as the attribute `specialAttributes`.

        Parameters
        ----------
        star_dir : `pathlib.Path`
            The path to the directory where the star is being constructed/read
            from.

        Returns
        -------
        dict
            A dictionary which will contain the contents of the specially-named
            JSON file in `star_dir. May be empty if no such file exists.

        """

        json_file = star_dir / f'{self.name}_special_attributes.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def getStellarParameters(self, paper_name):
        """
        Set the stellar parameters based on the paper given.

        Parameters
        ----------
        paper_name : str, ['Nordstrom2004', 'Casagrande2011']
            The name of the paper from which to use the stellar parameters.

        Returns
        -------
        None.

        """

        hd_name = re.compile(r'^HD')
        if hd_name.match(self.name):
            star_name = self.name[:2] + ' ' + self.name[2:]
        else:
            # If it's Vesta/the Sun:
            star_name = self.name

        if paper_name == 'Nordstrom2004':
            self._getNordstrom2004(star_name)
        elif paper_name == 'Casagrande2011':
            self._getCasagrande2011(star_name)
        else:
            f'{paper_name} is not a valid paper name!'

    def _getStellarProperty(self, value):
        """Load file containing stellar properties and return the requested
        value.

        Parameters
        ----------
        value, str : ['temperature', 'metallicity', 'absoluteMagnitude',
                      'apparentMagnitude', 'logg']
            The property of the star to return.

        Returns
        -------
        float or `unyt.unyt_quantity`
            The value of the requested property for the star, either as a float
            if it has no units or as a `unyt_quantity` if it does.

        """

        row_values = {'temperature': 3, 'metallicity': 4,
                      'absoluteMagnitude': 5, 'apparentMagnitude': 6,
                      'logg': 9}

        if value not in row_values.keys():
            raise RuntimeError('Improper value for keyword "value"!')

        stellar_props_file = vcl.data_dir / 'StellarSampleData.csv'
        data = np.loadtxt(stellar_props_file, delimiter=',', dtype=str)
        for row in data:
            if row[8] == self.name:
                if value == 'temperature':
                    return round((10 ** float(row[row_values[value]])) * u.K)
                else:
                    return float(row[row_values[value]])

        # If the star isn't found in any row, throw an error.
        raise RuntimeError(f"Couldn't find {self.name} in table!")

    def _getNordstrom2004(self, star_name):
        """
        Return values from Nordstrom et al. 2004 and set them in the star.

        star_name : str
            The name of the star to look for.

        Returns
        -------
        None.

        """

        stellar_props_file = vcl.data_dir / 'StellarSampleData.csv'
        data = np.loadtxt(stellar_props_file, delimiter=',', dtype=str)
        for row in data:
            if row[8] == star_name:
                self._temperature = round((10 ** float(row[3])) * u.K)
                self._metallicity = float(row[4])
                self._absoluteMagnitude = float(row[5])
                self._logg = float(row[9])
                break

    def _getCasagrande2011(self, star_name):
        """
        Return values from Casagrande et al. 2011 and set them in the star.

        star_name : str
            The name of the star to look for.

        Returns
        -------
        None.

        """

        stellar_props_file = vcl.data_dir /\
            'Casagrande2011_GCS_updated_values.tsv'
        data = np.loadtxt(stellar_props_file, delimiter=';', dtype=str,
                          comments='#', skiprows=48)
        for row in data:
            if row[1].strip() == star_name:
                try:
                    self._temperature = int(row[3].strip()) * u.K
                    self._metallicity = float(row[5].strip())
                    self._logg = float(row[2].strip())
                    break
                except ValueError:
                    tqdm.write(f'Missing value for {row[1]}!')

    def _getFiberSplitIndex(self):
        """Return the index of the first observation after the HARPS fiber
        change. If there are none, return None.

        There are three possibilities:
            1. All observations are after the fiber change date, in which case
               this method will return 0.
            2. Some observations are prior to the fiber change, and some after,
               in which case the returned value will be a positive integer.
            3. All observations are prior to the fiber change, in which case
               `None` will be returned as it runs off the end of the list.

        Returns
        -------
        int or None
            Either 0, a positive integer, or None, depending on the three cases
            mentioned above.
        """

        dates = [dt.datetime.fromisoformat(s) for s in
                 sorted(self._obs_date_bidict.keys())]

        for index, date in enumerate(dates):
            if date > self.fiber_change_date:
                return index

        # Else if no observation dates are after the fiber change:
        return None

    def getNumObs(self, array_slice=slice(None, None)):
        """Return the number of observations encompassed by the given array
        slice.

        Parameters
        ----------
        array_slice : slice
            A slice object to pass through to the star. If one of the values is
            `self.fiberSplitIndex` the number returned will be the number of
            observations before or after the fiber change date.

        Returns
        -------
        int
            The number of observations found in the given slice.

        """

        return len(self.fitMeansArray[array_slice])

    def getTransitionOffsetPattern(self, array_slice=slice(None, None),
                                   normalized=True):
        """Return the mean pattern of transition offsets from their expected
        position for this star.

        array_slice : slice
            A slice object to get either the pre- or post-fiber change
            observations.
        normalized : bool, Default : True
            Whether or not to normalize the value returned. The offsets will
            have their mean subtracted, then be divided by the value of the
            maximum of the remaining values. The standard deviations will also
            be divided by this value.

        Returns
        -------
        `unyt.unyt_array`
             A `unyt_array` containing two arrays corresponding to the means
             and standard deviations of all the various transitions in this
             star.

        """

        means, stddevs = [], []
        for key in self._transition_bidict.keys():
            column_index = self.t_index(key)
            offsets = ma.masked_array(self.fitOffsetsNormalizedArray[
                    array_slice, column_index])
            # A NaN value for an error will give a NaN value for the weighted
            # mean for a transition, so mask out values where that's the case.
            errors = ma.masked_invalid(self.fitErrorsArray[array_slice,
                                                           column_index])
            offsets.mask = errors.mask

            if np.any(offsets.mask):
                tqdm.write(f'{key} has at least one masked value.')
            weighted_mean = np.average(offsets, weights=1/errors**2)
            sample_std = np.std(offsets)

            means.append(weighted_mean)
            stddevs.append(sample_std)

        means *= self.fitOffsetsNormalizedArray.units
        stddevs *= self.fitOffsetsNormalizedArray.units

        if normalized:

            return u.unyt_array([means - np.nanmedian(means), stddevs])

        return u.unyt_array([means, stddevs])

    def p_index(self, label):
        """Return the index number of the column associated with this pair
        label.

        Parameters
        ----------
        label : str
            A label of a pair of transitions, including the number of the order
            in which it was measured, e.g.,
            '6769.640Ni1_6774.190Ni1_70'.

        Returns
        -------
        int
            The index number of the column corresponding to the given pair
            label in `self._pair_bidict`.

        """

        return self._pair_bidict.index_for[label]

    def p_label(self, index):
        """Return the pair label associated with this index number.

        Parameters
        ----------
        index : int
            An index number found in `self._pair_bidict.values()`.

        Returns
        -------
        str
            The pair label corresponding to the given index number in
            `self._pair_bidict.keys()`.

        """

        return self._pair_bidict.label_for[index]

    def t_index(self, label):
        """Return the index number of the column associated with this
        transition label.

        Parameters
        ----------
        label : str
            A label of a transition, including the number of the order on which
            it was fitted (zero-based), e.g., '6769.640Ni1_70'.

        Returns
        -------
        int
            The index number of the column corresponding to the given
            transition label in `self._transition_bidict.keys()`.

        """

        return self._transition_bidict.index_for[label]

    def t_label(self, index):
        """Return the transition label associated with this index number.

        Parameters
        ----------
        index : int
            An index number found in `self._transition_bidict.values()`.

        Returns
        -------
        str
            The transition label corresponding to the given index number.

        """

        return self._transition_bidict.label_for[index]

    def od_index(self, observation_date):
        """Return the index of the row associated with a given observation date
        (either as a `datetime.datetime` object or an ISO-formatted string.)

        Parameters
        ----------
        observation_date : `datetime.datetime` or (ISO-formatted) str
            The date to look for the index for. Internally the dates are saved
            as ISO-formatted strings so `datetime.datetime` objects will be
            automatically converted and tried in that format.

        Returns
        -------
        int
            The index number of the row corresponding to the given observation
            time in `self._obs_date_bidict`.

        """

        if isinstance(observation_date, str):
            return self._obs_date_bidict.index_for[observation_date]

        elif isinstance(observation_date, dt.datetime):
            return self._obs_date_bidict.index_for[observation_date.isoformat(
                                           timespec='milliseconds')]

    def od_date(self, index):
        """Return the observation datetime associated with this index number.

        Parameters
        ----------
        index : int
            An index number in `self._obs_date_bidict.values()`.

        Returns
        -------
        str
            The observation date corresponding to the given index number in
            ISO-standard format.

        """

        return self._obs_date_bidict.date_for[index]
