#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:59:22 2019

@author: dberke

A class to hold information about a single star, both metadata relating to the
star itself and arrays of information about absorption features fitted in the
various observations taken of it.

"""

from bidict import namedbidict
import datetime as dt
from glob import glob
import lzma
import os
from pathlib import Path
import pickle

import h5py
import hickle
import numpy as np
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.miscellaneous import wavelength2velocity as wave2vel


class Star(object):
    """A class to hold information gathered from analysis of (potentially many)
    observations of a single star.

    The `Star` class is intended to hold information relating to a single
    star: both information intrinsic to the star (absolute magnitude,
    metallicity, color, etc.) and information about the fits of given
    transitions in the spectra of observations of that star.

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
        between different reduction methods. Must be given if `star_dir` is
        given. Defaults to 'int' for 'integrated Gaussian' fits.
    transitions_list : list
        A list of `transition_line.Transition` objects. If `star_dir` is
        given, will be passed to `initializeFromFits`, otherwise no effect.
    pairs_list : list
        A list of `transition_pair.TransitionPair` objects. If `star_dir`
        is given, will be passed to `initializeFromFits`, otherwise no
        effect.
    load_data : bool, Default : True
        Controls whether to attempt to read a file containing data for the
        star.

    Attributes
    ----------
    fitMeansArray : `unyt.unyt_array`
        A two-dimensional array holding the mean of the fit (in wavelength
        space) for each absorption feature in each observation of the star.
        Rows correspond to observations, columns to transitions.
    fitErrorsArray : `unyt.unyt_array`
        A two-diemsional array holding the standard deviation of the measured
        mean of the fit for each absorption feature of each observation of the
        star. Rows correspond to observations, columns to transitions.
    fitOffsetsArray : `unyt.unyt_array`
        A two-dimensional array holding the offset from the expected wavelength
        of the measured mean of each absorption feature in each observation of
        the star. Rows correspond to observations, columns to transitions.
    pairSeparationsArray : `unyt.unyt_array`
        A two-dimensional array holding the velocity separation values for each
        pair of transitions for each observation of the star. Rows correspond
        to observations, columns to pairs.
    pairSepErrorsArray : `unyt.unyt_array`
        A two-dimensional array holding the standard deviation of the velocity
        separation values for each pair of transitions for each observation of
        the star. Rows correspond to observations, columns to pairs.
    bervArray : `unyt.unyt_array`
        A one-dimensional array holding the barycentric Earth radial velocity
        (BERV) for each observation of the star. Index number corresponds to
        row numbers in the two-dimensional arrays
    chiSquaredNuArray : `unyt.unyt_array`
        A two-dimensional array holding the reduced chi-squared value of the
        Gaussian fit to each transition for each observation of the star.
        Rows correspond to observations, columns to transitions.
    airmassArray : `np.ndarray`
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

    """

    # Define dataset names and corresponding attribute names to be saved
    # and loaded when dumping star data.
    unyt_arrays = {'transition_means': 'fitMeansArray',
                   'transition_errors': 'fitErrorsArray',
                   'offsets': 'fitOffsetsArray',
                   'pair_separations': 'pairSeparationsArray',
                   'pair_separation_errors': 'pairSepErrorsArray',
                   'BERV_array': 'bervArray'}

    other_attributes = {'reduced_chi_squareds': 'chiSquaredNuArray',
                        'airmasses': 'airmassArray',
                        '/obs_date_bidict': '_obs_date_bidict',
                        '/transition_bidict': '_transition_bidict',
                        '/pair_bidict': '_pair_bidict'}

    # Define some custom namedbidict objects which can be
    DateMap = namedbidict('ObservationDateMap', 'date', 'index')
    TransitionMap = namedbidict('TransitionMap', 'label', 'index')
    PairMap = namedbidict('PairMap', 'label', 'index')

    # Date of fiber change in HARPS:
    fiber_change_date = dt.datetime(year=2015, month=6, day=1,
                                    hour=0, minute=0, second=0)

    def __init__(self, name, star_dir=None, suffix='int',
                 transitions_list=None, pairs_list=None,
                 load_data=True):
        """Instantiate a `star.Star` object.

        """

        self.name = name

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

        if transitions_list:
            self.transitionsList = transitions_list
        if pairs_list:
            self.pairsList = pairs_list

        if (star_dir is not None):
            star_dir = Path(star_dir)
            h5filename = star_dir / f'{name}_data.hdf5'
            if load_data and h5filename.exists():
                self.constructFromHDF5(h5filename)
            else:
                self.constructFromDir(star_dir, suffix,
                                      pairs_list=pairs_list,
                                      transitions_list=transitions_list)
                self.getPairSeparations()
                self.dumpDataToDisk(h5filename)

        # Figure out when the observations for the star were taken, and set the
        # appropriate flags.
        if self.fiberSplitIndex is None:
            self.hasObsPre = True
        elif self.fiberSplitIndex == 0:
            self.hasObsPost = True
        else:
            self.hasObsPre = True
            self.hasObsPost = True

    def constructFromDir(self, star_dir, suffix, transitions_list=None,
                         pairs_list=None):
        """
        Collect information on fits in observations of the star, and organize
        it.

        Parameters
        ----------
        star_dir : `pathlib.Path`
            A path object representing the root directory to look in for fits
            to the star's spectra.
        suffix : str
            The suffix to affix to end of the sub-directories under the main
            star directory.

        Optional
        --------
        transitions_list : list
            A list of `transition_line.Transition` objects. If this is omitted
            the default list of transitions selected for use will be read, but
            this will be slower.
        pairs_list : list
            A list of `transition_pair.TransitionPair` objects. If this is
            omitted the default list of pairs selected for use will be read,
            but this will be slower.

        """

        # Check that the given directory exists.
        if not star_dir.exists():
            print(star_dir)
            raise RuntimeError('The given directory does not exist:'
                               f'{star_dir}')

        # Get a list of pickled fit results in the given directory.
        search_str = str(star_dir) + '/*/pickles_{}/*fits.lzma'.format(suffix)
        pickle_files = [Path(path) for path in glob(search_str)]

        means_list = []
        errors_list = []
        offsets_list = []
        chi_squared_list = []
        self.bervArray = np.empty(len(pickle_files))
        self.airmassArray = np.empty(len(pickle_files))

        # For each pickle file:
        for obs_num, pickle_file in enumerate(tqdm(pickle_files[:])):

            with lzma.open(pickle_file, 'rb') as f:
                fits_list = pickle.loads(f.read())

            # Save the observation date.
            # ??? Maybe save as datetime objects rather than strings?
            self._obs_date_bidict[fits_list[0].dateObs.isoformat(
                               timespec='milliseconds')] = obs_num
            # Save the BERV and airmass.
            self.bervArray[obs_num] = fits_list[0].BERV
            self.airmassArray[obs_num] = fits_list[0].airmass

            means_list.append([fit.mean.to(u.angstrom).value for
                               fit in fits_list])

            errors_list.append([fit.meanErrVel.to(u.m/u.s).value for
                                fit in fits_list])

            offsets_list.append([fit.velocityOffset.to(u.m/u.s).value
                                 for fit in fits_list])

            chi_squared_list.append([fit.chiSquaredNu for fit in fits_list])

        means_units = fits_list[0].mean.units
        errors_units = fits_list[0].meanErrVel.units
        offsets_units = fits_list[0].velocityOffset.units
        berv_units = fits_list[0].BERV.units

        self.fitMeansArray = u.unyt_array(np.asarray(means_list),
                                          means_units)
        self.fitErrorsArray = u.unyt_array(np.asarray(errors_list),
                                           errors_units)
        self.fitOffsetsArray = u.unyt_array(np.asarray(offsets_list),
                                            offsets_units)
        self.bervArray *= berv_units
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

    def dumpDataToDisk(self, file_path):
        """Save important data arrays to disk in HDF5 format.

        Saves various arrays which are time-consuming to create to disk in HDF5
        format for easy retrieval later.

        Parameters
        ----------
        file_path : `pathlib.Path` object
            The file name to save the data to.

        """

        if file_path.exists():
            backup_path = file_path.with_name(file_path.stem + ".bak")
            os.replace(file_path, backup_path)
        with h5py.File(file_path, mode='a') as f:

            for dataset_name, attr_name in self.unyt_arrays.items():
                getattr(self, attr_name).write_hdf5(file_path,
                                                    dataset_name=dataset_name)
            for path_name, attr_name in self.other_attributes.items():
                hickle.dump(getattr(self, attr_name), f, path=path_name)

    def constructFromHDF5(self, filename):
        """Retrieve datasets from HDF5 file.

        Loads data previously saved to disc into an initialized `Star` object,
        ready for use without needed to create or collate it again.

        Paramters
        ---------
        filename : `pathlib.Path` or str
            An HDF5 file name to retrieve previously-saved data from.

        """

        with h5py.File(filename, mode='r') as f:

            for dataset_name, attr_name in self.unyt_arrays.items():
                dataset = u.unyt_array.from_hdf5(filename,
                                                 dataset_name=dataset_name)
                setattr(self, attr_name, dataset)

            for path_name, attr_name in self.other_attributes.items():
                setattr(self, attr_name, hickle.load(f, path=path_name))

    @property
    def transitionsList(self):
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

# TODO: Add an exposure time array?
    @property
    def fiberSplitIndex(self):
        if not hasattr(self, '_fiberSplitIndex'):
            self._fiberSplitIndex = self._getFiberSplitIndex()

        return self._fiberSplitIndex

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

    def getTransitionOffsetPattern(self, array_slice=slice(None, None)):
        """Return the mean pattern of transition offsets from their expected
        position for this star.

        """

        means, stddevs = [], []
        for key in self._transition_bidict.keys():
            column_index = self.t_index(key)
            offsets = self.fitOffsetsArray[array_slice, column_index]
            errors = self.fitErrorsArray[array_slice, column_index]

            weighted_mean = np.average(offsets, weights=1/errors**2)
            sample_std = np.std(offsets)

            means.append(weighted_mean)
            stddevs.append(sample_std)

        means *= self.fitOffsetsArray.units
        stddevs *= self.fitOffsetsArray.units

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
