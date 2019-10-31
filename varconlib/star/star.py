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
import lzma
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

    # Date of fiber change in HARPS:
    fiber_change_date = dt.datetime(year=2015, month=6, day=1,
                                    hour=0, minute=0, second=0)

    def __init__(self, name, star_dir=None, suffix='int',
                 transitions_list=None, pairs_list=None,
                 load_data=True, dump_data=True):
        """Instantiate a Star object.

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
        dump_data : bool, Default : True
            Whether or not to save the data collected in an HDF5 file for
            faster recovery later.

        """

        self.name = name

        # Initialize some attributes to be filled later.
        self._obs_date_dict = {}
        self.fifitMeansArray = None
        self.fitErrorsArray = None
        self.pairSeparationsArray = None
        self.pairSepErrorsArray = None

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
                if dump_data:
                    self.dumpDataToDisk(h5filename)

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

        # For each pickle file:
        for obs_num, pickle_file in enumerate(tqdm(pickle_files[:])):

            # Import it, read the list of fits inside, save their means to
            # a Series with index made up of the fit labels, and their errors
            # to a similar Series.
            with lzma.open(pickle_file, 'rb') as f:
                fits_list = pickle.loads(f.read())

            # Save the observation date.
            self._obs_date_dict[fits_list[0].dateObs.isoformat(
                               timespec='milliseconds')] = obs_num
            means_list.append(np.array([fit.mean.to(u.angstrom).value for
                                        fit in fits_list]))
            means_units = fits_list[0].mean.units
            errors_list.append(np.array([fit.meanErrVel.to(u.m/u.s).value for
                                         fit in fits_list]))
            errors_units = fits_list[0].meanErrVel.units

        means_array = np.array(means_list)
        errors_array = np.array(errors_list)

        self.fitMeansArray = means_array * means_units
        self.fitErrorsArray = errors_array * errors_units

        transition_labels = []
        for transition in self.transitionsList:
            for order_num in transition.ordersToFitIn:
                transition_labels.append('_'.join((transition.label,
                                         str(order_num))))

        pair_labels = []
        for pair in self.pairsList:
            for order_num in pair.ordersToMeasureIn:
                pair_labels.append('_'.join((pair.label, str(order_num))))

        self._pair_label_dict = {pair_label: num for num,
                                 pair_label in
                                 enumerate(pair_labels)}
        self._transition_label_dict = {transition_label: num for
                                       num, transition_label in
                                       enumerate(transition_labels)}

    def getPairSeparations(self):
        """Create attributes containing pair separations and associated errors.

        This method creates attributes called pairSeparationsArray and
        pairSepErrorsArray containing lists of pair separations and associated
        errors in each row corresponding to an observation of this star.
        """

        # Set up the arrays for pair separations and errors
        pairSeparationsArray = np.empty([len(self._obs_date_dict),
                                         len(self._pair_label_dict)])
        pairSepErrorsArray = np.empty([len(self._obs_date_dict),
                                       len(self._pair_label_dict)])

        for pair in self.pairsList:
            for order_num in pair.ordersToMeasureIn:
                pair_label = '_'.join((pair.label, str(order_num)))
                label1 = '_'.join((pair._higherEnergyTransition.label,
                                   str(order_num)))
                label2 = '_'.join((pair._lowerEnergyTransition.label,
                                   str(order_num)))

                pairSeparationsArray[:, self._p_label(pair_label)]\
                    = wave2vel(self.fitMeansArray[:, self._t_label(label1)],
                               self.fitMeansArray[:, self._t_label(label2)])

                self.pairSeparationsArray = u.unyt_array(pairSeparationsArray,
                                                         units='m/s')

                pairSepErrorsArray[:, self._p_label(pair_label)]\
                    = np.sqrt(self.fitErrorsArray[:, self._t_label(label1)]**2,
                              self.fitErrorsArray[:, self._t_label(label2)]**2)

                self.pairSepErrorsArray = u.unyt_array(pairSepErrorsArray,
                                                       units='m/s')

    def dumpDataToDisk(self, file_path):
        """Save important data arrays to disk in HDF5 format.

        """

        dataset_names = ('transition_means', 'transition_errors',
                         'pair_separations', 'pair_separation_errors')
        datasets = (self.fitMeansArray, self.fitErrorsArray,
                    self.pairSeparationsArray, self.pairSepErrorsArray)

        if not file_path.exists():
            with h5py.File(file_path, mode='a') as f:

                for dataset_name, dataset in zip(dataset_names, datasets):
                    f.create_dataset(dataset_name, data=dataset)
                    f[dataset_name].attrs['units'] = str(dataset.units)
                hickle.dump(self._obs_date_dict, f, path='/obs_date_dict')
                hickle.dump(self._transition_label_dict, f,
                            path='/transition_label_dict')
                hickle.dump(self._pair_label_dict, f,
                            path='/pair_label_dict')

    def constructFromHDF5(self, filename):
        """Retrieve datasets from HDF5 file.

        """

        dataset_names = ('transition_means', 'transition_errors',
                         'pair_separations', 'pair_separation_errors')
        attr_names = ('fitMeansArray', 'fitErrorsArray',
                      'pairSeparationsArray', 'pairSepErrorsArray')

        with h5py.File(filename, mode='r') as f:

            for dataset_name, attr_name in zip(dataset_names, attr_names):
                units_added_dataset = u.unyt_array(f[dataset_name],
                                                   units=f[dataset_name]
                                                   .attrs['units'])
                setattr(self, attr_name, units_added_dataset)
            self._obs_date_dict = hickle.load(f, path='/obs_date_dict')
            self._transition_label_dict = hickle.load(f, path='/transition_'
                                                      'label_dict')
            self._pair_label_dict = hickle.load(f, path='/pair_label_dict')

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
                 sorted(self._obs_date_dict.keys())]

        for index, date in enumerate(dates):
            if date > self.fiber_change_date:
                return index

        # Else if no observation dates are after the fiber change:
        return None

    def _p_label(self, label):
        """Return the index number of the column associated with this pair
        label.

        Parameters
        ----------
        label : str
            A label of a pair of transitions, including the number of the order
            in which it was measured, e.g.,
            '6769.640Ni1_6774.190Ni1_70'.

        """

        if not hasattr(self, '_pair_label_dict'):
            raise AttributeError('self._pair_label_dict not yet instantiated')

        else:
            return self._pair_label_dict[label]

    def _t_label(self, label):
        """Return the index number of the column associated with this
        transition label.

        Parameters
        ----------
        label : str
            A label of a transition, including the number of the order on which
            it was fitted (zero-based), e.g., '6769.640Ni1_70'.

        """

        if not hasattr(self, '_transition_label_dict'):
            raise AttributeError('self._transition_label_dict not yet'
                                 ' instantiated')
        else:
            return self._transition_label_dict[label]

    def _o_label(self, observation_date):
        """Return the index associated with a given observation date (either as
        a `datetime.datetime` object or an ISO-formatted string.)

        Parameters
        ----------
        observation_date : `datetime.datetime` or (ISO-formatted) str
            The date to look for the index for. Internally the dates are saved
            as ISO-formatted strings so `datetime.datetime` objects will be
            automatically converted and tried in that format.

        """

        if isinstance(observation_date, str):
            return self._obs_date_dict[observation_date]

        elif isinstance(observation_date, dt.datetime):
            try:
                return self._obs_date_dict[observation_date.isoformat(
                                           timespec='milliseconds')]
            except KeyError:
                raise KeyError('self._obs_date_dict does not have a key'
                               ' corresponding to the ISO-formatted string'
                               ' from the given datetime')
