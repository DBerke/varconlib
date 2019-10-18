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

import numpy as np
from tqdm import tqdm

import varconlib as vcl
from varconlib.miscellaneous import wavelength2velocity as wave2vel


class Star(object):

    def __init__(self, name, star_dir=None, suffix='int',
                 transitions_list=None, pairs_list=None):
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

        """

        self.name = name

        if (star_dir is not None):
            self.initializeFromDir(Path(star_dir), suffix,
                                   pairs_list=pairs_list,
                                   transitions_list=transitions_list)

    def initializeFromDir(self, star_dir, suffix, transitions_list=None,
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

        if transitions_list is not None:
            if isinstance(transitions_list, list):
                if isinstance(transitions_list[0],
                              vcl.transition_line.Transition):
                    self.transitions_list = transitions_list
        else:
            # Read the list of chosen transitions.
            with open(vcl.final_selection_file, 'r+b') as f:
                self.transitions_list = pickle.load(f)

        if pairs_list is not None:
            if isinstance(pairs_list, list):
                if isinstance(pairs_list[0],
                              vcl.transition_pair.TransitionPair):
                    self.pairs_list = pairs_list
        else:
            # Read the list of chosen pairs.
            with open(vcl.final_pair_selection_file, 'r+b') as f:
                self.pairs_list = pickle.load(f)

        # Get a list of pickled fit results in the given directory.
        search_str = str(star_dir) + '/*/pickles_{}/*fits.lzma'.format(suffix)
        pickle_files = [Path(path) for path in glob(search_str)]

        self._obs_date_dict = {}
        means_list = []
        errors_list = []

        # For each pickle file:
        for obs_num, pickle_file in tqdm(enumerate(pickle_files)):

            # Import it, read the list of fits inside, save their means to
            # a Series with index made up of the fit labels, and their errors
            # to a similar Series.
            with lzma.open(pickle_file, 'rb') as f:
                fits_list = pickle.loads(f.read())

            # Save the observation date.
            self._obs_date_dict[fits_list[0].dateObs.isoformat(
                               timespec='milliseconds')] = obs_num
            transition_dict = {fit.label: num for num, fit in
                               enumerate(fits_list)}
            means_list.append([fit.mean for fit in fits_list])
            errors_list.append([fit.meanErrVel for fit in fits_list])

        self._transition_label_dict = transition_dict
        self.tMeansArray = np.array(means_list, ndmin=2)
        self.tErrorsArray = np.array(errors_list, ndmin=2)

        # Now generate the pair separation values:
        pair_labels = []
        for pair in self.pairs_list:
            for order_num in pair.ordersToMeasureIn:
                pair_labels.append('_'.join((pair.label, str(order_num))))

        self._pair_label_dict = {pair_label: num for num, pair_label in
                                 enumerate(pair_labels)}

        # Set up the arrays for pair separations and errors
        self.pSeparationsArray = np.empty([len(self._obs_date_dict),
                                           len(self._pair_label_dict)])
        self.pSepErrorsArray = np.empty([len(self._obs_date_dict),
                                         len(self._pair_label_dict)])

        for pair, pair_label in zip(self.pairs_list, pair_labels):
            for order_num in pair.ordersToMeasureIn:
                label1 = '_'.join((pair._higherEnergyTransition.label,
                                   str(order_num)))
                label2 = '_'.join((pair._lowerEnergyTransition.label,
                                   str(order_num)))

                self.pSeparationsArray[:, self._p_label(pair_label)]\
                    = wave2vel(self.tMeansArray[:, self._t_label(label1)],
                               self.tMeansArray[:, self._t_label(label2)])
                self.pSepErrorsArray[:, self._p_label(pair_label)]\
                    = np.sqrt(self.tErrorsArray[:, self._t_label(label1)] ** 2,
                              self.tErrorsArray[:, self._t_label(label2)] ** 2)

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

        if not hasattr(self, '_obs_date_dict'):
            raise AttributeError('self._obs_date_dict not instantiated yet')

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
