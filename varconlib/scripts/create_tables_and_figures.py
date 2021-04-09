#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:21:59 2020

@author: dberke

This script creates the necessary figures and tables for my two papers and
thesis.

"""

import argparse
import csv
from inspect import signature
from itertools import tee
from glob import glob
from math import ceil
import os
from pathlib import Path
import pickle

from adjustText import adjust_text
import cmasher as cmr
import h5py
import hickle
import numpy as np
import numpy.ma as ma
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting as fit
from varconlib.miscellaneous import (remove_nans, weighted_mean_and_error,
                                     get_params_file)
from varconlib.star import Star
from varconlib.transition_line import roman_numerals


def pairwise(iterable):
    """Return successive pairs from an iterable.

    E.g., s -> (s0,s1), (s1,s2), (s2, s3), ...

    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_weighted_mean(values_array, errs_array, time_slice, col_index):
    """Get the weighted mean of a column in an array avoiding NaNs.

    This function is intended to get the weighted mean of the values for either
    a transition or pair from star, from a give time period using the given
    column index.

    It is CRITICAL that you check if the star has observations in the pre- or
    post-fiber change era before calling this function; due to a quirk, a star
    with observation only in the pre-fiber change era will return all of its
    observations if given a timeslice for its post-change observations, so only
    call this after checking a star actually has observations for the era in
    question.

    Parameters
    ----------
    values_array : array-like
        The array from which to get the values (specifically, some array from
        a `varconlib.star.Star` object).
    errs_array : array-like
        An array of the same shape as the array given to `values_array`.
    time_slice : Slice
        A Slice object.
    col_index : int
        The index of the column to get the values from.

    Returns
    -------
    tuple, length-2 of floats
        A tuple containing the weighted mean and error on the weighted mean for
        the given slice and era.

    """

    values, mask = remove_nans(values_array[time_slice, col_index],
                               return_mask=True)
    errs = errs_array[time_slice, col_index][mask]

    try:
        return weighted_mean_and_error(values, errs)
    except ZeroDivisionError:
        return (np.nan * values.units, np.nan * values.units)


def create_HR_diagram_plot():
    """
    Create an HR diagram of the stars in our sample.

    Returns
    -------
    None.

    """

    star_names = [d.split('/')[-1] for d in
                  glob('/Users/dberke/data_output/[H]*')]

    star_list = []
    temp_lim = 6077 * u.K
    metal_lim = -0.45
    tqdm.write('Gathering stars...')
    for star_name in tqdm(star_names):
        star = Star(star_name, vcl.output_dir / star_name)
        if star.temperature > temp_lim:
            del star
        elif star.metallicity < metal_lim:
            del star
        else:
            star_list.append(star)

    colors, mags = [], []
    for star in tqdm(star_list):
        colors.append(star.color)
        mags.append(star.absoluteMagnitude)

    vesta = Star('Vesta', '/Users/dberke/data_output/Vesta')

    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel(r'$(b-y)$ color')
    ax.set_ylabel(r'M$_\mathrm{V}$')
    ax.set_ylim(bottom=5.8, top=3.87)
    ax.set_xlim(left=0.34, right=0.49)

    ax.plot(colors, mags,
            color='LemonChiffon',
            markeredgecolor='Black', marker='o', markersize=6,
            linestyle='')

    ax.plot(vesta.color, vesta.absoluteMagnitude,
            color='Gold', marker='D', markersize=8,
            markeredgecolor='Black',
            linestyle='')

    filename = output_dir / 'plots/Sample_HR_diagram.png'
    fig.savefig(str(filename))


def create_example_pair_sep_plots():
    """Create example plots for talks.

    Returns
    -------
    None.

    """
    tqdm.write('Reading data from stellar database file...')

    star_pair_separations = u.unyt_array.from_hdf5(
        db_file, dataset_name='star_pair_separations')
    star_pair_separations_EotWM = u.unyt_array.from_hdf5(
        db_file, dataset_name='star_pair_separations_EotWM')
    star_pair_separations_EotM = u.unyt_array.from_hdf5(
        db_file, dataset_name='star_pair_separations_EotM')
    star_temperatures = u.unyt_array.from_hdf5(
        db_file, dataset_name='star_temperatures')

    with h5py.File(db_file, mode='r') as f:

        star_metallicities = hickle.load(f, path='/star_metallicities')
        # star_magnitudes = hickle.load(f, path='/star_magnitudes')
        star_gravities = hickle.load(f, path='/star_gravities')
        transition_column_dict = hickle.load(f, path='/transition_column_index')
        pair_column_dict = hickle.load(f, path='/pair_column_index')

        star_names = hickle.load(f, path='/star_row_index')

    label = '4492.660Fe2_4503.480Mn1_25'
    # label = '4588.413Fe1_4599.405Fe1_28'
    # label = '4637.144Fe1_4638.802Fe1_29'
    # label = '4652.593Cr1_4653.460Cr1_29'
    # label = '4683.218Ti1_4684.870Fe1_30'
    # label = '4789.165Fe1_4794.208Co1_33'
    col = pair_column_dict[label]

    # Just use the pre-change era (0) for these plots for simplicity.
    mean = np.nanmean(star_pair_separations[0, :, col]).to(u.m/u.s)
    print(f'mean is {mean}')

    # First, create a masked version to catch any missing
    # entries:
    m_seps = ma.masked_invalid(star_pair_separations[0, :, col])
    m_seps = m_seps.reshape([len(m_seps), 1])

    # Then create a new array from the non-masked data:
    separations = u.unyt_array(m_seps[~m_seps.mask], units=u.km/u.s).to(u.m/u.s)

    m_eotwms = ma.masked_invalid(star_pair_separations_EotWM[0, :, col])
    m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
    eotwms = u.unyt_array(m_eotwms[~m_seps.mask], units=u.m/u.s)

    m_eotms = ma.masked_invalid(star_pair_separations_EotM[0, :, col])
    m_eotms = m_eotms.reshape([len(m_eotms), 1])
    # Use the same mask as for the offsets.
    eotms = u.unyt_array(m_eotms[~m_seps.mask],
                         units=u.m/u.s)
    # Create an error array which uses the greater of the error
    # on the mean or the error on the weighted mean.
    err_array = ma.array(np.maximum(eotwms, eotms).value)

    weighted_mean = np.average(separations, weights=err_array**-2).to(u.m/u.s)
    print(f'weighted mean is {weighted_mean}')

    temperatures = ma.masked_array(star_temperatures)
    temps = temperatures[~m_seps.mask]
    metallicities = ma.masked_array(star_metallicities)
    metals = metallicities[~m_seps.mask]
    gravities = ma.masked_array(star_gravities)
    loggs = gravities[~m_seps.mask]

    stars = ma.masked_array([key for key in
                             star_names.keys()]).reshape(
                                  len(star_names.keys()), 1)
    # stars = ma.masked_array([key for key in star_names.keys()])

    names = stars[~m_seps.mask]

    x_data = ma.array(np.stack((temps, metals, loggs), axis=0))

    # separations -= weighted_mean

    results_const = fit.find_sys_scatter(fit.constant_model,
                                         x_data,
                                         ma.array(separations
                                                  .to(u.km/u.s).value),
                                         err_array, (mean,),
                                         n_sigma=4.0,
                                         tolerance=0.001)

    mask_const = results_const['mask_list'][-1]
    residuals_const = ma.array(results_const['residuals'], mask=mask_const)
    x_data.mask = mask_const
    err_array.mask = mask_const
    names.mask = mask_const

    fig1 = plt.figure(figsize=(6, 6), tight_layout=True)
    fig2 = plt.figure(figsize=(6, 7), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig2,
                  height_ratios=(1, 1), hspace=0)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(gs[0, 0])
    ax3 = fig2.add_subplot(gs[1, 0], sharex=ax2)

    ax2.annotate(r'$\lambda4492.660\,\textrm{Fe\,II}-'
                 r'\lambda4503.480\,\textrm{Mn\,I}$',
                 xy=(0, 0), xytext=(0.03, 0.02),
                 textcoords='axes fraction', size=19,
                 horizontalalignment='left', verticalalignment='bottom')

    ax3.set_xlabel('[Fe/H]')

    for ax in (ax1, ax2):
        ax.set_ylabel('Normalized pair\nseparation (m/s)')
    ax3.set_ylabel('Residuals (m/s)')

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for ax in (ax2, ax3):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2.tick_params(labelbottom=False)

    ax1.set_xlim(left=-0.11, right=0.11)
    ax3.set_xlim(left=-0.73, right=0.44)
    ax3.set_ylim(bottom=-110, top=110)
    ax2.set_ylim(bottom=-270, top=270)

    ax3.axhline(y=0, linestyle='--',
                color='DarkCyan')

    # metallicities is index 1 here
    ax2.errorbar(ma.compressed(x_data[1]),
                 ma.compressed(residuals_const),
                 yerr=ma.compressed(err_array),
                 color='Chocolate',
                 linestyle='',
                 markersize=5, markeredgewidth=1.2,
                 marker='o', markeredgecolor='Black',
                 capsize=7, elinewidth=2.5, capthick=2.5,
                 ecolor='DarkOrange')

    mtls = []
    offsets = []
    errors = []
    for name, sep, err, mtl in zip(names, ma.compressed(residuals_const),
                                   ma.compressed(err_array),
                                   ma.compressed(x_data[1])):
        if name in sp1_stars:
            # print(f'Found {name} in SP1.')
            offsets.append(sep)
            errors.append(err)
            mtls.append(mtl)

    print(f'Found {len(offsets)} solar twins.')

    mtls = np.array(mtls)

    ax1.errorbar(mtls, offsets, yerr=errors,
                 color='MediumSeaGreen', linestyle='',
                 marker='D', markeredgecolor='Black',
                 markersize=8, markeredgewidth=1.5,
                 capsize=7, elinewidth=2.5, capthick=2.5,
                 ecolor='LightSeaGreen')

    # Plot SP1 stars on whole sample.
    # ax2.errorbar(mtls, offsets, yerr=errors,
    #              color='SeaGreen',
    #              linestyle='',
    #              marker='D', markeredgecolor='Black',
    #              markersize=5, markeredgewidth=1.5,
    #              capsize=7, elinewidth=2.5, capthick=2.5,
    #              ecolor='OliveDrab')

    # Now plot the same tranition's residuals after being corrected.
    results_quad = fit.find_sys_scatter(fit.quadratic_model,
                                        x_data,
                                        ma.array(separations
                                                 .to(u.km/u.s).value),
                                        err_array, (mean, 1, 1, 1, 1, 1, 1),
                                        n_sigma=4.0,
                                        tolerance=0.001)

    mask_quad = results_quad['mask_list'][-1]
    residuals_quad = ma.array(results_quad['residuals'], mask=mask_quad)
    sigma_s2s = results_quad['sys_err_list'][-1] * u.m/u.s
    x_data.mask = mask_quad
    err_array.mask = mask_quad
    names.mask = mask_quad

    ax3.annotate(r'$\sigma_\mathrm{2s2}=\,$'
                 f'{sigma_s2s:.2f}',
                 xy=(0, 0), xytext=(0.02, 0.04),
                 textcoords='axes fraction', size=17,
                 horizontalalignment='left', verticalalignment='bottom')

    # metallicities is index 1 here
    ax3.errorbar(ma.compressed(x_data[1]),
                 ma.compressed(residuals_quad),
                 yerr=ma.compressed(err_array),
                 color='Chocolate',
                 linestyle='',
                 markersize=5, markeredgewidth=1.2,
                 marker='o', markeredgecolor='Black',
                 capsize=7, elinewidth=2.5, capthick=2.5,
                 ecolor='DarkOrange')

    plot_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')
    fig1.savefig(str(plot_dir / f'{label}_SP1.png'))
    fig2.savefig(str(plot_dir / f'{label}_sample.png'))

    # plt.show()


def create_sigma_sys_hist():
    """
    Create a histogram of the systematic errors.

    Returns
    -------
    None.

    """

    data_dir = vcl.output_dir /\
        'stellar_parameter_fits_pairs_4.0sigma_first_selection/quadratic'
    data_file = data_dir / 'quadratic_pairs_fit_results.csv'

    with open(data_file, newline='', mode='r') as f:
        data_reader = csv.reader(f)
        data_reader.__next__()
        lines = [row for row in data_reader]

    sigmas_sys = []
    for line in lines:
        sigmas_sys.append(float(line[3]))

    sigmas_sys = np.array(sigmas_sys)
    # print(sigmas_sys)
    # print(np.mean(sigmas_sys))

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$\sigma_\mathrm{sys}$ (m/s)')

    ax.axvline(x=np.mean(sigmas_sys), linestyle='--',
               color='Gray', label='Mean')
    ax.axvline(x=np.median(sigmas_sys), linestyle='-',
               color='Gray', label='Median')
    # ax.axvline(x=np.mode(sigmas_sys), linestyle=':',
    #            color='Gray', label='Mode')

    ax.hist(sigmas_sys, color='Black', histtype='step',
            bins='fd')

    ax.legend()

    plt.show()


def create_parameter_dependence_plot(use_cached=False, min_bin_size=5):
    """
    Create a plot showing the change in sigma_s2s as a function of stellar
    parameters.

    Parameters
    ----------
    use_cached : bool, Default: False
        If False, will rerun entire binning and fitting procedure, which is very
        slow. (Though it must be done at least once first.) If True, will
        instead used saved values from running full procedure.
    min_bin_size : int
        The lower limit on the number of stars in a bin to proceed with finding
        a sigma_s2s value for it.

    Returns
    -------
    None.

    """

    # tqdm.write('Unpickling transitions list...')
    # with open(vcl.final_selection_file, 'r+b') as f:
    #     transitions_list = pickle.load(f)
    # vprint(f'Found {len(transitions_list)} transitions.')

    tqdm.write('Unpickling pairs list...')
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    model_func = fit.quadratic_model

    # Load data from HDF5 database file.
    db_file = vcl.databases_dir / 'stellar_db_uncorrected_hot_stars.hdf5'

    tqdm.write('Reading data from stellar database file...')
    # star_transition_offsets = u.unyt_array.from_hdf5(
    #         db_file, dataset_name='star_transition_offsets')
    # star_transition_offsets_EotWM = u.unyt_array.from_hdf5(
    #         db_file, dataset_name='star_transition_offsets_EotWM')
    # star_transition_offsets_EotM = u.unyt_array.from_hdf5(
    #         db_file, dataset_name='star_transition_offsets_EotM')

    star_pair_separations = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_pair_separations')
    star_pair_separations_EotWM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_pair_separations_EotWM')
    star_pair_separations_EotM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_pair_separations_EotM')
    star_temperatures = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_temperatures')

    with h5py.File(db_file, mode='r') as f:

        star_metallicities = hickle.load(f, path='/star_metallicities')
        star_magnitudes = hickle.load(f, path='/star_magnitudes')
        star_gravities = hickle.load(f, path='/star_gravities')
        column_dict = hickle.load(f, path='/pair_column_index')
        star_names = hickle.load(f, path='/star_row_index')

    # Handle various fitting and plotting setup:
    eras = {'pre': 0, 'post': 1}
    param_dict = {'temp': 0, 'mtl': 1, 'logg': 2}
    plot_types = ('temp', 'mtl', 'logg')

    params_list = []
    # Figure out how many parameters the model function takes, so we know how
    # many to dynamically give it later.
    num_params = len(signature(model_func).parameters)
    for i in range(num_params - 1):
        params_list.append(0.)

    # Collect a list of the labels to use.
    labels = []
    # for transition in tqdm(transitions_list):
    #     if transition.blendedness < 3:
    #         for order_num in transition.ordersToFitIn:
    #             label = '_'.join([transition.label, str(order_num)])
    #             labels.append(label)

    blends = set([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])
    for pair in tqdm(pairs_list):
        if pair.blendTuple in blends:
            for order_num in pair.ordersToMeasureIn:
                labels.append('_'.join([pair.label, str(order_num)]))

    bin_dict = {}
    # Set bins manually.
    bin_dict['temp'] = [5377, 5477, 5577, 5677,
                        5777, 5877, 5977, 6077,
                        6177, 6277]
    bin_dict['mtl'] = [-0.75, -0.6, -0.45, -0.3,
                       -0.15, 0, 0.15, 0.3, 0.45]
    bin_dict['logg'] = [4.14, 4.24, 4.34, 4.44, 4.54, 4.64]

    # Create an array to store all the individual sigma_sys values in in order
    # to get the means and STDs for each bin.
    row_len = len(labels)
    temp_col_len = len(bin_dict['temp']) - 1
    metal_col_len = len(bin_dict['mtl']) - 1
    logg_col_len = len(bin_dict['logg']) - 1

    # First axis is for pre- and post- fiber change values: 0 = pre, 1 = post
    temp_array = np.full([row_len, temp_col_len], np.nan)
    metal_array = np.full([row_len, metal_col_len], np.nan)
    logg_array = np.full([row_len, logg_col_len], np.nan)
    full_arrays_dict = {key: value for key, value in zip(plot_types,
                                                         (temp_array,
                                                          metal_array,
                                                          logg_array))}

    fig = plt.figure(figsize=(12, 4.5), tight_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    ax1.set_ylim(bottom=-3, top=120)
    ax1.set_xlim(left=bin_dict['temp'][0], right=bin_dict['temp'][-1])
    ax2.set_xlim(left=bin_dict['mtl'][0], right=bin_dict['mtl'][-1])
    ax3.set_xlim(left=bin_dict['logg'][0], right=bin_dict['logg'][-1])

    ax1.set_xlabel(r'$T_\mathrm{eff}$ (K)')
    ax1.set_ylabel(r'$\sigma_\mathrm{s2s}$ (m/s)')
    ax2.set_xlabel('[Fe/H]')
    ax3.set_xlabel(r'$\log{g}\,(\mathrm{cm\,s}^{-2})$')

    ax1.xaxis.set_major_locator(ticker.FixedLocator([5477, 5777, 6077]))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=0.3))
    ax3.xaxis.set_major_locator(ticker.FixedLocator([4.24, 4.34, 4.44,
                                                     4.54]))

    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)

    for ax in (ax1, ax2, ax3):
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    for plot_type, ax in zip(plot_types, (ax1, ax2, ax3)):
        for limit in bin_dict[plot_type]:
            ax.axvline(x=limit, color='SeaGreen', linestyle='--',
                       alpha=0.8, zorder=5)

    data_file = vcl.output_dir /\
        f'stellar_parameter_dependence_data_bin_{min_bin_size}.h5py'
    if not use_cached:
        for label_num, label in tqdm(enumerate(labels[:]),
                                     total=len(labels[:])):

            vprint(f'Analyzing {label}...')
            # The column number to use for this transition:
            try:
                col = column_dict[label]
            except KeyError:
                print(f'Incorrect key given: {label}')
                sys.exit(1)

            vprint(20 * '=')
            vprint(f'Working on pre-change era.')
            mean = np.nanmean(star_pair_separations[eras['pre'], :, col])

            # First, create a masked version to catch any missing entries:
            m_seps = ma.masked_invalid(star_pair_separations[
                        eras['pre'], :, col])
            m_seps = m_seps.reshape([len(m_seps), 1])
            # Then create a new array from the non-masked data:
            separations = u.unyt_array(m_seps[~m_seps.mask],
                                       units=u.m/u.s)
            vprint(f'Median of separations is {np.nanmedian(separations)}')

            m_eotwms = ma.masked_invalid(star_pair_separations_EotWM[
                    eras['pre'], :, col])
            m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
            eotwms = u.unyt_array(m_eotwms[~m_eotwms.mask],
                                  units=u.m/u.s)

            m_eotms = ma.masked_invalid(star_pair_separations_EotM[
                    eras['pre'], :, col])
            m_eotms = m_eotms.reshape([len(m_eotms), 1])
            # Use the same mask as for the separations.
            eotms = u.unyt_array(m_eotms[~m_seps.mask],
                                 units=u.m/u.s)
            # Create an error array which uses the greater of the error on
            # the mean or the error on the weighted mean.
            err_array = np.maximum(eotwms, eotms)

            vprint(f'Mean is {np.mean(separations)}')
            weighted_mean = np.average(separations, weights=err_array**-2)
            vprint(f'Weighted mean is {weighted_mean}')

            # Mask the various stellar parameter arrays with the same mask
            # so that everything stays in sync.
            temperatures = ma.masked_array(star_temperatures)
            temps = temperatures[~m_seps.mask]
            metallicities = ma.masked_array(star_metallicities)
            metals = metallicities[~m_seps.mask]
            magnitudes = ma.masked_array(star_magnitudes)
            mags = magnitudes[~m_seps.mask]
            gravities = ma.masked_array(star_gravities)
            loggs = gravities[~m_seps.mask]

            # stars = ma.masked_array([key for key in
            #                          star_names.keys()]).reshape(
            #                              len(star_names.keys()), 1)
            # names = stars[~m_seps.mask]

            # Stack the stellar parameters into vertical slices
            # for passing to model functions.
            x_data = np.stack((temps, metals, loggs), axis=0)

            # Create the parameter list for this run of fitting.
            params_list[0] = float(mean)

            beta0 = tuple(params_list)
            vprint(beta0)

            # Iterate over binned segments of the data to find what additional
            # systematic error is needed to get a chi^2 of ~1.
            arrays_dict = {name: array for name, array in
                           zip(plot_types,
                               (temps, metals, loggs))}

            popt, pcov = curve_fit(model_func, x_data, separations.value,
                                   sigma=err_array.value,
                                   p0=beta0,
                                   absolute_sigma=True,
                                   method='lm', maxfev=10000)

            model_values = model_func(x_data, *popt)
            residuals = separations.value - model_values

            # if args.nbins:
            #     nbins = int(args.nbins)
            #     # Use quantiles to get bins with the same number of elements
            #     # in them.
            #     vprint(f'Generating {args.nbins} bins.')
            #     bins = np.quantile(arrays_dict[name],
            #                        np.linspace(0, 1, nbins+1),
            #                        interpolation='nearest')
            #     bin_dict[name] = bins

            min_bin_size = min_bin_size
            sigma_sys_dict = {}
            star_bins_dict = {}
            num_params = 1
            for name in tqdm(plot_types):
                sigma_sys_list = []
                sigma_list = []
                bin_mid_list = []
                bin_num = -1
                star_bins_dict[name] = []
                for bin_lims in pairwise(bin_dict[name]):
                    bin_num += 1
                    lower, upper = bin_lims
                    bin_mid_list.append((lower + upper) / 2)
                    mask_array = ma.masked_outside(arrays_dict[name], *bin_lims)
                    num_points = mask_array.count()
                    star_bins_dict[name].append(num_points)
                    vprint(f'{num_points} values in bin ({lower},{upper})')
                    if num_points < min_bin_size:
                        vprint('Skipping this bin!')
                        sigma_list.append(np.nan)
                        sigma_sys_list.append(np.nan)
                        continue
                    temps_copy = temps[~mask_array.mask]
                    metals_copy = metals[~mask_array.mask]
                    mags_copy = mags[~mask_array.mask]
                    residuals_copy = residuals[~mask_array.mask]
                    errs_copy = err_array[~mask_array.mask].value
                    x_data_copy = np.stack((temps_copy, metals_copy, mags_copy),
                                           axis=0)

                    chi_squared_nu = fit.calc_chi_squared_nu(residuals_copy,
                                                             errs_copy,
                                                             num_params)
                    sigma_sys_delta = 0.01
                    sigma_sys = -sigma_sys_delta
                    chi_squared_nu = np.inf
                    variances = np.square(errs_copy)
                    while chi_squared_nu > 1.0:
                        sigma_sys += sigma_sys_delta
                        variance_sys = np.square(sigma_sys)
                        variances_iter = variances + variance_sys
                        # err_iter = np.sqrt(np.square(errs_copy) +
                        #                    np.square(sigma_sys))
                        weights = 1 / variances_iter
                        wmean, sum_weights = np.average(residuals_copy,
                                                        weights=weights,
                                                        returned=True)

                        chi_squared_nu = fit.calc_chi_squared_nu(
                            residuals_copy - wmean, np.sqrt(variances_iter),
                            num_params)

                    sigma_sys_list.append(sigma_sys)
                    sigma = np.std(residuals_copy)
                    sigma_list.append(sigma)
                    vprint(f'sigma_sys is {sigma_sys:.3f}')
                    vprint(f'chi^2_nu is {chi_squared_nu}')
                    if sigma_sys / sigma > 1.2:
                        print('---')
                        print(bin_lims)
                        print(mask_array)
                        print(metals)
                        print(residuals)
                        print(n_params)
                        print(num_params)
                        print(residuals_copy)
                        print(errs_copy)
                        print(sigma)
                        print(sigma_sys)
                        sys.exit()

                    # Store the result in the appropriate full array.
                    full_arrays_dict[name][label_num, bin_num] = sigma_sys

                sigma_sys_dict[f'{name}_sigma_sys'] = sigma_sys_list
                sigma_sys_dict[f'{name}_sigma'] = sigma_list
                sigma_sys_dict[f'{name}_bin_mids'] = bin_mid_list

        with h5py.File(data_file, mode='w') as f:
            hickle.dump(sigma_sys_dict, f, path='/sigma_sys_dict')
            hickle.dump(full_arrays_dict, f, path='/full_arrays_dict')
            hickle.dump(star_bins_dict, f, path='/star_bins_dict')

    elif use_cached:
        with h5py.File(data_file, mode='r') as f:
            sigma_sys_dict = hickle.load(f, path='/sigma_sys_dict')
            full_arrays_dict = hickle.load(f, path='/full_arrays_dict')
            star_bins_dict = hickle.load(f, path='/star_bins_dict')

    for name, ax in zip(plot_types, (ax1, ax2, ax3)):
        means = []
        stds = []
        arr = full_arrays_dict[name]
        for i in range(0, np.size(arr, 0)):
            ax.plot(sigma_sys_dict[f'{name}_bin_mids'], arr[i],
                    color='Black', alpha=0.05, zorder=1)
        for j in range(0, np.size(arr, 1)):
            means.append(np.nanmean(arr[:, j]))
            stds.append(np.nanstd(arr[:, j]))
        ax.errorbar(sigma_sys_dict[f'{name}_bin_mids'], means,
                    yerr=stds, color='Red', alpha=1,
                    marker='o', markersize=4, capsize=4,
                    elinewidth=2, zorder=10, linestyle='-',
                    label='Mean and RMS')
        # Annotate with the number of stars in each bin.
        for i in range(len(sigma_sys_dict[f'{name}_bin_mids'])):
            ax.annotate(fr'${star_bins_dict[name][i]}$',
                        (float(sigma_sys_dict[f'{name}_bin_mids'][i]), 110),
                        xytext=(float(sigma_sys_dict[f'{name}_bin_mids'][i]),
                                110),
                        textcoords='data',
                        verticalalignment='top', horizontalalignment='center',
                        fontsize=18, zorder=15)

    plot_path = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

    filename = plot_path /\
        f'Stellar_parameter_dependence_bin_{min_bin_size}.png'
    fig.savefig(str(filename))


def plot_duplicate_pairs(star):
    """
    Create a plot comparing the duplicate pairs for the given star.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to use for comparing its duplicate pairs.

    Returns
    -------
    None.

    """

    pair_sep_pre1, pair_model_pre1 = [], []
    pair_sep_err_pre1, pair_model_err_pre1 = [], []

    pair_sep_pre2, pair_model_pre2 = [], []
    pair_sep_err_pre2, pair_model_err_pre2 = [], []

    blends_to_use = set(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)))

    pair_order_numbers = []
    for pair in tqdm(star.pairsList):
        if len(pair.ordersToMeasureIn) == 2 and\
           pair.blendTuple in blends_to_use:
            pair_order_numbers.append(pair.ordersToMeasureIn[1])
            p_index1 = star.p_index('_'.join([pair.label,
                                              str(pair.ordersToMeasureIn[0])]))
            p_index2 = star.p_index('_'.join([pair.label,
                                              str(pair.ordersToMeasureIn[1])]))

            if star.hasObsPre:
                # Get the values for the first duplicate
                time_slice = slice(None, star.fiberSplitIndex)
                w_mean, eotwm = get_weighted_mean(
                    star.pairSeparationsArray,
                    star.pairSepErrorsArray,
                    time_slice,
                    p_index1)
                pair_sep_pre1.append(w_mean)
                pair_sep_err_pre1.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0, p_index1]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index1)
                pair_model_pre1.append(w_mean)
                pair_model_err_pre1.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0, p_index1]**2))

                # Get the values for the second duplicate
                time_slice = slice(None, star.fiberSplitIndex)
                w_mean, eotwm = get_weighted_mean(
                    star.pairSeparationsArray,
                    star.pairSepErrorsArray,
                    time_slice,
                    p_index2)
                pair_sep_pre2.append(w_mean)
                pair_sep_err_pre2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0, p_index2]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index2)
                pair_model_pre2.append(w_mean)
                pair_model_err_pre2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0, p_index2]**2))

    # pprint(pair_order_numbers)

    if star.hasObsPre:
        pair_sep_pre1 = np.array(pair_sep_pre1)
        pair_model_pre1 = np.array(pair_model_pre1)
        pair_sep_err_pre1 = np.array(pair_sep_err_pre1)
        pair_model_err_pre1 = np.array(pair_model_err_pre1)
        pair_sep_pre2 = np.array(pair_sep_pre2)
        pair_model_pre2 = np.array(pair_model_pre2)
        pair_sep_err_pre2 = np.array(pair_sep_err_pre2)
        pair_model_err_pre2 = np.array(pair_model_err_pre2)

    # Plot the results

    fig = plt.figure(figsize=(10, 4.5), tight_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0)
    ax_measured = fig.add_subplot(gs[0, 0])
    ax_corrected = fig.add_subplot(gs[0, 1])

    ax_measured.set_title(r'$\mathrm{Raw}$', fontsize=18)
    ax_corrected.set_title(r'$\mathrm{Corrected}$', fontsize=18)
    ax_measured.set_xlabel(r'$\Delta v_\mathrm{\,instance\ 2}-'
                           r'\Delta v_\mathrm{\,instance\ 1}\ \mathrm{(m/s)}$')
    ax_corrected.set_xlabel(r'$\Delta v_\mathrm{\,instance\ 2}-'
                            r'\Delta v_\mathrm{\,instance\ 1}\ \mathrm{(m/s)}$')

    order_boundaries = []
    for i in range(len(pair_order_numbers)):
        if i == 0:
            continue
        if pair_order_numbers[i-1] != pair_order_numbers[i]:
            order_boundaries.append(i - 0.5)

    # ax_measured.tick_params(labelbottom=False)

    for ax in (ax_measured, ax_corrected):
        # ax.yaxis.grid(which='major', color='Gray', alpha=0.7,
        #               linestyle='-')
        # ax.yaxis.grid(which='minor', color='Gray', alpha=0.6,
        #               linestyle='--')
        ax.axvline(x=0, linestyle='-', color='Gray')
        ax.set_xlim(left=-60, right=60)
        ax.set_ylim(bottom=-1, top=55)
        # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(left=False, right=False, labelleft=False)
        # ax.tick_params(labelleft=False)
        for b in order_boundaries:
            ax.axhline(y=b, linestyle='--', color='DimGray', alpha=1)

    # add_star_information(star, ax_pre, (0.07, 0.49))

    if star.hasObsPre:
        pair_indices = np.array([x for x in range(len(pair_sep_pre1))])
    else:
        pair_indices = np.array([x for x in range(len(pair_sep_post1))])
    model_pair_indices = pair_indices + 0.2

    if star.hasObsPre:
        pair_diffs = pair_sep_pre2 - pair_sep_pre1
        model_diffs = pair_model_pre2 - pair_model_pre1
        pair_errs = np.sqrt(pair_sep_err_pre1**2 + pair_sep_err_pre2**2)
        model_errs = np.sqrt(pair_model_err_pre1**2 + pair_model_err_pre2**2)

        pairs_chisq = fit.calc_chi_squared_nu(
            remove_nans(pair_diffs), remove_nans(pair_errs), 1)
        model_chisq = fit.calc_chi_squared_nu(
            remove_nans(model_diffs), remove_nans(model_errs), 1)
        pairs_sigma = np.nanstd(pair_diffs)
        model_sigma = np.nanstd(model_diffs)

        ax_measured.errorbar(pair_diffs, pair_indices,
                             xerr=pair_errs,
                             capsize=2, capthick=2,
                             elinewidth=2,
                             color='Chocolate', markeredgecolor='Black',
                             linestyle='', marker='o',
                             label=r'Pair $\chi^2_\nu$:'
                             f' {pairs_chisq:.2f}, RMS: {pairs_sigma:.2f}')
        ax_measured.annotate(r'$\chi^2_\nu:'
                             fr' {pairs_chisq:.2f}$'
                             '\n'
                             r'$\mathrm{RMS:}'
                             fr'\ {pairs_sigma:.2f}'
                             r'\,\mathrm{m/s}$',
                             xy=(0, 0), xytext=(-56, 28),
                             textcoords='data', size=19,
                             horizontalalignment='left',
                             verticalalignment='bottom')
        ax_corrected.errorbar(model_diffs, pair_indices,
                              xerr=model_errs,
                              color='DodgerBlue', capsize=2, capthick=2,
                              elinewidth=2,
                              ecolor='DodgerBlue', markeredgecolor='Black',
                              linestyle='', marker='D',
                              label=r'Model $\chi^2_\nu$:'
                              f' {model_chisq:.2f}, RMS: {model_sigma:.2f}')
        ax_corrected.annotate(r'$\chi^2_\nu:'
                              fr' {model_chisq:.2f}$'
                              '\n'
                              r'$\mathrm{RMS:}'
                              fr'\ {model_sigma:.2f}'
                              r'\,\mathrm{m/s}$',
                              xy=(0, 0), xytext=(-56, 28),
                              textcoords='data', size=19,
                              horizontalalignment='left',
                              verticalalignment='bottom')
        # ax_pre.legend()

    # plt.show(fig)
    output_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')
    outfile = output_dir /\
        f'{star.name}_duplicate_pairs.png'
    fig.savefig(str(outfile))
    plt.close('all')


def create_radial_velocity_plot():
    """
    Create a plot showing the effect of radial velocity on a specific pair.

    Returns
    -------
    None.

    """

    pair_label = '6138.313Fe1_6139.390Fe1_60'

    boundary_pix = 3072

    pair_seps_pre, offsets_pre = [], []
    errors_pre, pixels_pre = [], []

    parts = pair_label.split('_')
    blue_label = '_'.join((parts[0], parts[2]))
    red_label = '_'.join((parts[1], parts[2]))

    star_names = [d.split('/')[-1] for d in
                  glob('/Users/dberke/data_output/[HV]*')]

    star_list = []
    temp_lim = 6077 * u.K
    metal_lim = -0.45
    tqdm.write('Gathering stars...')
    for star_name in tqdm(star_names):
        star = Star(star_name, vcl.output_dir / star_name)
        if star.temperature > temp_lim:
            del star
        elif star.metallicity < metal_lim:
            del star
        else:
            star_list.append(star)

    for star in tqdm(star_list):

        pre_slice = slice(None, star.fiberSplitIndex)
        post_slice = slice(star.fiberSplitIndex, None)

        col_index = star.p_index(pair_label)

        if star.hasObsPre:

            pair_seps_pre.extend(star.pairSeparationsArray[
                pre_slice, col_index].to(u.km/u.s))
            offsets_pre.extend(star.pairModelOffsetsArray[
                pre_slice, col_index].to(u.m/u.s))
            errors_pre.extend(star.pairModelErrorsArray[pre_slice,
                                                        col_index])
            pixels_pre.extend(star.pixelArray[
                pre_slice, star.t_index(blue_label)])

        if (len(offsets_pre) != len(pixels_pre)) or\
           (len(offsets_pre) != len(pair_seps_pre)):
            print(star.name)
            exit(1)

    offsets_pre, mask_pre = remove_nans(np.array(offsets_pre),
                                        return_mask=True)
    errors_pre = np.array(errors_pre)[mask_pre]
    pixels_pre = np.array(pixels_pre)[mask_pre]
    pair_seps_pre = np.array(pair_seps_pre)[mask_pre]
    mean_sep_pre = np.mean(pair_seps_pre)

    fig = plt.figure(figsize=(5, 6), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig,
                  height_ratios=(1, 0.4), hspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.set_xlim(left=2870, right=3140)
    ax.set_ylim(bottom=-90, top=90)
    # Denote the CCD sub-boundary in pixels.
    ax.axvline(x=boundary_pix, linestyle='--', color='Blue',
               label='Blue crossing', zorder=5)

    for axis in (ax, ax_mean):
        axis.axhline(y=0, linestyle='--', color='Gray', zorder=1)

    ax_mean.set_ylim(bottom=-8, top=8)
    ax.tick_params(labelbottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_mean.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_mean.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    harps_pix_width = 0.829  # km/s
    ax.axvline(boundary_pix - np.round(mean_sep_pre / harps_pix_width),
               linestyle='-.', color='Red',
               label='Red crossing', zorder=5)

    ax.set_ylabel('Model offset (m/s)')
    ax_mean.set_ylabel('Mean (m/s)')
    ax_mean.set_xlabel('Pixel number')

    ax.errorbar(pixels_pre, offsets_pre,
                linestyle='',
                markeredgecolor=None,
                marker='.', color='Chocolate',
                alpha=0.3, zorder=2)

    # Create some bins to measure in:
    midpoints = []
    means_pre, eotms_pre = [], []
    bin_lims = [i for i in range(2872, 3172, 15)]
    for lims in tqdm(pairwise(bin_lims)):
        mask_pre = np.where((pixels_pre > lims[0]) &
                            (pixels_pre < lims[1]))

        midpoints.append((lims[0] + lims[1])/2)

        num_pre = len(offsets_pre[mask_pre])

        if num_pre > 1:
            means_pre.append(np.mean(offsets_pre[mask_pre]))
            eotms_pre.append(np.std(offsets_pre[mask_pre]) /
                             np.sqrt(len(offsets_pre[mask_pre])))
        elif num_pre == 1:
            means_pre.append(offsets_pre[mask_pre][0])
            eotms_pre.append(errors_pre[mask_pre][0])
        else:
            means_pre.append(np.nan)
            eotms_pre.append(np.nan)

    ax_mean.errorbar(midpoints, means_pre,
                     yerr=eotms_pre,
                     color='Black', marker='o',
                     markerfacecolor='White',
                     markersize=4, capsize=2)

    # ax.legend(loc='upper left')
    plot_name = output_dir / f'plots/{pair_label}_vs_pixel.png'
    fig.savefig(str(plot_name))
    plt.close('all')


def plot_vs_pair_blendedness(star):
    """
    Create a plot for a star of its pair model-corrected values by blendedness.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to create the plot using.

    Returns
    -------
    None.

    """

    # tqdm.write(f'Plotting {star.name}...')
    plots_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

    sorted_blend_tuples = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                           (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                           (2, 2), (2, 3), (2, 4), (2, 5),
                           (3, 3), (3, 4), (3, 5),
                           (4, 4), (4, 5),
                           (5, 5))

    pre_slice = slice(None, star.fiberSplitIndex)

    pair_blends_dict = {}
    total_pairs = 0
    for pair in star.pairsList:
        for order_num in pair.ordersToMeasureIn:
            total_pairs += 1
            pair_label = '_'.join((pair.label, str(order_num)))
            pair_blends_dict[pair_label] = pair.blendTuple

    sorted_means_pre, sorted_errs_pre = [], []
    sigmas_pre = []
    mean_errs_pre = []
    total = 0
    per_bin_wmeans_pre = []
    per_bin_errs_pre = []

    for blend_tuple in sorted_blend_tuples:
        bin_means_pre, bin_errs_pre = [], []
        bin_means_pre_nn, bin_errors_pre_nn = [], []
        for pair_label, value in pair_blends_dict.items():
            col_index = star.p_index(pair_label)
            if blend_tuple == value:
                total += 1
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    pre_slice, star._pair_bidict[pair_label])
                bin_means_pre.append(w_mean)
                bin_errs_pre.append(
                    np.sqrt(eotwm**2 +
                            star.pairSysErrorsArray[0, col_index]**2))

        bin_means_pre_nn, mask_pre = remove_nans(np.array(bin_means_pre),
                                                 return_mask=True)
        bin_errs_pre_nn = np.array(bin_errs_pre)[mask_pre]

        if len(bin_means_pre_nn) > 1:

            sigmas_pre.append(np.nanstd(bin_means_pre))
            wmean, eotwm = weighted_mean_and_error(bin_means_pre_nn,
                                                   bin_errs_pre_nn)
            per_bin_wmeans_pre.append(wmean)
            per_bin_errs_pre.append(eotwm)
            mean_errs_pre.append(np.mean(bin_errs_pre_nn))
        else:
            sigmas_pre.append(np.nan)
            per_bin_wmeans_pre.append(np.nan)
            per_bin_errs_pre.append(np.nan)
            mean_errs_pre.append(np.nan)

        sorted_means_pre.extend(bin_means_pre)
        sorted_errs_pre.extend(bin_errs_pre)

    sorted_means_pre = np.array(sorted_means_pre)
    sorted_errs_pre = np.array(sorted_errs_pre)

    # Plot the results.
    fig = plt.figure(figsize=(5, 4.2), tight_layout=True)
    gs = GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    ax.set_xlabel('RMS (m/s)')
    ax.set_ylabel('Mean error (m/s)')

    # ax.tick_params(bottom=False, labelbottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    num_divisions = len(sigmas_pre)
    indices = [i for i in range(num_divisions)]
    ax.set_xlim(left=-1, right=46)
    ax.set_ylim(bottom=0, top=95)

    colors = cmr.take_cmap_colors('cmr.ember_r', 3,
                                  cmap_range=(0.1, 0.6), return_fmt='hex')

    five_patch = mlines.Line2D([], [], color=colors[2], marker='h',
                               markersize=10, linestyle='',
                               markeredgecolor='Black',
                               label=r'$(M, 5)$')
    four_patch = mlines.Line2D([], [], color=colors[1], marker='D',
                               markersize=8, linestyle='',
                               markeredgecolor='Black',
                               label=r'$(M, 4)$')
    other_patch = mlines.Line2D([], [], color=colors[0], marker='o',
                                markersize=9, linestyle='',
                                markeredgecolor='Black',
                                label=r'$(\leq3, \leq3)$')

    for sigma, mean_err, blend_tuple in zip(sigmas_pre, mean_errs_pre,
                                            sorted_blend_tuples):
        if blend_tuple[1] == 5:
            ax.plot(sigma, mean_err,
                    color=colors[2], marker='h', markersize=10,
                    markeredgecolor='Black', linestyle='',
                    alpha=0.7)

        elif blend_tuple[1] == 4:
            ax.plot(sigma, mean_err,
                    color=colors[1], marker='D', markersize=8,
                    markeredgecolor='Black', linestyle='',
                    alpha=0.7)

        else:
            ax.plot(sigma, mean_err,
                    color=colors[0], marker='o', markersize=9,
                    markeredgecolor='Black', linestyle='',
                    alpha=0.7)

    ax.legend(handles=[five_patch, four_patch, other_patch],
              loc='upper left')

    ax.annotate('(0, 0)', xy=(3, 9), xytext=(1, 22),
                arrowprops={'arrowstyle': '-', 'color': 'Black'},
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=20, weight='bold')
    # strings = [str(blend_tuple) for blend_tuple in sorted_blend_tuples]
    # texts = [plt.text(sigmas_pre[i], mean_errs_pre[i], strings[i],
    #                   ha='center', va='center', fontsize=20)
    #          for i in range(len(strings))]

    # adjust_text(texts,
    #             force_text=(0.2, 0.35),
    #             arrowprops=dict(arrowstyle='-', color='Black'),
    #             lim=1000)

    filename = plots_dir / f'{star.name}_by_blendedness_(5,5).png'
    if not plots_dir.exists():
        os.mkdir(plots_dir)
    fig.savefig(str(filename))
    plt.close('all')


def plot_pair_depth_differences(star):
    """
    Create a plot to investigate pair depth differences for systematics.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to plot the differences for.

    Returns
    -------
    None.

    """

    tqdm.write(f'{star.name} has {star.numObs} observations'
               f' ({star.numObsPre} pre, {star.numObsPost} post)')

    plots_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

    filename = vcl.output_dir /\
        f'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
    fit_results_dict = get_params_file(filename)
    sigma_sys_dict = fit_results_dict['sigmas_sys']

    pair_depth_diffs = []
    pair_depth_means = []
    pair_model_sep_pre = []
    pair_model_err_pre = []

    sigmas_sys_pre = []

    pre_slice = slice(None, star.fiberSplitIndex)

    blends_to_use = set(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)))

    for pair in tqdm(star.pairsList):
        if pair.blendTuple in blends_to_use:
            for order_num in pair.ordersToMeasureIn:
                pair_label = '_'.join([pair.label, str(order_num)])
                col_index = star._pair_bidict[pair_label]
                label_high = '_'.join([pair._higherEnergyTransition.label,
                                      str(order_num)])
                label_low = '_'.join([pair._lowerEnergyTransition.label,
                                     str(order_num)])
                col_high = star._transition_bidict[label_high]
                col_low = star._transition_bidict[label_low]
                depths_high = remove_nans(star.normalizedDepthArray[:,
                                                                    col_high])
                depths_low = remove_nans(star.normalizedDepthArray[:, col_low])
                mean_high = np.nanmean(depths_high)
                mean_low = np.nanmean(depths_low)
                # h_d = pair._higherEnergyTransition.normalizedDepth
                # l_d = pair._lowerEnergyTransition.normalizedDepth
                # depth_diff = l_d - h_d
                pair_depth_means.append((mean_high + mean_low) / 2)
                pair_depth_diffs.append(abs(mean_low - mean_high))

                w_mean, eotwm = get_weighted_mean(star.pairModelOffsetsArray,
                                                  star.pairModelErrorsArray,
                                                  pre_slice, col_index)
                pair_model_sep_pre.append(w_mean)
                pair_model_err_pre.append(eotwm)
                sigmas_sys_pre.append(
                    sigma_sys_dict[pair_label + '_pre'].value)

    pair_depth_diffs = np.array(pair_depth_diffs)
    pair_depth_means = np.array(pair_depth_means)
    pair_model_sep_pre = np.array(pair_model_sep_pre)
    pair_model_err_pre = np.array(pair_model_err_pre)
    sigmas_sys_pre = np.array(sigmas_sys_pre)

    # Plot as a function of mean pair depth.
    fig = plt.figure(figsize=(5, 7), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=1, figure=fig,
                  width_ratios=(2, 1),
                  wspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_pre_wmean = fig.add_subplot(gs[0, 1], sharey=ax_pre)
    # ax_clb_pre = fig.add_subplot(gs[0, 1])

    ax_pre.set_xlabel('Model offset (m/s)')
    ax_pre_wmean.set_xlabel('Weighted\nmean (m/s)', fontsize=18)
    ax_pre.set_ylabel('Normalized mean depth of pair')

    ax_pre_wmean.tick_params(labelleft=False)

    ax_pre.set_xlim(left=-45, right=45)
    ax_pre.set_ylim(top=0.14, bottom=1)
    ax_pre.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_pre_wmean.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_pre_wmean.xaxis.set_major_locator(ticker.FixedLocator([-2, 0, 2]))
    grid_keywords = {'which': 'major', 'linestyle': '--',
                     'color': 'SeaGreen', 'alpha': 0.8,
                     'linewidth': 1.5}
    ax_pre.yaxis.grid(**grid_keywords)
    ax_pre_wmean.yaxis.grid(**grid_keywords)
    ax_pre_wmean.set_xlim(left=-3.5, right=3.5)
    for ax in (ax_pre, ax_pre_wmean):
        ax.axvline(x=0, color='Gray', linestyle='--', zorder=0)
        # ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax.xaxis.grid(which='major', color='Gray',
        #               linestyle=':', alpha=0.65)
        # ax.xaxis.grid(which='minor', color='Gray',
        #               linestyle=':', alpha=0.5)

    full_errs_pre = np.sqrt(pair_model_err_pre ** 2 +
                            sigmas_sys_pre ** 2)
    values, mask = remove_nans(pair_model_sep_pre, return_mask=True)
    # chisq = calc_chi_squared_nu(values,
    #                             full_errs_pre[mask], 1)
    ax_pre.errorbar(pair_model_sep_pre, pair_depth_means,
                    xerr=full_errs_pre,
                    # xerr=pair_model_err_pre,
                    linestyle='', marker='o', markersize=5,
                    markerfacecolor='Chocolate',
                    markeredgecolor='Black', ecolor='Peru',
                    zorder=5, capsize=0, capthick=0)

    # Get results for bins.
    bin_lims = np.linspace(0, 1, 11)

    midpoints, w_means, eotwms, chisq = [], [], [], []
    for i, lims in zip(range(len(bin_lims)), pairwise(bin_lims)):
        midpoints.append((lims[1] + lims[0]) / 2)
        mask = np.where((pair_depth_means > lims[0]) &
                        (pair_depth_means < lims[1]))
        values, nan_mask = remove_nans(pair_model_sep_pre[mask],
                                       return_mask=True)
        errs = full_errs_pre[mask][nan_mask]
        # errs = pair_model_err_pre[mask][nan_mask]
        try:
            w_mean, eotwm = weighted_mean_and_error(values, errs)
        except ZeroDivisionError:
            w_mean, eotwm = np.nan, np.nan
        w_means.append(w_mean)
        eotwms.append(eotwm)

        # chisq.append(calc_chi_squared_nu(values, errs, 1))
    ax_pre_wmean.errorbar(w_means, midpoints, xerr=eotwms,
                          color='Black', marker='o',
                          markerfacecolor='White',
                          capsize=3, capthick=2,
                          zorder=5)

    outfile = plots_dir / f'{star.name}_obs_by_mean_depth.png'
    fig.savefig(str(outfile))
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create all the necessary'
                                     ' figures and tables for my two papers and'
                                     ' thesis.')

    parser.add_argument('--tables', action='store_true',
                        help='Save out tables in LaTeX format to text files.')
    parser.add_argument('--figures', action='store_true',
                        help='Create and save plots and figures.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print out more information about the script's"
                        " output.")

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)
    vprint('Using verbose settings.')

    output_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables')
    if not output_dir.exists():
        os.mkdir(output_dir)

    db_file = vcl.databases_dir / 'stellar_db_uncorrected.hdf5'

    sp1_stars = ('HD138573', 'HD140538', 'HD146233', 'HD157347', 'HD171665',
                 'HD1835', 'HD183658', 'HD19467', 'HD20782', 'HD220507'
                 'HD222582', 'HD30495', 'HD45184', 'HD45289', 'HD76151',
                 'HD78429', 'HD78660', 'Vesta')

    if args.tables:

        tqdm.write('Unpickling transitions list.')
        with open(vcl.final_selection_file, 'r+b') as f:
            transitions_list = pickle.load(f)
        vprint(f'Found {len(transitions_list)} transitions.')

        tqdm.write('Unpickling pairs list.')
        with open(vcl.final_pair_selection_file, 'r+b') as f:
            pairs_list = pickle.load(f)
        vprint(f'Found {len(pairs_list)} pairs.')

        tables_dir = output_dir / 'tables'
        if not tables_dir.exists():
            os.mkdir(tables_dir)

        pairs_table_file = tables_dir / 'pairs_table.txt'

        transition_headers = [r'Wavelength (\AA, vacuum)',
                              r'Wavenumber (\si{\per\centi\meter})',
                              'Species',
                              r'Energy (\si{\per\centi\meter})',
                              'Orbital configuration',
                              'J',
                              r'Energy (\si{\per\centi\meter})',
                              'Orbital configuration',
                              'J',
                              'Orders to fit in']

        n = 3
        fraction = ceil(len(transitions_list) / n)

        slices = (slice(0, fraction), slice(fraction, 2 * fraction),
                  slice(2 * fraction, None))
        for i, s in enumerate(slices):
            transitions_formatted_list = []
            transitions_table_file = tables_dir / f'transitions_table_{i}.txt'
            for transition in tqdm(transitions_list[s]):
                line = [f'{transition.wavelength.to(u.angstrom).value:.3f}',
                        f'{transition.wavenumber.value:.3f}',
                        ''.join((r'\ion{', transition.atomicSymbol,
                                 '}{',
                                 roman_numerals[
                                     transition.ionizationState].lower(), '}')),
                        transition.lowerEnergy.value,
                        transition.lowerOrbital,
                        transition.lowerJ,
                        transition.higherEnergy.value,
                        transition.higherOrbital,
                        transition.higherJ,
                        transition.ordersToFitIn]

                transitions_formatted_list.append(line)

            transitions_output = tabulate(transitions_formatted_list,
                                          headers=transition_headers,
                                          tablefmt='latex_raw',
                                          floatfmt=('.3f', '.3f', '',
                                                    '.3f', '', '',
                                                    '.3f', '', '', ''))

            if transitions_table_file.exists():
                os.unlink(transitions_table_file)
            with open(transitions_table_file, 'w') as f:
                f.write(transitions_output)

    if args.figures:
        hd146233 = Star('HD146233', '/Users/dberke/data_output/HD146233')

        create_HR_diagram_plot()

        # create_example_pair_sep_plots()

        # create_sigma_sys_hist()

        # create_parameter_dependence_plot(use_cached=True, min_bin_size=5)

        # plot_duplicate_pairs(Star('Vesta', '/Users/dberke/data_output/Vesta'))
        # plot_duplicate_pairs(hd146233)

        # create_radial_velocity_plot()

        # plot_vs_pair_blendedness(hd146233)

        # plot_pair_depth_differences(hd146233)
        # plot_pair_depth_differences(Star('HD72769',
        #                                  '/Users/dberke/data_output/HD72769'))
        # plot_pair_depth_differences(Star('HD98281',
        #                                  '/Users/dberke/data_output/HD98281'))
