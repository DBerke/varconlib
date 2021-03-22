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
from math import ceil
import os
from pathlib import Path
import pickle

import h5py
import hickle
import numpy as np
import numpy.ma as ma
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting as fit
from varconlib.miscellaneous import remove_nans, weighted_mean_and_error
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


def create_parameter_dependence_plot(use_cached=False):
    """
    Create a plot showing the change in sigma_s2s as a function of stellar
    parameters.

    Parameters
    ----------
    use_cached : bool, Default: False
        If False, will rerun entire binning and fitting procedure, which is very
        slow. (Though it must be done at least once first.) If True, will
        instead used saved values from running full procedure.

    Returns
    -------
    None.

    """

    # Define the limits to plot in the various stellar parameters.
    temp_lims = (5400, 6300) * u.K
    mtl_lims = (-0.75, 0.45)
    logg_lims = (4.1, 4.6)

    tqdm.write('Unpickling transitions list...')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions.')

    model_func = fit.quadratic_model

    # Load data from HDF5 database file.
    db_file = vcl.databases_dir / 'stellar_db_uncorrected_hot_stars.hdf5'

    tqdm.write('Reading data from stellar database file...')
    star_transition_offsets = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets')
    star_transition_offsets_EotWM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotWM')
    star_transition_offsets_EotM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotM')
    star_temperatures = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_temperatures')

    with h5py.File(db_file, mode='r') as f:

        star_metallicities = hickle.load(f, path='/star_metallicities')
        star_magnitudes = hickle.load(f, path='/star_magnitudes')
        star_gravities = hickle.load(f, path='/star_gravities')
        column_dict = hickle.load(f, path='/transition_column_index')
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
    for transition in tqdm(transitions_list):
        if transition.blendedness < 3:
            for order_num in transition.ordersToFitIn:
                label = '_'.join([transition.label, str(order_num)])
                labels.append(label)

    bin_dict = {}
    # Set bins manually.
    bin_dict['temp'] = [5377, 5477, 5577, 5677,
                        5777, 5877, 5977, 6077,
                        6177, 6277]
    bin_dict['mtl'] = [-0.75, -0.6, -0.45, -0.3,
                       -0.15, 0, 0.15, 0.3, 0.45]
    bin_dict['logg'] = [4.04, 4.14, 4.24,
                        4.34, 4.44, 4.54, 4.64]

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

    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    ax1.set_ylim(bottom=0, top=200)

    ax1.set_xlabel(r'$T_\mathrm{eff}$')
    ax1.set_ylabel(r'$\sigma_\mathrm{s2s}$ (m/s)')
    ax2.set_xlabel('[Fe/H]')
    ax3.set_xlabel(r'$\log{g}$')

    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)

    for ax in (ax1, ax2, ax3):
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    for plot_type, lims, ax in zip(plot_types,
                                   (temp_lims, mtl_lims, logg_lims),
                                   (ax1, ax2, ax3)):
        for limit in bin_dict[plot_type]:
            ax.axvline(x=limit, color='MediumSeaGreen', linestyle='--',
                       alpha=0.3, zorder=1)

    data_file = vcl.output_dir / 'stellar_parameter_dependence_data.h5py'
    if not use_cached:
        for label_num, label in tqdm(enumerate(labels[3:9]),
                                     total=len(labels[3:9])):

            vprint(f'Analyzing {label}...')
            # The column number to use for this transition:
            try:
                col = column_dict[label]
            except KeyError:
                print(f'Incorrect key given: {label}')
                sys.exit(1)

            vprint(20 * '=')
            vprint(f'Working on pre-change era.')
            mean = np.nanmean(star_transition_offsets[eras['pre'], :, col])

            # First, create a masked version to catch any missing entries:
            m_offsets = ma.masked_invalid(star_transition_offsets[
                        eras['pre'], :, col])
            m_offsets = m_offsets.reshape([len(m_offsets), 1])
            # Then create a new array from the non-masked data:
            offsets = u.unyt_array(m_offsets[~m_offsets.mask],
                                   units=u.m/u.s)
            vprint(f'Median of offsets is {np.nanmedian(offsets)}')

            m_eotwms = ma.masked_invalid(star_transition_offsets_EotWM[
                    eras['pre'], :, col])
            m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
            eotwms = u.unyt_array(m_eotwms[~m_eotwms.mask],
                                  units=u.m/u.s)

            m_eotms = ma.masked_invalid(star_transition_offsets_EotM[
                    eras['pre'], :, col])
            m_eotms = m_eotms.reshape([len(m_eotms), 1])
            # Use the same mask as for the offsets.
            eotms = u.unyt_array(m_eotms[~m_offsets.mask],
                                 units=u.m/u.s)
            # Create an error array which uses the greater of the error on
            # the mean or the error on the weighted mean.
            err_array = np.maximum(eotwms, eotms)

            vprint(f'Mean is {np.mean(offsets)}')
            weighted_mean = np.average(offsets, weights=err_array**-2)
            vprint(f'Weighted mean is {weighted_mean}')

            # Mask the various stellar parameter arrays with the same mask
            # so that everything stays in sync.
            temperatures = ma.masked_array(star_temperatures)
            temps = temperatures[~m_offsets.mask]
            metallicities = ma.masked_array(star_metallicities)
            metals = metallicities[~m_offsets.mask]
            magnitudes = ma.masked_array(star_magnitudes)
            mags = magnitudes[~m_offsets.mask]
            gravities = ma.masked_array(star_gravities)
            loggs = gravities[~m_offsets.mask]

            stars = ma.masked_array([key for key in
                                     star_names.keys()]).reshape(
                                         len(star_names.keys()), 1)
            names = stars[~m_offsets.mask]

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

            popt, pcov = curve_fit(model_func, x_data, offsets.value,
                                   sigma=err_array.value,
                                   p0=beta0,
                                   absolute_sigma=True,
                                   method='lm', maxfev=10000)

            model_values = model_func(x_data, *popt)
            residuals = offsets.value - model_values

            # if args.nbins:
            #     nbins = int(args.nbins)
            #     # Use quantiles to get bins with the same number of elements
            #     # in them.
            #     vprint(f'Generating {args.nbins} bins.')
            #     bins = np.quantile(arrays_dict[name],
            #                        np.linspace(0, 1, nbins+1),
            #                        interpolation='nearest')
            #     bin_dict[name] = bins

            min_bin_size = 5  # 5 is maximum before losing stars in current bins
            sigma_sys_dict = {}
            num_params = 1
            for name in tqdm(plot_types):
                sigma_sys_list = []
                sigma_list = []
                bin_mid_list = []
                bin_num = -1
                for bin_lims in pairwise(bin_dict[name]):
                    bin_num += 1
                    lower, upper = bin_lims
                    bin_mid_list.append((lower + upper) / 2)
                    mask_array = ma.masked_outside(arrays_dict[name], *bin_lims)
                    num_points = mask_array.count()
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

        with h5py.File(data_file, mode='a') as f:
            hickle.dump(sigma_sys_dict, f, path='/sigma_sys_dict')
            hickle.dump(full_arrays_dict, f, path='/full_arrays_dict')

    elif use_cached:
        with h5py.File(data_file, mode='r') as f:
            sigma_sys_dict = hickle.load(f, path='/sigma_sys_dict')
            full_arrays_dict = hickle.load(f, path='/full_arrays_dict')

    for name, ax in zip(plot_types, (ax1, ax2, ax3)):
        means = []
        stds = []
        arr = full_arrays_dict[name]
        for i in range(0, np.size(arr, 0)):
            ax.plot(sigma_sys_dict[f'{name}_bin_mids'], arr[i],
                    color='Black', alpha=0.1, zorder=2)
        for j in range(0, np.size(arr, 1)):
            means.append(np.nanmean(arr[:, j]))
            stds.append(np.nanstd(arr[:, j]))
        ax.errorbar(sigma_sys_dict[f'{name}_bin_mids'], means,
                    yerr=stds, color='IndianRed', alpha=1,
                    marker='o', markersize=4, capsize=4,
                    elinewidth=2, zorder=3, linestyle='-',
                    label='Mean and RMS')
        ax.legend()

    plot_path = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

    filename = plot_path / 'Stellar_parameter_dependence.png'
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

    fig = plt.figure(figsize=(6, 8), tight_layout=True)
    gs = GridSpec(2, 1, figure=fig, hspace=0)
    ax_measured = fig.add_subplot(gs[0, 0])
    ax_corrected = fig.add_subplot(gs[1, 0])

    ax_measured.set_ylabel('Measured')
    ax_corrected.set_ylabel('Corrected')
    ax_corrected.set_xlabel(r'$\Delta v_\mathrm{\,instance\ 2}-'
                            r'\Delta v_\mathrm{\,instance\ 1}$ (m/s)')

    order_boundaries = []
    for i in range(len(pair_order_numbers)):
        if i == 0:
            continue
        if pair_order_numbers[i-1] != pair_order_numbers[i]:
            order_boundaries.append(i - 0.5)

    ax_measured.tick_params(labelbottom=False)

    for ax in (ax_measured, ax_corrected):
        # ax.yaxis.grid(which='major', color='Gray', alpha=0.7,
        #               linestyle='-')
        # ax.yaxis.grid(which='minor', color='Gray', alpha=0.6,
        #               linestyle='--')
        ax.axvline(x=0, linestyle='-', color='Gray')
        ax.set_xlim(left=-55, right=55)
        ax.set_ylim(bottom=-1, top=55)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(labelleft=False)
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
        ax_measured.annotate(r'$\chi^2_\nu$:'
                             f' {pairs_chisq:.2f}\n'
                             f'RMS: {pairs_sigma:.2f} m/s',
                             xy=(0, 0), xytext=(-52, 26),
                             textcoords='data', size=19,
                             horizontalalignment='left',
                             verticalalignment='bottom')
        ax_corrected.errorbar(model_diffs, model_pair_indices,
                              xerr=model_errs,
                              color='DodgerBlue', capsize=2, capthick=2,
                              elinewidth=2,
                              ecolor='DodgerBlue', markeredgecolor='Black',
                              linestyle='', marker='D',
                              label=r'Model $\chi^2_\nu$:'
                              f' {model_chisq:.2f}, RMS: {model_sigma:.2f}')
        ax_corrected.annotate(r'$\chi^2_\nu$:'
                              f' {model_chisq:.2f}\n'
                              f'RMS: {model_sigma:.2f} m/s',
                              xy=(0, 0), xytext=(-52, 26),
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
        # create_example_pair_sep_plots()

        # create_sigma_sys_hist()

        vprint('Creating parameter dependence plot.')
        create_parameter_dependence_plot(use_cached=False)

        # plot_duplicate_pairs(Star('Vesta', '/Users/dberke/data_output/Vesta'))
