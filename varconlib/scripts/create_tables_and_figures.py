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

import cmasher as cmr
import h5py
import hickle
import numpy as np
import numpy.ma as ma
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.stats import anderson_ksamp, ks_2samp
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting as fit
from varconlib.miscellaneous import (remove_nans, weighted_mean_and_error,
                                     get_params_file,
                                     velocity2wavelength, wavelength2index)
from varconlib.obs2d import HARPSFile2DScience
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


def find_star_category(star):
    """
    Return the category (SP1, 2, 3, or other) that a star belongs to.

    Parameters
    ----------
    star : `varconlib.star.Star`
        A star object to test the membership of.

    Returns
    -------
    str
        Either 'SP1', 'SP2', 'SP3', or 'Other'.

    """

    categories = {1: 'SP1', 2: 'SP2', 3: 'SP3'}

    temp_increment = 100 * u.K
    mtl_increment = 0.1
    logg_increment = 0.1
    solar_teff = 5777 * u.K
    solar_feh = 0.
    solar_logg = 4.44

    teff = star.temperature
    feh = star.metallicity
    logg = star.logg

    for i in range(1, 4):
        teff_low = solar_teff - i * temp_increment
        teff_high = solar_teff + i * temp_increment
        feh_low = solar_feh - i * mtl_increment
        feh_high = solar_feh + i * mtl_increment
        logg_low = solar_logg - i * logg_increment - 0.1
        logg_high = solar_logg + i * logg_increment + 0.1

        if (teff_low <= teff <= teff_high) and\
           (feh_low <= feh <= feh_high) and\
           (logg_low <= logg <= logg_high):
            return categories[i]

    # tqdm.write(f'{star.name:>8}: {teff} {feh} {logg}')
    return 'Other'


def create_HR_diagram_plot():
    """
    Create an HR diagram of the stars in our sample.

    Returns
    -------
    None.

    """

    # Get all stars from the Nordstrom et al. 2004 catalog.
    nordstrom_table = vcl.data_dir / 'Nordstrom+2004_table1.dat'
    all_colors, all_mags = np.loadtxt(nordstrom_table,
                                      dtype=str, unpack=True,
                                      delimiter='|', usecols=(10, 16))
    nordstrom_mags, nordstrom_colors = [], []
    for mag, color in zip(all_mags, all_colors):
        try:
            nordstrom_mags.append(float(mag))
            nordstrom_colors.append(float(color))
        except ValueError:
            pass

    # Get stars in our initial selection.
    nordstrom_stellar_sample_file = vcl.data_dir / "StellarSampleData_full.csv"
    star_mags, star_colors = np.loadtxt(nordstrom_stellar_sample_file,
                                        dtype=str, unpack=True,
                                        delimiter=',', usecols=(5, 7))

    star_mags = [float(x) for x in star_mags]
    star_colors = [float(x) for x in star_colors]

    star_names = [d.split('/')[-1] for d in
                  glob('/Users/dberke/data_output/[H]*')]

    sp1_list, sp3_list, remainder_list = [], [], []
    temp_lim = 6077 * u.K
    metal_lim = -0.45
    tqdm.write('Gathering stars...')
    for star_name in tqdm(star_names):
        star = Star(star_name, vcl.output_dir / star_name)
        # category = find_star_category(star)
        # tqdm.write(f'{star_name} is in {category}')
        if star.temperature > temp_lim:
            del star
        elif star.metallicity < metal_lim:
            del star
        else:
            category = find_star_category(star)
            if category == 'SP1':
                sp1_list.append(star)
            elif (category == 'SP32') or (category == 'SP3'):
                sp3_list.append(star)
            else:
                remainder_list.append(star)
            try:
                num_planets = star.specialAttributes['has_planets']
                tqdm.write(f'{star_name}: {num_planets} planets')
            except KeyError:
                pass

    sp1_colors, sp1_mags = [], []
    sp3_colors, sp3_mags = [], []
    other_colors, other_mags = [], []
    for star in tqdm(sp1_list):
        sp1_colors.append(star.color)
        sp1_mags.append(star.absoluteMagnitude)
    for star in tqdm(sp3_list):
        sp3_colors.append(star.color)
        sp3_mags.append(star.absoluteMagnitude)
    for star in tqdm(remainder_list):
        other_colors.append(star.color)
        other_mags.append(star.absoluteMagnitude)

    vesta = Star('Vesta', '/Users/dberke/data_output/Vesta')

    colors = cmr.take_cmap_colors('cmr.sunburst', 3,
                                  cmap_range=(0.4, 0.9), return_fmt='hex')

    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel(r'Colour, $b-y$', fontsize=18)
    ax.set_ylabel(r'Absolute magnitude, M$_\mathrm{V}$', fontsize=18)
    ax.set_ylim(bottom=5.66, top=3.95)
    ax.set_xlim(left=0.25, right=0.53)

    # Plot all the stars in the GCS survey.
#    ax.plot(nordstrom_colors, nordstrom_mags,
#            color='Gray', marker='.', markersize=1,
#            linestyle='', zorder=1,
#            label=r'GCS')

    # Plot the Sun.
    ax.plot(vesta.color, vesta.absoluteMagnitude,
            color='LightSkyBlue', marker='o', markersize=8,
            markeredgecolor='Black', markeredgewidth=1,
            linestyle='', zorder=6, alpha=1,
            label='Sun')
    # Plot solar twins.
    ax.plot(sp1_colors, sp1_mags,
            color=colors[0],
            markeredgecolor='Black', marker='D', markersize=6,
            linestyle='', alpha=1,
            label='Solar twins', zorder=5)
    # Plot solar analogues.
    ax.plot(sp3_colors, sp3_mags,
            color=colors[1],
            markeredgecolor='Black', marker='s', markersize=6.5,
            linestyle='', alpha=1,
            label='Solar analogues', zorder=4)
    # Plot solar-type stars.
    ax.plot(other_colors, other_mags,
            color=colors[2],
            markeredgecolor='Black', marker='h', markersize=7.5,
            linestyle='', alpha=1,
            label='Solar-type stars', zorder=3)
    # Plot the stars in our initial selection box.
    ax.plot(star_colors, star_mags,
            color='DimGray', marker='o', markersize=2,
            linestyle='', zorder=2,
            label='Initial selection')

    ax.legend(loc='lower left', fontsize=17,
              handletextpad=0.1, handlelength=0.8,
              borderpad=0.2)

    filename = output_dir / 'plots/Sample_HR_diagram.pdf'
    fig.savefig(str(filename), bbox_inches='tight', pad_inches=0.01)


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
    separations = u.unyt_array(m_seps[~m_seps.mask],
                               units=u.km/u.s).to(u.m/u.s)

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

    ax2.annotate(r'$\lambda4492.660\,\textrm{Fe\,\textsc{\lowercase{II}}},\,'
                 r'\lambda4503.480\,\textrm{Mn\,\textsc{\lowercase{I}}}$',
                 xy=(0, 0), xytext=(0.03, 0.02),
                 textcoords='axes fraction', size=19,
                 horizontalalignment='left', verticalalignment='bottom')

    ax3.set_xlabel('[Fe/H]')

    for ax in (ax1, ax2):
        ax.set_ylabel('Normalized pair\nseparation (m/s)', size=18)
    ax3.set_ylabel('Residuals (m/s)', size=18)

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for ax in (ax2, ax3):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2.tick_params(labelbottom=False)

    ax1.set_xlim(left=-0.11, right=0.11)
    ax3.set_xlim(left=-0.465, right=0.44)
    ax3.set_ylim(bottom=-110, top=110)
    ax2.set_ylim(bottom=-220, top=220)

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

    ax3.annotate(r'$\sigma_{**}=\,$'
                 f'{sigma_s2s:.2f}',
                 xy=(0, 0), xytext=(0.03, 0.97),
                 textcoords='axes fraction', size=18,
                 horizontalalignment='left', verticalalignment='top')

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
    fig2.savefig(str(plot_dir / f'{label}_sample.pdf'))

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

    # Handle various fitting and plotting setup:
    eras = {'pre': 0, 'post': 1}
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
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0,
                                                               p_index1]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index1)
                pair_model_pre1.append(w_mean)
                pair_model_err_pre1.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0,
                                                               p_index1]**2))

                # Get the values for the second duplicate
                time_slice = slice(None, star.fiberSplitIndex)
                w_mean, eotwm = get_weighted_mean(
                    star.pairSeparationsArray,
                    star.pairSepErrorsArray,
                    time_slice,
                    p_index2)
                pair_sep_pre2.append(w_mean)
                pair_sep_err_pre2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0,
                                                               p_index2]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index2)
                pair_model_pre2.append(w_mean)
                pair_model_err_pre2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[0,
                                                               p_index2]**2))

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
        bin_means_pre_nn = []
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
    fig = plt.figure(figsize=(4.5, 3.8), tight_layout=True)
    gs = GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    ax.set_xlabel('RMS (m/s)', fontsize=18)
    ax.set_ylabel('Mean error (m/s)', fontsize=18)

    # ax.tick_params(bottom=False, labelbottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # Customize tick label font size.
    ax.tick_params(labelsize=18)

    ax.set_xlim(left=-1, right=45)
    ax.set_ylim(bottom=0, top=82)

    colors = cmr.take_cmap_colors('cmr.neon_r', 3,
                                  cmap_range=(0, 1), return_fmt='hex')

    five_patch = mlines.Line2D([], [], color=colors[2], marker='h',
                               markersize=10, linestyle='',
                               markeredgecolor='Black', alpha=0.7,
                               label='5')
    four_patch = mlines.Line2D([], [], color=colors[1], marker='D',
                               markersize=8, linestyle='',
                               markeredgecolor='Black', alpha=0.7,
                               label='4')
    other_patch = mlines.Line2D([], [], color=colors[0], marker='o',
                                markersize=9, linestyle='',
                                markeredgecolor='Black', alpha=0.7,
                                label=r'$\leq\!3$')

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

    ax.legend(handles=[other_patch, four_patch, five_patch],
              loc='upper left', fontsize=18, ncol=3,
              shadow=True,
              title='Maximum blendedness',
              handlelength=1.3,
              handletextpad=0.0,
              columnspacing=1.3,
              title_fontsize=18)

#    ax.annotate('(0, 0)', xy=(3, 9), xytext=(1, 22),
#                arrowprops={'arrowstyle': '-', 'color': 'Black'},
#                horizontalalignment='left',
#                verticalalignment='bottom',
#                fontsize=20, weight='bold')
    # strings = [str(blend_tuple) for blend_tuple in sorted_blend_tuples]
    # texts = [plt.text(sigmas_pre[i], mean_errs_pre[i], strings[i],
    #                   ha='center', va='center', fontsize=20)
    #          for i in range(len(strings))]

    # adjust_text(texts,
    #             force_text=(0.2, 0.35),
    #             arrowprops=dict(arrowstyle='-', color='Black'),
    #             lim=1000)

    filename = plots_dir / f'{star.name}_by_blendedness_(5,5).pdf'
    if not plots_dir.exists():
        os.mkdir(plots_dir)
    ax.margins(0, 0)
    fig.savefig(str(filename), bbox_inches='tight', pad_inches=0.01)
    plt.close('all')


def plot_representative_blendedness_plots():
    """Create a plot showing examples of the various blendedness categories.

    """

    # Vesta observation with SNR = 226
    data_file = vcl.harps_dir /\
        'Vesta/data/reduced/'\
        '2011-09-29/HARPS.2011-09-29T23:30:27.911_e2ds_A.fits'

    data_file = '/Users/dberke/Vesta/data/reduced/'\
        '2011-09-29/HARPS.2011-09-29T23:30:27.911_e2ds_A.fits'

    obs = HARPSFile2DScience(data_file)

    transitions = {'category0': ('4575.498Fe1_27',
                                 '5595.288Ni1_51',
                                 '6167.066Fe1_61'),
                   'category1': ('4589.484Cr2_28',
                                 '5143.171Fe1_42',
                                 '6131.832Ni1_60'),
                   'category2': ('4492.660Fe2_25',
                                 '5187.997Ni1_43',
                                 '6245.542Si1_62'),
                   'category3': ('4223.402Fe1_16',
                                 '5536.376Fe2_49',
                                 '6165.470Ca1_61'),
                   'category4': ('4295.006Fe1_18',
                                 '5146.101Cr1_42',
                                 '5596.208Fe1_51'),
                   'category5': ('4309.365Mn2_19',
                                 '5147.740Fe1_42',
                                 '5209.550Cr1_44')}

    # Create the plot.
    fig = plt.figure(figsize=(18, 4), tight_layout=True)
    axes = fig.subplots(1, 6, sharey=True, gridspec_kw={'wspace': 0})

    for i, category in tqdm(enumerate(transitions.keys())):
        axes[i].set_xlabel(f'Blendedness: {category[-1]}')
        for j, t_label in enumerate(transitions[category]):
            tqdm.write(f'Working on {t_label}')
            wavelength = float(t_label[:8]) * u.angstrom
            order_num = int(t_label[-2:])

            wavelength_data = obs.barycentricArray[order_num]
            flux_data = obs.photonFluxArray[order_num]
            error_data = obs.errorArray[order_num]

            plot_range = velocity2wavelength(15 * u.km/u.s,
                                             wavelength)

            low_index = wavelength2index(wavelength - plot_range,
                                         wavelength_data)
            high_index = wavelength2index(wavelength + plot_range,
                                          wavelength_data)
            indices = [i for i in range(high_index - low_index)]

            axes[i].errorbar(indices,
                             flux_data[low_index:high_index],
                             yerr=error_data[low_index:high_index],
                             barsabove=True,
                             color='Black', ecolor='Chocolate',
                             linestyle='-', marker='')

    plt.show()


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
        'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
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
    ax_chi_squared = fig.add_subplot(gs[0, 1], sharey=ax_pre)

    ax_pre.set_xlabel('Model offset (m/s)')
    ax_chi_squared.set_xlabel(r'$\chi^2_{\nu}$', fontsize=18)
    ax_pre.set_ylabel('Normalized mean depth of pair')

    ax_chi_squared.tick_params(labelleft=False)

    ax_pre.set_xlim(left=-45, right=45)
    ax_pre.set_ylim(top=0.12, bottom=0.74)
    ax_pre.axvline(x=0, color='Gray', linestyle='--', zorder=0)
#    ax_pre.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#    ax_chi_squared.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_chi_squared.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 2]))
    ax_chi_squared.axvline(x=1, linestyle='-.', color='Gray',
                           zorder=0, alpha=1)
    grid_keywords = {'which': 'major', 'linestyle': ':',
                     'color': 'SeaGreen', 'alpha': 0.8,
                     'linewidth': 1.5}
    ax_pre.yaxis.grid(**grid_keywords)
    ax_chi_squared.yaxis.grid(**grid_keywords)
    ax_chi_squared.set_xlim(left=-0.07, right=2.07)
    for ax in (ax_pre, ax_chi_squared):
        # ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax.xaxis.grid(which='major', color='Gray',
        #               linestyle=':', alpha=0.65)
        # ax.xaxis.grid(which='minor', color='Gray',
        #               linestyle=':', alpha=0.5)

    full_errs_pre = np.sqrt(pair_model_err_pre ** 2 +
                            sigmas_sys_pre ** 2)
    values, mask = remove_nans(pair_model_sep_pre, return_mask=True)
#    chisq = calc_chi_squared_nu(values,
#                                full_errs_pre[mask], 1)
    ax_pre.errorbar(pair_model_sep_pre, pair_depth_means,
                    xerr=full_errs_pre,
                    # xerr=pair_model_err_pre,
                    linestyle='', marker='o', markersize=5,
                    markerfacecolor='Chocolate',
                    markeredgecolor='Black', ecolor='Peru',
                    zorder=5, capsize=0, capthick=0,
                    label=f'Teff: {star.temperature}\n'
                    f'[Fe/H]: {star.metallicity}\n'
                    f'log g: {star.logg}')

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

        chisq.append(fit.calc_chi_squared_nu(values, errs, 1))
    ax_chi_squared.errorbar(chisq, midpoints,
                            color='Black', marker='o',
                            markerfacecolor='White',
                            markersize=10, linewidth=2, markeredgewidth=1.5,
                            capsize=3, capthick=2,
                            zorder=5)

#    ax_pre.legend(loc='lower left', fontsize=14)
    category = find_star_category(star)
    outfile = plots_dir / f'{star.name}_obs_by_mean_depth_{category}.pdf'
    fig.savefig(str(outfile))
    plt.close('all')


def create_sigma_s2s_histogram():
    """Create a histogram showing the distribution of sigma_s2s values."""

    # Get values for the 17 pairs on the shortlist.
    pairs_file = vcl.data_dir / '17_pairs.txt'
    pair_labels = np.loadtxt(pairs_file, dtype=str)

    vesta = Star('Vesta', vcl.output_dir / 'Vesta')

    # Size 20 because 17 pairs + 3 duplicates
    pre_sigma_s2s_short_list = np.zeros(20, dtype=float)
    post_sigma_s2s_short_list = np.zeros(20, dtype=float)

    for i, label in enumerate(pair_labels):
        index = vesta.p_index(label)
        pre_sigma_s2s_short_list[i] = vesta.pairSysErrorsArray[0, index].value
        post_sigma_s2s_short_list[i] = vesta.pairSysErrorsArray[1, index].value

    # Get all the max blendedness < 3 pairs.
    pre_sigma_s2s_blend_2 = []
    post_sigma_s2s_blend_2 = []
    for pair in vesta.pairsList:
        if (pair.blendTuple[0] < 3) and (pair.blendTuple[1] < 3):
            for order_num in pair.ordersToMeasureIn:
                label = '_'.join([pair.label, str(order_num)])
                index = vesta.p_index(label)
                pre_sigma_s2s_blend_2.append(float(vesta.pairSysErrorsArray[
                        0, index].value))
                post_sigma_s2s_blend_2.append(float(vesta.pairSysErrorsArray[
                        1, index].value))

    total_short_list = np.concatenate((pre_sigma_s2s_short_list,
                                       post_sigma_s2s_short_list))
    total_blend_2 = np.concatenate((pre_sigma_s2s_blend_2,
                                    post_sigma_s2s_blend_2))

    # Make the plot for the 17 pairs with q-coefficients.
    fig1 = plt.figure(figsize=(5, 5), tight_layout=True)
    ax1 = fig1.add_subplot(1, 1, 1)

    ax1.set_ylabel('N')
    ax1.set_xlabel(r'Star-to-star scatter, $\sigma_{**}$ (m/s)')
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Print out the medians of the distributions.
    vprint('Median of pre (17 pairs):'
           f' {np.median(pre_sigma_s2s_short_list):.2f}')
    vprint('Median of post (17 pairs): '
           f'{np.median(post_sigma_s2s_short_list):.2f}')

    short_list_bins = [x+0.0001 for x in range(
            -1, ceil(total_short_list.max())+1)]
    ax1.hist(total_short_list,
             histtype='step', color='Gray', linestyle='-',
             bins=short_list_bins, linewidth=2.5,
             label='Total')
    ax1.hist(post_sigma_s2s_short_list,
             histtype='step', color='Black', linestyle='-',
             bins=short_list_bins, linewidth=2.5,
             label='Post')
    ax1.legend(loc='upper left')

    outfile = plots_dir / 'Sigma_s2s_histogram_17_pairs.pdf'
    fig1.savefig(str(outfile))

    fig2 = plt.figure(figsize=(5, 5), tight_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)

    ax2.set_ylabel('N')
    ax2.set_xlabel(r'Star-to-star scatter, $\sigma_{**}$ (m/s)')
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Print out the medians of the distributions.
    vprint('Median of pre (all blend 2):'
           f' {np.median(pre_sigma_s2s_blend_2):.2f}')
    vprint('Median of post (all blend 2): '
           f'{np.median(post_sigma_s2s_blend_2):.2f}')

    blend_2_bins = [x+0.0001 for x in range(-1, ceil(total_blend_2.max())+1)]
    ax2.hist(total_blend_2,
             histtype='step', color=cmr.torch(0.5), linestyle='-',
             bins=blend_2_bins, linewidth=2.5,
             label='Total')
    ax2.hist(pre_sigma_s2s_blend_2,
             histtype='step', color=cmr.torch(0.8), linestyle='-',
             bins=blend_2_bins, linewidth=2.5,
             label='Pre')
    ax2.hist(post_sigma_s2s_blend_2,
             histtype='step', color=cmr.torch(0.2), linestyle='-',
             bins=blend_2_bins, linewidth=2.5,
             label='Post')

    ax2.legend(loc='upper right')

    outfile = plots_dir / 'Sigma_s2s_histogram_blend_2_pairs.pdf'
    fig2.savefig(str(outfile))

    vprint(f'# of pre values: {len(pre_sigma_s2s_blend_2)}')
    vprint(f'# of post values: {len(post_sigma_s2s_blend_2)}')

    # Run a k-sample Anderson-Darling test to see if the two samples appear
    # likely to come from the same distribution.
#    k_value, crit_values, sig_level = anderson_ksamp(
#            [np.array(pre_sigma_s2s_blend_2),
#             np.array(post_sigma_s2s_blend_2)])
#
#    vprint('For the Anderson-Darling k-sample test:')
#    vprint(f'statistic: {k_value}')
#    vprint(f'critical values: {crit_values}')
#    vprint(f'significance level: {sig_level}')

    ks_value, p_value = ks_2samp(pre_sigma_s2s_blend_2,
                                 post_sigma_s2s_blend_2)
    vprint('For the Kologorov-Smirnov test:')
    vprint(f'KS statistic: {ks_value}')
    vprint(f'p-value: {p_value}')


def plot_solar_twins_results():
    """Plot results for 17 pairs with q-coefficients for solar twins"""

    def format_label(pair_label):
        """Format a pair label for printing with MNRAS ion format.

        Parameters
        ----------
        pair_label : str
             A pair label of the form "4492.660Fe2_4503.480Mn1_25"

        Returns
        -------
        dict
            A dictionary containing LaTeX-formatted representations of the two
            transitions in the pair label.
        """

        t1, t2, order_num = pair_label.split('_')
        # This mimics the look of ion labels in MNRAS.
        new_label1 = f"{t1[:8]}" + r"\ " + f"{t1[8:-1]}" + r"\," + \
            r"\textsc{\lowercase{" + f"{roman_numerals[t1[-1]]}" + r"}}"
        new_label2 = f"{t2[:8]}" + r"\ " + f"{t2[8:-1]}" + r"\," + \
            r"\textsc{\lowercase{" + f"{roman_numerals[t2[-1]]}" + r"}}"

        return {'ion1': new_label1, 'ion2': new_label2}

    roman_numerals = {'1': 'I', '2': 'II'}

    # Get labels of the 17 pairs on the shortlist.
    pairs_file = vcl.data_dir / '17_pairs.txt'
    pair_labels = np.loadtxt(pairs_file, dtype=str)

    # Get the 18 solar twins.
    stars = {star_name: Star(star_name, vcl.output_dir / star_name)
             for star_name in sp1_stars}

    # Set out lists of star for the top and bottom panels.
    block1_stars = ('Vesta', 'HD76151', 'HD78429',
                    'HD140538', 'HD146233', 'HD157347')
    block2_stars = ('HD20782', 'HD19467', 'HD45184',
                    'HD45289', 'HD138573', 'HD171665',)
    block3_stars = ('HD183658', 'HD220507', 'HD222582')
    block4_stars = ('HD1835', 'HD30495', 'HD78660', )

    block1_width = 25
    block1_ticks = 15
    block2_width = 50
    block2_ticks = 30
    block3_width = 80
    block3_ticks = 50
    block4_width = 125
    block4_ticks = 75

    fig = plt.figure(figsize=(18, 10), tight_layout=True)
    gs = GridSpec(ncols=20, nrows=4, figure=fig, wspace=0,
                  height_ratios=(len(block1_stars),
                                 len(block2_stars),
                                 len(block3_stars),
                                 len(block4_stars)))
    # Set the "velocity" title to be below the figure.
    fig.supxlabel('Velocity (m/s)', fontsize=18)

    fig_hist = plt.figure(figsize=(12, 5), tight_layout=True)
    gs_hist = GridSpec(ncols=10, nrows=2, figure=fig_hist, wspace=0)

    # Create a dict to hold all the axes.
    axes = {}
    # Create top panel (with pair labels)
    # Create tick locations to put the grid at.
    y_grid_locations = [y+0.5 for y in range(len(block1_stars))]
    for i, label in (enumerate(pair_labels)):
        ax = fig.add_subplot(gs[0, i])
        ax.axvline(x=0, color='Black', linestyle='--', linewidth=1.7,
                   zorder=1)
        # Set the limits of each axis.
        ax.set_ylim(top=-0.5, bottom=len(block1_stars)-0.5)
        ax.set_xlim(left=-block1_width, right=block1_width)
        # Add the grid.
        ax.yaxis.set_minor_locator(ticker.FixedLocator(y_grid_locations))
        ax.yaxis.grid(which='minor', color='LightGray', linewidth=1.8,
                      linestyle=':', zorder=0)
        # Remove all the ticks and labels on the y-axes (left-most will have
        # them specially added back in).
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelleft=False, labelright=False)
        ax.tick_params(axis='x', which='both', top=False, bottom=True,
                       labeltop=False, labelbottom=True, labelsize=12)
        ax.xaxis.set_major_locator(ticker.FixedLocator(
                (-block1_ticks, 0, block1_ticks)))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # This sets the width of the outside edges of the subaxes.
        for axis in ['top', 'right', 'bottom', 'left']:
            ax.spines[axis].set_linewidth(2.1)
            ax.spines[axis].set_zorder(20)

        # Add the tick labels for each pair at the top of the plot.
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.tick_params(top=False, labelsize=16)
        t1, t2, order_num = label.split('_')
        if i > 5:
            ax_twin.xaxis.set_major_locator(ticker.FixedLocator((-12,)))
            ax_twin.set_xticklabels(('{ion1}\n{ion2}'.format(
                    **format_label(label)),),
                                    fontdict={'rotation': 90,
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'})
        elif i in (0, 2, 4):
            ax_twin.xaxis.set_major_locator(ticker.FixedLocator((-11, 12)))
            ax_twin.set_xticklabels((str(order_num),
                                     '{ion1}\n{ion2}'.format(
                    **format_label(label)),),
                                    fontdict={'rotation': 90,
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'})
        elif i in (1, 3, 5):
            ax_twin.xaxis.set_major_locator(ticker.FixedLocator((2,)))
            ax_twin.set_xticklabels((f'{str(order_num)}',),
                                    fontdict={'rotation': 90,
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'})
        # Add axis to axes dictionary.
        axes[(0, i)] = ax

    # Create second panel
    y_grid_locations = [y+0.5 for y in range(len(block2_stars))]
    for i, label in (enumerate(pair_labels)):
        ax = fig.add_subplot(gs[1, i])
        ax.axvline(x=0, color='Black', linestyle='--', linewidth=1.7)
        ax.set_ylim(top=-0.5, bottom=len(block2_stars)-0.5)
        ax.set_xlim(left=-block2_width, right=block2_width)
        ax.yaxis.set_minor_locator(ticker.FixedLocator(y_grid_locations))
        ax.yaxis.grid(which='minor', color='LightGray', linewidth=1.8,
                      linestyle=':')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.FixedLocator(
                (-block2_ticks, 0, block2_ticks)))
        ax.tick_params(which='both', labelleft=False, labelbottom=True,
                       left=False, right=False, top=False, bottom=True,
                       labelsize=12)
        for axis in ['top', 'right', 'bottom', 'left']:
            ax.spines[axis].set_linewidth(2.1)
            ax.spines[axis].set_zorder(20)
        axes[(1, i)] = ax

    # Create third panel
    y_grid_locations = [y+0.5 for y in range(len(block3_stars))]
    for i, label in (enumerate(pair_labels)):
        ax = fig.add_subplot(gs[2, i])
        ax.axvline(x=0, color='Black', linestyle='--', linewidth=1.7)
        ax.set_ylim(top=-0.5, bottom=len(block3_stars)-0.5)
        ax.set_xlim(left=-block3_width, right=block3_width)
        ax.yaxis.set_minor_locator(ticker.FixedLocator(y_grid_locations))
        ax.yaxis.grid(which='minor', color='LightGray', linewidth=1.8,
                      linestyle=':')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.FixedLocator(
                (-block3_ticks, 0, block3_ticks)))
        ax.tick_params(which='both', labelleft=False, labelbottom=True,
                       left=False, right=False, top=False, bottom=True,
                       labelsize=12)
        for axis in ['top', 'right', 'bottom', 'left']:
            ax.spines[axis].set_linewidth(2.1)
            ax.spines[axis].set_zorder(20)
        axes[(2, i)] = ax

    # Create fourth panel
    y_grid_locations = [y+0.5 for y in range(len(block4_stars))]
    for i, label in (enumerate(pair_labels)):
        ax = fig.add_subplot(gs[3, i])
        ax.axvline(x=0, color='Black', linestyle='--', linewidth=1.7)
        ax.set_ylim(top=-0.5, bottom=len(block4_stars)-0.5)
        ax.set_xlim(left=-block4_width, right=block4_width)
        ax.yaxis.set_minor_locator(ticker.FixedLocator(y_grid_locations))
        ax.yaxis.grid(which='minor', color='LightGray', linewidth=1.8,
                      linestyle=':')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.FixedLocator(
                (-block4_ticks, 0, block4_ticks)))
        ax.tick_params(which='both', labelleft=False, labelbottom=True,
                       left=False, right=False, top=False, bottom=True,
                       labelsize=12)
        for axis in ['top', 'right', 'bottom', 'left']:
            ax.spines[axis].set_linewidth(2.1)
            ax.spines[axis].set_zorder(20)
        axes[(3, i)] = ax

    # Set the left-most axes to have y-labels for star names.
    for i in range(4):
        axes[(i, 0)].tick_params(labelleft=True)
    # Create the locations for minor ticks to put the star name labels at.
    for i, block in enumerate((block1_stars, block2_stars,
                               block3_stars, block4_stars)):
        y_ticks = [y for y in range(len(block))]
        axes[(i, 0)].yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    # Create the list of top stars...have to handle Vesta specially.
    top_labels = ['Sun']
    top_labels.extend([' '.join((x[:2], x[2:])) for x in block1_stars[1:]])

    axes[(0, 0)].set_yticklabels(top_labels,
                                 fontdict={'horizontalalignment': 'right',
                                           'fontsize': 15})
    for i, star_names in enumerate((block2_stars, block3_stars, block4_stars)):
        axes[(i+1, 0)].set_yticklabels([' '.join((x[:2], x[2:]))
                                       for x in star_names],
                                       fontdict={
                                               'horizontalalignment': 'right',
                                               'fontsize': 15})

    # Define colors for pre- and post- eras.
    pre_color = cmr.ember(0.7)
    post_color = cmr.freeze(0.55)

    # How significant to report outliers.
    sigma_significance = 2
    vprint(f'Looking for outliers beyond {sigma_significance} sigma')
    for i, pair_label in enumerate(pair_labels):
        # Create lists to hold the significance values:
        pre_stat, pre_sys = [], []
        post_stat, post_sys = [], []
        # Figure out some numbers for locating things from star name.
        for star_name in sp1_stars:
            if star_name in block1_stars:
                row = 0
                j = block1_stars.index(star_name)
            elif star_name in block2_stars:
                row = 1
                j = block2_stars.index(star_name)
            elif star_name in block3_stars:
                row = 2
                j = block3_stars.index(star_name)
            elif star_name in block4_stars:
                row = 3
                j = block4_stars.index(star_name)
            else:
                raise RuntimeError(f"{star_name} not in any list!")
            star = stars[star_name]
            pair_index = star.p_index(pair_label)
            fiber_split_index = star.fiberSplitIndex
            # Get the pre-change values.
            if star.hasObsPre:
                values, mask = remove_nans(star.pairModelOffsetsArray[
                        :fiber_split_index, pair_index], return_mask=True)
                errors = star.pairModelErrorsArray[:fiber_split_index,
                                                   pair_index][mask]
                try:
                    value, error = weighted_mean_and_error(values, errors)
                except ZeroDivisionError:
                    # This indicates no value for a particular 'cell', so just
                    # plot something there to indicate that.
                    axes[(row, i)].plot(0, j, color='Black', marker='x',
                                        markersize=7, zorder=10)
                    continue
                # Compute error with sigma_** included.
                sigma_s2s = star.pairSysErrorsArray[0, pair_index]
                full_error = np.sqrt(error**2 + sigma_s2s**2)
                sig_stat = float(abs(value / error).value)
                sig_sys = float(abs(value / full_error).value)
                pre_stat.append(sig_stat)
                pre_sys.append(sig_sys)
                if sig_sys > sigma_significance:
                    vprint(f'{star.name}: {pair_label}:'
                           f' (Pre) {sig_sys:.2f}')
                # First plot an errorbar with sigma_** included.
                axes[(row, i)].errorbar(value, j-0.15,
                                        xerr=full_error,
                                        ecolor=pre_color,
                                        marker='',
                                        capsize=3,
                                        capthick=1.5,
                                        elinewidth=1.4,
                                        zorder=11)
                # Then plot just the star's statistical error.
                axes[(row, i)].errorbar(value, j-0.15,
                                        xerr=error,
                                        markerfacecolor=pre_color,
                                        markeredgecolor='Black',
                                        ecolor=pre_color,
                                        markeredgewidth=2,  # controls capthick
                                        marker='o',
                                        markersize=9,
                                        capsize=5,
                                        elinewidth=4,
                                        zorder=12)
            # Get the post-change values.
            if star.hasObsPost:
                values, mask = remove_nans(star.pairModelOffsetsArray[
                        fiber_split_index:, pair_index], return_mask=True)
                errors = star.pairModelErrorsArray[fiber_split_index:,
                                                   pair_index][mask]
                try:
                    value, error = weighted_mean_and_error(values, errors)
                except ZeroDivisionError:
                    axes[(row, i)].plot(0, j, color='Black', marker='x',
                                        markersize=7)
                    continue
                sigma_s2s = star.pairSysErrorsArray[1, pair_index]
                full_error = np.sqrt(error**2 + sigma_s2s**2)
                sig_stat = float(abs(value / error).value)
                sig_sys = float(abs(value / full_error).value)
                post_stat.append(sig_stat)
                post_sys.append(sig_sys)
                if sig_sys > sigma_significance:
                    vprint(f'{star.name}: {pair_label}:'
                           f' (Post) {sig_sys:.2f}')
                axes[(row, i)].errorbar(value, j+0.15,
                                        xerr=full_error,
                                        ecolor=post_color,
                                        marker='',
                                        capsize=4,
                                        capthick=1.5,
                                        elinewidth=1.5,
                                        zorder=13)
                axes[(row, i)].errorbar(value, j+0.15,
                                        xerr=error,
                                        markerfacecolor=post_color,
                                        markeredgecolor='Black',
                                        ecolor=post_color,
                                        markeredgewidth=2,
                                        marker='D',
                                        markersize=8.5,
                                        capsize=5,
                                        elinewidth=4,
                                        zorder=14)

        # Create the histogram plots for the pair.
        if i > 9:
            i -= 10
            k = 1
        else:
            k = 0
        ax = fig_hist.add_subplot(gs_hist[k, i])
        ax.tick_params(labelleft=False)
        ax.hist(pre_stat, color=pre_color, histtype='step')
        ax.hist(post_stat, color=post_color, histtype='step')

    outfile = plots_dir / 'Pair_offsets_17_pairs.pdf'
    fig.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)

    histfile = plots_dir / 'Pair_offsets_histograms.pdf'
    fig_hist.savefig(str(histfile), bbox_inches='tight', pad_inches=0.01)

    # Create an excerpt of a single column.
    fig_ex = plt.figure(figsize=(5, 7), tight_layout=True)
    ax_ex = fig_ex.add_subplot(1, 1, 1)

    y_grid_locations = [y+0.5 for y in range(len(sp1_stars))]
    ax_ex.axvline(x=0, color='Black', linestyle='--', linewidth=1.7)
    ax_ex.set_ylim(top=-0.5, bottom=len(sp1_stars)-0.5)
    ax_ex.set_xlim(left=-40, right=40)
    ax_ex.yaxis.set_minor_locator(ticker.FixedLocator(y_grid_locations))
    ax_ex.yaxis.grid(which='minor', color='LightGray', linewidth=1.8,
                     linestyle=':')
    ax_ex.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#    ax_ex.xaxis.set_major_locator(ticker.FixedLocator(
#            [-50, -25, 0, 25, 50]))
    ax_ex.tick_params(which='both', labelleft=True, labelbottom=True,
                      left=False, right=False, top=False, bottom=True,
                      labelsize=12)
    for axis in ['top', 'right', 'bottom', 'left']:
        ax_ex.spines[axis].set_linewidth(2.1)
        ax_ex.spines[axis].set_zorder(20)
    ax_ex.set_xlabel('Model offset (m/s)', size=15)

    # Add labels to axis.
    # Create the locations for major ticks to put the star name labels at.
    y_ticks = [y for y in range(len(sp1_stars))]
    ax_ex.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    # Create the list of top stars...have to handle Vesta specially.
    ex_labels = ['Sun']
    ex_labels.extend([' '.join((x[:2], x[2:])) for x in sp1_stars[1:]])

    ax_ex.set_yticklabels(ex_labels,
                          fontdict={'horizontalalignment': 'right',
                                    'fontsize': 15})
    # Set the pair label to use.
    pair_label = pair_labels[10]  # 6138--6139
    pair_label = pair_labels[16]
    fig_ex.suptitle(r'{ion1},\ {ion2}'.format(
                    **format_label(pair_label)), size=16)
    for j, star_name in enumerate(sp1_stars):
        star = stars[star_name]
        pair_index = star.p_index(pair_label)
        fiber_split_index = star.fiberSplitIndex
        # Get the pre-change values.
        if star.hasObsPre:
            values, mask = remove_nans(star.pairModelOffsetsArray[
                    :fiber_split_index, pair_index], return_mask=True)
            errors = star.pairModelErrorsArray[:fiber_split_index,
                                               pair_index][mask]
            try:
                value, error = weighted_mean_and_error(values, errors)
            except ZeroDivisionError:
                # This indicates no value for a particular 'cell', so just
                # plot something there to indicate that.
                ax_ex.plot(0, j, color='Black', marker='x',
                           markersize=7, zorder=10)
                continue
            # Compute error with sigma_** included.
            sigma_s2s = star.pairSysErrorsArray[0, pair_index]
            full_error = np.sqrt(error**2 + sigma_s2s**2)
            significance = abs(value / full_error).value
            if significance > sigma_significance:
                vprint(f'{star.name}: {pair_label}:'
                       f' (Pre) {significance:.2f}')
            # First plot an errorbar with sigma_** included.
            ax_ex.errorbar(value, j-0.15,
                           xerr=full_error,
                           ecolor=pre_color,
                           marker='',
                           capsize=3,
                           capthick=1.5,
                           elinewidth=1.4,
                           zorder=11)
            # Then plot just the star's statistical error.
            ax_ex.errorbar(value, j-0.15,
                           xerr=error,
                           markerfacecolor=pre_color,
                           markeredgecolor='Black',
                           ecolor=pre_color,
                           markeredgewidth=2,  # controls capthick
                           marker='o',
                           markersize=9,
                           capsize=5,
                           elinewidth=4,
                           zorder=12)
        # Get the post-change values.
        if star.hasObsPost:
            values, mask = remove_nans(star.pairModelOffsetsArray[
                    fiber_split_index:, pair_index], return_mask=True)
            errors = star.pairModelErrorsArray[fiber_split_index:,
                                               pair_index][mask]
            try:
                value, error = weighted_mean_and_error(values, errors)
            except ZeroDivisionError:
                ax_ex.plot(0, j, color='Black', marker='x',
                           markersize=7)
                continue
            sigma_s2s = star.pairSysErrorsArray[1, pair_index]
            full_error = np.sqrt(error**2 + sigma_s2s**2)
            significance = abs(value / full_error).value
            if significance > sigma_significance:
                vprint(f'{star.name}: {pair_label}:'
                       f' (Post) {significance:.2f}')
            ax_ex.errorbar(value, j+0.15,
                           xerr=full_error,
                           ecolor=post_color,
                           marker='',
                           capsize=4,
                           capthick=1.5,
                           elinewidth=1.5,
                           zorder=13)
            ax_ex.errorbar(value, j+0.15,
                           xerr=error,
                           markerfacecolor=post_color,
                           markeredgecolor='Black',
                           ecolor=post_color,
                           markeredgewidth=2,
                           marker='D',
                           markersize=8.5,
                           capsize=5,
                           elinewidth=4,
                           zorder=14)

    outfile = plots_dir / f'Pair_offsets_17_pairs_excerpt_{pair_label}.pdf'
    fig_ex.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create all the necessary'
                                     ' figures and tables for my two papers'
                                     ' and thesis.')

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

    sp1_stars = ('Vesta', 'HD1835', 'HD19467', 'HD20782',
                 'HD30495', 'HD45184', 'HD45289', 'HD76151',
                 'HD78429', 'HD78660', 'HD138573', 'HD140538',
                 'HD146233', 'HD157347', 'HD171665', 'HD183658',
                 'HD220507', 'HD222582')

    plots_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

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
#        hd146233 = Star('HD146233', '/Users/dberke/data_output/HD146233')

#        create_HR_diagram_plot()

#         create_example_pair_sep_plots()

        # create_sigma_sys_hist()

#         create_parameter_dependence_plot(use_cached=True, min_bin_size=5)

        # plot_duplicate_pairs(Star('Vesta', '/Users/dberke/data_output/Vesta'))
        # plot_duplicate_pairs(hd146233)

        # create_radial_velocity_plot()

#        plot_vs_pair_blendedness(hd146233)

#        plot_representative_blendedness_plots()

#        plot_pair_depth_differences(hd146233)
#        plot_pair_depth_differences(Star('HD134060',
#                                    '/Users/dberke/data_output/HD134060'))

#        create_sigma_s2s_histogram()

        plot_solar_twins_results()
