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
from fractions import Fraction
from glob import glob
from inspect import signature
from itertools import tee
import lzma
from math import ceil
import os
from pathlib import Path
import pickle
import re
import sys

import cmasher as cmr
import h5py
import hickle
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.exceptions import (NewCoefficientsNotFoundError,
                                  BlazeFileNotFoundError)
import varconlib.fitting as fit
from varconlib.miscellaneous import (remove_nans, weighted_mean_and_error,
                                     get_params_file, shift_wavelength,
                                     wavelength2index,
                                     wavelength2velocity,
                                     velocity2wavelength)
from varconlib.obs2d import HARPSFile2DScience
from varconlib.star import Star
from varconlib.transition_line import roman_numerals

plt.rcParams['text.usetex'] = True

stars_used = set(['HD10180', 'HD102117', 'HD102438', 'HD104982', 'HD106116',
                  'HD108309', 'HD110619', 'HD111031', 'HD114853', 'HD11505',
                  'HD115617', 'HD117105', 'HD117207', 'HD117618', 'HD12387',
                  'HD124292', 'HD125881', 'HD126525', 'HD128674', 'HD134060',
                  'HD134987', 'HD136352', 'HD136894', 'HD138573', 'HD1388',
                  'HD140538', 'HD140901', 'HD141937', 'HD143114', 'HD144585',
                  'HD1461', 'HD146233', 'HD147512', 'HD150433', 'HD152391',
                  'HD154417', 'HD157338', 'HD157347', 'HD1581', 'HD161612',
                  'HD168443', 'HD168871', 'HD171665', 'HD172051', 'HD177409',
                  'HD177565', 'HD1835', 'HD183658', 'HD184768', 'HD189567',
                  'HD189625', 'HD190248', 'HD19467', 'HD196761', 'HD197818',
                  'HD199960', 'HD203432', 'HD20407', 'HD204385', 'HD205536',
                  'HD20619', 'HD2071', 'HD207129', 'HD20766', 'HD20782',
                  'HD20807', 'HD208704', 'HD210918', 'HD211415', 'HD212708',
                  'HD213575', 'HD217014', 'HD220507', 'HD222582', 'HD222669',
                  'HD28821', 'HD30495', 'HD31527', 'HD32724', 'HD361',
                  'HD37962', 'HD38277', 'HD38858', 'HD38973', 'HD39091',
                  'HD43587', 'HD43834', 'HD4391', 'HD44420', 'HD44447',
                  'HD44594', 'HD45184', 'HD45289', 'HD47186', 'HD4915',
                  'HD55693', 'HD59468', 'HD65907', 'HD6735', 'HD67458',
                  'HD68168', 'HD68978', 'HD69655', 'HD69830', 'HD70642',
                  'HD70889', 'HD7134', 'HD72769', 'HD73256', 'HD73524',
                  'HD7449', 'HD76151', 'HD78429', 'HD78558', 'HD78660',
                  'HD82943', 'HD83529', 'HD88742', 'HD90156', 'HD92719',
                  'HD92788', 'HD95521', 'HD96423', 'HD96700', 'HD96937',
                  'HD97037', 'HD97343', 'HD9782', 'HD98281', 'Vesta'])

sp1_stars = set(['Vesta', 'HD1835', 'HD19467', 'HD20782', 'HD30495',
                 'HD45184', 'HD25289', 'HD76151', 'HD78429', 'HD78660',
                 'HD138573', 'HD146233', 'HD157347', 'HD171665',
                 'HD183658', 'HD220507', 'HD222582'])


def pairwise(iterable):
    """Return successive pairs from an iterable.

    E.g., s -> (s0,s1), (s1,s2), (s2, s3), ...

    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def format_transition_label(transition_label):
    """Format a transition label for printing with MNRAS ion format.

    Parameters
    ----------
    transition_label : str
         A transition label of the form "4492.660Fe2_25"

    Returns
    -------
    str
        A LaTeX-formatted representation of the transition label.
    """

    front, order_num = transition_label.split('_')
    # This mimics the look of ion labels in MNRAS.
    new_label = f"{front[8:-1]}" + r"\," + r"\textsc{\lowercase{" +\
        f"{roman_numerals[int(front[-1])]}" + r"}}" + r"\ " + f"{front[:8]}"

    return new_label


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
#    nordstrom_table = vcl.data_dir / 'Nordstrom+2004_table1.dat'
#    all_colors, all_mags = np.loadtxt(nordstrom_table,
#                                      dtype=str, unpack=True,
#                                      delimiter='|', usecols=(10, 16))
#    nordstrom_mags, nordstrom_colors = [], []
#    for mag, color in zip(all_mags, all_colors):
#        try:
#            nordstrom_mags.append(float(mag))
#            nordstrom_colors.append(float(color))
#        except ValueError:
#            pass

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
    temp_lim = 6072 * u.K
    metal_lim = -0.45
    total_obs = 0
    total_too_hot = 0
    total_metal_poor = 0
    tqdm.write('Gathering stars...')
    for star_name in tqdm(star_names):
        star = Star(star_name, vcl.output_dir / star_name)
        # category = find_star_category(star)
        # tqdm.write(f'{star_name} is in {category}')
        if star.temperature > temp_lim:
            del star
            total_too_hot += 1
        elif star.metallicity < metal_lim:
            del star
            total_metal_poor += 1
        else:
            category = find_star_category(star)
            if category == 'SP1':
                sp1_list.append(star)
            elif (category == 'SP32') or (category == 'SP3'):
                sp3_list.append(star)
            else:
                remainder_list.append(star)
            total_obs += star.getNumObs()
            try:
                num_planets = star.specialAttributes['has_planets']
                tqdm.write(f'{star_name}: {num_planets} planets')
            except KeyError:
                pass

    vprint(f'Found {total_obs} total observations for all solar-type stars.')
    vprint(f'Found {total_too_hot} stars that were too hot.')
    vprint(f'Found {total_metal_poor} stars that were too metal-poor.')

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

    vprint(f'{len(star_colors)} in initial selection.')
    vprint(f'{len(other_colors)} in solar-type stars.')
    vprint(f'{len(sp3_colors)} in solar analogs (SP3).')
    vprint(f'{len(sp1_colors)} in solar twins (SP1).')


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

    ax2.annotate(r'$\textrm{Fe\,\textsc{\lowercase{II}}}\,\lambda4492.660,\,'
                 r'\textrm{Mn\,\textsc{\lowercase{I}}}\,\lambda4503.480$',
                 xy=(0, 0), xytext=(0.03, 0.02),
                 textcoords='axes fraction', size=18,
                 horizontalalignment='left', verticalalignment='bottom')

    ax3.set_xlabel('[Fe/H]', size=18)

    for ax in (ax1, ax2):
        ax.set_ylabel('Normalised pair\nseparation (m/s)', size=16)
    ax3.set_ylabel('Pair model offsets (m/s)', size=16)

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for ax in (ax2, ax3):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(labelsize=16)

    ax2.tick_params(labelbottom=False)

    ax1.set_xlim(left=-0.11, right=0.11)
    ax3.set_xlim(left=-0.465, right=0.44)
    ax3.set_ylim(bottom=-110, top=110)
    ax2.set_ylim(bottom=-220, top=220)

    ax3.axhline(y=0, linestyle='--',
                color='Black')

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
    chi_squared = results_quad['chi_squared_list'][-1]
    x_data.mask = mask_quad
    err_array.mask = mask_quad
    names.mask = mask_quad

    # Add sigma_** and chi-squared to plot as annotation.
    ax3.annotate(r'$\sigma_{**}=\,$'
                 f'{sigma_s2s:.2f}\n'
                 r'$\chi^2_\nu=\,$'
                 f'{chi_squared:.2f}',
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
    fig1.savefig(str(plot_dir / f'{label}_SP1.png'),
                 bbox_inches='tight', pad_inches=0.01)
    fig2.savefig(str(plot_dir / f'{label}_sample.pdf'),
                 bbox_inches='tight', pad_inches=0.01)

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

    ax.legend(shadow=True)

    plt.show()


def create_transition_density_plot():
    """Create a plot showing the density of transitions from the Kurucz list.

    """

    transitions_list = [4652.593, 4653.460, 4759.449,
                        4760.600, 4799.873, 4800.072,
                        5138.510, 5143.171, 5187.346,
                        5200.158, 6123.910, 6138.313,
                        6139.390, 6153.320, 6155.928,
                        6162.452, 6168.150, 6175.044,
                        6192.900, 6202.028, 6242.372,
                        6244.834] * u.angstrom
    transitions_list = [4653.460] * u.angstrom

    kurucz_file = vcl.data_dir / "gfallvac08oct17.dat"
    col_widths = (11, 7, 6, 12, 5, 11, 12, 5, 11, 6, 6, 6, 4, 2, 2, 3, 6, 3, 6,
                  5, 5, 3, 3, 4, 5, 5, 6)
    col_names = ("wavelength", "log gf", "elem", "energy1", "J1", "label1",
                 "energy2", "J2", "label2", "gammaRad", "gammaStark",
                 "vanderWaals", "ref", "nlte1",  "nlte2", "isotope1",
                 "hyperf1", "isotope2", "logIsotope", "hyperfshift1",
                 "hyperfshift2", "code", "landeGeven", "landeGodd",
                 "isotopeShift")
    col_dtypes = (float, float, "U6", float, float, "U11", float, float, "U11",
                  float, float, float, "U4", int, int, int, float, int, float,
                  int, int, "U3", "U3", "U4", int, int, float)

    print('Reading Kurucz line list...')
    kurucz_data = np.genfromtxt(kurucz_file, delimiter=col_widths,
                                autostrip=True,
                                skip_header=842959, skip_footer=987892,
                                names=col_names, dtype=col_dtypes,
                                usecols=(0, 2, 3, 4, 5, 6, 7, 8, 18))

    # Use the SNR=316 observation of Vesta.
    data_file = Path('/Users/dberke/Vesta/data/reduced/2011-09-29/'
                     'HARPS.2011-09-29T23:30:27.911_e2ds_A.fits')
    data_obs = HARPSFile2DScience(data_file)

    rad_vel = data_obs.radialVelocity

    for wavelength in transitions_list[:]:
        plot_half_width = 17 * u.km/u.s
        shifted_wl = shift_wavelength(wavelength, rad_vel)
        lower_lim = velocity2wavelength(-plot_half_width,
                                        shifted_wl) + shifted_wl
        upper_lim = velocity2wavelength(plot_half_width,
                                        shifted_wl) + shifted_wl
        unshifted_lower_lim = velocity2wavelength(-plot_half_width,
                                                  wavelength) + wavelength
        unshifted_upper_lim = velocity2wavelength(plot_half_width,
                                                  wavelength) + wavelength

        order = data_obs.findWavelength(shifted_wl,
                                        data_obs.barycentricArray)
        index_min = wavelength2index(lower_lim,
                                     data_obs.barycentricArray[order])
        index_max = wavelength2index(upper_lim,
                                     data_obs.barycentricArray[order]) + 1

        fig = plt.figure(figsize=(6.5, 4.5), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(left=lower_lim, right=upper_lim)
        ax.set_ylim(bottom=0.19, top=1.1)
#        ax.tick_params(top=False)
        ax.set_xlabel(r'Wavelength $(\mathrm{\mathring{A}})$')
        ax.set_ylabel('Normalised flux')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.axvline(shifted_wl, ymin=0, ymax=1, color='Black',
                   linestyle=':', linewidth=2, zorder=15)

        x_values = data_obs.barycentricArray[order, index_min:index_max]
        y_values = data_obs.photonFluxArray[order, index_min:index_max]
        y_values /= y_values.max()

        transitions = 0
        tqdm.write('Parsing Kurucz lines...')
        for transition in tqdm(kurucz_data, unit='transitions'):
            wl = transition['wavelength'] * u.nm
            if unshifted_lower_lim < wl < unshifted_upper_lim:
                transitions += 1
                ax.axvline(shift_wavelength(wl.to(u.angstrom), rad_vel),
                           ymin=0.92, ymax=0.96, color='Gray', linestyle='-',
                           linewidth=1.4, zorder=5)
        tqdm.write(f'Found {transitions} transitions in wavelength range.')
        ax.plot(x_values, y_values,
                color='SandyBrown',
                drawstyle='steps-mid', linewidth=2.8,
                marker='', zorder=10)

        plot_path = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')
        out_file = plot_path /\
            f'Transition_density_plot_{wavelength.value}.pdf'
        fig.savefig(str(out_file), bbox_inches='tight', pad_inches=0.01)


def create_parameter_dependence_plot(use_cached=False, min_bin_size=5):
    """
    Create a plot showing the change in sigma_s2s as a function of stellar
    parameters.

    Parameters
    ----------
    use_cached : bool, Default: False
        If False, will rerun entire binning and fitting procedure, which is
        very slow. (Though it must be done at least once first.) If True, will
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
#        star_magnitudes = hickle.load(f, path='/star_magnitudes')
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

    ax1.set_ylim(bottom=-3, top=124)
    ax1.set_xlim(left=bin_dict['temp'][0], right=bin_dict['temp'][-1])
    ax2.set_xlim(left=bin_dict['mtl'][0], right=bin_dict['mtl'][-1])
    ax3.set_xlim(left=bin_dict['logg'][0], right=bin_dict['logg'][-1])

    ax1.set_xlabel(r'$T_\mathrm{eff}$ (K)', size=15)
    ax1.set_ylabel(r'$\sigma_\mathrm{**}$ (m/s)', size=15)
    ax2.set_xlabel('[Fe/H]', size=15)
    ax3.set_xlabel(r'$\log{g}\,(\mathrm{cm\,s}^{-2})$', size=15)

    ax1.xaxis.set_major_locator(ticker.FixedLocator([5472, 5772, 6072]))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=0.3))
    ax3.xaxis.set_major_locator(ticker.FixedLocator([4.24, 4.34, 4.44,
                                                     4.54]))

    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)

    for ax in (ax1, ax2, ax3):
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(labelsize=16)

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
            gravities = ma.masked_array(star_gravities)
            loggs = gravities[~m_seps.mask]

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
                    mask_array = ma.masked_outside(arrays_dict[name],
                                                   *bin_lims)
                    num_points = mask_array.count()
                    star_bins_dict[name].append(num_points)
                    vprint(f'{num_points} values in bin ({lower},{upper})')
                    if num_points < min_bin_size:
                        vprint('Skipping this bin!')
                        sigma_list.append(np.nan)
                        sigma_sys_list.append(np.nan)
                        continue
                    residuals_copy = residuals[~mask_array.mask]
                    errs_copy = err_array[~mask_array.mask].value

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
                                121),
                        textcoords='data',
                        verticalalignment='top', horizontalalignment='center',
                        fontsize=16, zorder=15)

    plot_path = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

    filename = plot_path /\
        f'Stellar_parameter_dependence_bin_{min_bin_size}.pdf'
    fig.savefig(str(filename), bbox_inches='tight', pad_inches=0.01)


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

    fig = plt.figure(figsize=(10, 4.6), tight_layout=True)
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
        ax.set_xlim(left=-59, right=59)
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

        dup_color1 = cmr.watermelon(0.15)
        dup_ecolor1 = cmr.watermelon(0.08)
        dup_color2 = cmr.watermelon(0.85)
        dup_ecolor2 = cmr.watermelon(0.92)

        ax_measured.errorbar(pair_diffs, pair_indices,
                             xerr=pair_errs,
                             color=dup_color1, capsize=2, capthick=2,
                             elinewidth=2,
                             ecolor=dup_ecolor1, markeredgecolor='Black',
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
                              color=dup_color2, capsize=2, capthick=2,
                              elinewidth=2,
                              ecolor=dup_ecolor2, markeredgecolor='Black',
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
        f'{star.name}_duplicate_pairs.pdf'
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
#    red_label = '_'.join((parts[1], parts[2]))

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
#        post_slice = slice(star.fiberSplitIndex, None)

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

    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig,
                  height_ratios=(1, 0.4), hspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.set_xlim(left=2870, right=3140)
    ax.set_ylim(bottom=-90, top=75)

    harps_pix_width = 0.829  # km/s
    for axis in (ax, ax_mean):
        axis.axhline(y=0, linestyle='--', color='DarkGray',
                     linewidth=2, zorder=1)
        # Denote the CCD sub-boundary in pixels.
        axis.axvline(x=boundary_pix, linestyle='--', color=cmr.guppy(0.8),
                     label='Blue crossing', linewidth=2.5, zorder=5)
        axis.axvline(boundary_pix - np.round(mean_sep_pre / harps_pix_width),
                     linestyle='-.', color=cmr.guppy(0.2),
                     label='Red crossing', linewidth=2.5, zorder=5)

    ax_mean.set_ylim(bottom=-8, top=8)
    ax.tick_params(labelbottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_mean.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_mean.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_ylabel('Model offset (m/s)')
    ax_mean.set_ylabel('Mean (m/s)')
    ax_mean.set_xlabel('Pixel number')

    # Plot the model offsets.
    ax.errorbar(pixels_pre, offsets_pre,
                linestyle='',
                markeredgecolor=None,
                marker='.', color=cmr.redshift(0.5),
                markersize=1.8,
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


def plot_vs_max_pair_blendedness():
    """Create a plot of the chi-squared values of each pair againsts it maximum
    blendedness.
    """
    plots_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')

#    filename = vcl.output_dir /\
#        'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
#    fit_results_dict = get_params_file(filename)
#    coeffs_dict = fit_results_dict['coeffs']
#    model_func = fit_results_dict['model_func']

    tqdm.write('Unpickling pairs list.')
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)
    vprint(f'Found {len(pairs_list)} pairs in the list.')

    labels = []
    max_blends = []

    tqdm.write('Getting pair max blendedness...')
    for pair in pairs_list:
        for order_num in pair.ordersToMeasureIn:
            labels.append('_'.join((pair.label, str(order_num))))
            max_blends.append(np.max(pair.blendTuple))

    tqdm.write('Reading data from stellar database file...')

    with h5py.File(db_file, mode='r') as f:

        pair_column_dict = hickle.load(f, path='/pair_column_index')

        star_names = hickle.load(f, path='/star_row_index')

    cols = list(pair_column_dict.values())

    results_pre = {0: [],
                   1: [],
                   2: [],
                   3: [],
                   4: [],
                   5: []}
    results_post = {0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: []}

    results_array = np.zeros((2, len(stars_used), len(labels)))
    errors_array = np.zeros((2, len(stars_used), len(labels)))

    for i, star_name in enumerate(star_names):

        star = Star(star_name, f'/Users/dberke/data_output/{star_name}')

        pre_slice = slice(None, star.fiberSplitIndex)
        post_slice = slice(star.fiberSplitIndex, None)

        for col in tqdm(cols):

            if star.hasObsPre:
                weighted_mean, err = get_weighted_mean(
                        star.pairModelOffsetsArray,
                        star.pairModelErrorsArray,
                        pre_slice, col)

                results_array[0, i, col] = weighted_mean
                errors_array[0, i, col] = err
            else:
                results_array[0, i, col] = np.nan
                errors_array[0, i, col] = np.nan

            if star.hasObsPost:
                weighted_mean, err = get_weighted_mean(
                        star.pairModelOffsetsArray,
                        star.pairModelErrorsArray,
                        post_slice, col)

                results_array[1, i, col] = weighted_mean
                errors_array[1, i, col] = err
            else:
                results_array[1, i, col] = np.nan
                errors_array[1, i, col] = np.nan

    for col in tqdm(cols):

        blendedness = max_blends[col]
        vprint(f'Max blendedness for {labels[col]} is'
               f' {blendedness}')

        chi_sq_pre = fit.calc_chi_squared_nu(
                remove_nans(results_array[0, :, col]),
                remove_nans(errors_array[0, :, col]), 1)
        chi_sq_post = fit.calc_chi_squared_nu(
                remove_nans(results_array[1, :, col]),
                remove_nans(errors_array[1, :, col]), 1)

        vprint(f'  chi^2 pre is {chi_sq_pre:.1f} for {labels[col]}'
               f' with max blendedness = {blendedness}')
        vprint(f'  chi^2 post is {chi_sq_post:.1f} for {labels[col]}'
               f' with max blendedness = {blendedness}')
        results_pre[blendedness].append(chi_sq_pre)
        results_post[blendedness].append(chi_sq_post)

    fig = plt.figure(figsize=(10, 3.4), tight_layout=True)
    gs = GridSpec(ncols=6, nrows=1, figure=fig, wspace=0)
    fig.supxlabel(r'Maximum blendedness', fontsize=14, y=0.07)
    fig.supylabel(r'$\chi^2_\nu$')

    bins = np.linspace(0, 50, 100)
    y_major_ticks = (1, 3, 5, 7)
    y_minor_ticks = (2, 4, 6)
    plot_bottom, plot_top = 0.25, 8.5

    for i, key in enumerate(results_pre.keys()):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            first_ax = ax
        ax.set_ylim(bottom=plot_bottom, top=plot_top)
        if i != 0:
            ax.tick_params(axis='y', which='both',
                           labelleft=False)
        ax.tick_params(axis='x', which='both', labelbottom=False,
                       bottom=False, top=False, labelsize=12)
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_major_ticks))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(y_minor_ticks))
        ax.annotate(f'{len(results_pre[key])}', (0.95, 0.95),
                    xycoords='axes fraction',
                    xytext=(0.96, 0.93),
                    textcoords='axes fraction',
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=14)
        ax.set_xlabel(r'$\mathrm{N}_\mathrm{max}=' + rf'{i}$')
        ax.hist(results_pre[key], bins=bins,
                orientation='horizontal',
                density=True, histtype='step', linewidth=2.2,
                color=cmr.torch(0.75),
                label='Pre')

        ax.hist(results_post[key], bins=bins,
                orientation='horizontal',
                density=True, histtype='step', linewidth=2.2,
                color=cmr.torch(0.3),
                label='Post')

    first_ax.legend(loc='upper left', fontsize=13, markerscale=0.7,
                    shadow=True)

    filename = plots_dir / 'Chi-squared_vs_max_blendedness.pdf'
    fig.savefig(str(filename), bbox_inches='tight', pad_inches=0.01)


def create_representative_blendedness_plots():
    """Create a plot showing examples of the various blendedness categories.

    """

    # Vesta observation with SNR = 226
    data_file = '/Users/dberke/Vesta/data/reduced/'\
        '2011-09-29/HARPS.2011-09-29T23:30:27.911_e2ds_A.fits'

    obs = HARPSFile2DScience(data_file)

    vesta = Star('Vesta', '/Users/dberke/data_output/Vesta')
    obs_index = vesta.od_index('2011-09-29T23:30:27.910')  # It's obs 0

    transitions = {'category0': ('4219.893V1_16',
                                 '5224.639Fe1_44',
                                 '6153.320Fe1_61'),
                   'category1': ('4589.484Cr2_28',
                                 '5523.980Fe1_49',
                                 '6131.832Ni1_60'),
                   'category2': ('4492.660Fe2_25',
                                 '5187.997Ni1_43',
                                 '6245.542Si1_62'),
                   'category3': ('4502.533Ti2_25',
                                 '5227.994Ti2_44',
                                 '6165.470Ca1_61'),
                   'category4': ('4295.006Fe1_18',
                                 '5146.101Cr1_42',
                                 '5596.208Fe1_51'),
                   'category5': ('4309.365Mn2_19',
                                 '5147.740Fe1_42',
                                 '5168.927Fe1_43')}

    # Create the plot.
    fig = plt.figure(figsize=(13, 4), tight_layout=True)
    axes = fig.subplots(1, 6, sharey=True, gridspec_kw={'wspace': 0})
    fig.supxlabel('Relative velocity (km/s)', size=16,
                  y=0.07)
    axes[0].set_ylabel('Normalised flux')

    colors = cmr.take_cmap_colors('cmr.rainforest', 3,
                                  cmap_range=(0.45, 0.85),
                                  return_fmt='hex')

    for i, category in enumerate(transitions.keys()):
        axes[i].annotate(f'{category[-1]}',
                         (0.78, 0.07), xycoords='axes fraction',
                         fontsize=24)
        axes[i].set_xlim(left=-18*u.km/u.s, right=18*u.km/u.s)
        axes[i].axvline(x=0, color='Gray', linestyle='--', alpha=1,
                        linewidth=1.5, zorder=2)
        axes[i].axhline(y=1, linestyle=':', color='Gray',
                        linewidth=1.8, zorder=1)
        axes[i].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        for j, t_label in enumerate(transitions[category]):
            vprint(f'Working on {t_label}')
            wavelength = float(t_label[:8]) * u.angstrom
            order_num = int(t_label[-2:])

            t_index = vesta.t_index(t_label)
            offset = vesta.fitOffsetsNormalizedArray[obs_index, t_index]
            meas_wavelength = vesta.fitMeansArray[obs_index, t_index]
            vprint(f'Shifting by {offset:.2f}')
            exp_wavelength = shift_wavelength(wavelength, offset)

            wavelength_data = obs.barycentricArray[order_num]
            flux_data = obs.photonFluxArray[order_num]
#            error_data = obs.errorArray[order_num]

            low_index = wavelength2index(exp_wavelength, wavelength_data) - 30
            high_index = wavelength2index(exp_wavelength, wavelength_data) + 20

            indices = [wavelength2velocity(meas_wavelength, i).to(u.km/u.s)
                       for i in wavelength_data[low_index:high_index]]

            fluxes = flux_data[low_index:high_index]
            flux_max = fluxes.max()

            axes[i].errorbar(indices,
                             fluxes / flux_max,
#                             yerr=error_data[low_index:high_index]/flux_max,
#                             barsabove=True,
#                             marker=markers[j], markersize=4,
                             marker='',
#                             markeredgecolor='Black',
#                             markeredgewidth=1.5,
                             color=colors[j], ecolor='Black',
                             drawstyle='steps-mid',
                             linestyle='-', linewidth=2.3,
                             zorder=j+5)

    plot_name = plots_dir / 'Blendedness_examples.pdf'
    fig.savefig(str(plot_name), bbox_inches='tight', pad_inches=0.01)


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
    ax_pre.set_ylabel('Mean normalised depth of pair')

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
    ax1.legend(loc='upper left', fontsize=16, shadow=True)

    outfile = plots_dir / 'Sigma_s2s_histogram_17_pairs.pdf'
    fig1.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)

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

    ax2.legend(loc='upper right', fontsize=16)

    outfile = plots_dir / 'Sigma_s2s_histogram_blend_2_pairs.pdf'
    fig2.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)

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
    vprint('For the Kolmogorov-Smirnov test:')
    vprint(f'KS statistic: {ks_value}')
    vprint(f'p-value: {p_value}')


def plot_solar_twins_results(star_postfix=''):
    """Plot results for 17 pairs with q-coefficients for solar twins"""

    def format_pair_label(pair_label):
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
        new_label1 = f"{t1[8:-1]}" + r"\," + r"\textsc{\lowercase{" +\
            f"{roman_numerals[t1[-1]]}" + r"}}" + r"\ " + f"{t1[:8]}"
        new_label2 = f"{t2[8:-1]}" + r"\," + r"\textsc{\lowercase{" +\
            f"{roman_numerals[t2[-1]]}" + r"}}" + r"\ " + f"{t2[:8]}"

        return {'ion1': new_label1, 'ion2': new_label2}

    roman_numerals = {'1': 'I', '2': 'II'}

    # Get labels of the 17 pairs on the shortlist.
    pairs_file = vcl.data_dir / '17_pairs.txt'
    pair_labels = np.loadtxt(pairs_file, dtype=str)

    # Get the 18 solar twins.
    stars = {star_name: Star(star_name + star_postfix,
                             vcl.output_dir / star_name)
             for star_name in sp1_stars}

    # Set out lists of star for the top and bottom panels.
    block1_stars = ('Vesta', 'HD76151', 'HD78429',
                    'HD140538', 'HD146233', 'HD157347')
    block2_stars = ('HD20782', 'HD19467', 'HD45184',
                    'HD45289', 'HD171665',)
    block3_stars = ('HD138573', 'HD183658', 'HD220507', 'HD222582')
    block4_stars = ('HD1835', 'HD30495', 'HD78660', )

    block1_width = 25
    block1_ticks = 15
    block2_width = 45
    block2_ticks = 30
    block3_width = 75
    block3_ticks = 50
    block4_width = 125
    block4_ticks = 75

    fig = plt.figure(figsize=(18, 10.5), tight_layout=True)
    gs = GridSpec(ncols=20, nrows=4, figure=fig, wspace=0,
                  height_ratios=(len(block1_stars),
                                 len(block2_stars),
                                 len(block3_stars),
                                 len(block4_stars)))
    # Set the "velocity" title to be below the figure.
    fig.supxlabel('Difference between pair velocity separation and model (m/s)',
                  fontsize=18)

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
                    **format_pair_label(label)),),
                                    fontdict={'rotation': 90,
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'})
        elif i in (0, 2, 4):
            ax_twin.xaxis.set_major_locator(ticker.FixedLocator((-11, 12)))
            ax_twin.set_xticklabels((f'Order: {str(order_num)}',
                                     '{ion1}\n{ion2}'.format(
                                             **format_pair_label(label)),),
                                    fontdict={'rotation': 90,
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'})
        elif i in (1, 3, 5):
            ax_twin.xaxis.set_major_locator(ticker.FixedLocator((2,)))
            ax_twin.set_xticklabels((f'Order: {str(order_num)}',),
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
    post_color = cmr.cosmic(0.55)

    # How significant to report outliers.
    sigma_significance = 3
    vprint(f'Looking for outliers beyond {sigma_significance} sigma')
    # Create lists to hold the significance values:
    pre_stat, pre_sys = [], []
    post_stat, post_sys = [], []
    for i, pair_label in enumerate(pair_labels):
        # Create lists to hold the values and errors:
        pre_values, post_values = [], []
        pre_err_stat, post_err_stat = [], []
        pre_err_sys, post_err_sys = [], []
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
                plot = True
                try:
                    value, error = weighted_mean_and_error(values, errors)
                except ZeroDivisionError:
                    # This indicates no value for a particular 'cell', so just
                    # plot something there to indicate that.
                    axes[(row, i)].plot(0, j-0.15, color='Black', marker='x',
                                        markersize=7, zorder=10)
                    plot = False
                if plot:
                    # Compute error with sigma_** included.
                    sigma_s2s = star.pairSysErrorsArray[0, pair_index]
                    full_error = np.sqrt(error**2 + sigma_s2s**2)
                    sig_stat = float((value / error).value)
                    sig_sys = float((value / full_error).value)
                    pre_stat.append(sig_stat)
                    pre_sys.append(sig_sys)
                    if abs(sig_sys) > sigma_significance:
                        vprint(f'{star.name}: {pair_label}:'
                               f' (Pre) {sig_sys:.2f}')
                    pre_values.append(value)
                    pre_err_stat.append(error)
                    pre_err_sys.append(full_error)
                    if (star.name == 'HD1835') and\
                            (pair_label == '4759.449Ti1_4760.600Ti1_32'):
                        vprint('For HD 1835, 4759.449Ti1_4760.600Ti1_32:')
                        vprint(f'Value: {value:.3f}, error: {full_error:.3f}')
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
                                            markeredgewidth=2,
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
                plot = True
                try:
                    value, error = weighted_mean_and_error(values, errors)
                except ZeroDivisionError:
                    axes[(row, i)].plot(0, j+0.15, color='Black', marker='x',
                                        markersize=7)
                    plot = False
                if plot:
                    sigma_s2s = star.pairSysErrorsArray[1, pair_index]
                    full_error = np.sqrt(error**2 + sigma_s2s**2)
                    sig_stat = float((value / error).value)
                    sig_sys = float((value / full_error).value)
                    post_stat.append(sig_stat)
                    post_sys.append(sig_sys)
                    if abs(sig_sys) > sigma_significance:
                        vprint(f'{star.name}: {pair_label}:'
                               f' (Post) {sig_sys:.2f}')
                    post_values.append(value)
                    post_err_stat.append(error)
                    post_err_sys.append(full_error)
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
        # Print some metrics for the pair.
        pre_val_arr = np.array(pre_values)
        pre_err_arr_stat = np.array(pre_err_stat)
        pre_err_arr_sys = np.array(pre_err_sys)
        post_val_arr = np.array(post_values)
        post_err_arr_stat = np.array(post_err_stat)
        post_err_arr_sys = np.array(post_err_sys)
        wm_value_pre, error_pre = weighted_mean_and_error(
                pre_val_arr, pre_err_arr_sys)
        wm_value_post, error_post = weighted_mean_and_error(
                post_val_arr, post_err_arr_sys)
        chi_2_pre_stat = fit.calc_chi_squared_nu(
                pre_val_arr, pre_err_arr_stat, 1)
        chi_2_pre_sys = fit.calc_chi_squared_nu(
                pre_val_arr, pre_err_arr_sys, 1)
        chi_2_post_stat = fit.calc_chi_squared_nu(
                post_val_arr, post_err_arr_stat, 1)
        chi_2_post_sys = fit.calc_chi_squared_nu(
                post_val_arr, post_err_arr_sys, 1)
        vprint(f'For {pair_label}:')
        vprint('    Pre : Weighted mean:'
               f' {wm_value_pre:.2f} ± {error_pre:.2f} m/s')
        vprint(f'    Pre : chi^2: {chi_2_pre_stat:.2f}, {chi_2_pre_sys:.2f}')
        vprint(f'    Pre : mean error: {np.mean(pre_err_arr_sys):.2f} m/s')
        vprint('    Post: Weighted mean:'
               f' {wm_value_post:.2f} ± {error_post:.2f} m/s')
        vprint(f'    Post: chi^2: {chi_2_post_stat:.2f}, {chi_2_post_sys:.2f}')
        vprint(f'    Post: mean error: {np.mean(post_err_arr_sys):.2f} m/s')

    # Create the histogram plots for the pair.

    fig_hist = plt.figure(figsize=(5.5, 5.5), tight_layout=True)
    bins = np.linspace(-3, 3, num=25)
    ax_hist = fig_hist.add_subplot(1, 1, 1)
    ax_hist.set_xlabel(r'Significance ($\sigma$)')
    ax_hist.set_ylabel('N')
    ax_hist.xaxis.set_major_locator(ticker.FixedLocator((-3, -2, -1,
                                                         0, 1, 2, 3)))
    ax_hist.xaxis.set_minor_locator(ticker.FixedLocator(bins))
    ax_hist.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Add the pre and post distributions together here.
    pre_stat.extend(post_stat)
    pre_sys.extend(post_sys)

    one_sigma, two_sigma = 0, 0
    for x in pre_sys:
        y = abs(x)
        if y < 1:
            one_sigma += 1
            two_sigma += 1
        elif y < 2:
            two_sigma += 1

    vprint(f'{one_sigma/len(pre_sys):.1%} of values within 1 sigma.')
    vprint(f'{two_sigma/len(pre_sys):.1%} of values within 2 sigma.')

    ax_hist.hist(pre_stat, color='Gray', histtype='step',
                 bins=bins, linewidth=1.8, label='Stat. only')
    ax_hist.hist(pre_sys, color='Black', histtype='step',
                 bins=bins, linewidth=2.6, label='Stat. + Sys.')

    ax_hist.legend(loc='upper right', fontsize=16,
                   shadow=True)

    outfile = plots_dir / f'Pair_offsets_17_pairs{star_postfix}.pdf'
    fig.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)

    histfile = plots_dir / f'Pair_offsets_histograms{star_postfix}.pdf'
    fig_hist.savefig(str(histfile), bbox_inches='tight', pad_inches=0.01)

    # Create an excerpt of a single column.
    fig_ex = plt.figure(figsize=(5, 6), tight_layout=True)
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
    ax_ex.set_xlabel('Pair model offset (m/s)', size=15)

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
    tqdm.write(f'Using pair {pair_label} for excerpt')
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

    outfile = plots_dir /\
        f'Pair_offsets_17_pairs_excerpt_{pair_label.replace(".", "_")}.pdf'
    fig_ex.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)


def create_cosmic_ray_plots():
    """Create plots showing the effect of a cosmic ray hit."""

    hd45184 = Star('HD45184', '/Users/dberke/data_output/HD45184')

    saved_fits_file = vcl.output_dir /\
        'HD45184/HARPS.2012-05-10T23:31:23.210_e2ds_A/pickles_int/'\
        'HARPS.2012-05-10T23:31:23.210_e2ds_A_gaussian_fits.lzma'

    # Transition of interest is 4658.285Fe2_29, index 108
    t_label = '4658.285Fe2_29'
    t_index = hd45184.t_index(t_label)
    # Observation date is 2012-05-10T23:31:23.209, index 77
#    o_index = hd45184._obs_date_bidict['2012-05-10T23:31:23.209']

    # Find the weighted mean and error of this transition offset in this star.
    offsets, mask = remove_nans(hd45184.fitOffsetsArray[:, 108],
                                return_mask=True)
    errors = hd45184.fitErrorsArray[:, 108][mask]
    wmean, eotwm = weighted_mean_and_error(offsets, errors)
    vprint(f'Weighted mean of offsets: {wmean:.3f}')
    vprint(f'Error on the weighted mean: {eotwm:.3f}')
    vprint(f'RMS of offsets: {np.std(offsets):.3f}')

    obs_file = vcl.data_dir /\
        'spectra/HD45184/HARPS.2012-05-10T23:31:23.210_e2ds_A.fits'

    # Get the right data from the affected observation.
    obs = HARPSFile2DScience(obs_file)
    radial_velocity = obs.radialVelocity
    vprint(f'Working on {t_label}.')
    wavelength = float(t_label[:8]) * u.angstrom
    exp_wavelength = shift_wavelength(wavelength, radial_velocity)
    order_num = int(t_label[-2:])

    wavelength_data = obs.barycentricArray[order_num]
    flux_data = obs.photonFluxArray[order_num]
    error_data = obs.errorArray[order_num]

    mid_index = wavelength2index(exp_wavelength, wavelength_data)
    low_index = mid_index - 12
    high_index = mid_index + 12

    # Get fit information from the saved fit.
    with lzma.open(saved_fits_file, 'rb') as f:
        fits_list = pickle.loads(f.read())
    model_fit = fits_list[t_index]
#    print(model_fit.label)

    fig = plt.figure(figsize=(6, 6), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=1, figure=fig,
                  height_ratios=(2.8, 1))

    ax_plot = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0])

    ax_plot.set_ylabel('Flux (photons)')
    ax_plot.set_xlabel(r'Wavelength (\AA)')
#    ax_plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>5.1e}'))
    ax_plot.set_ylim(bottom=flux_data[low_index:high_index].min() * 0.96,
                     top=flux_data[low_index:high_index].max() * 1.02)
    ax_plot.set_xlim(left=wavelength_data[low_index],
                     right=wavelength_data[high_index-1])

    ax_hist.set_xlim(left=-160, right=450)
    ax_hist.set_ylabel('N')
    ax_hist.set_xlabel('Transition offsets in HD 45184 (m/s)')
    bins = np.linspace(-150, 450, 57)

    # Plot the histogram:
    ax_hist.hist(hd45184.fitOffsetsArray[:, 108].value, bins=bins,
                 histtype='step', color='Black', linewidth=1.5,
                 zorder=5)
    ax_hist.axvline(x=wmean, color='Gray', linestyle='--',
                    zorder=6, linewidth=2)
    ax_hist.axvline(x=model_fit.velocityOffset.to(u.m/u.s),
                    color='Gray', linestyle=':', linewidth=2,
                    zorder=6)

    # Plot the spectrum:
    ax_plot.errorbar(wavelength_data[low_index:high_index],
                     flux_data[low_index:high_index],
                     yerr=error_data[low_index:high_index],
                     color=cmr.torch(0.84), linestyle='-', marker='',
                     ecolor='Gray', linewidth=4,
                     barsabove=True, capsize=4,
                     capthick=1.5, alpha=0.8, drawstyle='default',
                     zorder=5)

    # Plot the spectrum line a bit darker.
    ax_plot.errorbar(wavelength_data[mid_index-3:mid_index+4],
                     flux_data[mid_index-3:mid_index+4],
                     color=cmr.torch(0.8), drawstyle='default',
                     linestyle='-', marker='', linewidth=5,
                     zorder=8)

    # Plot the data points used in fitting with different color.
    ax_plot.errorbar(wavelength_data[mid_index-3:mid_index+4],
                     flux_data[mid_index-3:mid_index+4],
                     yerr=error_data[mid_index-3:mid_index+4],
                     color=cmr.torch(0.8), markeredgecolor='Black',
                     markersize=7,
                     linestyle='', marker='o', linewidth=5,
                     ecolor='Black', barsabove=False,
                     capsize=4, capthick=1.4,
                     zorder=10)

    # Create x-values for the fit.
    x = np.linspace(wavelength_data[low_index].value - 1,
                    wavelength_data[high_index].value + 1, 1000)
    # Plot the fit:
    ax_plot.plot(x, fit.integrated_gaussian(x, *model_fit.popt),
                 color=cmr.torch(0.3), alpha=0.9, linestyle='-',
                 zorder=9, linewidth=3)

    ax_plot.axvline(model_fit.mean.to(u.angstrom),
                    color='Gray', linestyle=':', linewidth=2,
                    zorder=7, label='Measured offset')
    ax_plot.axvline(shift_wavelength(model_fit.mean.to(u.angstrom),
                                     -model_fit.velocityOffset),
                    color='Gray', linestyle='--', linewidth=2,
                    zorder=6, label='Weighted mean\nof offsets')

    ax_plot.legend(loc='lower left',
                   fontsize=14, shadow=True)

    ax_plot.annotate('Affected\npixel',
                     (wavelength_data[mid_index-3]-0.003*u.angstrom,
                      flux_data[mid_index-3]+5),
                     xytext=(4658.14, 25500),
                     arrowprops={'arrowstyle': '-'},
                     fontsize=14, horizontalalignment='center',
                     zorder=6)
    ax_plot.annotate('Possibly\naffected',
                     (wavelength_data[mid_index-2]-0.003*u.angstrom,
                      flux_data[mid_index-2]+5),
                     xytext=(4658.14, 23200),
                     arrowprops={'arrowstyle': '-'},
                     fontsize=14, horizontalalignment='center',
                     verticalalignment='bottom',
                     zorder=6)

    outfile = str(plots_dir / 'Cosmic_ray_effect.pdf')
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(outfile.replace('.pdf', '.png'),
                bbox_inches='tight', pad_inches=0.01)


def create_feature_fitting_example_plot():
    """Create a plot showing the two features in a pair with the automatic fit.

    """

    # Vesta observation with SNR = 226
    data_file = '/Users/dberke/Vesta/data/reduced/'\
        '2011-09-29/HARPS.2011-09-29T23:30:27.911_e2ds_A.fits'

    obs = HARPSFile2DScience(data_file)

    # Get the fit.
    saved_fits_file = vcl.output_dir /\
        'Vesta/HARPS.2011-09-29T23:30:27.911_e2ds_A/pickles_int/'\
        'HARPS.2011-09-29T23:30:27.911_e2ds_A_gaussian_fits.lzma'
    with lzma.open(saved_fits_file, 'rb') as f:
        fits_list = pickle.loads(f.read())

    vesta = Star('Vesta', '/Users/dberke/data_output/Vesta')
    obs_index = vesta.od_index('2011-09-29T23:30:27.910')  # It's obs 0

    transitions = ('6138.313Fe1_60', '6139.390Fe1_60')

    fig = plt.figure(figsize=(8.5, 3.3), tight_layout=True)
    axes = fig.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0})
    axes[1].sharey(axes[0])
    axes[0].set_ylim(bottom=0.27, top=1.1)
    plot_width = 14.5 * u.km/u.s
    for ax in axes:
        ax.set_xlim(left=-plot_width, right=plot_width)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        x_min = -2.9
        x_max = 2.5

        # Set up lines for plotting residuals.
        vertical_offset = 0.95
        # Create the ±1 sigma lines:
        ax.hlines((vertical_offset - 0.05, vertical_offset + 0.05),
                  xmin=x_min, xmax=x_max, linestyle='-',
                  color=cmr.neutral(0.4), zorder=3)
        # Create the offset "0" line:
        ax.hlines(vertical_offset, xmin=x_min, xmax=x_max, linestyle='--',
                  color=cmr.neutral(0.5), zorder=4)
        ax.axvspan(xmin=x_min, xmax=x_max, alpha=0.8,
                   color=cmr.neutral(0.9),
                   zorder=2)

    fig.supxlabel('Relative velocity (km/s)', size=14,
                  y=0.08)
    axes[0].set_ylabel('Normalised flux', size=14)

    for i, t_label in enumerate(transitions):
#        wavelength = float(t_label[:8]) * u.angstrom
        order_num = int(t_label[-2:])
        t_index = vesta.t_index(t_label)
        # Get the fit for this transition:
        model_fit = fits_list[t_index]
#        offset = vesta.fitOffsetsNormalizedArray[obs_index, t_index]
        meas_wavelength = vesta.fitMeansArray[obs_index, t_index]
#        exp_wavelength = shift_wavelength(wavelength,
#                                          -model_fit.velocityOffset)

        wavelength_data = obs.barycentricArray[order_num]
        flux_data = obs.photonFluxArray[order_num]
        error_data = obs.errorArray[order_num]
        pixel_lower_data = obs.pixelLowerArray[order_num]
        pixel_upper_data = obs.pixelUpperArray[order_num]

        mid_index = wavelength2index(meas_wavelength, wavelength_data)
        low_index = mid_index - 37
        high_index = mid_index + 37

        rel_velocities = [wavelength2velocity(meas_wavelength, wl).to(u.km/u.s)
                          for wl in wavelength_data[low_index:high_index]]

        mid_vel_index = flux_data[low_index:high_index].argmin()

        fluxes = flux_data[low_index:high_index]
        errors = error_data[low_index:high_index]
        flux_max = fluxes.max()

        # Plot the entire spectrum.
        axes[i].errorbar(rel_velocities,
                         fluxes/flux_max,
#                         yerr=error_data[low_index:high_index]/flux_max,
#                         barsabove=True,
#                         marker='o', markersize=4,
#                         markeredgecolor='Black',
#                         markerfacecolor=cmr.torch(0.95),
#                         markeredgewidth=1,
                         drawstyle='steps-mid',
                         color=cmr.torch(0.8), ecolor='Black',
                         linestyle='-', linewidth=1.8, alpha=1,
                         zorder=5)
        # Plot just the central sevan pixels more enhanced.
        axes[i].errorbar(rel_velocities[mid_vel_index-3:mid_vel_index+4],
                         fluxes[mid_vel_index-3:mid_vel_index+4]/flux_max,
#                         marker='o', markersize=6,
#                         markeredgecolor=cmr.torch(0.65),
#                         markerfacecolor=cmr.torch(1.),
#                         markeredgewidth=1,
                         drawstyle='steps-mid',
                         linestyle='-', color=cmr.torch(0.6),
                         linewidth=2.7, zorder=10)

        # x-values for plotting fit:
        x = np.linspace(wavelength_data[mid_index-3],
                        wavelength_data[mid_index+3], 30)
        x_vel = wavelength2velocity(meas_wavelength, x).to(u.km/u.s)

        x_full = np.linspace(wavelength_data[mid_index-11],
                             wavelength_data[mid_index+11], 50)
        x_vel_full = wavelength2velocity(meas_wavelength, x_full).to(u.km/u.s)

#        vprint(x_full)
#        vprint(x_vel_full)

        # Plot the fit, with an enhanced section in the 7 central pixels.
        axes[i].plot(x_vel_full,
                     fit.gaussian(x_full.value, *model_fit.popt)/flux_max,
                     color=cmr.neutral(0.6), alpha=1, linestyle='--',
                     linewidth=1.5, zorder=15)
        # Here's the enhanced part.
        axes[i].plot(x_vel,
                     fit.gaussian(x.value, *model_fit.popt)/flux_max,
                     color=cmr.neutral(0.1), alpha=1, linestyle='-',
                     linewidth=2., zorder=15)

        # Plot a line at 0 velocity.
        axes[i].axvline(x=0*u.km/u.s, color=cmr.neutral(0.7), linestyle='-',
                        zorder=7)

        # Add annotations of transition and chi-squared values.
        axes[i].annotate(format_transition_label(transitions[i]),
                         (0.05, 0.06), xycoords='axes fraction', size=15)
        axes[i].annotate(r'$\chi^2_\nu:$' + f' {model_fit.chiSquaredNu:.2f}',
                         (0.95, 0.06), xycoords='axes fraction', size=15,
                         horizontalalignment='right')

        # Plot the residuals.
        x_values = wavelength_data[mid_index-3:mid_index+4]
        x_low_values = pixel_lower_data[mid_index-3:mid_index+4]
        x_high_values = pixel_upper_data[mid_index-3:mid_index+4]
        flux_values = fluxes[mid_vel_index-3:mid_vel_index+4]
        error_values = errors[mid_vel_index-3:mid_vel_index+4]
        pixel_values = [p for p in zip(x_low_values.value,
                                       x_high_values.value)]
        vprint(pixel_values)
        vprint(model_fit.popt)
        residuals = []
        for flux, pixel_edge in zip(flux_values, pixel_values):
            residuals.append(flux - fit.integrated_gaussian(
                    pixel_edge, *model_fit.popt))
#        residuals = [flux_values - fit.integrated_gaussian(
#                pixel_values, *model_fit.popt)]
        significances = np.array(residuals) / error_values
        vprint(f'For {t_label} the significances are:')
        vprint(significances)
        vprint('Residuals:')
        vprint(residuals)
        vprint('Errors:')
        vprint(error_values)
#        print('--------')
#        print(significances/20+1)

        # Scale the significances to display on the plot.
        axes[i].plot(rel_velocities[mid_vel_index-3:mid_vel_index+4],
                     (significances/20)+vertical_offset,
                     linestyle='', marker='D',
                     markersize=4,
                     markerfacecolor=cmr.torch(0.95),
                     markeredgecolor=cmr.torch(0.25),
                     markeredgewidth=1.3, zorder=13)

    outfile = plots_dir / 'Fitting_example_plot.pdf'
    fig.savefig(str(outfile), bbox_inches='tight', pad_inches=0.01)


def create_transitions_table():
    """
    Create tables of pairs and transitions.
    """

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

    vprint(f'Found {len(transitions_list)} transitions.')
    low_blend_transitions = [transition for transition in transitions_list
                             if transition.blendedness < 3]
    vprint(f'Found {len(low_blend_transitions)} transitions'
           ' with blendedness < 3.')

    if args.verbose:
        for i, transition in enumerate(low_blend_transitions):
            pairs_found_in = []
            for p, pair in enumerate(pairs_list):
                if transition in pair:
                    pairs_found_in.append(p)

            vprint(f'{transition.label}: {pairs_found_in}')

    transition_headers = [r'$\lambda^a$',
                          r'$\omega^b$',
                          'Ion$^c$',
                          r'$E^d$',
                          'Configuration$^e$',
                          '$J^f$',
                          r'$E^g$',
                          'Configuration$^h$',
                          '$J^i$',
                          'Orders$^j$']

    n = 5
    fraction = ceil(len(low_blend_transitions) / n)

#        slices = (slice(0, fraction), slice(fraction, 2 * fraction),
#                  slice(2 * fraction, None))
    slices = [slice(m * fraction, (m+1) * fraction) for m in range(n)]
    for i, s in enumerate(slices):
        transitions_formatted_list = []
        transitions_table_file = tables_dir / f'transitions_table_{i}.txt'
        for transition in tqdm(low_blend_transitions[s]):
            line = [f'{transition.wavelength.to(u.angstrom).value:.3f}',
                    f'{transition.wavenumber.value:.3f}',
                    ''.join((transition.atomicSymbol,
                             r'\,\textsc{',
                             roman_numerals[
                                 transition.ionizationState].lower(), '}')),
                    transition.lowerEnergy.value,
                    transition.lowerOrbital.replace('.', ''),
                    str(Fraction(transition.lowerJ)),
                    transition.higherEnergy.value,
                    transition.higherOrbital.replace('.', ''),
                    str(Fraction(transition.higherJ)),
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


def get_program_ids():
    """
    Return a list of program IDs from stars used.
    """

    star_files = [d for d in
                  glob('/Volumes/External Storage/HARPS/*/*/*/*/*e2ds_A.fits')]
#    print(len(star_files))
#    print(star_files[:10])

    id_all_dict = {}
    id_sp1_dict = {}
    odd_ids_dict = {'Udry': [],
                    'Pepe': []}
    odd_ids = set(['Udry', 'Pepe'])

    for file in tqdm(star_files[:]):
        parts = file.split('/')
#        print(parts)
        # Check if it's a star that we used:
        if not parts[4] in stars_used:
            continue
        # Check if the date file has an underscore appended, for files we
        # didn't use.
        if parts[7][0] == '_':
            print(f'Found a bad date: {parts[4]}: {parts[7]}')
            continue
        # If it's a file that doesn't even have wavelength arrays, continue:
        try:
            obs = HARPSFile2DScience(file)
        except (NewCoefficientsNotFoundError,
                BlazeFileNotFoundError):
            continue
        # Assume it's a star we use at this point.
        prog_id = obs.getHeaderCard('HIERARCH ESO OBS PROG ID')
#        print(f'PROG ID is {prog_id}')
        try:
            id_all_dict[prog_id] += 1
        except KeyError:
            id_all_dict[prog_id] = 1

        if prog_id in odd_ids:
            odd_ids_dict[prog_id].append(file)

        if parts[4] in sp1_stars:
            try:
                id_sp1_dict[prog_id] += 1
            except KeyError:
                id_sp1_dict[prog_id] = 1

    print('For all stars:')
    for key in id_all_dict.keys():
        print(key, sep=', ')
    for key in id_all_dict.keys():
        print(f'{key}: {id_all_dict[key]}')
    print(f'{len(id_all_dict.keys())} total program IDs.')
    print(f'Total observations: {sum([v for v in id_all_dict.values()])}')

    print('For solar twins:')
    print(id_sp1_dict)
    for key in id_sp1_dict.keys():
        print(f'{key}: {id_sp1_dict[key]}')
    print(f'{len(id_sp1_dict.keys())} total program IDs.')
    print(f'Total observations: {sum([v for v in id_sp1_dict.values()])}')

    for file in odd_ids_dict['Pepe']:
        print(file)
    print(odd_ids_dict['Udry'])


def create_fit_info_tables():
    """
    Create a table of fit coefficients and sigma_** for pairs.
    """

    def parse_label(label, ASCII=False):

        p1, p2, order = label.split('_')

        wv1, el1, ion1 = p1[:8], p1[8:-1], p1[-1]
        wv2, el2, ion2 = p2[:8], p2[8:-1], p2[-1]

        if ASCII:
            return p1, p2, order

        label1 = ''.join(
                [el1, r'\,\textsc{',
                 roman_numerals[int(ion1)], r'}\,', wv1])
        label2 = ''.join(
                [el2, r'\,\textsc{',
                 roman_numerals[int(ion2)], r'}\,', wv2])

        return label1, label2, order

    def format_sci_notation(matchobj):

        if matchobj.group(1) is not None:
            return str(matchobj.group(1)) +\
                r'\num{' + str(matchobj.group(2)) + '}'
        else:
            return r'\num{' + str(matchobj.group(2)) + '}'

    tables_dir = output_dir / 'tables'

    tqdm.write('Unpickling pairs list.')
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)
    vprint(f'Found {len(pairs_list)} pairs in the list.')

    tqdm.write('Loading fitting results file...')
    filename = vcl.output_dir /\
        'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
    fit_results_dict = get_params_file(filename)
    coeffs_dict = fit_results_dict['coeffs']
    sigma_sys_dict = fit_results_dict['sigmas_sys']

    pair_headers = ['', 'Pair label', 'Ord.', r'\sigsys', r'$a$',
                    r'$b_1$', r'$c_1$', r'$d_1$',
                    r'$b_2$', r'$c_2$', r'$d_2$']

    good_pairs = []
    for pair in tqdm(pairs_list):
        if not all(n < 3 for n in pair.blendTuple):
            vprint(f'Passing pair with {pair.blendTuple}.')
            continue
        good_pairs.append(pair)
    print(f'Found {len(good_pairs)} pairs with max blendednes <= 2.')

    total_instances = 0
    for pair in good_pairs:
        total_instances += len(pair.ordersToMeasureIn)
    print(f'Total of {total_instances} instances.')

    for era in ('pre', 'post'):
        # Max rows per table.
        max_rows = 26
        table_num = 1

        tqdm.write(f'Collecting information for each pair for {era}...')
        table_lines = []
        for pair in good_pairs:
            for order_num in pair.ordersToMeasureIn:
                label = '_'.join([pair.label, str(order_num)])
                vprint(20 * '-')
                vprint(f'Analyzing {label}...')
                dict_label = '_'.join([label, era])

                line = [*parse_label(label), sigma_sys_dict[dict_label]]
                line.extend(coeffs_dict[dict_label])
                table_lines.append(line)

        # Do some mumbo-jumbo to figure out the number of tables that will be
        # made.
        num_pages = len(table_lines) // max_rows + 2

        page_lines = []
        make_table = False
        breakout = False
        while (len(page_lines) <= max_rows):
            try:
                page_lines.append(table_lines.pop())
            except IndexError:
                make_table = True
                breakout = True

            if len(page_lines) == max_rows:
                make_table = True

            if make_table:

                pairs_table_file = tables_dir /\
                    f'pairs_table_{era}_{num_pages-table_num}.txt'
                table_num += 1

                pairs_output = tabulate(
                        reversed(page_lines),
                        headers=pair_headers,
                        tablefmt='latex_raw',
                        floatfmt=('', '', '', '.2f',
                                  '.2f', '.2f', '.2f',
                                  '.2f', '.2e', '.2f', '.2f')).replace(' -',
                                                                       ' $-$')

                # This matches numbers in scientific notation (minus sign
                # optional) and wraps them in a \num{} command.
                pairs_output = re.sub(r"(\$\-\$)?(\d\.\d{2}e\-?\d{2})",
                                      format_sci_notation, pairs_output)
                if pairs_table_file.exists():
                    os.unlink(pairs_table_file)
                with open(pairs_table_file, 'w') as f:
                    f.write(pairs_output)

                page_lines = []

                make_table = False
                if breakout:
                    break

    # Now make the ASCII version for the paper.
    pair_headers = ['Transition 1', 'Transition 2', 'Order', 'sigsys', 'a',
                    'b_1', 'c_1', 'd_1',
                    'b_2', 'c_2', 'd_2']

    for era in ('pre', 'post'):
        tqdm.write(f'Collecting information for each pair for {era}...')
        table_lines = []
        for pair in tqdm(pairs_list):
            for order_num in pair.ordersToMeasureIn:
                label = '_'.join([pair.label, str(order_num)])
                vprint(20 * '-')
                vprint(f'Analyzing {label}...')
                dict_label = '_'.join([label, era])

                line = [*parse_label(label, ASCII=True),
                        u.unyt_quantity(sigma_sys_dict[dict_label]).value]
                line.extend(coeffs_dict[dict_label])
                table_lines.append(line)

        pairs_coeffs_file = tables_dir / f'pair_coefficients_table_{era}.csv'
        if pairs_coeffs_file.exists():
            os.unlink(pairs_coeffs_file)
        with open(pairs_coeffs_file, 'w') as csvfile:
            tablewriter = csv.writer(csvfile, delimiter=',')
            tablewriter.writerow(pair_headers)
            tablewriter.writerows(table_lines)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create all the necessary'
                                     ' figures and tables for my two papers'
                                     ' and thesis.')

    parser.add_argument('--tables', action='store_true',
                        help='Save out tables in LaTeX format to text files.')
    parser.add_argument('--figures', action='store_true',
                        help='Create and save plots and figures.')

    parser.add_argument('--star-postfix', action='store', type=str,
                        default='',
                        help='The exact postfix (with leading underscore)'
                        ' to append to the star names.')
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

#        create_transitions_table()

#        get_program_ids()

        create_fit_info_tables()

    if args.figures:
        hd146233 = Star('HD146233', '/Users/dberke/data_output/HD146233')

#        create_HR_diagram_plot()
#
#        create_example_pair_sep_plots()
#
#        create_sigma_sys_hist()
#
#        create_transition_density_plot()
#
#        create_parameter_dependence_plot(use_cached=True, min_bin_size=5)
#
#        plot_duplicate_pairs(Star('HD146233',
#                                  '/Users/dberke/data_output/HD146233'))
#
#        create_radial_velocity_plot()
#
#        plot_vs_pair_blendedness(Star('HD146233',
#                                      '/Users/dberke/data_output/HD146233'))
#        plot_vs_max_pair_blendedness()
#
#        create_representative_blendedness_plots()
#
#        plot_pair_depth_differences(Star('HD134060',
#                                    '/Users/dberke/data_output/HD134060'))
#
#        create_sigma_s2s_histogram()
#
        plot_solar_twins_results(args.star_postfix)
#
#        create_cosmic_ray_plots()
#
#        create_feature_fitting_example_plot()
