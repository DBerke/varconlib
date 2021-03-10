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
from tabulate import tabulate
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting as fit
from varconlib.transition_line import roman_numerals


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


parser = argparse.ArgumentParser(description='Create all the necessary figures'
                                 ' and tables for my two papers and thesis.')

parser.add_argument('--tables', action='store_true',
                    help='Save out tables in LaTeX format to text files.')
parser.add_argument('--figures', action='store_true',
                    help='Create and save plots and figures.')

parser.add_argument('-v', '--verbose', action='store_true',
                    help="Print out more information about the script's"
                    " output.")

args = parser.parse_args()

vprint = vcl.verbose_print(args.verbose)

output_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables')
if not output_dir.exists():
    os.mkdir(output_dir)

db_file = vcl.databases_dir / 'stellar_db_uncorrected.hdf5'

sp1_stars = ('HD138573', 'HD140538', 'HD146233', 'HD157347', 'HD171665',
             'HD1835', 'HD183658', 'HD19467', 'HD20782', 'HD220507' 'HD222582',
             'HD30495', 'HD45184', 'HD45289', 'HD76151', 'HD78429', 'HD78660',
             'Vesta')

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
                             roman_numerals[transition.ionizationState].lower(),
                             '}')),
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
                                      floatfmt=('.3f', '.3f', '', '.3f', '', '',
                                                '.3f' '', '', ''))

        if transitions_table_file.exists():
            os.unlink(transitions_table_file)
        with open(transitions_table_file, 'w') as f:
            f.write(transitions_output)

if args.figures:
    create_example_pair_sep_plots()

    # create_sigma_sys_hist()
