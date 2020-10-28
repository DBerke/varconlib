#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:38:30 2020

@author: dberke

A script to plot the per-star pair-wise velocity separations for each pair of
tansitions by various parameters.

"""

import argparse
import csv
from inspect import signature
import os
from pathlib import Path
import pickle
import sys

from astropy.coordinates import SkyCoord
import astropy.units as units
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from numpy import cos, sin
import pandas as pd
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.fitting import (calc_chi_squared_nu, constant_model,
                               find_sys_scatter)
from varconlib.miscellaneous import remove_nans, get_params_file
from varconlib.star import Star
from varconlib.transition_line import roman_numerals

import varconlib.fitting

params_dict = {'temperature': 'Teff (K)',
               'metallicity': '[Fe/H]',
               'logg': 'log(g)'}


types_dict = {'#star_name': str,
              'delta(v)_pair (m/s)': np.float,
              'err_stat_pair (m/s)': np.float,
              'err_sys_pair (m/s)': np.float,
              'transition1 (m/s)': np.float,
              't_stat_err1 (m/s)': np.float,
              't_sys_err1 (m/s)': np.float,
              'chi^2_nu1': np.float,
              'transition2 (m/s)': np.float,
              't_stat_err2 (m/s)': np.float,
              't_sys_err2 (m/s)': np.float,
              'chi^2_nu2': np.float}


plot_axis_labels = {'temperature': r'$\mathrm{T}_\mathrm{eff}\,$(K)',
                    'metallicity': r'$\mathrm{[Fe/H]}$',
                    'logg': r'$\log(g),\mathrm{cm\,s}^{-2}$'}


def read_csv_file(pair_label, csv_dir, era):
    """
    Import data on a pair from a CSV file.

    Parameters
    ----------
    pair_label : str
        The label of a pair for which the data is to be read.
    csv_dir : `pathlib.Path`
        The path to the main directory where the data files are kept.
    era : str, ['pre', 'post']
        A string denoting whether to read the data for the pre- or post-fiber
        change era.

    Returns
    -------
    None.

    """

    infile = csv_dir / f'{era}/{pair_label}_pair_separations_{era}.csv'
    return pd.read_csv(infile)


def parameter_plot(parameter, passed_ax):
    """
    Plot pairs for a given parameter upon a given axis.

    Parameters
    ----------
    parameter : str, ('temperature', 'metallicity', 'logg')
        The parameter to plot against
    passed_ax : `matplotlib.axis.Axes`
        An `Axes` object on which to plot the values.

    Returns
    -------
    None.

    """


def plot_vs(parameter):
    """
    Plot pair-wise velocity separations as a function of the given parameter.

    Parameters
    ----------
    parameter : str
        The name of the parameter to plot against.

    Returns
    -------
    None.

    """

    tqdm.write('Writing out data for each pair.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir / f'pair_result_plots/{parameter}'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)

        ax_pre.set_xlabel(f'${plot_axis_labels[parameter]}$')
        ax_post.set_xlabel(f'${plot_axis_labels[parameter]}$')
        ax_pre.set_ylabel(r'$\Delta V$ (pair, m/s)')
        ax_post.set_ylabel(r'$\Delta V$ (pair, m/s)')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)

        ax_pre.errorbar(star_data[params_dict[parameter]],
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(star_data[params_dict[parameter]],
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


def plot_distance():
    """Plot pair separations as a function of distance from the Sun."""

    tqdm.write('Making plots for each pair as a function of heliocentric'
               ' distance.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir /\
                f'pair_result_plots/heliocentric_distance'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)
        ax_pre.set_xlim(left=-1, right=54)
        ax_pre.set_xlabel('Heliocentric distance (pc)')
        ax_post.set_xlabel('Heliocentric distance (pc)')
        ax_pre.set_ylabel(r'$\Delta v$ (m/s, pre)')
        ax_post.set_ylabel(r'$\Delta v$ (m/s, post)')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.axhline(0, color='Black')
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)

        ax_pre.errorbar(star_data['distance (pc)'],
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(star_data['distance (pc)'],
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


def plot_galactic_distance():
    """Plot pair separations as a function of distance from the Galactic center.


    Returns
    -------
    None

    """

    RA = star_data['RA'][:-1]
    DEC = star_data['DEC'][:-1]
    dist = star_data['distance (pc)'][:-1]

    coordinates = SkyCoord(ra=RA, dec=DEC, distance=dist,
                           unit=(units.hourangle, units.degree,
                                 units.pc))
    # for c in coordinates:
    #     print(c.galactocentric.x, c.galactocentric.y)
    # exit()

    distances = [np.sqrt(c.galactocentric.x**2 + c.galactocentric.y**2 +
                         c.galactocentric.z**2).value
                 for c in coordinates]
    # Add Sun's galactocentric distance at the end manually.
    distances.append(8300)
    # distances *= u.pc
    # distances = [c.galactocentric.z.value for c in coordinates]
    # distances.append(0)
    # distances *= u.pc

    tqdm.write('Making plots for each pair as a function of galactocentric'
               ' distance.')
    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])
            vprint(f'Collecting data for {pair_label}.')

            pair_plots_dir = vcl.output_dir /\
                f'pair_result_plots/galactocentric_distance'
            if not pair_plots_dir.exists():
                os.mkdir(pair_plots_dir)
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax_pre = fig.add_subplot(2, 1, 1)
        ax_post = fig.add_subplot(2, 1, 2,
                                  sharex=ax_pre,
                                  sharey=ax_pre)
        ax_pre.set_xlim(left=8245, right=8340)
        ax_pre.set_xlabel('Galactocentric distance (pc)')
        ax_post.set_xlabel('Galactocentric distance (pc)')
        ax_pre.set_ylabel(r'$\Delta v$ (m/s, pre)')
        ax_post.set_ylabel(r'$\Delta v$ (m/s, post)')

        diffs_pre = ma.masked_invalid(data_pre['delta(v)_pair (m/s)'])
        errs_pre = ma.masked_invalid(
            np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                    data_pre['err_sys_pair (m/s)']**2))

        diffs_post = ma.masked_invalid(data_post['delta(v)_pair (m/s)'])
        errs_post = ma.masked_invalid(
            np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                    data_post['err_sys_pair (m/s)']**2))

        weighted_mean_pre, weight_sum_pre = ma.average(diffs_pre,
                                                       weights=errs_pre**-2,
                                                       returned=True)
        eotwm_pre = 1 / np.sqrt(weight_sum_pre)
        weighted_mean_post, weight_sum_post = ma.average(diffs_post,
                                                         weights=errs_post**-2,
                                                         returned=True)
        eotwm_post = 1 / np.sqrt(weight_sum_post)

        vprint(f'EotWM_pre for {pair_label} is {eotwm_pre}')
        vprint(f'EotWM_post for {pair_label} is {eotwm_post}')

        for ax in (ax_pre, ax_post):
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.yaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.xaxis.grid(which='major', color='Gray',
                          linestyle='-', alpha=0.65)
            ax.xaxis.grid(which='minor', color='Gray',
                          linestyle=':', alpha=0.5)
            ax.axhline(0, color='Black')

        ax_pre.axhline(weighted_mean_pre, color='Black', linestyle='--')
        ax_pre.fill_between([8245, 8340], weighted_mean_pre+eotwm_pre,
                            y2=weighted_mean_pre-eotwm_pre,
                            color='Gray', alpha=0.4)
        ax_post.axhline(weighted_mean_post, color='Black', linestyle='--')
        ax_post.fill_between([8245, 8340], weighted_mean_post+eotwm_post,
                             y2=weighted_mean_post-eotwm_post,
                             color='Gray', alpha=0.4)

        ax_pre.errorbar(distances,
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 +
                                     data_pre['err_sys_pair (m/s)']**2),
                        color='Chocolate',
                        markeredgecolor='Black', marker='o',
                        linestyle='')
        ax_post.errorbar(distances,
                         data_post['delta(v)_pair (m/s)'],
                         yerr=np.sqrt(data_post['err_stat_pair (m/s)']**2 +
                                      data_post['err_sys_pair (m/s)']**2),
                         color='DodgerBlue',
                         markeredgecolor='Black', marker='o',
                         linestyle='')

        outfile = pair_plots_dir / f'{pair_label}.png'

        fig.savefig(str(outfile))
        plt.close('all')


def plot_pair_stability(star):
    """
    Plot the stability of a single pair for a single star over time.

    Parameters
    ----------
    star : `varconlib.star.Star`
        A Star object.

    Returns
    -------
    None.

    """

    try:
        col_num = star.p_index(args.pair_label)
    except KeyError:
        raise

    x = [i for i in range(star.numObs)]
    bervs = star.bervArray
    diffs = star.pairSeparationsArray[:, col_num]
    errs_stat = star.pairSepErrorsArray[:, col_num]
    # print(diffs)

    diffs_no_nans, mask = remove_nans(diffs, return_mask=True)
    # print(diffs_no_nans)
    m_diffs = ma.array(diffs_no_nans.to(u.m/u.s).value)
    # print(m_diffs)
    m_errs = ma.array(errs_stat[mask].value)
    bervs_masked = bervs[mask]

    weighted_mean = np.average(m_diffs, weights=m_errs**-2)
    print(weighted_mean)

    sigma = np.std(diffs_no_nans).to(u.m/u.s)

    results = find_sys_scatter(constant_model, bervs_masked,
                               m_diffs,
                               m_errs, (weighted_mean,),
                               n_sigma=3, tolerance=0.001,
                               verbose=True)

    sys_err = results['sys_err_list'][-1] * u.m / u.s
    print(results['chi_squared_list'][-1])

    # vprint(np.std(diffs))

    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('BERV (km/s)')
    ax.set_ylabel('Pair difference (m/s)')

    # ax.axvline(x=star.fiberSplitIndex, linestyle='--', color='Black')
    ax.errorbar(bervs_masked, m_diffs, yerr=m_errs, linestyle='',
                marker='o',
                color='DarkSalmon', ecolor='Black', markeredgecolor='Black',
                label=r'$\sigma:$'
                f' {sigma:.3f},'
                r' $\sigma_\mathrm{sys}:$'
                f' {sys_err:.3f}')

    ax.legend()
    plt.show()


def plot_sigma_sys_vs_pair_separation(star):
    """
    Plot the sigma_sys for each pair as a function of the average separation.

    Parameters
    ----------
    star : `varconlib.star.Star`
        A `Star` for which to do the plotting.

    Returns
    -------
    None.

    """

    print(f'{star.name} has {star.numObs} observations.')
    n_sigma = 3

    plots_dir = Path('/Users/dberke/Pictures/'
                     'pair_separation_investigation/vs_sigma_sys')

    average_seps_pre = []
    average_seps_post = []
    sigma_sys_list_pre = []
    sigma_sys_list_post = []
    for pair, col_num in tqdm(star._pair_bidict.items()):

        x = ma.array([i for i in range(star.numObs)])
        separations = star.pairSeparationsArray[:, col_num]
        errs_stat = star.pairSepErrorsArray[:, col_num]

        seps_no_nans, mask = remove_nans(separations, return_mask=True)
        m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
        m_errs = ma.array(errs_stat[mask].value)

        weighted_mean = np.average(m_seps, weights=m_errs**-2)

        sigma = np.std(seps_no_nans).to(u.m/u.s)

        results = find_sys_scatter(constant_model, x,
                                   m_seps,
                                   m_errs, (weighted_mean,),
                                   n_sigma=n_sigma, tolerance=0.001,
                                   verbose=False)

        sys_err = results['sys_err_list'][-1] * u.m / u.s
        average_seps_pre.append((weighted_mean * u.m/u.s).to(u.km/u.s))
        sigma_sys_list_pre.append(sys_err)

    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Weighted mean pair separation (km/s)')
    ax.set_ylabel(r'$\sigma_\mathrm{sys}$ (m/s)')
    ax.plot(average_seps_pre, sigma_sys_list_pre,
            linestyle='', marker='o')

    filepath = plots_dir / f'{star.name}_{star.numObs}_obs_{n_sigma}sigma.png'
    fig.savefig(str(filepath))
    # plt.show()


def plot_model_diff_vs_pair_separation(star, model):
    """
    Create a plot showing the difference from a model vs. the pair separation.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to analyze.
    model : str
        The name of the model to test against.

    Returns
    -------
    None.

    """

    print(f'{star.name} has {star.numObs} observations.')
    n_sigma = 5

    plots_dir = Path('/Users/dberke/Pictures/'
                     'pair_separation_investigation')

    # Get the star pair corrections arrays for the given model.
    star.createPairsModelCorrectedArrays(model, n_sigma=5)

    # model_func = getattr(varconlib.fitting, f'{model}_model')
    # num_params = len(signature(model_func).parameters) - 1
    num_params = 1

    filename = vcl.output_dir /\
        f'fit_params/{model}_pairs_params.hdf5'
    fit_results_dict = get_params_file(filename)
    sigma_sys_dict = fit_results_dict['sigmas_sys']

    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    average_separations_pre = []
    model_offsets_pre = []
    pair_sep_errs_pre = []
    average_separations_post = []
    model_offsets_post = []
    pair_sep_errs_post = []

    sigmas_sys_pre = []
    sigmas_sys_post = []

    for pair_label, col_num in tqdm(star._pair_bidict.items()):

        sigmas_sys_pre.append(sigma_sys_dict[pair_label + '_pre'])
        sigmas_sys_post.append(sigma_sys_dict[pair_label + '_post'])

        if star.hasObsPre:
            separations = star.pairSeparationsArray[pre_slice, col_num]
            errs_stat = star.pairSepErrorsArray[pre_slice, col_num]

            seps_no_nans, mask = remove_nans(separations, return_mask=True)
            m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_errs = ma.array(errs_stat[mask].value)

            weighted_mean, weight_sum = np.average(m_seps, weights=m_errs**-2,
                                                   returned=True)
            error_on_weighted_mean = (1 / np.sqrt(weight_sum))

            average_separations_pre.append((weighted_mean
                                            * u.m/u.s).to(u.km/u.s))
            pair_sep_errs_pre.append(error_on_weighted_mean * u.m/u.s)

            # Now get the weighted mean of the remaining offset from the model.
            corrected_separations = star.pairParamsOffsetsArray[pre_slice,
                                                                col_num]
            corrected_errs_stat = star.pairParamsErrorsArray[pre_slice,
                                                             col_num]

            seps_no_nans, mask = remove_nans(corrected_separations,
                                             return_mask=True)
            m_c_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_c_errs = ma.array(errs_stat[mask].value)

            weighted_c_mean, weight_c_sum = np.average(m_c_seps,
                                                       weights=m_c_errs**-2,
                                                       returned=True)

            model_offsets_pre.append(weighted_c_mean * u.m/u.s)

        if star.hasObsPost:
            separations = star.pairSeparationsArray[post_slice, col_num]
            errs_stat = star.pairSepErrorsArray[post_slice, col_num]

            seps_no_nans, mask = remove_nans(separations, return_mask=True)
            m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_errs = ma.array(errs_stat[mask].value)

            weighted_mean, weight_sum = np.average(m_seps, weights=m_errs**-2,
                                                   returned=True)
            error_on_weighted_mean = (1 / np.sqrt(weight_sum))

            average_separations_post.append((weighted_mean
                                            * u.m/u.s).to(u.km/u.s))
            pair_sep_errs_post.append(error_on_weighted_mean * u.m/u.s)

            # Now get the weighted mean of the remaining offset from the model.
            corrected_separations = star.pairParamsOffsetsArray[post_slice,
                                                                col_num]
            corrected_errs_stat = star.pairParamsErrorsArray[post_slice,
                                                             col_num]

            seps_no_nans, mask = remove_nans(corrected_separations,
                                             return_mask=True)
            m_c_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_c_errs = ma.array(errs_stat[mask].value)

            weighted_c_mean, weight_c_sum = np.average(m_c_seps,
                                                       weights=m_c_errs**-2,
                                                       returned=True)

            model_offsets_post.append(weighted_c_mean * u.m/u.s)

    # Plot the results.
    fig = plt.figure(figsize=(11, 7), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=2, figure=fig,
                  width_ratios=(6, 1))
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_hist_pre = fig.add_subplot(gs[0, 1], sharey=ax_pre)
    ax_post = fig.add_subplot(gs[1, 0], sharex=ax_pre, sharey=ax_pre)
    ax_hist_post = fig.add_subplot(gs[1, 1], sharey=ax_pre)

    ax_pre.set_xlim(left=0, right=805)
    ax_pre.set_ylim(bottom=-100, top=100)
    ax_post.set_xlabel('Weighted mean pair separation (km/s)')
    ax_pre.set_ylabel('Offset from model value (pre) (m/s)')
    ax_post.set_ylabel('Offset from model value (post) (m/s)')
    for ax in (ax_pre, ax_post, ax_hist_pre, ax_hist_post):
        ax.axhline(y=0, linestyle='-', color='Black')

    full_errs_pre = np.sqrt(u.unyt_array(pair_sep_errs_pre, units='m/s') ** 2 +
                            u.unyt_array(sigmas_sys_pre, units='m/s') ** 2)
    full_errs_post = np.sqrt(u.unyt_array(pair_sep_errs_post, units='m/s') ** 2
                             + u.unyt_array(sigmas_sys_post, units='m/s') ** 2)
    # vprint(f'{np.median(pair_sep_errs_pre)},   {sigma_sys_pre}')
    # vprint(f'{np.median(pair_sep_errs_post)},   {sigma_sys_post}')

    # Calculate the chi^2 value for the residuals.
    chi_squared_nu_pre = calc_chi_squared_nu(model_offsets_pre,
                                             full_errs_pre,
                                             num_params).value
    chi_squared_nu_post = calc_chi_squared_nu(model_offsets_post,
                                              full_errs_post,
                                              num_params).value
    ax_pre.errorbar(average_separations_pre, model_offsets_pre,
                    yerr=full_errs_pre,
                    linestyle='', marker='o',
                    color='Chocolate',
                    markeredgecolor='Black',
                    label=r'$\chi^2_\nu$:'
                    f' {chi_squared_nu_pre:.2f}, {star.numObsPre} obs')
    ax_post.errorbar(average_separations_post, model_offsets_post,
                     yerr=full_errs_post,
                     linestyle='', marker='o',
                     color='DodgerBlue',
                     markeredgecolor='Black',
                     label=r'$\chi^2_\nu$:'
                     f' { chi_squared_nu_post:.2f}, {star.numObsPost} obs')
    bottom, top = ax_pre.get_ylim()
    bins = [x for x in range(int(bottom), int(top), 1)]
    ax_hist_pre.hist(model_offsets_pre, bins=bins, color='Black',
                     histtype='step', orientation='horizontal')
    ax_hist_post.hist(model_offsets_post, bins=bins, color='Black',
                      histtype='step', orientation='horizontal')

    ax_pre.legend()
    ax_post.legend()
    filepath = plots_dir /\
        f'{star.name}_{star.numObs}_obs_{n_sigma}sigma_{model}_offsets.png'
    fig.savefig(str(filepath))


def create_example_plots():
    """Create example plots."""

    pairs_of_interest = ('5571.164Fe1_5577.637Fe1_50',
                         '6123.910Ca1_6138.313Fe1_60',
                         '6138.313Fe1_6139.390Fe1_60')
    # manual_sys_errs = (7.04, 11.75, 3.31)  # In m/s.
    sys_errs = []
    num_pairs = len(pairs_of_interest)

    sys_err_file = vcl.output_dir /\
        'pair_separation_files/pair_excess_scatters.csv'
    with open(sys_err_file, 'r') as f:
        lines = f.readlines()
    for pair_label in pairs_of_interest:
        for line in lines:
            line = line.split(',')
            if line[0] == pair_label:
                sys_errs.append(float(line[1]))

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)

    RA = star_data['RA'][:-1]
    DEC = star_data['DEC'][:-1]
    dist = star_data['distance (pc)'][:-1]

    coordinates = SkyCoord(ra=RA, dec=DEC, distance=dist,
                           unit=(units.hourangle, units.degree,
                                 units.pc))

    distances = [np.sqrt(c.galactocentric.x**2 + c.galactocentric.y**2 +
                         c.galactocentric.z**2).value
                 for c in coordinates]
    # Add Sun's galactocentric distance at the end manually.
    distances.append(8300)

    # Galactocentric plot
    fig = plt.figure(figsize=(8, 10), constrained_layout=False)
    gs = fig.add_gridspec(nrows=len(pairs_of_interest), ncols=1, hspace=0.04,
                          left=0.11, right=0.96, bottom=0.06, top=0.99)

    axes_dict = {}
    for i in range(num_pairs):
        axes_dict[f'ax{i}'] = fig.add_subplot(gs[i, 0])

    for i in range(num_pairs - 1):
        axes_dict[f'ax{i}'].tick_params(which='both',
                                        labelbottom=False, bottom=False)
    axes_dict[f'ax{num_pairs - 1}'].set_xlabel(
        'Galactocentric distance (pc)')

    plots_dir = Path('/Users/dberke/Pictures/paper_plots_and_tables/plots')
    if not plots_dir.exists():
        os.mkdir(plots_dir)

    for pair_label, ax, err in zip(pairs_of_interest,
                                   axes_dict.values(),
                                   sys_errs):

        data_pre = read_csv_file(pair_label, csv_dir, 'pre')
        data_pre = data_pre.astype(types_dict)
        data_post = read_csv_file(pair_label, csv_dir, 'post')
        data_post = data_post.astype(types_dict)

        ax.set_xlim(left=8245, right=8340)
        ax.set_ylim(bottom=-65, top=65)
        ax.set_ylabel(r'$\Delta v_\mathrm{pair}$ (m/s)')
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax.yaxis.grid(which='major', color='Gray',
        #               linestyle='-', alpha=0.65)
        # ax.yaxis.grid(which='minor', color='Gray',
        #               linestyle=':', alpha=0.5)
        # ax.xaxis.grid(which='major', color='Gray',
        #               linestyle='-', alpha=0.65)
        # ax.xaxis.grid(which='minor', color='Gray',
        #               linestyle=':', alpha=0.5)
        ax.axhline(0, color='Black', linestyle='--')

        # diffs_pre = ma.masked_invalid(data_pre['delta(v)_pair (m/s)'])
        # errs_pre = ma.masked_invalid(data_pre['err_stat_pair (m/s)'])

        # diffs_post = ma.masked_invalid(data_post['delta(v)_pair (m/s)'])
        # errs_post = ma.masked_invalid(data_post['err_stat_pair (m/s)'])

        # diffs = pd.concat([data_pre['delta(v)_pair (m/s)'],
        #                   data_post['delta(v)_pair (m/s)']])
        # errs = pd.concat([data_pre['err_stat_pair (m/s)'],
        #                  data_post['err_stat_pair (m/s)']])

        # print(data_pre['err_stat_pair (m/s)'])

        # TODO: Figure out how to combine the pre- and post- data.

        ax.errorbar(distances,
                    data_pre['delta(v)_pair (m/s)'],
                    yerr=np.sqrt(data_pre['err_stat_pair (m/s)']**2 + err**2),
                    color='Black', markerfacecolor='DodgerBlue',
                    ecolor='DodgerBlue',
                    markeredgecolor='Black', marker='o',
                    linestyle='',
                    label=format_pair_label(pair_label, use_latex=True) +
                    f', Sys. Err: {err:.3f} (m/s)')
        # ax.annotate(format_pair_label(pair_label),
        #             (0.01, 0.11),
        #             xycoords='axes fraction',
        #             verticalalignment='top')
        ax.legend()

    outfile = plots_dir / 'Galactocentric_distance.png'
    fig.savefig(str(outfile))

    # Parameter plots

    for parameter in tqdm(('temperature', 'metallicity', 'logg')):
        fig = plt.figure(figsize=(8, 10), constrained_layout=False)
        gs = fig.add_gridspec(nrows=len(pairs_of_interest),
                              ncols=1, hspace=0.04,
                              left=0.11, right=0.96,
                              bottom=0.06, top=0.99)

        axes_dict = {}
        for i in range(num_pairs):
            axes_dict[f'ax{i}'] = fig.add_subplot(gs[i, 0])

        for i in range(num_pairs - 1):
            axes_dict[f'ax{i}'].tick_params(which='both',
                                            labelbottom=False, bottom=False)
        axes_dict[f'ax{num_pairs - 1}'].set_xlabel(
            plot_axis_labels[parameter])

        for pair_label, ax, err in zip(pairs_of_interest,
                                       axes_dict.values(),
                                       sys_errs):
            data_pre = read_csv_file(pair_label, csv_dir, 'pre')
            data_pre = data_pre.astype(types_dict)
            data_post = read_csv_file(pair_label, csv_dir, 'post')
            data_post = data_post.astype(types_dict)

            # ax.set_xlim(left=8245, right=8340)
            ax.set_ylim(bottom=-65, top=65)
            ax.set_ylabel(r'$\Delta v_\mathrm{pair}$ (m/s)')
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            # ax.yaxis.grid(which='major', color='Gray',
            #               linestyle='-', alpha=0.65)
            # ax.yaxis.grid(which='minor', color='Gray',
            #               linestyle=':', alpha=0.5)
            # ax.xaxis.grid(which='major', color='Gray',
            #               linestyle='-', alpha=0.65)
            # ax.xaxis.grid(which='minor', color='Gray',
            #               linestyle=':', alpha=0.5)
            ax.axhline(0, color='Black', linestyle='--')
            y_errs = np.sqrt(data_pre['err_stat_pair (m/s)'] ** 2 +
                             err ** 2)

            vprint(f'Chi^2_nu for {parameter}, {pair_label} is')
            vprint(calc_chi_squared_nu(data_pre['delta(v)_pair (m/s)'],
                                       y_errs, 1))

            ax.errorbar(star_data[params_dict[parameter]],
                        data_pre['delta(v)_pair (m/s)'],
                        yerr=y_errs,
                        color='Black', markerfacecolor='DodgerBlue',
                        ecolor='DodgerBlue',
                        markeredgecolor='Black', marker='o',
                        linestyle='',
                        label=format_pair_label(pair_label, use_latex=True) +
                        f', Sys. Err: {err:.3f} (m/s)')
            # ax.annotate(format_pair_label(pair_label),
            #             (0.01, 0.11),
            #             xycoords='axes fraction',
            #             verticalalignment='top')
            ax.legend()

        outfile = plots_dir / f'{parameter}.png'
        fig.savefig(str(outfile))
        plt.close('all')


def format_pair_label(pair_label, use_latex=False):
    """Format a pair label for prettier printing.

    Parameters
    ----------
    pair_label : str
        A pair label of the form '6138.313Fe1_6139.390Fe1_60'.
    use_latex : bool, Default : *False*
        Whether to return a string with LaTeX code to add a lambda symbol in
        front of the wavelength or not.

    Returns
    -------
    str
        A formatted string of the form 'Fe I 6138.313, Fe I 6139.390', if
        use_latex=False.

    """

    transition1, transition2, _ = pair_label.split('_')
    wavelength1 = transition1[:8]
    ion1 = transition1[8:]
    element1 = ion1[:-1]
    state1 = ion1[-1]
    wavelength2 = transition2[:8]
    ion2 = transition2[8:]
    element2 = ion2[:-1]
    state2 = ion2[-1]

    if use_latex:
        return f'{element1} {roman_numerals[int(state1)]}' +\
               rf' $\lambda{wavelength1}$, ' +\
               f'{element2} {roman_numerals[int(state2)]}' +\
               rf' $\lambda{wavelength2}$'
    else:
        return f'{element1} {roman_numerals[int(state1)]}' +\
               f' {wavelength1}, ' +\
               f'{element2} {roman_numerals[int(state2)]}' +\
               f' {wavelength2}'


def get_star(star_name):
    """
    Return a `varconlib.star.Star` object using this name.

    Parameters
    ----------
    star_name : str
        The name of a star corresponding the name of a directory for that star
        in `varconlib.output_dir`.

    Returns
    -------
    `varconlib.star.Star`
        A `Star` object made by using the name given and the default values.

    """

    return Star(star_name, vcl.output_dir / star_name)


# Main script body.
parser = argparse.ArgumentParser(description="Plot results for each pair"
                                 " of transitions for various parameters.")

parameter_options = parser.add_argument_group(
    title="Plot parameter options",
    description="Select what parameters to plot the pairs-wise velocity"
    " separations by.")

parser.add_argument('star', nargs='?', default=None, const=None, type=str,
                    help='The name of a single, specific star to make a plot'
                    ' from. If not given will default to using all stars.')

parser.add_argument('-m', '--model', type=str, action='store',
                    help='The name of a model to test against.')
parser.add_argument('--heliocentric-distance', action='store_true',
                    help='Plot as a function of distance from the Sun.')
parser.add_argument('--galactocentric-distance', action='store_true',
                    help='Plot as a function of distance from galactic center.')
parser.add_argument('--sigma-sys-vs-pair-separations', action='store_true',
                    help='Plot the sigma_sys for each pair as a function'
                    ' of its weighted-mean separation.')
parser.add_argument('--model-diff-vs-pair-separations', action='store_true',
                    help='Plot the model difference for each pair as a function'
                    ' of it weighted-mean separation.')

parameter_options.add_argument('-T', '--temperature',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='temperature',
                               help="Plot as a function of stellar"
                               "temperature.")
parameter_options.add_argument('-M', '--metallicity',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='metallicity',
                               help="Plot as a function of stellar"
                               " metallicity.")
parameter_options.add_argument('-G', '--logg',
                               dest='parameters_to_plot',
                               action='append_const',
                               const='logg',
                               help="Plot as a function of stellar"
                               " surface gravity.")

parser.add_argument('--pair-label', action='store', type=str,
                    help='The full label of a specific pair to make a plot for'
                    " a single star of that pair's stability over time.")
parser.add_argument('--example-plot', action='store_true',
                    help='Plot an example of some good pairs.')

parser.add_argument('-v', '--verbose', action='store_true',
                    help="Print more output about what's happening.")

args = parser.parse_args()

# Define vprint to only print when the verbose flag is given.
vprint = vcl.verbose_print(args.verbose)

# Get the star from the name.
star = get_star(args.star)

csv_dir = vcl.output_dir / 'pair_separation_files'

star_properties_file = csv_dir / 'star_properties.csv'

star_data = pd.read_csv(star_properties_file)

# Import the list of pairs to use.
with open(vcl.final_pair_selection_file, 'r+b') as f:
    pairs_list = pickle.load(f)

pairs_dict = {}
for pair in tqdm(pairs_list):
    for order_num in pair.ordersToMeasureIn:
        pair_label = "_".join([pair.label, str(order_num)])
        pairs_dict[pair_label] = pair

if args.parameters_to_plot:
    for parameter in tqdm(args.parameters_to_plot):
        vprint(f'Plotting vs. {parameter}')
        plot_vs(parameter)

if args.heliocentric_distance:
    plot_distance()

if args.galactocentric_distance:
    plot_galactic_distance()

if args.example_plot:
    create_example_plots()

if args.star is not None and args.pair_label:
    plot_pair_stability(star)

if args.star is not None and args.sigma_sys_vs_pair_separations:
    plot_sigma_sys_vs_pair_separation(star)

if args.star is not None and args.model is not None\
        and args.model_diff_vs_pair_separations:
    plot_model_diff_vs_pair_separation(star, args.model.replace('-', '_'))
