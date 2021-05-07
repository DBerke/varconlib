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
from itertools import tee
import os
from pathlib import Path
import pickle
import time

from astropy.coordinates import SkyCoord
import astropy.units as units
import cmasher as cmr
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from p_tqdm import p_map, t_map
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import unyt as u

import varconlib as vcl
from varconlib.fitting import (calc_chi_squared_nu, constant_model,
                               find_sys_scatter)
from varconlib.miscellaneous import (remove_nans, get_params_file,
                                     weighted_mean_and_error)
from varconlib.star import Star
from varconlib.transition_line import roman_numerals

import varconlib.fitting

params_dict = {'temperature': 'Teff (K)',
               'metallicity': '[Fe/H]',
               'logg': 'log(g)'}


types_dict = {'#star_name': str,
              'delta(v)_pair (m/s)': float,
              'err_stat_pair (m/s)': float,
              'err_sys_pair (m/s)': float,
              'transition1 (m/s)': float,
              't_stat_err1 (m/s)': float,
              't_sys_err1 (m/s)': float,
              'chi^2_nu1': float,
              'transition2 (m/s)': float,
              't_stat_err2 (m/s)': float,
              't_sys_err2 (m/s)': float,
              'chi^2_nu2': float}


plot_axis_labels = {'temperature': r'$\mathrm{T}_\mathrm{eff}\,$(K)',
                    'metallicity': r'$\mathrm{[Fe/H]}$',
                    'logg': r'$\log(g),\mathrm{cm\,s}^{-2}$'}


def pairwise(iterable):
    """Return pairs of results from iterable:

    s -> (s0,s1), (s1,s2), (s2, s3), ...

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
                'pair_result_plots/heliocentric_distance'
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
    """
    Plot pair separations as a function of distance from the Galactic center.


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
                'pair_result_plots/galactocentric_distance'
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


def plot_pair_stability(star, pair_label):
    """
    Plot the stability of a single pair for a single star over time.

    Parameters
    ----------
    star : `varconlib.star.Star`
        A Star object.
    pair_label : str
        The pair label for the pair to use.

    Returns
    -------
    None.

    """

    try:
        col_num = star.p_index(pair_label)
    except KeyError:
        raise

    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    fig = plt.figure(figsize=(10, 9), tight_layout=True)
    gs = GridSpec(nrows=5, ncols=1, figure=fig,
                  height_ratios=(5, 4, 1, 5, 4), hspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[3, 0], sharex=ax_pre, sharey=ax_pre)
    ax_bins_pre = fig.add_subplot(gs[1, 0], sharex=ax_pre)
    ax_bins_post = fig.add_subplot(gs[4, 0], sharex=ax_post,
                                   sharey=ax_bins_pre)
    for ax in (ax_pre, ax_post, ax_bins_pre, ax_bins_post):
        ax.axhline(y=0, color='Gray', linestyle=':', alpha=0.9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_pre.set_ylabel('Model offset\n(pre) (m/s')
    ax_post.set_ylabel('Model offset\n(post) (m/s')
    ax_bins_pre.set_ylabel('Weighted\nmean (m/s)')
    ax_bins_post.set_ylabel('Weighted\nmean (m/s)')
    ax_bins_post.set_xlabel('BERV (km/s)')

    separation_limits = [i for i in range(-25, 30, 5)]

    if star.hasObsPre:
        bervs = star.bervArray[pre_slice]
        diffs = star.pairModelOffsetsArray[pre_slice, col_num]
        errs_stat = star.pairModelErrorsArray[pre_slice, col_num]

        diffs_no_nans, nan_mask = remove_nans(diffs, return_mask=True)
        # print(diffs_no_nans)
        m_diffs = ma.array(diffs_no_nans.to(u.m/u.s).value)
        # print(m_diffs)
        m_errs = ma.array(errs_stat[nan_mask].value)
        bervs_masked = bervs[nan_mask]

        weighted_mean = np.average(m_diffs, weights=m_errs**-2)
        # print(weighted_mean)

        sigma = np.std(diffs_no_nans).to(u.m/u.s)

        results = find_sys_scatter(constant_model, bervs_masked,
                                   m_diffs,
                                   m_errs, (weighted_mean,),
                                   n_sigma=3, tolerance=0.001,
                                   verbose=False)

        sys_err = results['sys_err_list'][-1] * u.m / u.s
        # print(results['chi_squared_list'][-1])

        # vprint(np.std(diffs))

        # ax.axvline(x=star.fiberSplitIndex, linestyle='--', color='Black')
        ax_pre.errorbar(bervs_masked, m_diffs, yerr=m_errs,
                        linestyle='', marker='o',
                        color='Chocolate',
                        # ecolor='Black',
                        markeredgecolor='Black',
                        label=r'$\sigma:$'
                        f' {sigma:.3f},'
                        r' $\sigma_\mathrm{sys}:$'
                        f' {sys_err:.3f}')

        midpoints, w_means, eotwms = [], [], []
        bin_num = len(separation_limits) - 1
        for i, lims in zip(range(bin_num),
                           pairwise(separation_limits)):
            mask = np.where((bervs_masked > lims[0]) &
                            (bervs_masked < lims[1]))
            if len(m_diffs[mask]) == 0:
                midpoints.append(np.nan)
                w_means.append(np.nan)
                eotwms.append(np.nan)
                continue
            midpoints.append((lims[1] + lims[0]) / 2)
            w_mean, eotwm = weighted_mean_and_error(m_diffs[mask],
                                                    m_errs[mask])
#            chisq = calc_chi_squared_nu(m_diffs[mask],
#                                        m_errs[mask], 1)
            w_means.append(w_mean)
            eotwms.append(eotwm)

        midpoints = np.array(midpoints)
        w_means = np.array(w_means)
        eotwms = np.array(eotwms)

        # sigma_values = model_offsets_pre / full_errs_pre

        # ax_sigma_pre.errorbar(average_separations_pre,
        #                       sigma_values,
        #                       color='Chocolate', linestyle='',
        #                       marker='.')
        ax_bins_pre.errorbar(midpoints, w_means, yerr=eotwms,
                             linestyle='-', color='Green',
                             marker='',
                             capsize=4)
        # ax_sigma_pre.errorbar(midpoints, w_means / eotwms,
        #                       linestyle=':', marker='o',
        #                       color='ForestGreen')
        # ax_sigma_hist_pre.hist(sigma_values,
        #                        bins=[x for x in np.linspace(-5, 5, num=50)],
        #                        color='Black', histtype='step',
        #                        orientation='horizontal')

    if star.hasObsPost:
        bervs = star.bervArray[post_slice]
        diffs = star.pairModelOffsetsArray[post_slice, col_num]
        errs_stat = star.pairModelErrorsArray[post_slice, col_num]

        diffs_no_nans, nan_mask = remove_nans(diffs, return_mask=True)
        # print(diffs_no_nans)
        m_diffs = ma.array(diffs_no_nans.to(u.m/u.s).value)
        # print(m_diffs)
        m_errs = ma.array(errs_stat[nan_mask].value)
        bervs_masked = bervs[nan_mask]

        weighted_mean = np.average(m_diffs, weights=m_errs**-2)
        # print(weighted_mean)

        sigma = np.std(diffs_no_nans).to(u.m/u.s)

        results = find_sys_scatter(constant_model, bervs_masked,
                                   m_diffs,
                                   m_errs, (weighted_mean,),
                                   n_sigma=3, tolerance=0.001,
                                   verbose=False)

        sys_err = results['sys_err_list'][-1] * u.m / u.s
        # print(results['chi_squared_list'][-1])

        # vprint(np.std(diffs))

        # ax.axvline(x=star.fiberSplitIndex, linestyle='--', color='Black')
        ax_post.errorbar(bervs_masked, m_diffs, yerr=m_errs,
                         linestyle='', marker='o',
                         color='DodgerBlue',
                         # ecolor='Black',
                         markeredgecolor='Black',
                         label=r'$\sigma:$'
                         f' {sigma:.3f},'
                         r' $\sigma_\mathrm{sys}:$'
                         f' {sys_err:.3f}')

        midpoints, w_means, eotwms = [], [], []
        bin_num = len(separation_limits) - 1
        for i, lims in zip(range(bin_num),
                           pairwise(separation_limits)):
            mask = np.where((bervs_masked > lims[0]) &
                            (bervs_masked < lims[1]))
            if len(m_diffs[mask]) == 0:
                midpoints.append(np.nan)
                w_means.append(np.nan)
                eotwms.append(np.nan)
                continue
            midpoints.append((lims[1] + lims[0]) / 2)
            w_mean, eotwm = weighted_mean_and_error(m_diffs[mask],
                                                    m_errs[mask])
#            chisq = calc_chi_squared_nu(m_diffs[mask],
#                                        m_errs[mask], 1)
            w_means.append(w_mean)
            eotwms.append(eotwm)

        midpoints = np.array(midpoints)
        w_means = np.array(w_means)
        eotwms = np.array(eotwms)

        # sigma_values = model_offsets_pre / full_errs_pre

        # ax_sigma_pre.errorbar(average_separations_pre,
        #                       sigma_values,
        #                       color='Chocolate', linestyle='',
        #                       marker='.')
        ax_bins_post.errorbar(midpoints, w_means, yerr=eotwms,
                              linestyle='-', color='Green',
                              marker='',
                              capsize=4)
        # ax_sigma_pre.errorbar(midpoints, w_means / eotwms,
        #                       linestyle=':', marker='o',
        #                       color='ForestGreen')
        # ax_sigma_hist_pre.hist(sigma_values,
        #                        bins=[x for x in np.linspace(-5, 5, num=50)],
        #                        color='Black', histtype='step',
        #                        orientation='horizontal')

        # ax.legend()
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
#    average_seps_post = []
    sigma_sys_list_pre = []
#    sigma_sys_list_post = []

    model_seps = []
    model_sigma_sys = []

    for pair, col_num in tqdm(star._pair_bidict.items()):

        x = ma.array([i for i in range(star.numObs)])
        separations = star.pairSeparationsArray[:, col_num]
        errs_stat = star.pairSepErrorsArray[:, col_num]

        seps_no_nans, mask = remove_nans(separations, return_mask=True)
        m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
        m_errs = ma.array(errs_stat[mask].value)

        weighted_mean = np.average(m_seps, weights=m_errs**-2)

#        sigma = np.std(seps_no_nans).to(u.m/u.s)

        results = find_sys_scatter(constant_model, x,
                                   m_seps,
                                   m_errs, (weighted_mean,),
                                   n_sigma=n_sigma, tolerance=0.001,
                                   verbose=False)

        sys_err = results['sys_err_list'][-1] * u.m / u.s
        average_seps_pre.append((weighted_mean * u.m/u.s).to(u.km/u.s))
        sigma_sys_list_pre.append(sys_err)

        model_values = star.pairModelArray[:, col_num]
        model_errs = star.pairModelErrorsArray[:, col_num]
        print(model_values.shape)
        print(model_errs.shape)
        weighted_mean2 = np.average(model_values, weights=model_errs**-2)

        model_results = find_sys_scatter(constant_model, x,
                                         model_values,
                                         model_errs, (weighted_mean2,),
                                         n_sigma=n_sigma, tolerance=0.001,
                                         verbose=False)
        model_seps.append((weighted_mean2 * u.m/u.s).to(u.km/u.s))
        model_sigma_sys.append(model_results['sys_err_list'[-1]] * u.m / u.s)

    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Weighted mean pair separation (km/s)')
    ax.set_ylabel(r'$\sigma_\mathrm{sys}$ (m/s)')
    ax.plot(average_seps_pre, sigma_sys_list_pre,
            linestyle='', marker='o', color='DarkOrange')
    ax.plot(model_seps, model_sigma_sys,
            linestyle='', marker='x', color='MediumAquaMarine')

    filepath = plots_dir / f'{star.name}_{star.numObs}_obs_{n_sigma}sigma.png'
    fig.savefig(str(filepath))
    # plt.show()


def plot_model_diff_vs_pair_separation(star, model, n_sigma=4.0):
    """
    Create a plot showing the difference from a model vs. the pair separation.

    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to analyze.
    model : str
        The name of the model to test against.
    n_sigma : float
        The number of sigma to use for culling outliers in the pair model
        fitting.

    Returns
    -------
    tuple
        A tuple containing the star name, then the number of observations, the
        reduced chi-squared, and the weighted mean and error on the weighted
        mean for both the pre- and post- fiber change eras, with NaNs instead
        if there were no relevant values for an era.

    """

    tqdm.write(f'{star.name} has {star.numObs} observations'
               f' ({star.numObsPre} pre, {star.numObsPost} post)')

    plots_dir = Path('/Users/dberke/Pictures/'
                     'pair_separation_investigation')

    # Get the star pair corrections arrays for the given model.
    if n_sigma != 4.0:
        star.createPairModelCorrectedArrays(model_func=model, n_sigma=n_sigma)

    # model_func = getattr(varconlib.fitting, f'{model}_model')
    # num_params = len(signature(model_func).parameters) - 1
    num_params = 1

    filename = vcl.output_dir /\
        f'fit_params/{model}_pairs_{n_sigma:.1f}sigma_params.hdf5'
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

    # Initialize these variables as NaN so as not to break the code returning
    # them if the star only has observations from one era:
    chi_squared_nu_pre, w_mean_pre, eotwm_pre = np.nan, np.nan, np.nan
    chi_squared_nu_post, w_mean_post, eotwm_post = np.nan, np.nan, np.nan

    for pair_label, col_num in tqdm(star._pair_bidict.items()):

        if star.hasObsPre:
            separations = star.pairSeparationsArray[pre_slice, col_num]
            errs_stat = star.pairSepErrorsArray[pre_slice, col_num]

            # If all separations are non-existent for this pair, continue.
            if np.isnan(separations).all():
                continue

            # Now get the weighted mean of the remaining offset from the model.
            corrected_separations = star.pairModelOffsetsArray[pre_slice,
                                                               col_num]
            corrected_errs_stat = star.pairModelErrorsArray[pre_slice,
                                                            col_num]

            if np.isnan(corrected_separations).all():
                continue

            sigmas_sys_pre.append(sigma_sys_dict[pair_label + '_pre'])
            seps_no_nans, mask = remove_nans(separations, return_mask=True)
            m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_errs = ma.array(errs_stat[mask].value)

            try:
                weighted_mean,\
                    error_on_weighted_mean = weighted_mean_and_error(
                            m_seps, m_errs)
            except ZeroDivisionError:
                print(separations)
                print(errs_stat)
                raise

            average_separations_pre.append((weighted_mean
                                            * u.m/u.s).to(u.km/u.s))
            pair_sep_errs_pre.append(error_on_weighted_mean * u.m/u.s)

            seps_no_nans, mask = remove_nans(corrected_separations,
                                             return_mask=True)
            m_c_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_c_errs = ma.array(errs_stat[mask].value)

            try:
                weighted_c_mean,\
                    weight_c_sum = np.average(m_c_seps,
                                              weights=m_c_errs**-2,
                                              returned=True)
            except ZeroDivisionError:
                print(separations, errs_stat)
                print(corrected_separations, corrected_errs_stat)
                print(m_c_seps, m_c_errs)
                raise

            model_offsets_pre.append(weighted_c_mean * u.m/u.s)

        if star.hasObsPost:
            separations = star.pairSeparationsArray[post_slice, col_num]
            errs_stat = star.pairSepErrorsArray[post_slice, col_num]

            if np.isnan(separations).all():
                continue

            # Now get the weighted mean of the remaining offset from the model.
            corrected_separations = star.pairModelOffsetsArray[post_slice,
                                                               col_num]
            corrected_errs_stat = star.pairModelErrorsArray[post_slice,
                                                            col_num]

            if np.isnan(corrected_separations).all():
                continue

            sigmas_sys_post.append(sigma_sys_dict[pair_label + '_post'])
            seps_no_nans, mask = remove_nans(separations, return_mask=True)
            m_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_errs = ma.array(errs_stat[mask].value)

            weighted_mean, error_on_weighted_mean = weighted_mean_and_error(
                m_seps, m_errs)

            average_separations_post.append((weighted_mean
                                            * u.m/u.s).to(u.km/u.s))
            pair_sep_errs_post.append(error_on_weighted_mean * u.m/u.s)

            seps_no_nans, mask = remove_nans(corrected_separations,
                                             return_mask=True)
            m_c_seps = ma.array(seps_no_nans.to(u.m/u.s).value)
            m_c_errs = ma.array(errs_stat[mask].value)

            weighted_c_mean, weight_c_sum = np.average(m_c_seps,
                                                       weights=m_c_errs**-2,
                                                       returned=True)

            model_offsets_post.append(weighted_c_mean * u.m/u.s)

    # Plot the results.
    fig = plt.figure(figsize=(14, 10.5), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=7, figure=fig,
                  width_ratios=(8.5, 1),
                  height_ratios=(2, 1, 1, 0.6, 2, 1, 1), hspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[4, 0], sharex=ax_pre, sharey=ax_pre)
    ax_hist_pre = fig.add_subplot(gs[0, 1], sharey=ax_pre)
    ax_hist_post = fig.add_subplot(gs[4, 1], sharey=ax_post)
    ax_chi_pre = fig.add_subplot(gs[1, 0], sharex=ax_pre)
    ax_chi_post = fig.add_subplot(gs[5, 0], sharex=ax_post, sharey=ax_chi_pre)
    ax_wmean_pre = fig.add_subplot(gs[2, 0], sharex=ax_pre)
    ax_wmean_post = fig.add_subplot(gs[6, 0], sharex=ax_post,
                                    sharey=ax_wmean_pre)
    ax_sigma_hist_pre = fig.add_subplot(gs[1:3, 1])
    ax_sigma_hist_post = fig.add_subplot(gs[5:7, 1],
                                         sharey=ax_sigma_hist_pre)

    ax_pre.set_xlim(left=0, right=805)
    ax_pre.set_ylim(bottom=-70, top=70)
    ax_sigma_hist_pre.set_ylim(bottom=-3, top=3)
    for ax in (ax_pre, ax_post, ax_hist_pre, ax_hist_post,
               ax_chi_pre, ax_chi_post):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_wmean_post.set_xlabel('Weighted mean pair separation (km/s)')
    ax_pre.set_ylabel('Model offset\npre (m/s)')
    ax_post.set_ylabel('Model offset\npost (m/s)')
    ax_chi_pre.set_ylabel(r'$\chi^2_\nu$')
    ax_chi_post.set_ylabel(r'$\chi^2_\nu$')
    ax_wmean_pre.set_ylabel('Weighted\nmean pre\n(m/s)')
    ax_wmean_post.set_ylabel('Weighted\nmean post\n(m/s)')
    ax_sigma_hist_pre.set_ylabel('Significance')
    ax_sigma_hist_post.set_ylabel('Significance')
    for ax in (ax_pre, ax_post, ax_hist_pre, ax_hist_post,
               ax_sigma_hist_pre, ax_sigma_hist_post,
               ax_wmean_pre, ax_wmean_post):
        ax.axhline(y=0, linestyle='--', color='Gray')
    for ax in (ax_chi_pre, ax_chi_post):
        ax.axhline(y=1, linestyle=':', color='Gray')
        ax.set_ylim(bottom=0, top=2)

    # Add some information about the star to the figure:
    add_star_information(star, ax_wmean_pre, (0.1, 0.5))

    if star.hasObsPre:
        full_errs_pre = np.sqrt(u.unyt_array(pair_sep_errs_pre,
                                             units='m/s') ** 2 +
                                u.unyt_array(sigmas_sys_pre,
                                             units='m/s') ** 2)
        chi_squared_nu_pre = calc_chi_squared_nu(model_offsets_pre,
                                                 full_errs_pre,
                                                 num_params).value

        label = r'$\chi^2_\nu$:' +\
            f' {chi_squared_nu_pre:.2f}, {star.numObsPre} obs'

        ax_pre.errorbar(average_separations_pre, model_offsets_pre,
                        yerr=full_errs_pre,
                        linestyle='', marker='o',
                        color='Chocolate',
                        markeredgecolor='Black',
                        label=label)
        ax_pre.legend(loc='upper right')
    if star.hasObsPost:
        full_errs_post = np.sqrt(u.unyt_array(pair_sep_errs_post,
                                              units='m/s') ** 2 +
                                 u.unyt_array(sigmas_sys_post,
                                              units='m/s') ** 2)
        chi_squared_nu_post = calc_chi_squared_nu(model_offsets_post,
                                                  full_errs_post,
                                                  num_params).value

        label = r'$\chi^2_\nu$:' +\
            f' {chi_squared_nu_post:.2f}, {star.numObsPost} obs'

        ax_post.errorbar(average_separations_post, model_offsets_post,
                         yerr=full_errs_post,
                         linestyle='', marker='o',
                         color='DodgerBlue',
                         markeredgecolor='Black',
                         label=label)
        ax_post.legend(loc='upper right')

    # Plot on the separation histogram axes.
    gaussians_pre = []
    gaussians_post = []
    if star.hasObsPre:
        for offset, err in zip(model_offsets_pre, full_errs_pre):
            gaussians_pre.append(norm(loc=0, scale=err))
    if star.hasObsPost:
        for offset, err in zip(model_offsets_post, full_errs_post):
            gaussians_post.append(norm(loc=0, scale=err))

    bottom, top = ax_pre.get_ylim()
    bins = [x for x in range(int(bottom), int(top), 1)]

    pdf_pre = []
    pdf_post = []
    # Add up the PDFs for each point.
    for x in tqdm(bins):
        if star.hasObsPre:
            pdf_pre.append(np.sum([g.pdf(x) for g in gaussians_pre]))
        if star.hasObsPost:
            pdf_post.append(np.sum([g.pdf(x) for g in gaussians_post]))

    if star.hasObsPre:
        ax_hist_pre.hist(np.array(model_offsets_pre), bins=bins,
                         color='Black',
                         histtype='step', orientation='horizontal')
        ax_hist_pre.step(pdf_pre, bins, color='Green',
                         where='mid', linestyle='-')
        w_mean_pre, eotwm_pre = weighted_mean_and_error(model_offsets_pre,
                                                        full_errs_pre)
        w_mean_pre = w_mean_pre.value
        eotwm_pre = eotwm_pre.value
        ax_hist_pre.annotate(f'{w_mean_pre:.2f}±\n'
                             f'{eotwm_pre:.2f} m/s',
                             (0.99, 0.99),
                             xycoords='axes fraction',
                             verticalalignment='top',
                             horizontalalignment='right',
                             fontsize=20)
    if star.hasObsPost:
        ax_hist_post.hist(np.array(model_offsets_post), bins=bins,
                          color='Black',
                          histtype='step', orientation='horizontal')
        ax_hist_post.step(pdf_post, bins, color='Green',
                          where='mid', linestyle='-')
        w_mean_post, eotwm_post = weighted_mean_and_error(model_offsets_post,
                                                          full_errs_post)
        w_mean_post = w_mean_post.value
        eotwm_post = eotwm_post.value
        ax_hist_post.annotate(f'{w_mean_post:.2f}±\n'
                              f'{eotwm_post:.2f} m/s',
                              (0.99, 0.99),
                              xycoords='axes fraction',
                              verticalalignment='top',
                              horizontalalignment='right',
                              fontsize=20)

    # Do the binned checks.
    separation_limits = [i for i in range(0, 900, 100)]
    if star.hasObsPre:
        average_separations_pre = np.array(average_separations_pre)
        model_offsets_pre = np.array(model_offsets_pre)
    if star.hasObsPost:
        average_separations_post = np.array(average_separations_post)
        model_offsets_post = np.array(model_offsets_post)

    if star.hasObsPre:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(separation_limits)-1),
                           pairwise(separation_limits)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((average_separations_pre > lims[0]) &
                            (average_separations_pre < lims[1]))
            w_mean, eotwm = weighted_mean_and_error(model_offsets_pre[mask],
                                                    full_errs_pre[mask])
            chisq.append(calc_chi_squared_nu(model_offsets_pre[mask],
                                             full_errs_pre[mask], 1).value)

            w_means.append(w_mean)
            eotwms.append(eotwm)

        midpoints = np.array(midpoints)
        w_means = np.array(w_means)
        eotwms = np.array(eotwms)

        sigma_values = model_offsets_pre / full_errs_pre

        ax_wmean_pre.errorbar(midpoints, w_means, yerr=eotwms,
                              linestyle='-', color='Black',
                              marker='o')
        ax_chi_pre.plot(midpoints, chisq, linestyle='-',
                        color='SaddleBrown', marker='o')
        ax_sigma_hist_pre.hist(sigma_values,
                               bins=[x for x in np.linspace(-5, 5, num=50)],
                               color='Black', histtype='step',
                               orientation='horizontal')

    if star.hasObsPost:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(separation_limits)-1),
                           pairwise(separation_limits)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((average_separations_post > lims[0]) &
                            (average_separations_post < lims[1]))
            w_mean, eotwm = weighted_mean_and_error(model_offsets_post[mask],
                                                    full_errs_post[mask])
            chisq.append(calc_chi_squared_nu(model_offsets_post[mask],
                                             full_errs_post[mask], 1).value)
            w_means.append(w_mean)
            eotwms.append(eotwm)

            # if lims[0] == 400 and lims[1] == 500:
            #     outfile = plots_dir /\
            #         f'{n_sigma}sigma_bin_values_{star.name}.csv'
            #     print(outfile)
            #     names = ma.array([k for k in star._pair_bidict.keys()])
            #     with open(outfile, 'w', newline='') as f:
            #         datawriter = csv.writer(f)
            #         datawriter.writerow(('pair_label', 'value',
            #                              'error', 'significance'))
            #         for value, err, label in zip(model_offsets_post[mask],
            #                                      full_errs_post[mask].value,
            #                                      names[mask]):
            #             datawriter.writerow((label, value, err, value/err))

        for label, offset, err in zip(star._pair_bidict.keys(),
                                      model_offsets_post,
                                      full_errs_post):
            if abs(offset/err).value > 3:
                print(label, offset, err, offset/err.value)

        midpoints = np.array(midpoints)
        w_means = np.array(w_means)
        eotwms = np.array(eotwms)

        sigma_values = model_offsets_post / full_errs_post

        ax_wmean_post.errorbar(midpoints, w_means, yerr=eotwms,
                               linestyle='-', color='Black',
                               marker='o')
        ax_chi_post.plot(midpoints, chisq, linestyle='-',
                         color='RoyalBlue', marker='o')
        ax_sigma_hist_post.hist(sigma_values,
                                bins=[x for x in np.linspace(-5, 5, num=50)],
                                color='Black', histtype='step',
                                orientation='horizontal')

    # Save out the plot.
    plots_dir = plots_dir / f'{n_sigma}-sigma'
    if not plots_dir.exists():
        os.mkdir(plots_dir)
    filepath = plots_dir /\
        f'{star.name}_{star.numObs}_obs_{n_sigma}sigma_{model}_offsets.png'
    fig.savefig(str(filepath))
    plt.close('all')
    return (star.name,
            star.numObsPre, chi_squared_nu_pre,
            w_mean_pre, eotwm_pre,
            star.numObsPost, chi_squared_nu_post,
            w_mean_post, eotwm_post)


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
    pair_sep_post1, pair_model_post1 = [], []
    pair_sep_err_post1, pair_model_err_post1 = [], []

    pair_sep_pre2, pair_model_pre2 = [], []
    pair_sep_err_pre2, pair_model_err_pre2 = [], []
    pair_sep_post2, pair_model_post2 = [], []
    pair_sep_err_post2, pair_model_err_post2 = [], []

    pair_order_numbers = []
    for pair in tqdm(star.pairsList):
        if len(pair.ordersToMeasureIn) == 2:
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

            if star.hasObsPost:
                # Get the values for the first instance.
                time_slice = slice(star.fiberSplitIndex, None)
                w_mean, eotwm = get_weighted_mean(
                    star.pairSeparationsArray,
                    star.pairSepErrorsArray,
                    time_slice,
                    p_index1)
                pair_sep_post1.append(w_mean)
                pair_sep_err_post1.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[1,
                                                               p_index1]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index1)
                pair_model_post1.append(w_mean)
                pair_model_err_post1.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[1,
                                                               p_index1]**2))

                # Get the values for the second instance.
                time_slice = slice(star.fiberSplitIndex, None)
                w_mean, eotwm = get_weighted_mean(
                    star.pairSeparationsArray,
                    star.pairSepErrorsArray,
                    time_slice,
                    p_index2)
                pair_sep_post2.append(w_mean)
                pair_sep_err_post2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[1,
                                                               p_index2]**2))
                w_mean, eotwm = get_weighted_mean(
                    star.pairModelOffsetsArray,
                    star.pairModelErrorsArray,
                    time_slice,
                    p_index2)
                pair_model_post2.append(w_mean)
                pair_model_err_post2.append(
                    np.sqrt(eotwm**2 + star.pairSysErrorsArray[1,
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

    if star.hasObsPost:
        pair_sep_post1 = np.array(pair_sep_post1)
        pair_model_post1 = np.array(pair_model_post1)
        pair_sep_err_post1 = np.array(pair_sep_err_post1)
        pair_model_err_post1 = np.array(pair_model_err_post1)
        pair_sep_post2 = np.array(pair_sep_post2)
        pair_model_post2 = np.array(pair_model_post2)
        pair_sep_err_post2 = np.array(pair_sep_err_post2)
        pair_model_err_post2 = np.array(pair_model_err_post2)

    # Plot the results

    fig = plt.figure(figsize=(16, 12), tight_layout=True)
    gs = GridSpec(ncols=1, nrows=2, figure=fig,
                  height_ratios=(1, 1))
    ax_pre = fig.add_subplot(gs[0, 0])
    if (star.hasObsPre and star.hasObsPost):
        ax_post = fig.add_subplot(gs[1, 0],
                                  sharex=ax_pre)
    else:
        ax_post = fig.add_subplot(gs[1, 0])

    ax_post.set_xlabel('Pair index')
    # ax_post.set_ylabel(r'$\Delta$(Separation) post')
    ax_pre.set_ylabel(f'(Instance 2 – Instance 1) {star.numObsPre} obs'
                      ' (pre, m/s)')
    ax_post.set_ylabel(f'(Instance 2 – Instance 1) {star.numObsPost} obs'
                       ' (post, m/s)')

    order_boundaries = []
    for i in range(len(pair_order_numbers)):
        if i == 0:
            continue
        if pair_order_numbers[i-1] != pair_order_numbers[i]:
            order_boundaries.append(i - 0.5)

    for ax in (ax_pre, ax_post):
        # ax.yaxis.grid(which='major', color='Gray', alpha=0.7,
        #               linestyle='-')
        # ax.yaxis.grid(which='minor', color='Gray', alpha=0.6,
        #               linestyle='--')
        ax.axhline(y=0, linestyle='-', color='Gray')
        ax.set_xlim(left=-1, right=134)
        for b in order_boundaries:
            ax.axvline(x=b, linestyle=':', color='DimGray', alpha=0.75)

    add_star_information(star, ax_pre, (0.07, 0.49))

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

        pairs_chisq = calc_chi_squared_nu(remove_nans(pair_diffs),
                                          remove_nans(pair_errs), 1)
        model_chisq = calc_chi_squared_nu(remove_nans(model_diffs),
                                          remove_nans(model_errs), 1)
        pairs_sigma = np.nanstd(pair_diffs)
        model_sigma = np.nanstd(model_diffs)

        ax_pre.errorbar(pair_indices, pair_diffs,
                        yerr=pair_errs,
                        color='Chocolate', markeredgecolor='Black',
                        linestyle='', marker='o',
                        label=r'Pair $\chi^2_\nu$:'
                        f' {pairs_chisq:.2f}, RMS: {pairs_sigma:.2f}')
        ax_pre.errorbar(model_pair_indices, model_diffs,
                        yerr=model_errs,
                        color='MediumSeaGreen', markeredgecolor='Black',
                        linestyle='', marker='D',
                        label=r'Model $\chi^2_\nu$:'
                        f' {model_chisq:.2f}, RMS: {model_sigma:.2f}')
        ax_pre.legend()

    if star.hasObsPost:
        pair_diffs = pair_sep_post2 - pair_sep_post1
        pair_errs = np.sqrt(pair_sep_err_post1**2 + pair_sep_err_post2**2)
        model_diffs = pair_model_post2 - pair_model_post1
        model_errs = np.sqrt(pair_model_err_post1**2 + pair_model_err_post2**2)

        pairs_chisq = calc_chi_squared_nu(remove_nans(pair_diffs),
                                          remove_nans(pair_errs), 1)
        model_chisq = calc_chi_squared_nu(remove_nans(model_diffs),
                                          remove_nans(model_errs), 1)
        pairs_sigma = np.nanstd(pair_diffs)
        model_sigma = np.nanstd(model_diffs)

        ax_post.errorbar(pair_indices, pair_diffs,
                         yerr=pair_errs,
                         color='DodgerBlue', markeredgecolor='Black',
                         linestyle='', marker='o',
                         label=r'Pair $\chi^2_\nu$:'
                         f' {pairs_chisq:.2f}, RMS: {pairs_sigma:.2f}')
        ax_post.errorbar(model_pair_indices, model_diffs,
                         yerr=model_errs,
                         color='GoldenRod', markeredgecolor='Black',
                         linestyle='', marker='D',
                         label=r'Model $\chi^2_\nu$:'
                         f' {model_chisq:.2f}, RMS: {model_sigma:.2f}')
        ax_post.legend()

    # plt.show(fig)
    output_dir = Path('/Users/dberke/Pictures/duplicate_pairs')
    outfile = output_dir /\
        f'{star.name}_{star.radialVelocity.value:.2f}kms.png'
    fig.savefig(str(outfile))
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

    plots_dir = Path('/Users/dberke/Pictures/'
                     'pair_depth_differences_investigation')
    if not plots_dir.exists():
        os.mkdir(plots_dir)

    filename = vcl.output_dir /\
        'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
    fit_results_dict = get_params_file(filename)
    sigma_sys_dict = fit_results_dict['sigmas_sys']

    pair_depth_diffs = []
    pair_depth_means = []
    pair_model_sep_pre, pair_model_sep_post = [], []
    pair_model_err_pre, pair_model_err_post = [], []

    sigmas_sys_pre = []
    sigmas_sys_post = []

    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    for pair in tqdm(star.pairsList):
        for order_num in pair.ordersToMeasureIn:
            pair_label = '_'.join([pair.label, str(order_num)])
            col_index = star._pair_bidict[pair_label]
            label_high = '_'.join([pair._higherEnergyTransition.label,
                                  str(order_num)])
            label_low = '_'.join([pair._lowerEnergyTransition.label,
                                 str(order_num)])
            col_high = star._transition_bidict[label_high]
            col_low = star._transition_bidict[label_low]
            depths_high = remove_nans(star.normalizedDepthArray[:, col_high])
            depths_low = remove_nans(star.normalizedDepthArray[:, col_low])
            mean_high = np.nanmean(depths_high)
            mean_low = np.nanmean(depths_low)
            # h_d = pair._higherEnergyTransition.normalizedDepth
            # l_d = pair._lowerEnergyTransition.normalizedDepth
            # depth_diff = l_d - h_d
            pair_depth_means.append((mean_high + mean_low) / 2)
            pair_depth_diffs.append(abs(mean_low - mean_high))

            if star.hasObsPre:
                w_mean, eotwm = get_weighted_mean(star.pairModelOffsetsArray,
                                                  star.pairModelErrorsArray,
                                                  pre_slice, col_index)
                pair_model_sep_pre.append(w_mean)
                pair_model_err_pre.append(eotwm)
                sigmas_sys_pre.append(
                    sigma_sys_dict[pair_label + '_pre'].value)

            if star.hasObsPost:
                w_mean, eotwm = get_weighted_mean(star.pairModelOffsetsArray,
                                                  star.pairModelErrorsArray,
                                                  post_slice, col_index)
                pair_model_sep_post.append(w_mean)
                pair_model_err_post.append(eotwm)
                sigmas_sys_post.append(
                    sigma_sys_dict[pair_label + '_post'].value)

    pair_depth_diffs = np.array(pair_depth_diffs)
    pair_depth_means = np.array(pair_depth_means)
    pair_model_sep_pre = np.array(pair_model_sep_pre)
    pair_model_sep_post = np.array(pair_model_sep_post)
    pair_model_err_pre = np.array(pair_model_err_pre)
    pair_model_err_post = np.array(pair_model_err_post)
    sigmas_sys_pre = np.array(sigmas_sys_pre)
    sigmas_sys_post = np.array(sigmas_sys_post)

    # Plot as a function of pair depth separation.
    point_size = 18

    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=7, figure=fig,
                  height_ratios=(4.8, 1.5, 1.5, 1, 4.8, 1.5, 1.5),
                  width_ratios=(40, 1),
                  hspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_pre_chi = fig.add_subplot(gs[1, 0])
    ax_pre_wmean = fig.add_subplot(gs[2, 0])
    ax_post = fig.add_subplot(gs[4, 0])
    ax_post_chi = fig.add_subplot(gs[5, 0])
    ax_post_wmean = fig.add_subplot(gs[6, 0])
    ax_clb_pre = fig.add_subplot(gs[0, 1])
    ax_clb_post = fig.add_subplot(gs[4, 1])

    ax_pre.set_ylabel('Model-offset pre\n(m/s)')
    ax_post.set_ylabel('Model-offset post\n(m/s)')
    ax_pre_chi.set_ylabel(r'$\chi^2_\nu$')
    ax_post_chi.set_ylabel(r'$\chi^2_\nu$')
    ax_pre_wmean.set_ylabel('Weighted\nmean (m/s)')
    ax_post_wmean.set_ylabel('Weighted\nmean (m/s)')
    ax_post_wmean.set_xlabel('Pair normalized depth difference')

    ax_pre.set_xticklabels('')
    ax_pre_chi.set_xticklabels('')
    ax_post.set_xticklabels('')
    ax_post_chi.set_xticklabels('')

    for ax in (ax_pre, ax_post):
        ax.set_ylim(bottom=-150, top=150)
        ax.axhline(y=0, color='Gray', linestyle='--')
    for ax in (ax_pre_wmean, ax_post_wmean):
        ax.axhline(y=0, color='Gray', linestyle='--')
    for ax in (ax_pre, ax_post, ax_pre_wmean, ax_post_wmean,
               ax_pre_chi, ax_post_chi):
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.grid(which='major', color='Gray',
                      linestyle=':', alpha=0.8)
        ax.xaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.5)
        ax.set_xlim(left=-0.001, right=0.35)
    for ax in (ax_pre_chi, ax_post_chi):
        ax.axhline(y=1, color='Black', linestyle='--')
        ax.set_ylim(bottom=0, top=3)

    if star.hasObsPre:
        full_errs_pre = np.sqrt(pair_model_err_pre ** 2 +
                                sigmas_sys_pre ** 2)
        values, mask = remove_nans(pair_model_sep_pre, return_mask=True)
        chisq = calc_chi_squared_nu(values,
                                    full_errs_pre[mask], 1)
        ax_pre.errorbar(pair_depth_diffs, pair_model_sep_pre,
                        yerr=full_errs_pre,
                        linestyle='', marker='.',
                        color='Peru',
                        zorder=0,
                        label=r'$\chi^2_\nu$:'
                        f' {chisq:.2f}, {star.numObsPre} obs')
        clb_pre = ax_pre.scatter(pair_depth_diffs, pair_model_sep_pre,
                                 marker='o', s=point_size,
                                 c=pair_depth_means,
                                 cmap=cmr.get_sub_cmap('cmr.ember',
                                                       0.1, 0.85),
                                 zorder=2)
        fig.colorbar(clb_pre, ax=ax_pre, pad=0.01, cax=ax_clb_pre,
                     label='Mean pair depth')
        ax_pre.legend()

    if star.hasObsPost:
        full_errs_post = np.sqrt(pair_model_err_post ** 2 +
                                 sigmas_sys_post ** 2)
        values, mask = remove_nans(pair_model_sep_post, return_mask=True)
        chisq = calc_chi_squared_nu(values,
                                    full_errs_post[mask], 1)
        ax_post.errorbar(pair_depth_diffs, pair_model_sep_post,
                         yerr=full_errs_post,
                         linestyle='', marker='.',
                         color='DeepSkyBlue',
                         zorder=0)
        clb_post = ax_post.scatter(pair_depth_diffs, pair_model_sep_post,
                                   marker='o', s=point_size,
                                   c=pair_depth_means,
                                   cmap=cmr.get_sub_cmap('cmr.cosmic',
                                                         0.1, 0.85),
                                   zorder=2,
                                   label=r'$\chi^2_\nu$:'
                                   f' {chisq:.2f}, {star.numObsPost} obs')
        fig.colorbar(clb_post, ax=ax_post, pad=0.01, cax=ax_clb_post,
                     label='Mean pair depth')
        ax_post.legend()

    add_star_information(star, ax_pre_wmean, (0.1, 0.51))

    # Get results for bins.
    bin_lims = np.linspace(0, 0.35, 15)

    if star.hasObsPre:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(bin_lims)), pairwise(bin_lims)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((pair_depth_diffs > lims[0]) &
                            (pair_depth_diffs < lims[1]))
            values, nan_mask = remove_nans(pair_model_sep_pre[mask],
                                           return_mask=True)
            errs = full_errs_pre[mask][nan_mask]
            try:
                w_mean, eotwm = weighted_mean_and_error(values, errs)
            except ZeroDivisionError:
                w_mean, eotwm = np.nan, np.nan
            w_means.append(w_mean)
            eotwms.append(eotwm)

            chisq.append(calc_chi_squared_nu(values, errs, 1))
        ax_pre_wmean.errorbar(midpoints, w_means, yerr=eotwms,
                              color='Green')
        ax_pre_chi.plot(midpoints, chisq, color='SaddleBrown',
                        marker='o', markersize=5)

    if star.hasObsPost:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(bin_lims)), pairwise(bin_lims)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((pair_depth_diffs > lims[0]) &
                            (pair_depth_diffs < lims[1]))
            values, nan_mask = remove_nans(pair_model_sep_post[mask],
                                           return_mask=True)
            errs = full_errs_post[mask][nan_mask]
            try:
                w_mean, eotwm = weighted_mean_and_error(values, errs)
            except ZeroDivisionError:
                w_mean, eotwm = np.nan, np.nan
            w_means.append(w_mean)
            eotwms.append(eotwm)

            chisq.append(calc_chi_squared_nu(values, errs, 1))
        ax_post_wmean.errorbar(midpoints, w_means, yerr=eotwms,
                               color='Green')
        ax_post_chi.plot(midpoints, chisq, color='RoyalBlue',
                         marker='o', markersize=5)
    # plt.show(fig)
    outfile = plots_dir /\
        f'{star.name}_{star.numObs}_obs_by_depth_difference.png'
    fig.savefig(str(outfile))
    plt.close('all')

    # Plot as a function of mean pair depth.
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    gs = GridSpec(ncols=2, nrows=7, figure=fig,
                  height_ratios=(4.8, 1.8, 1.9, 1, 4.8, 1.8, 1.9),
                  width_ratios=(40, 1),
                  hspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_pre_chi = fig.add_subplot(gs[1, 0])
    ax_pre_wmean = fig.add_subplot(gs[2, 0])
    ax_post = fig.add_subplot(gs[4, 0])
    ax_post_chi = fig.add_subplot(gs[5, 0])
    ax_post_wmean = fig.add_subplot(gs[6, 0])
    ax_clb_pre = fig.add_subplot(gs[0, 1])
    ax_clb_post = fig.add_subplot(gs[4, 1])

    ax_pre.set_ylabel('Model offset pre\n(m/s)')
    ax_post.set_ylabel('Model offset post\n(m/s))')
    ax_pre_chi.set_ylabel(r'$\chi^2_\nu$')
    ax_post_chi.set_ylabel(r'$\chi^2_\nu$')
    ax_pre_wmean.set_ylabel('Weighted\nmean (m/s)')
    ax_post_wmean.set_ylabel('Weighted\nmean (m/s)')
    ax_post_wmean.set_xlabel('Pair mean depth')

    ax_pre.set_xticklabels('')
    ax_pre_chi.set_xticklabels('')
    ax_post.set_xticklabels('')
    ax_post_chi.set_xticklabels('')

    for ax in (ax_pre, ax_post):
        ax.set_ylim(bottom=-150, top=150)
        ax.axhline(y=0, color='Gray', linestyle='--')
    for ax in (ax_pre_wmean, ax_post_wmean):
        ax.axhline(y=0, color='Gray', linestyle='--')
    for ax in (ax_pre, ax_post, ax_pre_wmean, ax_post_wmean,
               ax_pre_chi, ax_post_chi):
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.grid(which='major', color='Gray',
                      linestyle=':', alpha=0.65)
        ax.xaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.5)
        ax.set_xlim(left=0, right=1)
    for ax in (ax_pre_chi, ax_post_chi):
        ax.axhline(y=1, color='Black', linestyle='--')
        ax.set_ylim(bottom=0, top=2)

    if star.hasObsPre:
        full_errs_pre = np.sqrt(pair_model_err_pre ** 2 +
                                sigmas_sys_pre ** 2)
        values, mask = remove_nans(pair_model_sep_pre, return_mask=True)
        chisq = calc_chi_squared_nu(values,
                                    full_errs_pre[mask], 1)
        ax_pre.errorbar(pair_depth_means, pair_model_sep_pre,
                        yerr=full_errs_pre,
                        linestyle='', marker='.',
                        color='Peru',
                        zorder=0,
                        label=r'$\chi^2_\nu$:'
                        f' {chisq:.2f}, {star.numObsPre} obs')
        clb_pre = ax_pre.scatter(pair_depth_means, pair_model_sep_pre,
                                 marker='o', s=point_size,
                                 c=pair_depth_diffs,
                                 cmap=cmr.get_sub_cmap('cmr.ember',
                                                       0.1, 0.85),
                                 zorder=2)
        fig.colorbar(clb_pre, pad=0.01, cax=ax_clb_pre,
                     label=r'Mean pair depth $\Delta$')
        ax_pre.legend()

    if star.hasObsPost:
        full_errs_post = np.sqrt(pair_model_err_post ** 2 +
                                 sigmas_sys_post ** 2)
        values, mask = remove_nans(pair_model_sep_post, return_mask=True)
        chisq = calc_chi_squared_nu(values,
                                    full_errs_post[mask], 1)
        ax_post.errorbar(pair_depth_means, pair_model_sep_post,
                         yerr=full_errs_post,
                         linestyle='', marker='.',
                         color='DeepSkyBlue',
                         zorder=0)
        clb_post = ax_post.scatter(pair_depth_means, pair_model_sep_post,
                                   marker='o', s=point_size,
                                   c=pair_depth_diffs,
                                   cmap=cmr.get_sub_cmap('cmr.cosmic',
                                                         0.1, 0.85),
                                   zorder=2,
                                   label=r'$\chi^2_\nu$:'
                                   f' {chisq:.2f}, {star.numObsPost} obs')
        fig.colorbar(clb_post, pad=0.01, cax=ax_clb_post,
                     label=r'Mean pair depth $\Delta$')
        ax_post.legend()

    add_star_information(star, ax_pre_wmean, (0.1, 0.5))

    # Get results for bins.
    # bin_lims = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    bin_lims = np.linspace(0, 1, 11)

    if star.hasObsPre:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(bin_lims)), pairwise(bin_lims)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((pair_depth_means > lims[0]) &
                            (pair_depth_means < lims[1]))
            values, nan_mask = remove_nans(pair_model_sep_pre[mask],
                                           return_mask=True)
            errs = full_errs_pre[mask][nan_mask]
            try:
                w_mean, eotwm = weighted_mean_and_error(values, errs)
            except ZeroDivisionError:
                w_mean, eotwm = np.nan, np.nan
            w_means.append(w_mean)
            eotwms.append(eotwm)

            chisq.append(calc_chi_squared_nu(values, errs, 1))
        ax_pre_wmean.errorbar(midpoints, w_means, yerr=eotwms,
                              color='Green')
        ax_pre_chi.plot(midpoints, chisq, color='SaddleBrown',
                        marker='.')

    if star.hasObsPost:
        midpoints, w_means, eotwms, chisq = [], [], [], []
        for i, lims in zip(range(len(bin_lims)), pairwise(bin_lims)):
            midpoints.append((lims[1] + lims[0]) / 2)
            mask = np.where((pair_depth_means > lims[0]) &
                            (pair_depth_means < lims[1]))
            values, nan_mask = remove_nans(pair_model_sep_post[mask],
                                           return_mask=True)
            errs = full_errs_post[mask][nan_mask]
            try:
                w_mean, eotwm = weighted_mean_and_error(values, errs)
            except ZeroDivisionError:
                w_mean, eotwm = np.nan, np.nan
            w_means.append(w_mean)
            eotwms.append(eotwm)

            chisq.append(calc_chi_squared_nu(values, errs, 1))
        ax_post_wmean.errorbar(midpoints, w_means, yerr=eotwms,
                               color='Green')
        ax_post_chi.plot(midpoints, chisq, color='RoyalBlue',
                         marker='.')
    # plt.show(fig)
    outfile = plots_dir / f'{star.name}_{star.numObs}_obs_by_mean_depth.png'
    fig.savefig(str(outfile))
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
    plots_dir = Path('/Users/dberke/Pictures/blendedness_investigation')

    sorted_blend_tuples = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                           (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                           (2, 2), (2, 3), (2, 4), (2, 5),
                           (3, 3), (3, 4), (3, 5),
                           (4, 4), (4, 5),
                           (5, 5))

    sorted_blend_tuples = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                           (1, 1), (1, 2), (1, 3), (1, 4),
                           (2, 2), (2, 3), (2, 4),
                           (3, 3), (3, 4),
                           (4, 4))

    sorted_blend_tuples = ((0, 0), (0, 1), (0, 2), (0, 3),
                           (1, 1), (1, 2), (1, 3),
                           (2, 2), (2, 3),
                           (3, 3))

    sorted_blend_tuples = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]

    pre_slice = slice(None, star.fiberSplitIndex)
    post_slice = slice(star.fiberSplitIndex, None)

    pair_blends_dict = {}
    total_pairs = 0
    for pair in star.pairsList:
        for order_num in pair.ordersToMeasureIn:
            total_pairs += 1
            pair_label = '_'.join((pair.label, str(order_num)))
            pair_blends_dict[pair_label] = pair.blendTuple

    sorted_means_pre, sorted_errs_pre = [], []
    sorted_means_post, sorted_errs_post = [], []
    chi_squareds_pre, chi_squareds_post = [], []
    sigmas_pre, sigmas_post = [], []
    divisions = []
    total = 0
    per_bin_wmeans_pre, per_bin_wmeans_post = [], []
    per_bin_errs_pre, per_bin_errs_post = [], []

    for blend_tuple in sorted_blend_tuples:
        bin_means_pre, bin_errs_pre = [], []
        bin_means_pre_nn, bin_errors_pre_nn = [], []
        bin_means_post, bin_errs_post = [], []
        bin_means_post_nn, bin_means_post_nn = [], []
        for pair_label, value in pair_blends_dict.items():
            col_index = star.p_index(pair_label)
            if blend_tuple == value:
                total += 1
                if star.hasObsPre:
                    w_mean, eotwm = get_weighted_mean(
                        star.pairModelOffsetsArray,
                        star.pairModelErrorsArray,
                        pre_slice, star._pair_bidict[pair_label])
                    bin_means_pre.append(w_mean)
                    bin_errs_pre.append(
                        np.sqrt(eotwm**2 +
                                star.pairSysErrorsArray[0, col_index]**2))
                if star.hasObsPost:
                    w_mean, eotwm = get_weighted_mean(
                        star.pairModelOffsetsArray,
                        star.pairModelErrorsArray,
                        post_slice, star._pair_bidict[pair_label])
                    bin_means_post.append(w_mean)
                    bin_errs_post.append(
                        np.sqrt(eotwm**2 +
                                star.pairSysErrorsArray[1, col_index]**2))
        divisions.append(total - 0.5)
        if star.hasObsPre:
            bin_means_pre_nn, mask_pre = remove_nans(np.array(bin_means_pre),
                                                     return_mask=True)
            bin_errs_pre_nn = np.array(bin_errs_pre)[mask_pre]

        if star.hasObsPost:
            bin_means_post_nn, mask_post = remove_nans(np.array(bin_means_post),
                                                       return_mask=True)
            bin_errs_post_nn = np.array(bin_errs_post)[mask_post]

        if len(bin_means_pre_nn) >= 2:

            chi_squareds_pre.append(calc_chi_squared_nu(
                bin_means_pre_nn, bin_errs_pre_nn, 1))
            sigmas_pre.append(np.nanstd(bin_means_pre))
            wmean, eotwm = weighted_mean_and_error(bin_means_pre_nn,
                                                   bin_errs_pre_nn)
            per_bin_wmeans_pre.append(wmean)
            per_bin_errs_pre.append(eotwm)
        else:
            chi_squareds_pre.append(np.nan)
            sigmas_pre.append(np.nan)
            per_bin_wmeans_pre.append(np.nan)
            per_bin_errs_pre.append(np.nan)

        if len(bin_means_post_nn) >= 2:
            chi_squareds_post.append(calc_chi_squared_nu(
                bin_means_post_nn, bin_errs_post_nn, 1))
            sigmas_post.append(np.nanstd(bin_means_post))
            wmean, eotwm = weighted_mean_and_error(bin_means_post_nn,
                                                   bin_errs_post_nn)
            per_bin_wmeans_post.append(wmean)
            per_bin_errs_post.append(eotwm)
        else:
            chi_squareds_post.append(np.nan)
            sigmas_post.append(np.nan)
            per_bin_wmeans_post.append(np.nan)
            per_bin_errs_post.append(np.nan)

        sorted_means_pre.extend(bin_means_pre)
        sorted_errs_pre.extend(bin_errs_pre)
        sorted_means_post.extend(bin_means_post)
        sorted_errs_post.extend(bin_errs_post)

    sorted_means_pre = np.array(sorted_means_pre)
    sorted_errs_pre = np.array(sorted_errs_pre)
    sorted_means_post = np.array(sorted_means_post)
    sorted_errs_post = np.array(sorted_errs_post)

    # Plot the results.
    fig = plt.figure(figsize=(18, 10), tight_layout=True)
    gs = GridSpec(ncols=1, nrows=7, figure=fig,
                  height_ratios=(3.1, 1, 1, 0.6, 3.1, 1, 1), hspace=0)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[4, 0])
    ax_chi_pre = fig.add_subplot(gs[1, 0])
    ax_chi_post = fig.add_subplot(gs[5, 0])
    ax_wmean_pre = fig.add_subplot(gs[2, 0])
    ax_wmean_post = fig.add_subplot(gs[6, 0])

    ax_pre.set_ylabel('Model-corrected pair\noffsets pre (m/s)')
    ax_post.set_ylabel('Model-corrected pair\noffsets post (m/s)')
    ax_chi_pre.set_ylabel(r'$\chi^2_\nu$')
    ax_chi_post.set_ylabel(r'$\chi^2_\nu$')
    ax_wmean_pre.set_ylabel('Weighted\nmean\n/sigma (m/s)')
    ax_wmean_post.set_ylabel('Weighted\nmean\n/sigma (m/s)')
    label_positions = (0.01, 0.155, 0.315, 0.49, 0.85, 0.99)
    label_positions = [(div / total) - np.sqrt(div / total) * 0.005
                       for div in divisions]

    for ax in (ax_pre, ax_post, ax_chi_pre, ax_chi_post,
               ax_wmean_pre, ax_wmean_post):
        # Set the x-limits for the plot
        ax.set_xlim(left=-1, right=total)
        ax.axhline(y=0, color='Black', alpha=1, linestyle='--')
    for ax in (ax_wmean_pre, ax_wmean_post):
        ax.set_ylim(bottom=-40, top=40)
    for ax in (ax_chi_pre, ax_chi_post):
        ax.axhline(y=1, color='Black', linestyle='--')
        ax.set_ylim(bottom=0, top=2)
    for ax in (ax_pre, ax_post):
        ax.set_ylim(bottom=-110, top=110)
        for div, b_tuple, label_pos, sig_pre, sig_post in zip(
                divisions, sorted_blend_tuples, label_positions,
                sigmas_pre, sigmas_post):
            ax.axvline(x=div, color='DarkSlateGray', alpha=0.8)
            ax_pre.annotate(f'{sig_pre:.2f} m/s',
                            xy=(div, 0),
                            xytext=(label_pos, 0.01),
                            textcoords='axes fraction',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            rotation=90)
            ax_pre.annotate(f'{b_tuple}',
                            xy=(div, 0), xytext=(label_pos, 0.99),
                            textcoords='axes fraction',
                            horizontalalignment='center',
                            verticalalignment='top',
                            rotation=90)
            ax_post.annotate(f'{sig_post:.2f} m/s',
                             xy=(div, 0),
                             xytext=(label_pos, 0.01),
                             textcoords='axes fraction',
                             horizontalalignment='center',
                             verticalalignment='bottom',
                             rotation=90)
            ax_post.annotate(f'{b_tuple}',
                             xy=(div, 0), xytext=(label_pos, 0.99),
                             textcoords='axes fraction',
                             horizontalalignment='center',
                             verticalalignment='top',
                             rotation=90)

    add_star_information(star, ax_wmean_pre, (0.07, 0.5))

    indices = [x for x in range(total)]
    boundaries = [0]
    boundaries.extend(divisions)
    bin_mids = [(a + b) / 2 for a, b in pairwise(boundaries)]

    max_tuple = sorted_blend_tuples[-1]
    legend_loc = {(2, 2): 0.51,
                  (3, 3): 0.50,
                  (4, 4): 0.41,
                  (5, 5): 0.38}

    if star.hasObsPre:
        values, mask = remove_nans(sorted_means_pre, return_mask=True)
        chisq_pre = calc_chi_squared_nu(values,
                                        sorted_errs_pre[mask], 1)
        ax_pre.errorbar(x=indices, y=sorted_means_pre, yerr=sorted_errs_pre,
                        color='Chocolate', markeredgecolor='Black',
                        ecolor='Peru',
                        linestyle='', marker='o', markersize=4,
                        label=f'{star.numObsPre} obs,'
                        r' $\chi^2_\nu$:'
                        f' {chisq_pre:.2f}')
        ax_chi_pre.plot(bin_mids, chi_squareds_pre,
                        color='SaddleBrown', linestyle='-',
                        marker='.')
        ax_wmean_pre.errorbar(x=bin_mids, y=per_bin_wmeans_pre,
                              yerr=per_bin_errs_pre,
                              linestyle='-', color='Black',
                              marker='.')
        ax_wmean_pre.plot(bin_mids, sigmas_pre,
                          linestyle='-', color='Green',
                          marker='.')
        ax_pre.legend(loc=(legend_loc[max_tuple], 0.01))

    if star.hasObsPost:
        values, mask = remove_nans(sorted_means_post, return_mask=True)
        chisq_post = calc_chi_squared_nu(values,
                                         sorted_errs_post[mask], 1)
        ax_post.errorbar(x=indices, y=sorted_means_post, yerr=sorted_errs_post,
                         color='DodgerBlue', markeredgecolor='Black',
                         ecolor='DeepSkyBlue',
                         linestyle='', marker='o', markersize=4,
                         label=f'{star.numObsPost} obs,'
                         r' $\chi^2_\nu$:'
                         f' {chisq_post:.2f}')
        ax_chi_post.plot(bin_mids, chi_squareds_post,
                         color='RoyalBlue', linestyle='-',
                         marker='.')
        ax_wmean_post.errorbar(x=bin_mids, y=per_bin_wmeans_post,
                               yerr=per_bin_errs_post,
                               linestyle='-', color='Black',
                               marker='.')
        ax_wmean_post.plot(bin_mids, sigmas_post,
                           linestyle='-', color='Green',
                           marker='.')
        ax_post.legend(loc=(legend_loc[max_tuple], 0.01))

    filename = plots_dir / f'{star.name}_by_blendedness_{max_tuple}.png'
    if not plots_dir.exists():
        os.mkdir(plots_dir)
    fig.savefig(str(filename))
    plt.close('all')
    # plt.show()


def plot_vs_radial_velocity(star_list):
    """
    Plot pair offsets vs. radial velocity.

    Parameters
    ----------
    star_list : list of star.Star
        A list of stars to use to create the plot.

    Returns
    -------
    None.

    """

    plots_dir = Path('/Users/dberke/Pictures/'
                     'sample_radial_velocity_dependence')

    pairs_to_use = ('4652.593Cr1_4653.460Cr1_29',
                    '4652.593Cr1_4653.460Cr1_30',
                    '4759.449Ti1_4760.600Ti1_32',
                    '4759.449Ti1_4760.600Ti1_33',
                    '4799.873Ti2_4800.072Fe1_33',
                    '4799.873Ti2_4800.072Fe1_34',
                    '5138.510Ni1_5143.171Fe1_42',
                    '5187.346Ti2_5200.158Fe1_43',
                    '6123.910Ca1_6138.313Fe1_60',
                    '6123.910Ca1_6139.390Fe1_60',
                    '6138.313Fe1_6139.390Fe1_60',
                    '6153.320Fe1_6155.928Na1_61',
                    '6153.320Fe1_6162.452Na1_61',
                    '6153.320Fe1_6168.150Ca1_61',
                    '6155.928Na1_6162.452Na1_61',
                    '6162.452Na1_6168.150Ca1_61',
                    '6162.452Na1_6175.044Fe1_61',
                    '6168.150Ca1_6175.044Fe1_61',
                    '6192.900Ni1_6202.028Fe1_61',
                    '6242.372Fe1_6244.834V1_62')

    pairs_to_use = ('4575.498Fe1_4576.000Fe1_27',
                    '4652.593Cr1_4653.460Cr1_30',
                    '6138.313Fe1_6139.390Fe1_60',
                    '5589.125Fe1_5589.410Ni1_50')
    # pairs_to_use = ('5589.125Fe1_5589.410Ni1_50',)
    x_limits = {'4575.498Fe1_4576.000Fe1_27': (2380, 2650),
                '4652.593Cr1_4653.460Cr1_30': (330, 560),
                '5589.125Fe1_5589.410Ni1_50': (3490, 3780),
                '6138.313Fe1_6139.390Fe1_60': (2870, 3140)}
    y_limits = {'4575.498Fe1_4576.000Fe1_27': (-150, 150),
                '4652.593Cr1_4653.460Cr1_30': (-100, 100),
                '5589.125Fe1_5589.410Ni1_50': (-200, 200),
                '6138.313Fe1_6139.390Fe1_60': (-75, 75)}
    boundaries = {'4575.498Fe1_4576.000Fe1_27': 2560,
                  '4652.593Cr1_4653.460Cr1_30': 512,
                  '5589.125Fe1_5589.410Ni1_50': 3584,
                  '6138.313Fe1_6139.390Fe1_60': 3072}
    bin_limits = {'4575.498Fe1_4576.000Fe1_27': (2385, 2660),
                  '4652.593Cr1_4653.460Cr1_30': (287, 587),
                  '5589.125Fe1_5589.410Ni1_50': (3484, 3809),
                  '6138.313Fe1_6139.390Fe1_60': (2872, 3172)}

    # pairs_to_use = []
    # for pair in tqdm(pairs_list):
    #     for order_num in pair.ordersToMeasureIn:
    #         pair_label = '_'.join((pair.label, str(order_num)))
    #         pairs_to_use.append(pair_label)

    for pair_label in tqdm(pairs_to_use):

        pair_seps_pre, offsets_pre = [], []
        errors_pre, pixels_pre = [], []
        pair_seps_post, offsets_post = [], []
        errors_post, pixels_post = [], []

        parts = pair_label.split('_')
        blue_label = '_'.join((parts[0], parts[2]))
        red_label = '_'.join((parts[1], parts[2]))

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

            if star.hasObsPost:

                pair_seps_post.extend(star.pairSeparationsArray[
                    post_slice, col_index].to(u.km/u.s))
                offsets_post.extend(star.pairModelOffsetsArray[
                    post_slice, col_index].to(u.m/u.s))
                errors_post.extend(star.pairModelErrorsArray[post_slice,
                                                             col_index])
                pixels_post.extend(star.pixelArray[
                    post_slice, star.t_index(blue_label)])

            if (len(offsets_pre) != len(pixels_pre)) or\
               (len(offsets_pre) != len(pair_seps_pre)):
                print(star.name)
                exit(1)

        offsets_pre, mask_pre = remove_nans(np.array(offsets_pre),
                                            return_mask=True)
        offsets_post, mask_post = remove_nans(np.array(offsets_post),
                                              return_mask=True)
        errors_pre = np.array(errors_pre)[mask_pre]
        errors_post = np.array(errors_post)[mask_post]
        pixels_pre = np.array(pixels_pre)[mask_pre]
        pixels_post = np.array(pixels_post)[mask_post]
        pair_seps_pre = np.array(pair_seps_pre)[mask_pre]
        pair_seps_post = np.array(pair_seps_post)[mask_post]
        mean_sep_pre = np.mean(pair_seps_pre)
        mean_sep_post = np.mean(pair_seps_post)
        # print(mean_sep_pre)
        # print(mean_sep_post)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        gs = GridSpec(nrows=5, ncols=1, figure=fig,
                      height_ratios=(1, 0.3, 0.1, 1, 0.3), hspace=0)
        ax_pre = fig.add_subplot(gs[0, 0])
        ax_post = fig.add_subplot(gs[3, 0])
        ax_mean_pre = fig.add_subplot(gs[1, 0], sharex=ax_pre)
        ax_mean_post = fig.add_subplot(gs[4, 0], sharex=ax_post,
                                       sharey=ax_mean_pre)

        x_lims = x_limits[pair_label]
        y_lims = y_limits[pair_label]
        boundary_pix = boundaries[pair_label]

        for ax in (ax_pre, ax_post):
            ax.set_xlim(left=x_lims[0], right=x_lims[1])
            ax.set_ylim(bottom=y_lims[0], top=y_lims[1])
            ax.axvline(x=boundary_pix, linestyle='--', color='CadetBlue',
                       label='Blue crosses')

        for ax in (ax_pre, ax_post, ax_mean_pre, ax_mean_post):
            ax.axhline(y=0, linestyle='--', color='Gray')

        # for ax in (ax_mean_pre, ax_mean_post):
        #     ax.set_ylim(bottom=-7, top=7)

        ax_pre.axvline(boundary_pix - np.round(mean_sep_pre / 0.829),
                       linestyle=':', color='IndianRed',
                       label='Red crosses')
        ax_post.axvline(boundary_pix - np.round(mean_sep_post / 0.829),
                        linestyle=':', color='IndianRed',
                        label='Red crosses')

        ax_pre.set_ylabel('Pre model offset (m/s)')
        ax_post.set_ylabel('Post model offset (m/s)')
        ax_mean_pre.set_ylabel(r'$\mu$')
        ax_mean_post.set_ylabel(r'$\mu$')
        ax_post.set_xlabel('Pixel of blue transitions')

        ax_pre.errorbar(pixels_pre, offsets_pre,
                        # yerr=errors_pre,
                        linestyle='',
                        markeredgecolor=None,
                        marker='.', color='Chocolate',
                        alpha=0.4)
        ax_post.errorbar(pixels_post, offsets_post,
                         # yerr=errors_post,
                         linestyle='',
                         markeredgecolor=None,
                         marker='.', color='DodgerBlue',
                         alpha=0.4)

        # ax_pre.legend(loc='lower left')
        # ax_post.legend(loc='lower left')

        # Create some bins to measure in:
        midpoints = []
        means_pre, eotms_pre = [], []
        means_post, eotms_post = [], []
        print(bin_limits[pair_label])
        bin_lims = [i for i in range(bin_limits[pair_label][0],
                                     bin_limits[pair_label][1], 25)]
        for lims in tqdm(pairwise(bin_lims)):
            mask_pre = np.where((pixels_pre > lims[0]) &
                                (pixels_pre < lims[1]))
            mask_post = np.where((pixels_post > lims[0]) &
                                 (pixels_post < lims[1]))

            midpoints.append((lims[0] + lims[1])/2)

            num_pre = len(offsets_pre[mask_pre])
            num_post = len(offsets_post[mask_post])

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

            if num_post > 1:
                means_post.append(np.mean(offsets_post[mask_post]))
                eotms_post.append(np.std(offsets_post[mask_post]) /
                                  np.sqrt(len(offsets_post[mask_post])))
            elif num_post == 1:
                means_post.append(offsets_post[mask_post][0])
                eotms_post.append(errors_post[mask_post][0])
            else:
                means_post.append(np.nan)
                eotms_post.append(np.nan)

        ax_mean_pre.errorbar(midpoints, means_pre,
                             yerr=eotms_pre,
                             color='Black', marker='x')
        ax_mean_post.errorbar(midpoints, means_post,
                              yerr=eotms_post,
                              color='Black', marker='x')

        plot_name = plots_dir / f'{pair_label}_vs_pixel.png'
        fig.savefig(str(plot_name))
        plt.close('all')


def get_weighted_mean(values_array, errs_array, time_slice, col_index):
    """
    Get the weighted mean of a column in an array avoiding NaNs.

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


def add_star_information(star, axis, coords):
    """Write information on a star to a plot.


    Parameters
    ----------
    star : `varconlib.star.Star`
        The star to get information from.
    axis : `matplotlib Axes` instance
        The axis to attach the information to.
    coords : tuple
        A 2-tuple of floats between 0 and 1, denoting the figure coordinates
        where to place the information.

    Returns
    -------
    None.

    """

    information = [r'T$_\mathrm{eff}$:' + fr' {star.temperature}',
                   fr'[Fe/H]: {star.metallicity}',
                   r'$\log{g}$:' + fr' {star.logg}']
    for key, value in star.specialAttributes.items():
        information.append(f'{key}: {value}'.replace('_', '-'))

    info = '      '.join(information)
    axis.annotate(info, coords,
                  xycoords='figure fraction',
                  fontsize=18)


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

parser.add_argument('stars', nargs='*', default=None, const=None, type=str,
                    help='A list of stars to make plots from. If not given'
                    ' will default to using all stars.')

parser.add_argument('-m', '--model', type=str, action='store',
                    help='The name of a model to test against.')
parser.add_argument('--sigma', type=float, action='store',
                    help='The number of sigma at which to trim outliers.')

parser.add_argument('--heliocentric-distance', action='store_true',
                    help='Plot as a function of distance from the Sun.')
parser.add_argument('--galactocentric-distance', action='store_true',
                    help='Plot as a function of distance from galactic'
                    ' center.')
parser.add_argument('--sigma-sys-vs-pair-separations', action='store_true',
                    help='Plot the sigma_sys for each pair as a function'
                    ' of its weighted-mean separation.')
parser.add_argument('--model-diff-vs-pair-separations', action='store_true',
                    help='Plot the model difference for each pair as a'
                    " function of it's weighted-mean separation.")
parser.add_argument('--plot-duplicate-pairs', action='store_true',
                    help='Plot differences in measured and model-corrected'
                    ' pair separations for duplicate pairs for a given star.')
parser.add_argument('--plot-depth-differences', action='store_true',
                    help='Create a plot of systematic differences as a'
                    ' function of pair depth differences.')
parser.add_argument('--plot-vs-blendedness', action='store_true',
                    help='Plot pairs sorted by blendedness.')
parser.add_argument('--plot-vs-radial-velocity', action='store_true',
                    help='Plot pair linear slopes as a function of radial'
                    ' velocity.')

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

start_time = time.time()

# Get the star from the name.
if len(args.stars) == 1:
    star = get_star(args.stars[0])
else:
    stars = [get_star(star_name) for star_name in tqdm(args.stars)]

csv_dir = vcl.output_dir / 'pair_separation_files'

star_properties_file = csv_dir / 'star_properties.csv'

star_data = pd.read_csv(star_properties_file)

# Import the list of pairs to use.
with open(vcl.final_pair_selection_file, 'r+b') as f:
    pairs_list = pickle.load(f)

pairs_dict = {}
for pair in pairs_list:
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

if args.stars is not None and args.pair_label:
    plot_pair_stability(star, args.pair_label)

if args.stars is not None and args.sigma_sys_vs_pair_separations:
    plot_sigma_sys_vs_pair_separation(star)

# Create plots as a function of pair separation distance.
if args.stars is not None and args.model is not None\
        and args.sigma is not None and args.model_diff_vs_pair_separations:
    if len(args.stars) == 1:
        plot_model_diff_vs_pair_separation(star, args.model.replace('-', '_'),
                                           n_sigma=args.sigma)
    elif len(args.stars) > 1:
        results_file = vcl.output_dir /\
            f'pair_separation_files/star_pair_separation_{args.sigma}sigma.csv'
        results = []
        for star in tqdm(args.stars):
            results.append(plot_model_diff_vs_pair_separation(
                get_star(star), args.model.replace('-', '_'),
                n_sigma=args.sigma))
        with open(results_file, 'w', newline='') as f:
            datawriter = csv.writer(f)
            header = ('#star_name',
                      '#obs_pre', 'chisq_pre',
                      'w_mean_pre', 'eotwm_pre',
                      '#obs_post', 'chisq_post',
                      'w_mean_post', 'eotwm_post')
            datawriter.writerow(header)
            for row in results:
                datawriter.writerow(row)

if args.stars is not None and args.plot_duplicate_pairs:
    if len(args.stars) == 1:
        plot_duplicate_pairs(star)
    elif len(args.stars) > 1:
        for star in tqdm(args.stars):
            plot_duplicate_pairs(get_star(star))

if args.stars is not None and args.plot_depth_differences:
    if len(args.stars) == 1:
        plot_pair_depth_differences(star)
    if len(args.stars) > 1:
        t_map(plot_pair_depth_differences, stars)
        # for star_name in tqdm(args.stars):
        #     plot_pair_depth_differences(get_star(star_name))

if args.stars is not None and args.plot_vs_blendedness:
    if len(args.stars) == 1:
        plot_vs_pair_blendedness(star)
    if len(args.stars) > 1:
        p_map(plot_vs_pair_blendedness, stars)
        # for star_name in tqdm(args.stars):
        #     plot_vs_pair_blendedness(get_star(star_name))

if args.stars is not None and args.plot_vs_radial_velocity:
    plot_vs_radial_velocity(stars)

duration = time.time() - start_time
tqdm.write(f'Finished in {duration:.1f} seconds.')
