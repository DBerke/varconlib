#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:41:51 2020

@author: dberke

A script to check if sigma_sys changes as a function of stellar parameters or
not.

"""

import argparse
from inspect import signature
from pathlib import Path
import pickle

import h5py
import hickle
from itertools import tee
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting.fitting as fit
from varconlib.scripts.multi_fit_stars import plot_data_points


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def create_parameter_comparison_figures(ylims=None,
                                        temp_lims=(5300 * u.K, 6200 * u.K),
                                        mtl_lims=(-0.75, 0.4),
                                        mag_lims=(4, 5.8),
                                        logg_lims=(4.1, 4.6)):
    """Create and returns a figure with pre-set subplots.

    This function creates the background figure and subplots for use with the
    --compare-stellar-parameter-* flags.

    Optional
    ----------
    ylims : 2-tuple of floats or ints
        A tuple of length 2 containing the upper and lower limits of the
        subplots in the figure.
    temp_lims : 2-tuple of floats or ints (optional dimensions of temperature)
        A tuple of length containing upper and lower limits for the x-axis of
        the temperature subplot.
    mtl_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the metallicity subplot.
    mag_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the absolute magnitude subplot.
    logg_lims : 2-tuple of floats or ints
        A tuple of length containing upper and lower limits for the x-axis of
        the log(g) subplot.

    Returns
    -------
    tuple
        A tuple containing the figure itself and the various axes of the
        subplots within it.

    """

    comp_fig = plt.figure(figsize=(12, 8), tight_layout=True)
    gs = GridSpec(ncols=3, nrows=2, figure=comp_fig,
                  width_ratios=(5, 5, 5))

    temp_ax_pre = comp_fig.add_subplot(gs[0, 0])
    temp_ax_post = comp_fig.add_subplot(gs[1, 0],
                                        sharex=temp_ax_pre,
                                        sharey=temp_ax_pre)
    mtl_ax_pre = comp_fig.add_subplot(gs[0, 1],
                                      sharey=temp_ax_pre)
    mtl_ax_post = comp_fig.add_subplot(gs[1, 1],
                                       sharex=mtl_ax_pre,
                                       sharey=mtl_ax_pre)
    mag_ax_pre = comp_fig.add_subplot(gs[0, 2],
                                      sharey=temp_ax_pre)
    mag_ax_post = comp_fig.add_subplot(gs[1, 2],
                                       sharex=mag_ax_pre,
                                       sharey=mag_ax_pre)
    # hist_ax_pre = comp_fig.add_subplot(gs[0, 3],
    #                                    sharey=temp_ax_pre)
    # hist_ax_post = comp_fig.add_subplot(gs[1, 3],
    #                                     sharex=hist_ax_pre,
    #                                     sharey=hist_ax_pre)

    all_axes = (temp_ax_pre, temp_ax_post, mtl_ax_pre, mtl_ax_post,
                mag_ax_pre, mag_ax_post)#, hist_ax_pre, hist_ax_post)
    # Set the plot limits here. The y-limits for temp_ax1 are
    # used for all subplots.
    if ylims is not None:
        temp_ax_pre.set_ylim(bottom=ylims[0],
                             top=ylims[1])
    temp_ax_pre.set_xlim(left=temp_lims[0],
                         right=temp_lims[1])
    mtl_ax_pre.set_xlim(left=mtl_lims[0],
                        right=mtl_lims[1])
    mag_ax_pre.set_xlim(left=mag_lims[0],
                        right=mag_lims[1])

    # Axis styles for all subplots.
    for ax in all_axes:
        if not args.full_range:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(
                                      base=100))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(
                                      base=50))
        else:
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.axhline(y=0, color='Black', linestyle='--')
        ax.yaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.65)
        ax.yaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.5)
        # if ax not in (hist_ax_pre, hist_ax_post):
        ax.xaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.65)

    for ax in (temp_ax_pre, temp_ax_post):
        ax.set_xlabel('Temperature (K)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    for ax in (mtl_ax_pre, mtl_ax_post):
        ax.set_xlabel('Metallicity [Fe/H]')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    for ax in (mag_ax_pre, mag_ax_post):
        ax.set_xlabel('Absolute Magnitude')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=0.25))

    # Just label the left-most two subplots' y-axes.
    for ax, era in zip((temp_ax_pre, temp_ax_post),
                       ('Pre', 'Post')):
        ax.set_ylabel(f'{era}-fiber change offset (m/s)')

    axes_dict = {'temp_pre': temp_ax_pre, 'temp_post': temp_ax_post,
                 'mtl_pre': mtl_ax_pre, 'mtl_post': mtl_ax_post,
                 'mag_pre': mag_ax_pre, 'mag_post': mag_ax_post,}
                 # 'hist_pre': hist_ax_pre, 'hist_post': hist_ax_post}

    return comp_fig, axes_dict


def main():
    """Run the main routine for the script."""

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions.')

    # Define the limits to plot in the various stellar parameters.
    temp_lims = (5400, 6300) * u.K
    mtl_lims = (-0.75, 0.45)
    mag_lims = (4, 5.8)

    model_func = fit.linear_model
    model_name = '_'.join(model_func.__name__.split('_')[:-1])

    db_file = vcl.databases_dir / f'stellar_db_{model_name}_params.hdf5'
    # Load data from HDF5 database file.
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
    param_dict = {'temp': 0, 'mtl': 1, 'mag': 2}

    params_list = []
    # Figure out how many parameters the model function takes, so we know how
    # many to dynamically give it later.
    for i in range(len(signature(model_func).parameters)-1):
        params_list.append(0.)

    label = '4589.484Cr2_28'
    # label = '6192.900Ni1_61'
    # label = '6178.520Ni1_61'

    # The column number to use for this transition:
    col = column_dict[label]

    comp_fig, axes_dict = create_parameter_comparison_figures(
            ylims=None,
            temp_lims=temp_lims,
            mtl_lims=mtl_lims,
            mag_lims=mag_lims)

    for time in eras.keys():

        vprint(20 * '=')
        vprint(f'Working on {time}-change era.')
        mean = np.nanmean(star_transition_offsets[eras[time],
                          :, col])

        # First, create a masked version to catch any missing entries:
        m_offsets = ma.masked_invalid(star_transition_offsets[
                    eras[time], :, col])
        m_offsets = m_offsets.reshape([len(m_offsets), 1])
        # Then create a new array from the non-masked data:
        offsets = u.unyt_array(m_offsets[~m_offsets.mask],
                               units=u.m/u.s)
        vprint(f'Median of offsets is {np.nanmedian(offsets)}')

        m_eotwms = ma.masked_invalid(star_transition_offsets_EotWM[
                eras[time], :, col])
        m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
        eotwms = u.unyt_array(m_eotwms[~m_eotwms.mask],
                              units=u.m/u.s)

        m_eotms = ma.masked_invalid(star_transition_offsets_EotM[
                eras[time], :, col])
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

        stars = ma.masked_array([key for key in
                                 star_names.keys()]).reshape(
                                     len(star_names.keys()), 1)
        names = stars[~m_offsets.mask]

        # nbin = 6
        # temp_bins = np.quantile(temps, np.linspace(0, 1, nbin+1),
        #                         interpolation='nearest')
        # metal_bins = np.quantile(metals, np.linspace(0, 1, nbin+1),
        #                          interpolation='nearest')
        # mag_bins = np.quantile(magss, np.linspace(0, 1, nbin+1),
        #                        interpolation='nearest')

        # Stack the stellar parameters into vertical slices
        # for passing to model functions.
        # x_data = np.stack((temps, metals, mags), axis=0)

        # Create the parameter list for this run of fitting.
        params_list[0] = float(mean)

        beta0 = tuple(params_list)
        vprint(beta0)

        # Iterate over binned segments of the data to find what additional
        # systematic error is needed to get a chi^2 of ~1.

        arrays_dict = {name: array for name, array in
                       zip(('temp', 'mtl', 'mag'),
                           (temps, metals, mags))}

        # First need to set up the bins:
        bin_dict = {}
        nbins = int(args.nbins)
        for name in arrays_dict:

            bins = np.quantile(arrays_dict[name], np.linspace(0, 1, nbins+1),
                               interpolation='nearest')
            bin_dict[name] = bins

        sigma_sys_dict = {}
        for name in tqdm(arrays_dict.keys()):
            sigma_sys_list = []
            bin_mid_list = []
            for bin_lims in tqdm(pairwise(bin_dict[name])):
                lower, upper = bin_lims
                bin_mid_list.append((lower + upper)/2)
                mask_array = ma.masked_outside(arrays_dict[name], *bin_lims)
                temps_copy = temps[~mask_array.mask]
                metals_copy = metals[~mask_array.mask]
                mags_copy = mags[~mask_array.mask]
                offsets_copy = offsets[~mask_array.mask]
                errs_copy = err_array[~mask_array.mask]
                x_data_copy = np.stack((temps_copy, metals_copy, mags_copy),
                                       axis=0)

                results = fit.find_sigma_sys(model_func, x_data_copy,
                                             offsets_copy.value,
                                             errs_copy.value,
                                             beta0)
                sigma_sys = results['sys_err_list'][-1]
                sigma_sys_list.append(sigma_sys)
                # tqdm.write(f'sigma_sys is {sigma_sys:.3f}')

            sigma_sys_dict[f'{name}_sigma_sys'] = sigma_sys_list
            sigma_sys_dict[f'{name}_bin_mids'] = bin_mid_list

        # Add the optimized parameters and covariances to the
        # dictionary. Make sure we separate them by time period.
        # coefficients_dict[label + '_' + time] = popt
        # covariance_dict[label + '_' + time] = pcov

        # sigma = np.nanstd(residuals)

        # sigmas_dict[label + '_' + time] = sigma
        # sigma_sys_dict[label + '_' + time] = sys_err

        residuals = u.unyt_array(results['residuals'],
                                 units=u.m/u.s)

        for plot_type, lims in zip(('temp', 'mtl', 'mag'),
                                   (temp_lims, mtl_lims, mag_lims)):
            ax = axes_dict[f'{plot_type}_{time}']
            # plot_data_points(ax, x_data[param_dict[plot_type]],
            #                  residuals, thick_err=err_array,
            #                  # thin_err=iter_err_array,
            #                  thin_err=None,
            #                  era=time)
            ax.plot(sigma_sys_dict[f'{plot_type}_bin_mids'],
                    sigma_sys_dict[f'{plot_type}_sigma_sys'],
                    color='Green', marker='o')

            # ax.annotate(r'$\sigma_\mathrm{sys}$:'
            #             f' {sys_err:.2f}',
            #             (0.01, 0.99),
            #             xycoords='axes fraction',
            #             verticalalignment='top')
            # ax.annotate(fr'$\chi^2_\nu$: {chi_squared_nu.value:.4f}'
            #             '\n'
            #             fr'$\sigma$: {sigma:.2f}',
            #             (0.99, 0.99),
            #             xycoords='axes fraction',
            #             horizontalalignment='right',
            #             verticalalignment='top')
            data = np.array(ma.masked_invalid(residuals).compressed())

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the additional'
                                     ' error needed to reach a chi-squared'
                                     ' value of 1 as a function of various'
                                     ' stellar parameters.')
    parser.add_argument('--full-range', action='store_true',
                        help='Plot the full range of values instead of'
                        ' restricting to a  fixed range.')
    parser.add_argument('--nbins', action='store', type=int,
                        default=5,
                        help='The number of bins to use (default: 5).')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')
    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
