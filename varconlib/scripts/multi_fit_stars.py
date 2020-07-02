#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:18:50 2020

@author: dberke

A script to read in data on transition offsets vs. several stellar parameters
from a database, and perform multi-component fitting to it.
"""

import argparse
import csv
from inspect import signature
import os
from pathlib import Path
import pickle
from pprint import pprint
from time import sleep
import sys

from adjustText import adjust_text
import h5py
import hickle
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

# Define style parameters to use for stellar parameter plots.
style_pre = {'color': 'Chocolate',
             'ecolor_thick': 'DarkOrange',
             'ecolor_thin': 'BurlyWood'}
style_post = {'color': 'DodgerBlue',
              'ecolor_thick': 'CornFlowerBlue',
              'ecolor_thin': 'LightSkyBlue'}
style_ref = {'color': 'DarkGreen',
             'ecolor_thick': 'ForestGreen',
             'ecolor_thin': 'DarkSeaGreen'}
style_markers = {'markeredgecolor': 'Black',
                 'markeredgewidth': 1,
                 'alpha': 0.7,
                 'markersize': 4}
style_caps = {'capsize_thin': 4,
              'capsize_thick': 7,
              'linewidth_thin': 2,
              'linewidth_thick': 3,
              'cap_thin': 1.5,
              'cap_thick': 2.5}


def create_comparison_figure(ylims=None,
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
    gs = GridSpec(ncols=4, nrows=2, figure=comp_fig,
                  width_ratios=(5, 5, 5, 3))

    temp_ax_pre = comp_fig.add_subplot(gs[0, 0])
    temp_ax_post = comp_fig.add_subplot(gs[1, 0],
                                        sharex=temp_ax_pre,
                                        sharey=temp_ax_pre)
    mtl_ax_pre = comp_fig.add_subplot(gs[0, 1],
                                      sharey=temp_ax_pre)
    mtl_ax_post = comp_fig.add_subplot(gs[1, 1],
                                       sharex=mtl_ax_pre,
                                       sharey=mtl_ax_pre)
    logg_ax_pre = comp_fig.add_subplot(gs[0, 2],
                                       sharey=temp_ax_pre)
    logg_ax_post = comp_fig.add_subplot(gs[1, 2],
                                        sharex=logg_ax_pre,
                                        sharey=logg_ax_pre)
    hist_ax_pre = comp_fig.add_subplot(gs[0, 3],
                                       sharey=temp_ax_pre)
    hist_ax_post = comp_fig.add_subplot(gs[1, 3],
                                        sharex=hist_ax_pre,
                                        sharey=hist_ax_pre)

    all_axes = (temp_ax_pre, temp_ax_post, mtl_ax_pre, mtl_ax_post,
                logg_ax_pre, logg_ax_post, hist_ax_pre, hist_ax_post)
    # Set the plot limits here. The y-limits for temp_ax1 are
    # used for all subplots.
    if ylims is not None:
        temp_ax_pre.set_ylim(bottom=ylims[0],
                             top=ylims[1])
    temp_ax_pre.set_xlim(left=temp_lims[0],
                         right=temp_lims[1])
    mtl_ax_pre.set_xlim(left=mtl_lims[0],
                        right=mtl_lims[1])
    logg_ax_pre.set_xlim(left=logg_lims[0],
                         right=logg_lims[1])

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
        if ax not in (hist_ax_pre, hist_ax_post):
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
    for ax in (logg_ax_pre, logg_ax_post):
        ax.set_xlabel(r'log $g$ $(\mathrm{cm}/\mathrm{s}^2)$')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=0.05))

    # Just label the left-most two subplots' y-axes.
    for ax, era in zip((temp_ax_pre, temp_ax_post),
                       ('Pre', 'Post')):
        ax.set_ylabel(f'{era}-fiber change offset (m/s)')

    axes_dict = {'temp_pre': temp_ax_pre, 'temp_post': temp_ax_post,
                 'mtl_pre': mtl_ax_pre, 'mtl_post': mtl_ax_post,
                 'logg_pre': logg_ax_pre, 'logg_post': logg_ax_post,
                 'hist_pre': hist_ax_pre, 'hist_post': hist_ax_post}

    return comp_fig, axes_dict


def plot_data_points(axis, x_pos, y_pos, thick_err, thin_err, era=None,
                     ref=False):
    """Plot a data point for a star.

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        An axes to plot the data on.
    x_pos : iterable of floats or `unyt.unyt_quantity`
        The x-positions of the points to plot.
    y_pos : iterable of floats or `unyt.unyt_quantity`
        The y-positions of the points to plot. The length must match the length
        of `x_pos`.
    thick_err : iterable of floats or `unyt.unyt_quantity` or None
        The values of the thick error bars to plot. The length must match the
        length of `x_pos`.
    thin_err : iterable of floats or `unyt.unyt_quantity` or None
        The values of the thin error bars to plot. The length must match the
        length of `x_pos`.
    era : string, ['pre', 'post'], or None, Default : None
        Whether the time period of the plot is pre- or post-fiber
        change. The only allowed string values are 'pre' and 'post'. Controls
        color of the points. If `ref` is *True*, the value of `era` is
        ignored, and can be left unspecified, otherwise it needs a
        value to be given.
    ref : bool, Default : False
        Whether this data point is for the reference star. If *True*,
        will use a special separate color scheme.

    Returns
    -------
    None.

    """

    if ref:
        params = style_ref
    elif era == 'pre':
        params = style_pre
    elif era == 'post':
        params = style_post
    else:
        raise ValueError("Keyword 'era' received an unknown value"
                         f" (valid values are 'pre' & 'post'): {era}")

    if thin_err is not None:
        axis.errorbar(x=x_pos, y=y_pos,
                      yerr=thin_err, linestyle='',
                      marker='', capsize=style_caps['capsize_thin'],
                      color=params['color'],
                      ecolor=params['ecolor_thin'],
                      elinewidth=style_caps['linewidth_thin'],
                      capthick=style_caps['cap_thin'])
    elif thick_err is not None:
        axis.errorbar(x=x_pos, y=y_pos,
                      yerr=thick_err, linestyle='',
                      marker='o', markersize=style_markers['markersize'],
                      markeredgewidth=style_markers['markeredgewidth'],
                      markeredgecolor=style_markers['markeredgecolor'],
                      alpha=style_markers['alpha'],
                      capsize=style_caps['capsize_thick'],
                      color=params['color'],
                      ecolor=params['ecolor_thick'],
                      elinewidth=style_caps['linewidth_thick'],
                      capthick=style_caps['cap_thick'])


def find_star(star_params, stellar_params, star_names):
    """Return the name of the star which matches the given parameters.

    Parameters
    ----------
    star_params : iterable
        An iterable (in practice, an array slice of length 3) which contains
        values of stellar parameters, temperature in position 0, metallicity in
        position 1, and magnitude in position 2.
    stellar_params : array-like
        A 3xN array of stellar parameters, with the same order as `star_params`.
    star_names : `bidict.bidict`
        A `bidict` containing star names in strings as keys, and index numbers
        as ints for the values.

    Returns
    -------
    str
        The name of the star associated with the given parameters.

    """

    for i in range(len(stellar_params[0])):
        if np.all(np.isclose(star_params, stellar_params[:, i])):
            return star_names[i]


def main():
    """Run the main routine of the script."""

    # Define the limits to plot in the various stellar parameters.
    temp_lims = (5400, 6300) * u.K
    mtl_lims = (-0.75, 0.45)
    # mag_lims = (4, 5.8)
    logg_lims = (4.1, 4.6)

    # Define the model to use:
    if args.constant:
        model_func = fit.constant_model
    elif args.linear:
        model_func = fit.linear_model
    elif args.quadratic:
        model_func = fit.quadratic_model
    elif args.cubic:
        model_func = fit.cubic_model
    elif args.quartic:
        model_func = fit.quartic_model
    elif args.quintic:
        model_func = fit.quintic_model
    elif args.cross_term:
        model_func = fit.cross_term_model
    elif args.quadratic_cross_term:
        model_func = fit.quadratic_cross_term_model
    elif args.quadratic_magnitude:
        model_func = fit.quadratic_mag_model
    elif args.quad_cross_terms:
        model_func = fit.quad_full_cross_terms_model

    model_name = '_'.join(model_func.__name__.split('_')[:-1])

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions.')

    if not args.use_corrected_transitions:
        db_file = vcl.databases_dir / 'stellar_db_uncorrected.hdf5'
    else:
        db_file = vcl.databases_dir / f'stellar_db_{model_name}_params.hdf5'
        if not db_file.exists():
            raise FileNotFoundError('The given stellar database does not exist:'
                                    f' {db_file}')

    # Load data from HDF5 database file.
    tqdm.write('Reading data from stellar database file...')
    star_transition_offsets = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets')
    star_transition_offsets_EotWM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotWM')
    star_transition_offsets_EotM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotM')
    # star_transition_offsets_stds = u.unyt_array.from_hdf5(
    #         db_file, dataset_name='star_standard_deviations')
    star_temperatures = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_temperatures')

    with h5py.File(db_file, mode='r') as f:

        star_metallicities = hickle.load(f, path='/star_metallicities')
        # star_magnitudes = hickle.load(f, path='/star_magnitudes')
        star_gravities = hickle.load(f, path='/star_gravities')
        column_dict = hickle.load(f, path='/transition_column_index')
        star_names = hickle.load(f, path='/star_row_index')

    # Handle various fitting and plotting setup:
    eras = {'pre': 0, 'post': 1}
    param_dict = {'temp': 0, 'mtl': 1, 'logg': 2}

    # Create lists to store information about each fit in:
    index_nums = []
    chi_squareds_pre, sigmas_pre, sigma_sys_pre = [], [], []
    chi_squareds_post, sigmas_post, sigma_sys_post = [], [], []
    index_num = 0

    params_list = []
    # Figure out how many parameters the model function takes, so we know how
    # many to dynamically give it later.
    for i in range(len(signature(model_func).parameters)-1):
        params_list.append(0.)

    # Define the folder to put plots in.
    output_dir = Path(vcl.config['PATHS']['output_dir'])
    extra = '_corrected' if args.use_corrected_transitions else ''
    plots_folder = output_dir / f'stellar_parameter_fits/{model_name}{extra}'
    vprint(f'Creating plots in {plots_folder}')
    if not plots_folder.exists():
        os.makedirs(plots_folder)

    # Create a dictionary of fit coefficients assigned to each transition's
    # label
    coefficients_dict = {}
    covariance_dict = {}
    sigmas_dict = {}
    sigma_sys_dict = {}

    tqdm.write('Creating plots for each transition...')
    for transition in tqdm(transitions_list):
        for order_num in transition.ordersToFitIn:
            index_nums.append(index_num)
            index_num += 1
            label = '_'.join([transition.label, str(order_num)])
            vprint(20 * '-')
            vprint(f'Analyzing {label}...')

            # The column number to use for this transition:
            col = column_dict[label]
            ylimits = (-300 * u.m / u.s,
                       300 * u.m / u.s) if not args.full_range else None

            comp_fig, axes_dict = create_comparison_figure(
                            ylims=ylimits,
                            temp_lims=temp_lims,
                            mtl_lims=mtl_lims,
                            logg_lims=logg_lims)

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

#                m_stds = ma.masked_invalid(star_transition_offsets_stds[
#                            eras[time], :, col])
#                m_stds = m_stds.reshape([len(m_stds), 1])
#                stds = u.unyt_array(m_stds[~m_stds.mask],
#                                    units=u.m/u.s)

                m_eotwms = ma.masked_invalid(star_transition_offsets_EotWM[
                        eras[time], :, col])
                m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
                eotwms = u.unyt_array(m_eotwms[~m_offsets.mask],
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
                # magnitudes = ma.masked_array(star_magnitudes)
                # mags = magnitudes[~m_offsets.mask]
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

                results = fit.find_sigma_sys(model_func, x_data, offsets.value,
                                             err_array.value, beta0)

                residuals = u.unyt_array(results['residuals'],
                                         units=u.m/u.s)
                chi_squared_nu = results['chi_squared_list'][-1]
                sys_err = results['sys_err_list'][-1] * u.m / u.s

                vprint(f'Terminated with sys_err = {sys_err}')
                vprint(f'Finished {label}_{time} in'
                       f' {len(results["sys_err_list"])} steps.')
                # Add the optimized parameters and covariances to the
                # dictionary. Make sure we separate them by time period.
                coefficients_dict[label + '_' + time] = results['popt']
                covariance_dict[label + '_' + time] = results['pcov']

                sigma = np.nanstd(residuals)

                sigmas_dict[label + '_' + time] = sigma
                sigma_sys_dict[label + '_' + time] = sys_err

                if time == 'pre':
                    chi_squareds_pre.append(chi_squared_nu)
                    sigmas_pre.append(sigma.value)
                    sigma_sys_pre.append(sys_err.value)
                else:
                    chi_squareds_post.append(chi_squared_nu)
                    sigmas_post.append(sigma.value)
                    sigma_sys_post.append(sys_err.value)

                for plot_type, lims in zip(('temp', 'mtl', 'logg'),
                                           (temp_lims, mtl_lims, logg_lims)):
                    ax = axes_dict[f'{plot_type}_{time}']
                    plot_data_points(ax, x_data[param_dict[plot_type]],
                                     residuals, thick_err=err_array,
                                     # thin_err=iter_err_array,
                                     thin_err=None,
                                     era=time)
                    if args.label_outliers:
                        # Find outliers more than 3 sigma away from zero so we
                        # can label them.
                        labels = []
                        for x, y, e in zip(range(len(
                                x_data[param_dict[plot_type]])), residuals,
                                err_array):
                            sig_lim = 3 * e
                            if abs(y) > sig_lim:
                                star_name = find_star(x_data[:, x],
                                                      x_data, names)

                                labels.append(ax.text(
                                    x_data[param_dict[plot_type], x],
                                    y.value, star_name,
                                    horizontalalignment='left',
                                    verticalalignment='top',
                                    size=8, weight='bold', color='Red'))
                        # print(labels)
                        adjust_text(labels,
                                    ax=ax,
                                    only_move={'points': 'y',
                                               'text': 'xy',
                                               'objects': 'xy'},
                                    arrowprops=dict(arrowstyle='-',
                                                    color='OliveDrab'),
                                    autoalign=True,
                                    lim=1000, fontsize=9)

                    ax.annotate(f'Blendedness: {transition.blendedness}\n'
                                r'$\sigma_\mathrm{sys}$:'
                                f' {sys_err:.2f}',
                                (0.01, 0.99),
                                xycoords='axes fraction',
                                verticalalignment='top')
                    ax.annotate(fr'$\chi^2_\nu$: {chi_squared_nu:.4f}'
                                '\n'
                                fr'$\sigma$: {sigma:.2f}',
                                (0.99, 0.99),
                                xycoords='axes fraction',
                                horizontalalignment='right',
                                verticalalignment='top')
                    data = np.array(ma.masked_invalid(residuals).compressed())
                    axes_dict[f'hist_{time}'].hist(data,
                                                   bins='fd',
                                                   color='Black',
                                                   histtype='step',
                                                   orientation='horizontal')

            file_name = plots_folder / f'{label}_{model_name}.png'
            vprint(f'Saving file {label}.png')
            vprint('\n')

            comp_fig.savefig(str(file_name))
            plt.close('all')

    # Save metadata from this run's fits to CSV:
    csv_file = plots_folder / f'{model_name}_fit_results.csv'

    with open(csv_file, 'w', newline='') as f:
        datawriter = csv.writer(f)
        header = ('#index', 'chi_squared_pre', 'sigma_pre', 'sigma_sys_pre',
                  'chi_squared_post', 'sigma_post', 'sigma_sys_post')
        datawriter.writerow(header)
        for row in zip(index_nums, chi_squareds_pre, sigmas_pre, sigma_sys_pre,
                       chi_squareds_post, sigmas_post, sigma_sys_post):
            datawriter.writerow(row)

    # Save the function used and the parameters found for each transition to
    # an HDF5 file for use in other scripts.
    if not args.use_corrected_transitions:
        hdf5_file = output_dir / f'fit_params/{model_name}_params.hdf5'
    else:
        hdf5_file = output_dir /\
            f'fit_params/{model_name}_corrected_params.hdf5'
    if not hdf5_file.parent.exists():
        os.mkdir(hdf5_file.parent)

    vprint(f'Writing HDF5 file with fit parameters at {hdf5_file}')
    if hdf5_file.exists():
        os.unlink(hdf5_file)
    with h5py.File(hdf5_file, mode='a') as f:
        f.attrs['type'] = 'A file containing a fitting function and the' +\
                          ' parameters for it for each transition in' +\
                          '/params_dict'
        hickle.dump(model_func, f, path='/fitting_function')
        hickle.dump(coefficients_dict, f, path='/coeffs_dict')
        hickle.dump(covariance_dict, f, path='/covariance_dict')
        hickle.dump(sigmas_dict, f, path='/sigmas_dict')
        hickle.dump(sigma_sys_dict, f, path='/sigma_sys_dict')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use stored data from stars'
                                     ' to fit transition offsets to stellar'
                                     ' parameters.')
    parser.add_argument('--full-range', action='store_true',
                        help='Plot the full range of values instead of'
                        ' restricting to a  fixed range.')
    parser.add_argument('--use-corrected-transitions', action='store_true',
                        help='Use stellar database with outliers removed from'
                        ' previously using the selected model.')
    parser.add_argument('--label-outliers', action='store_true',
                        help='Label the points which are more than'
                        ' 3 sigma away from the mean.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')

    # parser.add_argument('--temp', action='store', type=float, nargs=2,
    #                     metavar=('T_low', 'T_high'),
    #                     help='The limits in temperature of stars to use.')
    # parser.add_argument('--mtl', action='store', type=float, nargs=2,
    #                     metavar=('FeH_low', 'FeH_high'),
    #                     help='The limits in metallicity of stars to use.')
    # parser.add_argument('--mag', action='store', type=float, nargs=2,
    #                     metavar=('M_low', 'M_high'),
    #                     help='The limits in magnitude of stars to use.')

    func = parser.add_mutually_exclusive_group(required=True)
    func.add_argument('--constant', action='store_true',
                      help='Use a constant function.')
    func.add_argument('--linear', action='store_true',
                      help='Use a function linear in all three variables.')
    func.add_argument('--quadratic', action='store_true',
                      help='Use a function quadratic in all three variables.')
    func.add_argument('--cubic', action='store_true',
                      help='Use a cubic function for all three variables.')
    func.add_argument('--quartic', action='store_true',
                      help='Use a quartic function for all three variables.')
    func.add_argument('--quintic', action='store_true',
                      help='Use a quintic function for all three variables.')
    func.add_argument('--cross-term', action='store_true',
                      help='Use a linear model with cross term ([Fe/H]/Teff).')
    func.add_argument('--quadratic-cross-term', action='store_true',
                      help='Use a quadratic model with cross terms between'
                      ' metallicity and temperature.')
    func.add_argument('--quadratic-magnitude', action='store_true',
                      help='Use a cross term with quadratic magnitude.')
    func.add_argument('--quad-cross-terms', action='store_true',
                      help='Use a quadratic model with full cross terms.')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
