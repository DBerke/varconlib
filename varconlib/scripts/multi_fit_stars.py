#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:18:50 2020

@author: dberke

A script to read in data on transition offsets vs. several stellar parameters
from a database, and perform multi-component fitting to it.
"""

import argparse
import os
from pathlib import Path
import pickle

import h5py
import hickle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
import scipy.odr as odr
from tqdm import tqdm
import unyt as u

import varconlib as vcl


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


def create_parameter_comparison_figures(ylims=None,
                                        temp_lims=(5300 * u.K, 6200 * u.K),
                                        mtl_lims=(-0.75, 0.4),
                                        mag_lims=(4, 5.8),
                                        logg_lims=(4.1, 4.6)):
    """Creates and returns a figure with pre-set subplots.

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

    comp_fig = plt.figure(figsize=(16, 8), tight_layout=True)
    gs = GridSpec(ncols=4, nrows=2, figure=comp_fig)

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
    logg_ax_pre = comp_fig.add_subplot(gs[0, 3],
                                       sharey=temp_ax_pre)
    logg_ax_post = comp_fig.add_subplot(gs[1, 3],
                                        sharex=logg_ax_pre,
                                        sharey=logg_ax_pre)

    all_axes = (temp_ax_pre, temp_ax_post, mtl_ax_pre, mtl_ax_post,
                mag_ax_pre, mag_ax_post, logg_ax_pre, logg_ax_post)
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
    logg_ax_pre.set_xlim(left=logg_lims[0],
                         right=logg_lims[1])

    # Axis styles for all subplots.
    for ax in all_axes:
#        ax.yaxis.set_major_locator(ticker.MultipleLocator(
#                                   base=100))
#        ax.yaxis.set_minor_locator(ticker.MultipleLocator(
#                                   base=50))
        ax.axhline(y=0, color='Black', linestyle='--')
        ax.yaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.85)
        ax.xaxis.grid(which='major', color='Gray',
                      linestyle='--', alpha=0.85)
        ax.yaxis.grid(which='minor', color='Gray',
                      linestyle=':', alpha=0.75)

    for ax in (temp_ax_pre, temp_ax_post):
        ax.set_xlabel('Temperature (K)')
    for ax in (mtl_ax_pre, mtl_ax_post):
        ax.set_xlabel('Metallicity [Fe/H]')
    for ax in (mag_ax_pre, mag_ax_post):
        ax.set_xlabel('Absolute Magnitude')
    for ax in (logg_ax_pre, logg_ax_post):
        ax.set_xlabel(r'$\log(g)$')

    # Just label the left-most two subplots' y-axes.
    for ax in (temp_ax_pre, temp_ax_post):
        ax.set_ylabel('Pre-fiber change offset (m/s)')

    axes_dict = {'temp_pre': temp_ax_pre, 'temp_post': temp_ax_post,
                 'mtl_pre': mtl_ax_pre, 'mtl_post': mtl_ax_post,
                 'mag_pre': mag_ax_pre, 'mag_post': mag_ax_post,
                 'logg_pre': logg_ax_pre, 'logg_post': logg_ax_post}

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
        The y-positions of the points to plot.
    thick_err : iterable of floats or `unyt.unyt_quantity`
        The values of the thick error bars to plot.
    thin_err : iterable of floats or `unyt.unyt_quantity`
        The values of the thin error bars to plot.
    era : string, ['pre', 'post'], Default : None
        Whether the time period of the plot is pre- or post-fiber
        change. Only allowed values are 'pre' and 'post'. Controls
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

    axis.errorbar(x=x_pos, y=y_pos,
                  yerr=thin_err, linestyle='',
                  marker='', capsize=style_caps['capsize_thin'],
                  color=params['color'],
                  ecolor=params['ecolor_thin'],
                  elinewidth=style_caps['linewidth_thin'],
                  capthick=style_caps['cap_thin'])
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


def offset_model(beta, data):
    """Function representing the model for multi-parameter fits of offsets.

    ODR appears to need return values specified as indices from given vectors,
    so as a reference:
        data[0] = temperature
        data[1] = metallicity
        data[2] = absolute magnitude

    Parameters
    ----------

    beta : iterable (length 8)
        A vector of coefficients for use in the equation.

    data : iterable (length 3)
        A vector containing a data point (or equal-length iterables of data
        points) for each of the three variables to consider.

    """

#    return beta[0] + beta[1] * data[0]# + beta[2] * data[1] +\
#        beta[3] * data[2]# + beta[4] * data[0] ** 2 +\
#        beta[5] * data[1] ** 2 + beta[6] * data[2] ** 2# +\
#        beta[7] * data[1] / data[0] +\
#        beta[8] * data[1] / data[2]
    if args.constant:
        return beta[0] + 0 * data[0]
    elif args.linear_temp:
        return beta[0] + beta[1] * data[0]
    elif args.quad_temp:
        return beta[0] + beta[1] * data[0] + beta[2] * data[0] * data[0]
    elif args.linear_mtl:
        return beta[0] + beta[1] * data[1]
    elif args.quad_mtl:
        return beta[0] + beta[1] * data[1] + beta[2] * data[1] ** 2
    elif args.linear_mag:
        return beta[0] + beta[1] * data[2]
    elif args.quad_mag:
        return beta[0] + beta[1] * data[2] + beta[2] * data[2] ** 2
    elif args.linear:
        return beta[0] + beta[1] * data[0] + beta[2] * data[1] +\
            beta[3] * data[2]


def make_x_values(limits, num_points, position):
    """Create an array to use as x-values when plotting slices of hypersurface.

    Parameters
    ----------
    limits : iterable, length-2
        A pair of (lower, upper) limits to use for the extremes.
    num_points : int or float
        The number of points to use in `np.linspace`.
    position : int, [0, 1, 2]
        The row index (out of three rows) to place the generated points in. The
        other rows will be zeros.

    """

    temp = np.full([num_points], 5777 * u.K)
    mtl = np.zeros([num_points])
    mag = np.full([num_points], 4.83)

    array = np.stack((temp, mtl, mag), axis=0)
    array[position] = np.linspace(limits[0], limits[1], num_points)

    return array


def main():
    """The main routine of the script."""

    # Define the limits to plot in the various stellar parameters.
    temp_lims = (5400, 6300) * u.K
    mtl_lims = (-0.63, 0.52)
    mag_lims = (4, 5.8)
    logg_lims = (4.1, 4.6)

    tqdm.write('Unpickling transitions list..')
    with open(vcl.final_selection_file, 'r+b') as f:
        transitions_list = pickle.load(f)
    vprint(f'Found {len(transitions_list)} transitions.')

    plots_folder = Path(vcl.config['PATHS']['output_dir']) /\
        'stellar_parameter_fits'
    if not plots_folder.exists():
        os.makedirs(plots_folder)

    db_file = vcl.stellar_results_file

    # Load data from HDF5 database file.
    tqdm.write('Reading data from stellar database file...')
    star_transition_offsets = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets')
    star_transition_offsets_EotWM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotWM')
    star_transition_offsets_EotM = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_transition_offsets_EotM')
    star_transition_offsets_stds = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_standard_deviations')
    star_temperatures = u.unyt_array.from_hdf5(
            db_file, dataset_name='star_temperatures')

    with h5py.File(db_file, mode='r') as f:

        star_metallicities = hickle.load(f, path='/star_metallicities')
        star_magnitudes = hickle.load(f, path='/star_magnitudes')
        star_gravities = hickle.load(f, path='/star_gravities')
        column_dict = hickle.load(f, path='transition_column_index')

    # Handle various fitting and plotting setup:
    # Define the model to fit.
    hypersurface = odr.Model(offset_model)

    eras = {'pre': 0, 'post': 1}
    param_dict = {'temp': 0, 'mtl': 1, 'mag': 2}

    # Create some x values for plotting later:
    numpoints = 50
    x_temps = make_x_values(temp_lims.to_value(), numpoints,
                            param_dict['temp'])
    x_mtls = make_x_values(mtl_lims, numpoints, param_dict['mtl'])
    x_mags = make_x_values(mag_lims, numpoints, param_dict['mag'])

    tqdm.write('Creating plots for each transition...')
    for transition in tqdm(transitions_list[:]):
        for order_num in transition.ordersToFitIn:
            label = '_'.join([transition.label, str(order_num)])

            # The column number to use for this transition:
            col = column_dict[label]

            median = np.nanmedian(star_transition_offsets[:, :,
                                  col])

            comp_fig, axes_dict = create_parameter_comparison_figures(
#                            ylims=(median - 300 * u.m / u.s,
#                                   median + 300 * u.m / u.s),
                            temp_lims=(5400 * u.K, 6300 * u.K),
                            mtl_lims=(-0.63, 0.52))

            for ax in axes_dict.values():
                ax.annotate(f'Blendedness: {transition.blendedness}',
                            (0.01, 0.95),
                            xycoords='axes fraction')

#            for ax, attr in zip(('temp_pre', 'mtl_pre',
#                                 'mag_pre', 'logg_pre'),
#                                (star_temperatures,
#                                 star_metallicities,
#                                 star_magnitudes,
#                                 star_gravities)):
#                plot_data_points(
#                    axes_dict[ax], attr,
#                    star_transition_offsets[eras['pre'], :, col],
#                    star_transition_offsets_EotWM[eras['pre'], :, col],
#                    star_transition_offsets_EotM[eras['pre'], :, col],
#                    era='pre')
#
#            for ax, attr in zip(('temp_post', 'mtl_post',
#                                 'mag_post', 'logg_post'),
#                                (star_temperatures,
#                                 star_metallicities,
#                                 star_magnitudes,
#                                 star_gravities)):
#                plot_data_points(
#                    axes_dict[ax], attr,
#                    star_transition_offsets[eras['post'], :, col],
#                    star_transition_offsets_EotWM[eras['post'], :, col],
#                    star_transition_offsets_EotM[eras['post'], :, col],
#                    era='post')

            # Perform the ODR fitting and plot the resulting functions.
            for time in eras.keys():

                # First, create a masked version to catch any missing entries:
                m_offsets = ma.masked_invalid(star_transition_offsets[
                            eras[time], :, col])
                m_offsets = m_offsets.reshape([len(m_offsets), 1])
                # Then create a new array from the non-masked data:
                offsets = m_offsets[~m_offsets.mask]

                m_stds = ma.masked_invalid(star_transition_offsets_stds[
                            eras[time], :, col])
                m_stds = m_stds.reshape([len(m_stds), 1])
                stds = m_stds[~m_stds.mask]

                m_eotwms = ma.masked_invalid(star_transition_offsets_EotWM[
                        eras[time], :, col])
                m_eotwms = m_eotwms.reshape([len(m_eotwms), 1])
                eotwms = m_eotwms[~m_eotwms.mask]

                temperatures = ma.masked_array(star_temperatures)
                temps = temperatures[~m_offsets.mask]
                metallicities = ma.masked_array(star_metallicities)
                metals = metallicities[~m_offsets.mask]
                magnitudes = ma.masked_array(star_magnitudes)
                mags = magnitudes[~m_offsets.mask]

#                for array in (temps, metals, mags):
#                    print(array)

                x_data = np.stack((temps, metals, mags), axis=0)
#                print(x_data.shape)
#                print(x_data[:, :5])
                data = odr.RealData(x_data, y=offsets, sy=eotwms)
#                data = odr.RealData(x=temps,
#                                    y=offsets)
#                print(list(data.x.shape))
#                print(list(data.y.shape))
#                print(data.x)
#                print(data.y)
                odr_instance = odr.ODR(data, hypersurface,
                                       beta0=[median, 1, 1, 1, 1, 1, 1])
                odr_instance.maxit = 100000
                # Create log files for output:
                outfile = plots_folder / 'ODR_output.txt'
                errfile = plots_folder / 'ODR_errors.txt'
                for f in (outfile, errfile):
                    if f.exists():
                        os.unlink(f)
                odr_instance.rptfile = str(outfile)
                odr_instance.errfile = str(errfile)
                output = odr_instance.run()
                if args.verbose:
                    output.pprint()
                params = output.beta

                results = u.unyt_array(offset_model(params, x_data),
                                       units=u.m/u.s)
#                print(type(results))
#                print(results)
#                print(offsets)
                diffs = offsets - results

                for plot_type, lims, xs in zip(('temp', 'mtl', 'mag'),
                                               (temp_lims, mtl_lims, mag_lims),
                                               (x_temps, x_mtls, x_mags)):
#                    print(offset_model(params, xs[:, :3]))
                    axes_dict[f'{plot_type}_{time}'].plot(
                            x_data[param_dict[plot_type]],
                            diffs,
                            color='LightGray', markeredgecolor='Black',
                            linestyle='', markersize=4,
                            marker='o')
                    axes_dict[f'{plot_type}_{time}'].annotate(
                            [format(p, '.3e') for p in params if not
                             np.isclose(p, 1)],
                            (0.01, 0.01),
                            xycoords='axes fraction',
                            color='Red')

            file_name = plots_folder / f'{label}.png'
            vprint(f'Saving file {label}.png')

            comp_fig.savefig(str(file_name))
            plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use stored data from stars'
                                     ' to fit transition offsets to stellar'
                                     ' parameters.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')

    parser.add_argument('--temp', action='store', type=float, nargs=2,
                        metavar=('T_low', 'T_high'),
                        help='The limits in temperature of stars to use.')
    parser.add_argument('--mtl', action='store', type=float, nargs=2,
                        metavar=('FeH_low', 'FeH_high'),
                        help='The limits in metallicity of stars to use.')
    parser.add_argument('--mag', action='store', type=float, nargs=2,
                        metavar=('M_low', 'M_high'),
                        help='The limits in magnitude of stars to use.')

    func = parser.add_mutually_exclusive_group(required=True)
    func.add_argument('--constant', action='store_true',
                      help='Use a constant function.')
    func.add_argument('--linear-temp', action='store_true',
                      help='Use a function linear in temperature:\n'
                      'a + b1 T')
    func.add_argument('--quad-temp', action='store_true',
                      help='Use a quadratic function in temperature:\n'
                      'a + b1 T + b2 T^2')
    func.add_argument('--linear-mtl', action='store_true',
                      help='Use a function linear in metallicity:\n'
                      'a + c1 FeH')
    func.add_argument('--quad-mtl', action='store_true',
                      help='Use a function quadratic in metallicity:\n'
                      'a + c1 FeH + c2 FeH^2')
    func.add_argument('--linear-mag', action='store_true',
                      help='Use a function linear in magnitude:\n'
                      'a + d1 M')
    func.add_argument('--quad-mag', action='store_true',
                      help='Use a function quadratic in magnitude:\n'
                      'a + d1 M + d2 M^2')
    func.add_argument('--linear', action='store_true',
                      help='Use a function linear in all three variables:\n'
                      'a + b1 T + c1 FeH + d1 M')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
