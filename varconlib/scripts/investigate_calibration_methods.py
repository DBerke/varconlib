#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:27:54 2019

@author: dberke

Script to plot various different combinations of wavelength calibration
procedures against each other.

"""

import argparse
from pathlib import Path

from astropy.io import fits
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import trange, tqdm
import unyt as u

import varconlib as vcl
from varconlib.obs2d import HARPSFile2DScience
from varconlib.miscellaneous import wavelength2velocity

plt.rc('text', usetex=True)
plt.rc('font', size=12, weight='bold')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams['axes.linewidth'] = 1.5

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--update', action='store_true',
                    help='Flag to update the wavelength solution.')
parser.add_argument('--wavelength-delta', action='store_true',
                    help='Create a plot showing the differences in'
                    ' wavelength between old and new solutions.')
parser.add_argument('--order-overlap', action='store_true',
                    help='Plot the overlap between old and new wavelength'
                    ' solutions over the H-alpha line.')
parser.add_argument('--velocity-delta', action='store_true',
                    help='Create a plote showing the differences in velocity'
                    ' between old and new solutions.')
parser.add_argument('--paper-figure', action='store_true',
                    help='Create a plot comparing the new and old wavelength'
                    ' solutions.')

args = parser.parse_args()

test_old_file = vcl.data_dir / 'test_old.fits'
test_new_coeffs_file = vcl.data_dir / 'test_new_coefficients.fits'
test_pix_pos_file = vcl.data_dir / 'test_pix_positions.fits'
test_new_file = vcl.data_dir / 'test_new.fits'

test_flat_file = vcl.data_dir / 'ceres2.flat.fits'

if args.update:
    update_status = ['WAVE', 'BARY']  # use '['WAVE', 'BARY']'
else:
    update_status = []

test_old = HARPSFile2DScience(test_old_file, new_coefficients=False,
                              pixel_positions=False)

if args.wavelength_delta or args.velocity_delta:
    print('Creating new coefficients file...')
    test_new_coeffs = HARPSFile2DScience(test_new_coeffs_file,
                                         new_coefficients=True,
                                         pixel_positions=False,
                                         update=update_status)
    print('Creating pixel positions file...')
    test_pix_pos = HARPSFile2DScience(test_pix_pos_file,
                                      update=update_status,
                                      new_coefficients=False,
                                      pixel_positions=True)

    pix_wv = test_pix_pos.wavelengthArray
    coeffs_wv = test_new_coeffs.wavelengthArray

print('Creating new calibration with both...')
test_new = HARPSFile2DScience(test_new_file, update=update_status,
                              new_coefficients=True,
                              pixel_positions=True)

old_wv = test_old.wavelengthArray
new_wv = test_new.wavelengthArray

with fits.open(test_flat_file) as hdulist:
    nx = hdulist[0].header['NAXIS1']
    cdelta1 = hdulist[0].header['CDELT1']
    crval1 = hdulist[0].header['CRVAL1']
    data = hdulist[0].data * 70000

wavelength = np.linspace(0, len(data), len(data))*cdelta1 + crval1
wavelength *= u.angstrom

tqdm.write('Creating plots.')
if args.wavelength_delta:
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title('New pixel positions $-$ old')
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title('New coefficients $-$ old')
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title('Both $-$ old')

    x_lims = (3750, 6950)

    ax3.set_xlabel('Wavelength ($\AA$)')
    ylabel = '$\Delta\lambda (\AA)$'
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    ax1.set_xlim(x_lims)
    ax2.set_xlim(x_lims)
    ax3.set_xlim(x_lims)

    ax1.axhline(0, color='Gray', linewidth=1, alpha=0.8)
    ax2.axhline(0, color='Gray', linewidth=1, alpha=0.8)
    ax3.axhline(0, color='Gray', linewidth=1, alpha=0.8)

    for order in trange(0, 72):
        ax1.plot(old_wv[order, :], pix_wv[order, :]-old_wv[order, :],
                 color='Green', linestyle='-')
        ax2.plot(old_wv[order, :], coeffs_wv[order, :]-old_wv[order, :],
                 color='Red', linestyle='-')
        ax3.plot(old_wv[order, :], new_wv[order, :]-old_wv[order, :],
                 color='Blue', linestyle='-')
    #    ax3.axvline(old_wv[order, 0], 0.49, 1, color='Gray',
    #                linestyle='--', alpha=0.7)
    #    ax3.axvline(old_wv[order, -1], 0, 0.51, color='DarkGray',
    #                linestyle='--', alpha=0.7)
    #    ax1.plot(x_lims, x_lims, color='Black')
    #    ax2.plot(x_lims, x_lims, color='Black')
    #    ax3.plot(x_lims, x_lims, color='Black')

    plt.show()

if args.order_overlap:
    fig2 = plt.figure(figsize=(13, 7))
    ax = fig2.add_subplot(1, 1, 1)

    plot_order = 67
    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel('Photons')
    ax.set_title(r'HARPS Orders {} & {}'.format(plot_order - 1, plot_order))
    ax.set_xlim(left=old_wv[plot_order, 0], right=old_wv[plot_order, -1])
    ax.axvline(6564.614)

    ax.plot(old_wv[plot_order], test_old.photonFluxArray[plot_order],
            color='MediumSeaGreen', linestyle='-', label='Old', alpha=0.7)

    #ax.plot(test_old.barycentricArray[plot_order],
    #        test_old.photonFluxArray[plot_order],
    #        color='Purple', linestyle='--', label='Barycentric', alpha=0.7)

    # RV-corrected
    #ax.plot(test_old.rvCorrectedArray[plot_order],
    #        test_old.photonFluxArray[plot_order],
    #        color='Sienna', linestyle=':', label='RV-corrected', alpha=0.7)


    ax.plot(old_wv[plot_order-1], test_old.photonFluxArray[plot_order-1],
            color='MediumSeaGreen', linestyle='-', alpha=0.7)
    #ax.plot(pix_wv[67], test_pix_pos.photonFluxArray[67],
    #        color='Gold', linestyle=':', label='Pixel positions')
    ax.plot(coeffs_wv[plot_order], test_new_coeffs.photonFluxArray[plot_order],
            color='RoyalBlue', linestyle='--', label='New coefficients',
            alpha=0.7)
    #ax.plot(new_wv[67], test_new.photonFluxArray[67],
    #        color='IndianRed', linestyle=':', label='New')
    ax.plot(coeffs_wv[plot_order-1],
            test_new_coeffs.photonFluxArray[plot_order-1],
            color='RoyalBlue', linestyle='-', alpha=0.7)
    ax.plot(wavelength, data, color='Gray', alpha=0.7, label='1D comparison')

    ax.legend()


if args.velocity_delta:
    fig3 = plt.figure(figsize=(9, 9))
    ax4 = fig3.add_subplot(3, 1, 1)
    ax4.set_title('New pixel positions $-$ old')
    ax5 = fig3.add_subplot(3, 1, 2)
    ax5.set_title('New coefficients $-$ old')
    ax6 = fig3.add_subplot(3, 1, 3)
    ax6.set_title('Both $-$ old')

    x_lims = (3750, 6950)

    ax6.set_xlabel('Wavelength ($\AA$)')
    ylabel = '$\Delta$ v (m/s)'
    ax4.set_ylabel(ylabel)
    ax5.set_ylabel(ylabel)
    ax6.set_ylabel(ylabel)
    ax4.set_xlim(x_lims)
    ax5.set_xlim(x_lims)
    ax6.set_xlim(x_lims)

    ax4.axhline(0, color='Gray', linewidth=1, alpha=0.8)
    ax5.axhline(0, color='Gray', linewidth=1, alpha=0.8)
    ax6.axhline(0, color='Gray', linewidth=1, alpha=0.8)

    for order in trange(0, 72):
        vel_pix_old = wavelength2velocity(old_wv[order, :],
                                          pix_wv[order, :])
        vel_coeffs_old = wavelength2velocity(old_wv[order, :],
                                             coeffs_wv[order, :])
        vel_new_old = wavelength2velocity(old_wv[order, :],
                                          new_wv[order, :])
        ax4.plot(old_wv[order, :], vel_pix_old,
                 color='Green', linestyle='-')
        ax5.plot(old_wv[order, :], vel_coeffs_old,
                 color='Red', linestyle='-')
        ax6.plot(old_wv[order, :], vel_new_old,
                 color='Blue', linestyle='-')

    plt.show()

if args.paper_figure:
    fig = plt.figure(figsize=(7, 5), tight_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_ylim(bottom=-50 * u.m / u.s, top=50 * u.m / u.s)
    ax2.set_ylim(bottom=-50 * u.m / u.s, top=50 * u.m / u.s)
#    ax1.set_xlim(left=3770 * u.angstrom, right=5325 * u.angstrom)
#    ax2.set_xlim(left=5325 * u.angstrom, right=6930 * u.angstrom)
    ax1.set_xlim(left=4895 * u.angstrom, right=5105  * u.angstrom)
    ax2.set_xlim(left=6195 * u.angstrom, right=6405 * u.angstrom)
    ylabel = r'$\Delta v$ (m/s)'
    xlabel = r'Wavelength (\AA)'

    for ax in (ax1, ax2):
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.axhline(0, color='Black', linewidth=2, alpha=1)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#        ax.yaxis.grid(which='major', color='SlateGray', alpha=0.7,
#                      linestyle='-')
#        ax.yaxis.grid(which='minor', color='Gray', alpha=0.5,
#                      linestyle='-.')
#        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=50))
#        ax.xaxis.grid(which='major', color='SlateGray', alpha=0.7,
#                      linestyle='--')
#        ax.xaxis.grid(which='minor', color='Gray', alpha=0.5,
#                      linestyle=':')
#        ax.xaxis.set_tick_params(which='major', width=2, length=6)
#        ax.xaxis.set_tick_params(which='minor', width=1.5, length=4)
#        ax.yaxis.set_tick_params(which='major', width=2, length=5)
#        ax.yaxis.set_tick_params(which='minor', width=1.5, length=3)

    for order in trange(0, 72):
        if order % 2:
            order_color = 'MediumSlateBlue'
            order_color = cmr.torch(0.75)
            order_linestyle = '-'
        else:
            order_color = 'MidnightBlue'
            order_color = cmr.torch(0.2)
            order_linestyle = '-'
        vel_new_old = wavelength2velocity(new_wv[order, :], old_wv[order, :])
        ax1.plot(new_wv[order, :], vel_new_old,
                 color=order_color, linestyle=order_linestyle,
                 linewidth=2)
        ax2.plot(new_wv[order, :], vel_new_old,
                 color=order_color, linestyle=order_linestyle,
                 linewidth=2)

    out_file = '/Users/dberke/Pictures/paper_plots_and_tables/plots/' +\
        'HARPS_old-new_calibration.pdf'
#    plt.show()
    fig.savefig(out_file, bbox_inches='tight', pad_inches=0.01)
