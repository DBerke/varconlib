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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

import obs2d
from varconlib import wavelength2velocity

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--update', action='store_true',
                    help='Flag to update the wavelength solution.')
args = parser.parse_args()

data_dir = Path('/Users/dberke/code/data/')
test_old_file = data_dir / 'test_old.fits'
test_new_coeffs_file = data_dir / 'test_new_coefficients.fits'
test_pix_pos_file = data_dir / 'test_pix_positions.fits'
test_new_file = data_dir / 'test_new.fits'

test_flat_file = Path('/Users/dberke/Downloads/ceres2.flat.fits')

if args.update:
    update_status = ['WAVE', 'BARY']  # use '['WAVE', 'BARY']'
else:
    update_status = []

test_old = obs2d.HARPSFile2DScience(test_old_file, use_new_coefficients=False,
                                    use_pixel_positions=False)
print('Creating new coefficients file...')
test_new_coeffs = obs2d.HARPSFile2DScience(test_new_coeffs_file,
                                           use_new_coefficients=True,
                                           use_pixel_positions=False,
                                           update=update_status)
print('Creating pixel positions file...')
test_pix_pos = obs2d.HARPSFile2DScience(test_pix_pos_file,
                                        update=update_status,
                                        use_new_coefficients=False,
                                        use_pixel_positions=True)
print('Creating new calibration with both...')
test_new = obs2d.HARPSFile2DScience(test_new_file, update=update_status,
                                    use_new_coefficients=True,
                                    use_pixel_positions=True)

with fits.open(test_flat_file) as hdulist:
    nx = hdulist[0].header['NAXIS1']
    cdelta1 = hdulist[0].header['CDELT1']
    crval1 = hdulist[0].header['CRVAL1']
    data = hdulist[0].data * 70000

wavelength = np.linspace(0, len(data), len(data))*cdelta1 + crval1


old_wv = test_old.wavelengthArray
pix_wv = test_pix_pos.wavelengthArray
coeffs_wv = test_new_coeffs.wavelengthArray
new_wv = test_new.wavelengthArray
#for order in range(0, 72):
#    offset = (coeffs_wv[order, -1] - coeffs_wv[order, 0]) / 2
#    coeffs_wv[order] -= offset
#    new_wv[order] -= offset



#print(old_wv[71, :10])
#print(pix_wv[71, :10])
#print(coeffs_wv[71, :10])
#print(new_wv[71, :10])

print('Creating plots.')
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


#plt.show()

fig2 = plt.figure(figsize=(13, 7))
ax = fig2.add_subplot(1, 1, 1)


plot_order = 67
ax.set_xlabel('Wavelength ($\AA$)')
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
        color='RoyalBlue', linestyle='--', label='New coefficients', alpha=0.7)
#ax.plot(new_wv[67], test_new.photonFluxArray[67],
#        color='IndianRed', linestyle=':', label='New')
ax.plot(coeffs_wv[plot_order-1], test_new_coeffs.photonFluxArray[plot_order-1],
        color='RoyalBlue', linestyle='-', alpha=0.7)
ax.plot(wavelength, data, color='Gray', alpha=0.7, label='1D comparison')

ax.legend()


fig3 = plt.figure(figsize=(9,9))
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
    vel_pix_old = [wavelength2velocity(y, x) for x, y in
                   tqdm(zip(pix_wv[order, :], old_wv[order, :]),
                        total=4096)]
    vel_coeffs_old = [wavelength2velocity(y, x) for x, y in
                      tqdm(zip(coeffs_wv[order, :], old_wv[order, :]),
                           total=4096)]
    vel_new_old = [wavelength2velocity(y, x) for x, y in
                   tqdm(zip(new_wv[order, :], old_wv[order, :]),
                        total=4096)]
    ax4.plot(old_wv[order, :], vel_pix_old,
             color='Green', linestyle='-')
    ax5.plot(old_wv[order, :], vel_coeffs_old,
             color='Red', linestyle='-')
    ax6.plot(old_wv[order, :], vel_new_old,
             color='Blue', linestyle='-')


plt.show()
