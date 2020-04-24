#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:18:30 2019

@author: dberke


A script to plot the results of testing Gaussian vs. integrated Gaussian
fitting procedures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
import unyt as u

from varconlib.fitting import gaussian
import varconlib as vcl

parser = argparse.ArgumentParser()
parser.add_argument('SNR', action='store', type=str,
                    help='SNR suffix value of the files to read.')

parser.add_argument('-n', '--noise', action='store_true', default=False,
                    help='Plot results with noise added.')

args = parser.parse_args()
print(args.noise)
SNR = args.SNR

base_dir = Path("/Users/dberke/code/data")
file_gauss_noise = base_dir / "fittingTestGaussianSNR{}.txt".format(SNR)
file_igauss_noise = base_dir / "fittingTestIntGaussianSNR{}.txt".format(SNR)
if (not file_gauss_noise.exists()) or (not file_igauss_noise.exists()):
    print(file_gauss_noise)
    print(file_igauss_noise)
    raise FileNotFoundError("Couldn't find one of these files.")

file_gauss_nonoise = base_dir / "fittingTestGaussianNoiseless.txt"
file_igauss_nonoise = base_dir / "fittingTestIntGaussianNoiseless.txt"

with open(file_gauss_noise, 'r') as f:
    data_gauss_noise = np.loadtxt(f, dtype=float, skiprows=1,
                                  usecols=(0, 1, 3))

with open(file_gauss_nonoise, 'r') as f:
    data_gauss_nonoise = np.loadtxt(f, dtype=float, skiprows=1,
                                    usecols=(0, 1, 3))

with open(file_igauss_noise, 'r') as f:
    data_igauss_noise = np.loadtxt(f, dtype=float, skiprows=1,
                                   usecols=(0, 1, 3))

with open(file_igauss_nonoise, 'r') as f:
    data_igauss_nonoise = np.loadtxt(f, dtype=float, skiprows=1,
                                     usecols=(0, 1, 3))

file_params_noise = base_dir / 'fittingTestIntGaussianParametersNoise.txt'
file_params_nonoise = base_dir / 'fittingTestIntGaussianParametersNoNoise.txt'

with open(file_params_noise, 'r') as f:
    data_params_noise = np.loadtxt(f, dtype=float, skiprows=1,
                                   usecols=(0, 1, 2, 3, 4))

with open(file_params_nonoise, 'r') as f:
    data_params_nonoise = np.loadtxt(f, dtype=float, skiprows=1,
                                     usecols=(0, 1, 2, 3, 4))

print('Read data from all files.')

sigma_gauss = np.std(data_gauss_noise[:, 1])
sigma_igauss = np.std(data_igauss_noise[:, 1])
print('Standard deviations are:')
print('{:.6f} m/s (Gaussian)'.format(sigma_gauss))
print('{:.6f} m/s (integrated Gaussian)'.format(sigma_igauss))

med_gauss = np.median(data_gauss_noise[:, 1])
med_igauss = np.median(data_igauss_noise[:, 1])
print('Medians are:')
print('{:.5f} m/s (Gaussian)'.format(med_gauss))
print('{:.5f} m/s (integrated Gaussian)'.format(med_igauss))

mean_gauss = np.mean(data_gauss_noise[:, 1])
mean_igauss = np.mean(data_igauss_noise[:, 1])
print('Means are:')
print('{:.5f} m/s (Gaussian)'.format(mean_gauss))
print('{:.5f} m/s (integrated Gaussian)'.format(mean_igauss))



#fig = plt.figure(figsize=(10, 8))
#ax1 = fig.add_subplot(2, 2, 1)
#ax2 = fig.add_subplot(2, 2, 2)
#ax3 = fig.add_subplot(2, 2, 3)
#ax4 = fig.add_subplot(2, 2, 4)
#
#ax1.set_title('Gaussian mock data, noise')
#ax2.set_title('Gaussian mock data, no noise')
#ax3.set_title('Integrated Gaussian mock data, noise')
#ax4.set_title('Integrated Gaussian mock data, no noise')
#
#y_label = 'Offset (m/s)'
#
#ax1.set_ylabel(y_label)
#ax3.set_ylabel(y_label)
#
#x_label = 'Pixel phase'
#
#ax3.set_xlabel(x_label)
#ax4.set_xlabel(x_label)
#
#limits = {'left': -0.51, 'right': 0.51}
#
#ax1.set_xlim(**limits)
#ax2.set_xlim(**limits)
#ax3.set_xlim(**limits)
#ax4.set_xlim(**limits)
#
#g_color = 'SlateBlue'
#i_color = 'MediumSeaGreen'
#size = 5
#
#axes = (ax1, ax2, ax3, ax4)
#data_groups = (data_gauss_noise, data_gauss_nonoise,
#               data_igauss_noise, data_igauss_nonoise)
#
#for ax, data in zip(axes, data_groups):
#    ax.plot(data[:, 0], data[:, 1], alpha=0.8,
#            color=g_color, markersize=size, marker='+', linestyle='',
#            label="Gaussian fit")
#    ax.plot(data[:, 0], data[:, 2], alpha=0.6,
#            color=i_color, markersize=size-1, marker='x', linestyle='',
#            label="Integrated Gaussian fit")
#
#    ax.legend()
#
#fig2 = plt.figure(figsize=(8, 7))
#ax5 = fig2.add_subplot(2, 2, 1)
#ax6 = fig2.add_subplot(2, 2, 2)
#ax7 = fig2.add_subplot(2, 2, 3)
#ax8 = fig2.add_subplot(2, 2, 4)
#
#ax5.set_title('Gaussian mock data, noise')
#ax6.set_title('Gaussian mock data, no noise')
#ax7.set_title('Integrated Gaussian mock data, noise')
#ax8.set_title('Integrated Gaussian mock data, no noise')
#
#ax7.set_xlabel(y_label)
#ax8.set_xlabel(y_label)
#
#axes2 = (ax5, ax6, ax7, ax8)
#
#bin_num = 30
#
#for ax, data in zip(axes2, data_groups):
#    ax.hist(data[:, 1], bins=bin_num, alpha=0.8, color=g_color, histtype='bar',
#            label='Gaussian')
#    ax.hist(data[:, 2], bins=bin_num, alpha=0.5, color=i_color, histtype='bar',
#            label='Integrated Gaussian')
#
#    ax.legend()

fig3, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2, sharex='col',
                                                 figsize=(10, 7))
fig3.subplots_adjust(hspace=0)
#fig3.subplots_adjust(wspace=0)

ax9.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
#ax10.yaxis.set_label_position('right')
#ax10.yaxis.tick_right()
#ax10.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax11.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
#ax12.yaxis.set_label_position('right')
#ax12.yaxis.tick_right()
ax12.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))

if args.noise:
    print('Using noise-added results')
    title = 'Fitted Parameters (Gauss.) vs. Mock Data (Int. Gauss.) (noise)'
    pix_phase = data_params_noise[:, 0]
    amplitude = data_params_noise[:, 1]
    mu = data_params_noise[:, 2]
    sigma = data_params_noise[:, 3]
    baseline = data_params_noise[:, 4]

else:
    print('Using no-noise results.')
    title = 'Fitted Parameters (Gauss.) vs. Mock Data (Int. Gauss.) (no noise)'
    pix_phase = data_params_nonoise[:, 0]
    amplitude = data_params_nonoise[:, 1]
    mu = data_params_nonoise[:, 2]
    sigma = data_params_nonoise[:, 3]
    baseline = data_params_nonoise[:, 4]

#fig3.suptitle(title)
ax9.set_ylabel('Amplitude (normalized)')
ax10.set_ylabel('Mu (m/s)')
ax11.set_ylabel('Sigma (normalized)')
ax12.set_ylabel('Baseline (normalized)')

ax11.set_xlabel('Pixel phase')
ax12.set_xlabel('Pixel phase')

ax9.axhline(0, color='Red')
ax10.axhline(0, color='Red')
ax11.axhline(0, color='Red')
ax12.axhline(0, color='Red')


mu_vel = [vcl.wavelength2velocity(x, 5500) for x in mu]

norm_amplitude = (-100000000 - amplitude) / -100000000
#norm_mu = mu / 5500
norm_sigma = (0.05 - sigma) / 0.05
norm_baseline = (180000000 - baseline) / 180000000

ax9.axhline(np.median(norm_amplitude), color='DarkOliveGreen', linestyle='--')
ax10.axhline(np.median(mu_vel), color='DarkOliveGreen', linestyle='--')
ax11.axhline(np.median(norm_sigma), color='DarkOliveGreen', linestyle='--')
ax12.axhline(np.median(norm_baseline), color='DarkOliveGreen', linestyle='--')


ax9.fill_between((-0.53, 0.53), np.median(norm_amplitude) -
                 np.std(norm_amplitude),
                 np.median(norm_amplitude) + np.std(norm_amplitude),
                 color='Tomato', alpha=0.3)
ax10.fill_between((-0.53, 0.53), np.median(mu_vel) - np.std(mu_vel),
                  np.median(mu_vel) + np.std(mu_vel),
                  color='Tomato', alpha=0.3)
ax11.fill_between((-0.53, 0.53), np.median(norm_sigma) - np.std(norm_sigma),
                  np.median(norm_sigma) + np.std(norm_sigma),
                  color='Tomato', alpha=0.3)
ax12.fill_between((-0.53, 0.53), np.median(norm_baseline) -
                  np.std(norm_baseline),
                  np.median(norm_baseline) + np.std(norm_baseline),
                  color='Tomato', alpha=0.3)

ax9.plot(pix_phase, norm_amplitude, linestyle='',
         marker='o', alpha=0.6, markersize=4)
ax10.plot(pix_phase, mu_vel, linestyle='',
          marker='o', alpha=0.6, markersize=4)
ax11.plot(pix_phase, norm_sigma, linestyle='',
          marker='o', alpha=0.6, markersize=4)
ax12.plot(pix_phase, norm_baseline, linestyle='',
          marker='o', alpha=0.6, markersize=4)

plt.tight_layout()
plt.show()
