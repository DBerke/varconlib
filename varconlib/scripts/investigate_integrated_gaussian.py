#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:41:31 2019

@author: dberke
"""

# A script to generate a mock Gaussian feature and measure the difference
# between measuring it in the center of mock pixels or by integrating under it.

import argparse
from copy import copy
import math
from pprint import pprint

from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import unyt as u

from varconlib.fitting import gaussian, integrated_gaussian
from varconlib.miscellaneous import wavelength2velocity

# Don't plot wavelengths as 10^3.
matplotlib.rcParams['axes.formatter.useoffset'] = False

class MockAbsorptionFeature(object):
    """A Gaussian with negative amplitude and Gaussian photon noise added.
    """

    def __init__(self, x_range, amplitude, median, sigma, baseline,
                 noise=False):
        """

        x_range : list of either ints or tuples
            If given a list of integers, will use a Gaussian function.
            If given a list of pixel (start, end) values, will use an
            integrated Gaussian.
        amplitude : float
            The amplitude of the Gaussian. Must be Real.
        mu : float
            The median (also the center) of the Gaussian. Must be Real.
        sigma : float
            The standard deviation of the Gaussian. Must be non-zero.
        baseline : float
            The baseline of the Gaussian. Must be Real.
        noise : bool, Defaul : False
            Whether to add Gaussian noise to the generated mock values.

        """

        if isinstance(x_range[0], (int, float, np.float64)):
            self.baseCurve = np.array([gaussian(x, amplitude, median, sigma,
                                       baseline) for x in x_range])
#            pprint(self.baseCurve)
#            print('Used Gaussian.')
        elif isinstance(x_range[0], (tuple, np.ndarray)):
            self.baseCurve = np.array([integrated_gaussian(x, amplitude,
                                       median, sigma, baseline) for
                                       x in x_range])
#            pprint(self.baseCurve)
#            print('Used integrated Gaussian')
        else:
            print(f'First item of x_range is {type(x_range[0])}')
            print(x_range[:2])
            raise ValueError

        if noise:
            flux_list = []
            for pixel in self.baseCurve:
                noisy_value = np.random.normal(loc=0, scale=math.sqrt(pixel))
                flux_list.append(pixel + noisy_value)
            self.baseCurve = np.array(flux_list)

        self.noise = np.sqrt(self.baseCurve)


# Start main script here
desc = 'Fit a mock Gaussian feature with photon noise and plot results.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-amp', '--amplitude', action='store', type=float,
                    help='The amplitude of the Gaussian.')
parser.add_argument('-mu', '--median', action='store', type=float,
                    help='The median (mu) of the Gaussian.')
parser.add_argument('-sigma', '--stddev', action='store', type=float,
                    help='The standard deviation (sigma) of the Gaussian.')
parser.add_argument('-base', '--baseline', action='store', type=float,
                    help='The baseline (offset from 0) of the Gaussian.')

type_group = parser.add_mutually_exclusive_group()
type_group.add_argument('-g', '--gaussian', action='store_true', default=False,
                        help='Create mock data from a Gaussian.')
type_group.add_argument('-i', '--integrated_gaussian', action='store_true',
                        default=False,
                        help='Create mock data from an integrated Gaussian.')

parser.add_argument('-f', '--pixel-phase', type=float, action='store',
                    help='Pixel offset between [-0.5, 0.5] to apply.')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Produce a plot of the output.')
parser.add_argument('-n', '--noise', action='store_true', default=False,
                    help='Flag to add noise to the simulated data.')

args = parser.parse_args()

amplitude = args.amplitude
mu = args.median * u.angstrom
sigma = args.stddev * u.angstrom
baseline = args.baseline

if not (-0.5 <= args.pixel_phase <= 0.5):
    raise ValueError('Pixel phase outside [-0.5, 0.5]!')

pixel_size = 0.015 * u.angstrom
pix_phase = pixel_size * args.pixel_phase

#print('Pixel phase of {} applied.'.format(args.pixel_phase))

initial_params = [amplitude + np.random.normal(loc=0, scale=amplitude/-100),
                  mu.value + np.random.normal(loc=0, scale=sigma/2),
                  sigma.value + np.random.normal(loc=0, scale=sigma/6),
                  baseline + np.random.normal(loc=0, scale=baseline/100)]
#print(initial_params)

num_pix = 50

start = mu - (pixel_size * (num_pix / 2) + pix_phase)

# Generate a list of tuples of start and stop values of pixels.
pixel_low_edges = []
pixel_high_edges = []
edge = start - 0.0075 * u.angstrom

for i in range(num_pix):
    pixel_low_edges.append(copy(edge))
    edge += pixel_size
    pixel_high_edges.append(copy(edge))

midpoints = np.array([(x + y) / 2 for x, y in zip(pixel_low_edges,
                      pixel_high_edges)])
end = midpoints[-1] * u.angstrom

#print(pixel_low_edges[25:27])
#print(midpoints[25:27])
#print(pixel_high_edges[25:27])
pixels = np.array([x for x in zip(pixel_low_edges, pixel_high_edges)])
#print(pixels[:10])

if args.gaussian:
    mock_feature = MockAbsorptionFeature(midpoints, amplitude, mu.value,
                                         sigma.value, baseline,
                                         noise=args.noise)
elif args.integrated_gaussian:
    mock_feature = MockAbsorptionFeature(pixels, amplitude, mu.value,
                                         sigma.value, baseline,
                                         noise=args.noise)

popt, pcov = curve_fit(gaussian,
                       midpoints,
                       mock_feature.baseCurve,
                       sigma=mock_feature.noise, absolute_sigma=True,
                       p0=initial_params,
                       method='lm', maxfev=500)

popt2, pcov2 = curve_fit(integrated_gaussian,
                         (pixel_low_edges, pixel_high_edges),
                         mock_feature.baseCurve,
                         sigma=mock_feature.noise, absolute_sigma=True,
                         p0=initial_params,
                         method='lm', maxfev=1000)
#print(popt)
#print(popt2)
opt_amp1 = popt[0]
opt_mu1 = popt[1]
opt_sigma1 = popt[2]
opt_baseline1 = popt[3]


opt_amp2 = popt2[0]
opt_mu2 = popt2[1]
opt_sigma2 = popt2[2]
opt_baseline2 = popt2[3]

offset_gauss = wavelength2velocity(mu, opt_mu1 * u.angstrom)
offset_igauss = wavelength2velocity(mu, opt_mu2 * u.angstrom)

#print('Offset for Gaussian fit: {}'.format(offset_gauss))
#print('Offset for int. Gaussian fit: {}'.format(offset_igauss))

center_values = np.array([gaussian(x, opt_amp1, opt_mu1, opt_sigma1,
                          opt_baseline1) for x in midpoints])
integrated_values = np.array([integrated_gaussian(p, opt_amp2, opt_mu2,
                             opt_sigma2, opt_baseline2)
                             for p in pixels])

gauss_residuals = mock_feature.baseCurve - center_values
igauss_residuals = mock_feature.baseCurve - center_values


gauss_chi_squared = sum((gauss_residuals / mock_feature.noise) ** 2)
igauss_chi_squared = sum((igauss_residuals / mock_feature.noise) ** 2)

#print(gauss_chi_squared)
#print(igauss_chi_squared)
#
#nu = len(mock_feature.baseCurve) -  4
#print(nu)
#gauss_chi_squared_nu = gauss_chi_squared / nu
#igauss_chi_squared_nu = igauss_chi_squared / nu
#
#print(gauss_chi_squared_nu)
#print(igauss_chi_squared_nu)

#print('{:.3f} {:.6f} {:.6f}'.format(args.pixel_phase, offset_gauss,
#      offset_igauss))

print('{:.3f} {:.8f} {:.8f} {:.8f} {:.8f}'.format(args.pixel_phase, opt_amp1,
      opt_mu1, opt_sigma1, opt_baseline1))

if args.plot:
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0)

    ax.set_ylabel('Normalized simulated flux')
    # Define the points in terms of velocity offsets
    vel_offsets = [wavelength2velocity(mu, x * u.angstrom) for x in midpoints]

#    if args.gaussian:
#        ax.set_title('Mock Gaussian.')
#    elif args.integrated_gaussian:
#        ax.set_title('Mock integrated Gaussian.')

#    ax.set_xlim(left=-5 * sigma + mu, right=5 * sigma + mu)

    ax.step(vel_offsets, mock_feature.baseCurve/baseline, linestyle='-',
            marker=None,
            color='Black', where='mid', label='Mock data')
    #ax.errorbar(midpoints, mock_feature.baseCurve, yerr=mock_feature.noise,
    #            linestyle='', marker=None, ecolor='Gray')
    plot_range = np.linspace(start.value, end.value, 500)
    ax.plot(vel_offsets, gaussian(midpoints, amplitude,
                                  mu.value, sigma.value, baseline)/baseline,
            color='Red',
            linestyle='-', label='Gaussian (initial parameters)')
#    ax.plot(plot_range, gaussian(plot_range, *initial_params),
#            color='SteelBlue', linestyle='--', label='Init params Gauss')
#
#    ax.plot(midpoints, [integrated_gaussian(pixel, *initial_params) for pixel
#                     in pixels],
#            color='Goldenrod', linestyle=':', label='Init params Int. Gauss.')
#
    ax.plot(vel_offsets, gaussian(midpoints, opt_amp1, opt_mu1,
                                  opt_sigma1, opt_baseline1)/baseline,
            marker='', color='Blue', linestyle='--',
            label='Fitted Gaussian')
#    ax.plot(midpoints, integrated_values, marker='x', color='Green',
#            linestyle='',
#            label='Integrated Gaussian')
#    ax.plot(plot_range, gaussian(plot_range, *popt2), color='Red',
#            label='Fitted Int Gauss')

    ax.legend()
    delta_fluxes = [x - y for x, y in zip(center_values, integrated_values)]

#    ax2 = fig.add_subplot(2, 1, 2)

#    title = '$\mu={}, \sigma={}$, depth = {:.4f}'.format(args.median,
#            args.stddev, -1 * args.amplitude / args.baseline)


#    residuals1 = (mock_feature.baseCurve -
#                  center_values) / mock_feature.noise
#    residuals2 = (mock_feature.baseCurve -
#                  integrated_values) / mock_feature.noise
    residuals3 = (mock_feature.baseCurve/baseline -
                  gaussian(midpoints, amplitude, mu.value, sigma.value,
                           baseline)/baseline) / mock_feature.noise/baseline
    residuals4 = (mock_feature.baseCurve/baseline -
                  gaussian(midpoints, opt_amp1, opt_mu1,
                           opt_sigma1, opt_baseline1)/baseline) /\
                  mock_feature.noise/baseline



#    ax2.set_title(title)
    ax2.set_ylabel('Residuals')

    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

#    ax2.plot(vel_offsets, residuals1, marker='D',
#             color=(0.5, 0.5, 0.5, 0.5), markeredgecolor='RebeccaPurple',
#             label='Data$-$Gaussian')
#    ax2.plot(vel_offsets, residuals2, marker='o',
#             color=(0.5, 0.5, 0.5, 0.5), markeredgecolor='Teal',
#             label='Data$-$Int. Gaussian')
    ax2.plot(vel_offsets, residuals3, marker='+',
             color=(0.5, 0.6, 0.4, 0.3), markeredgecolor='Tomato',
             label='Int. Gauss. (data)$-$Gauss.')
    ax2.plot(vel_offsets, residuals4, marker='x',
             color=(0.5, 0.6, 0.4, 0.3), markeredgecolor='LightSkyBlue',
             label='Int. Gauss. (data)$-$Fitted Gauss.')
#    ax2.axvline(offset_gauss, ymin=0.4, ymax=0.5, color='PaleVioletRed')
#    ax2.axvline(offset_igauss, ymin=0.5, ymax=0.6, color='DodgerBlue')

    ax2.legend()

    ax3.set_ylabel('Residuals')
    ax3.set_xlabel('Velocity (m/s)')

    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

    ax3.set_ylim(bottom=-3.2e-17, top=3.2e-17)

    ax3.plot(vel_offsets, residuals3, marker='+',
             color=(0.5, 0.6, 0.4, 0.3), markeredgecolor='Tomato',
             label='Int. Gauss. (data)$-$Gauss.')
    ax3.plot(vel_offsets, residuals4, marker='x',
             color=(0.5, 0.6, 0.4, 0.3), markeredgecolor='LightSkyBlue',
             label='Int. Gauss. (data)$-$Fitted Gauss.')

    ax3.legend()
    plt.show()
