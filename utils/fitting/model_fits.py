#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:44:15 2019

@author: dberke

Code to define a class for a model fit to an absorption line.

"""

import configparser
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unyt as u
from tqdm import tqdm
import varconlib as vcl

matplotlib.rcParams['axes.formatter.useoffset'] = False

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)


class GaussianFit(object):
    """A class to fit an absorption line and store information about the fit.

    """

    def __init__(self, transition, observation, verbose=False):
        """Construct a fit to an absorption feature using a gaussian.

        Parameters
        ----------
        transition : `transition_line.Transition` object
            A `Transition` object representing the absorption feature to fit.
        observation : `obs2d.HARPSFile2DScience` object
            A `HARPSFile2DScience` object to find the absorption feature in.

        Optional
        --------
        verbose : bool, Default : False
            Whether to print out extra diagnostic information while running
            the function.

        """

        self.transition = transition
        self.observation = observation

        # Define some useful numbers and variables

        # The ranges in velocity space to search around to find the minimum of
        # an absorption line.
        search_range_vel = 5 * u.km / u.s

        # The range in velocity space to consider to find the continuum.
        continuum_range_vel = 25 * u.km / u.s

        # The number of pixels either side of the flux minimum to use in the
        # fit.
        pixel_range = 3

        radial_velocity = self.observation.radialVelocity
        # Shift the wavelength being searched for to correct for the radial
        # velocity of the star.
        nominal_wavelength = self.transition.wavelength
        self.correctedWavelength = vcl.shift_wavelength(nominal_wavelength,
                                                        radial_velocity)
        self.correctedWavelength.convert_to_units(u.nm)
        if verbose:
            tqdm.write('Given RV {:.2f}: line {:.4f} should be at {:.4f}'.
                       format(radial_velocity,
                              nominal_wavelength.to(u.nm),
                              self.correctedWavelength.to(u.nm)))
        # Find which order of the echelle spectrum the wavelength falls in.
        self.order = self.observation.findWavelength(self.correctedWavelength,
                                                     mid_most=True)

        baryArray = self.observation.barycentricArray[self.order]
        fluxArray = self.observation.photonFluxArray[self.order]
        errorArray = self.observation.errorArray[self.order]

        # Figure out the range in wavelength space to search around the nominal
        # wavelength for the flux minimum, as well as the range to take for
        # measuring the continuum.
        search_range = vcl.velocity2wavelength(search_range_vel,
                                               self.correctedWavelength)
        continuum_range = vcl.velocity2wavelength(continuum_range_vel,
                                                  self.correctedWavelength)

        low_search_index = vcl.wavelength2index(self.correctedWavelength -
                                                search_range,
                                                baryArray)
        high_search_index = vcl.wavelength2index(self.correctedWavelength +
                                                 search_range,
                                                 baryArray)
        self.lowContinuumIndex = vcl.wavelength2index(self.correctedWavelength
                                                      - continuum_range,
                                                      baryArray)
        self.highContinuumIndex = vcl.wavelength2index(self.correctedWavelength
                                                       + continuum_range,
                                                       baryArray)
        self.centralIndex = low_search_index + \
            fluxArray[low_search_index:high_search_index].argmin()

        self.continuumLevel = fluxArray[self.lowContinuumIndex:
                                        self.highContinuumIndex].max()

        self.fluxMinimum = fluxArray[self.centralIndex]

        self.lowFitIndex = self.centralIndex - pixel_range
        self.highFitIndex = self.centralIndex + pixel_range + 1

        # Grab the wavelengths, fluxes, and errors from the region to be fit.
        self.wavelengths = baryArray[self.lowFitIndex:self.highFitIndex]
        self.fluxes = fluxArray[self.lowFitIndex:self.highFitIndex]
        self.errors = errorArray[self.lowFitIndex:self.highFitIndex]

        self.lineDepth = self.continuumLevel - self.fluxMinimum
        self.normalizedLineDepth = self.lineDepth / self.continuumLevel

        self.initial_guess = (self.lineDepth * -1,
                              self.correctedWavelength.to(u.angstrom).value,
                              0.05,
                              self.continuumLevel)
        if verbose:
            tqdm.write('Attempting to fit line at {:.4f} with initial guess:'.
                       format(self.correctedWavelength))
        if verbose:
            print('Initial parameters are:\n{}\n{}\n{}'.format(
                  *self.initial_guess))

        # Do the fitting:
        try:
            self.popt, self.pcov = curve_fit(vcl.gaussian,
                                             self.wavelengths.value,
                                             self.fluxes,
                                             sigma=self.errors,
                                             absolute_sigma=True,
                                             p0=self.initial_guess,
                                             method='lm', maxfev=10000)
        except (OptimizeWarning, RuntimeError):
            print(self.continuumLevel)
            print(self.lineDepth)
            print(self.initial_guess)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(self.wavelengths.value,
                        self.fluxes,
                        yerr=self.errors,
                        color='Blue', marker='o', linestyle='')
            ax.plot(self.wavelengths.value,
                    vcl.gaussian(self.wavelengths.value, *self.initial_guess),
                    color='Black')
            outfile = Path('/Users/dberke/Pictures/debug_norm.png')
            fig.savefig(str(outfile))
            plt.close(fig)
            raise

        if verbose:
            print(self.popt)
            print(self.pcov)

        # Recover the fitted values for the parameters:
        self.amplitude = self.popt[0]
        self.centroid = self.popt[1] * u.angstrom
        self.standardDev = self.popt[2] * u.angstrom

        if self.amplitude > 0:
            if verbose:
                tqdm.write('Bad fit for {}'.format(
                        self.transition.wavelength))
            raise RuntimeError('Positive amplitude from fit.')

        # Find 1-σ errors from the covariance matrix:
        self.perr = np.sqrt(np.diag(self.pcov))

        self.amplitudeErr = self.perr[0]
        self.centroidErr = self.perr[1] * u.angstrom
        self.standardDevErr = self.perr[2] * u.angstrom

        # Compute the χ^2 value:
        residuals = self.fluxes - \
            vcl.gaussian(self.wavelengths.value, *self.popt)

        self.chiSquared = sum((residuals / self.errors) ** 2)
        self.chiSquaredNu = self.chiSquared / 3  # ν = 7 (pixels) - 4 (params)
        if (self.chiSquaredNu > 1):
            self.centroidErr *= np.sqrt(self.chiSquaredNu)

        # Find the full width at half max.
        # 2.354820 ≈ 2 * sqrt(2 * ln(2)), the relationship of FWHM to the
        # standard deviation of a Gaussian.
        self.FWHM = 2.354820 * self.standardDev
        self.FWHMErr = 2.354820 * self.standardDevErr
        self.velocityFWHM = vcl.wavelength2velocity(self.centroid,
                                                    self.centroid +
                                                    self.FWHM)
        self.velocityFWHMErr = vcl.wavelength2velocity(self.centroid,
                                                       self.centroid +
                                                       self.FWHMErr)

        # Compute the offset between the input wavelength and the wavelength
        # found in the fit.
        self.offset = self.correctedWavelength - self.centroid
        self.velocityOffset = vcl.wavelength2velocity(self.correctedWavelength,
                                                      self.centroid)
        # TODO: Error for velocity offset.

        if verbose:
            print(self.continuumLevel)
            print(self.fluxMminimum)
            print(self.wavelengths)

    def plotFit(self, close_up, context, verbose=False):
        """Plot a graph of this fit.

        This method will produce a very close-in plot of just the fitted region
        itself, in order to check out the fit has worked out.

        Optional
        --------
        close_up : string or `pathlib.Path`
            The file name to save a close-up plot of the fit to.
        context : string of `pathlib.Path`
            The file name to save a wider context (±25 km/s) around the fitted
            feature to.
        verbose : bool, Default : False
            If *True*, the function will print out additional information as it
            runs.

        """

        # Set up the figure.
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('Flux (photo-electrons)')
#        ax.xaxis.set_major_locator(ticker.FixedLocator(self.wavelengths
#                                                       .to(u.angstrom).value))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        ax.set_xlim(left=self.observation.barycentricArray
                    [self.order, self.lowContinuumIndex - 1],
                    right=self.observation.barycentricArray
                    [self.order, self.highContinuumIndex + 1])
        # Set y-limits so a fit doesn't balloon the plot scale out.
        ax.set_ylim(top=self.continuumLevel * 1.1,
                    bottom=self.fluxMinimum * 0.9)

        # Plot the expected and measured wavelengths.
        ax.axvline(self.correctedWavelength.to(u.angstrom),
                   color='LightSteelBlue', linestyle=':',
                   label='RV-corrected transition λ',)
        ax.axvline(self.centroid.to(u.angstrom),
                   color='IndianRed',
                   label='Fit ({:.2f})'.
                   format(self.velocityOffset.to(u.m/u.s)),
                   linestyle='-')
        # Plot the actual data.
        self.observation.plotErrorbar(self.order, ax,
                                      min_index=self.lowContinuumIndex,
                                      max_index=self.highContinuumIndex,
                                      color='SandyBrown', ecolor='Sienna',
                                      label='Flux', barsabove=True)

        # Generate some x-values across the plot range.
        x = np.linspace(self.observation.barycentricArray[self.order,
                        self.lowContinuumIndex].value,
                        self.observation.barycentricArray[self.order,
                        self.highContinuumIndex].value, 1000)
        # Plot the initial guess for the gaussian.
        ax.plot(x, vcl.gaussian(x, *self.initial_guess),
                color='SlateGray', label='Initial guess',
                linestyle='--', alpha=0.8)
        # Plot the fitted gaussian.
        ax.plot(x, vcl.gaussian(x, *self.popt),
                color='DarkGreen',
                linestyle='-', label='Fitted Gaussian')

        ax.legend()
        # Save the resultant plot.
        fig.savefig(str(context))
        if verbose:
            tqdm.write('Created wider context plot at {}'.format(context))

        # Now create a close-in version to focus on the fit.
        ax.set_xlim(left=self.observation.barycentricArray
                    [self.order, self.lowFitIndex - 1],
                    right=self.observation.barycentricArray
                    [self.order, self.highFitIndex])
        ax.set_ylim(top=self.fluxes.max() * 1.08,
                    bottom=self.fluxes.min() * 0.93)

        fig.savefig(str(close_up))
        if verbose:
            tqdm.write('Created close up plot at {}'.format(close_up))
        plt.close(fig)
