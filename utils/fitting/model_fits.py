#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:44:15 2019

@author: dberke

Code to define a class for a model fit to an absorption line.

"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm
import unyt as u

from exceptions import (PositiveAmplitudeError, WavelengthNotFoundInArrayError)
import varconlib as vcl

# This line prevents the wavelength formatting from being in the form of
# scientific notation.
matplotlib.rcParams['axes.formatter.useoffset'] = False

class GaussianFit(object):
    """A class to fit an absorption line and store information about the fit.

    """

    def __init__(self, transition, observation, radial_velocity=None,
                 close_up_plot_path=None, context_plot_path=None,
                 integrated=True, verbose=False):
        """Construct a fit to an absorption feature using a Gaussian or
        integrated Gaussian.

        Parameters
        ----------
        transition : `transition_line.Transition` object
            A `Transition` object representing the absorption feature to fit.
        observation : `obs2d.HARPSFile2DScience` object
            A `HARPSFile2DScience` object to find the absorption feature in.

        Optional
        --------
        radial_velocity : `unyt.unyt_quantity`
            A radial velocity (dimensions of length / time) for the object in
            the observation. Most of the time the radial velocity should be
            picked up from the observation itself, but for certain objects
            such as asteroids the supplied radial velocity may not be correct.
            In such cases, this parameter can be used to override the given
            radial velocity.
        close_up_plot_path : string or `pathlib.Path`
            The file name to save a close-up plot of the fit to.
        context_plot_path : string or `pathlib.Path`
            The file name to save a wider context plot (±25 km/s) around the
            fitted feature to.
        integrated : bool, Default : True
            Controls whether to attempt to fit a feature with an integrated
            Gaussian instead of a Gaussian.
        verbose : bool, Default : False
            Whether to print out extra diagnostic information while running
            the function.

        """

        # Store the transition
        self.transition = transition
        # Grab the observation date from the observation
        self.dateObs = observation.dateObs

        # Define some useful numbers and variables

        # The ranges in velocity space to search around to find the minimum of
        # an absorption line.
        search_range_vel = 5 * u.km / u.s

        # The range in velocity space to consider to find the continuum.
        continuum_range_vel = 25 * u.km / u.s

        # The number of pixels either side of the flux minimum to use in the
        # fit.
        pixel_range = 3

        # If no radial velocity is given, use the radial velocity from the
        # supplied observation. This is mostly for use with things like
        # asteroids that might not have a radial velocity assigned.
        if radial_velocity is None:
            radial_velocity = observation.radialVelocity

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
        self.order = observation.findWavelength(self.correctedWavelength,
                                                observation.barycentricArray,
                                                mid_most=True)
        if verbose:
            tqdm.write('Wavelength found in order {}'.format(self.order))

        if self.order is None:
            raise WavelengthNotFoundInArrayError('Wavelength not found!')

        self.baryArray = observation.barycentricArray[self.order]
        self.fluxArray = observation.photonFluxArray[self.order]
        self.errorArray = observation.errorArray[self.order]

        # Figure out the range in wavelength space to search around the nominal
        # wavelength for the flux minimum, as well as the range to take for
        # measuring the continuum.
        search_range = vcl.velocity2wavelength(search_range_vel,
                                               self.correctedWavelength)

        continuum_range = vcl.velocity2wavelength(continuum_range_vel,
                                                  self.correctedWavelength)

        low_search_index = vcl.wavelength2index(self.correctedWavelength -
                                                search_range,
                                                self.baryArray)

        high_search_index = vcl.wavelength2index(self.correctedWavelength +
                                                 search_range,
                                                 self.baryArray)

        self.lowContinuumIndex = vcl.wavelength2index(self.correctedWavelength
                                                      - continuum_range,
                                                      self.baryArray)
        self.highContinuumIndex = vcl.wavelength2index(self.correctedWavelength
                                                       + continuum_range,
                                                       self.baryArray)
        self.centralIndex = low_search_index + \
            self.fluxArray[low_search_index:high_search_index].argmin()

        self.continuumLevel = self.fluxArray[self.lowContinuumIndex:
                                             self.highContinuumIndex].max()

        self.fluxMinimum = self.fluxArray[self.centralIndex]

        self.lowFitIndex = self.centralIndex - pixel_range
        self.highFitIndex = self.centralIndex + pixel_range + 1

        # Grab the wavelengths, fluxes, and errors from the region to be fit.
        self.wavelengths = self.baryArray[self.lowFitIndex:self.highFitIndex]
        self.fluxes = self.fluxArray[self.lowFitIndex:self.highFitIndex]
        self.errors = self.errorArray[self.lowFitIndex:self.highFitIndex]

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
            print('Initial parameters are:\n{}\n{}\n{}\n{}'.format(
                  *self.initial_guess))

        # Do the fitting:
        try:
            if integrated:
                wavelengths_lower = observation.pixelLowerArray
                wavelengths_upper = observation.pixelUpperArray

                pixel_edges_lower = wavelengths_lower[self.order,
                                                      self.lowFitIndex:
                                                          self.highFitIndex]
                pixel_edges_upper = wavelengths_upper[self.order,
                                                      self.lowFitIndex:
                                                          self.highFitIndex]
                self.popt, self.pcov = curve_fit(vcl.integrated_gaussian,
                                                 (pixel_edges_lower.value,
                                                  pixel_edges_upper.value),
                                                 self.fluxes,
                                                 sigma=self.errors,
                                                 absolute_sigma=True,
                                                 p0=self.initial_guess,
                                                 method='lm', maxfev=10000)
            else:
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
            self.plotFit(close_up_plot_path, context_plot_path,
                         plot_fit=False, verbose=False)
            raise

        if verbose:
            print(self.popt)
            print(self.pcov)

        # Recover the fitted values for the parameters:
        self.amplitude = self.popt[0]
        self.median = self.popt[1] * u.angstrom
        self.sigma = self.popt[2] * u.angstrom

        if self.amplitude > 0:
            if verbose:
                tqdm.write('Bad fit for {}'.format(
                        self.transition.wavelength))
            self.plotFit(close_up_plot_path, context_plot_path,
                         plot_fit=True, verbose=False)
            raise PositiveAmplitudeError('Positive amplitude from fit.')

        # Find 1-σ errors from the covariance matrix:
        self.perr = np.sqrt(np.diag(self.pcov))

        self.amplitudeErr = self.perr[0]
        self.medianErr = self.perr[1] * u.angstrom
        self.medianErrVel = abs(vcl.wavelength2velocity(self.median,
                                                        self.median +
                                                        self.medianErr))
        self.sigmaErr = self.perr[2] * u.angstrom

        # Compute the χ^2 value:
        residuals = self.fluxes - \
            vcl.gaussian(self.wavelengths.value, *self.popt)

        self.chiSquared = sum((residuals / self.errors) ** 2)
        self.chiSquaredNu = self.chiSquared / 3  # ν = 7 (pixels) - 4 (params)
        if (self.chiSquaredNu > 1):
            self.medianErr *= np.sqrt(self.chiSquaredNu)
        if verbose:
            tqdm.write('χ^2_ν = {}'.format(self.chiSquaredNu))

        # Find the full width at half max.
        # 2.354820 ≈ 2 * sqrt(2 * ln(2)), the relationship of FWHM to the
        # standard deviation of a Gaussian.
        self.FWHM = 2.354820 * self.sigma
        self.FWHMErr = 2.354820 * self.sigmaErr
        self.velocityFWHM = vcl.wavelength2velocity(self.median,
                                                    self.median +
                                                    self.FWHM)
        self.velocityFWHMErr = vcl.wavelength2velocity(self.median,
                                                       self.median +
                                                       self.FWHMErr)

        # Compute the offset between the input wavelength and the wavelength
        # found in the fit.
        self.offset = self.correctedWavelength - self.median
        self.offsetErr = self.medianErr
        self.velocityOffset = vcl.wavelength2velocity(self.correctedWavelength,
                                                      self.median)

        self.velocityOffsetErr = vcl.wavelength2velocity(self.median,
                                                         self.median +
                                                         self.offsetErr)

        if verbose:
            print(self.continuumLevel)
            print(self.fluxMinimum)
            print(self.wavelengths)

    def plotFit(self, close_up_plot_path, context_plot_path, plot_fit=True,
                verbose=False):
        """Plot a graph of this fit.

        This method will produce a 'close-up' plot of just the fitted region
        itself, in order to check out the fit has worked out, and a wider
        'context' plot of the area around the feature.

        Optional
        --------
        close_up_plot_path : string or `pathlib.Path`
            The file name to save a close-up plot of the fit to.
        context_plot_path : string or `pathlib.Path`
            The file name to save a wider context plot (±25 km/s) around the
            fitted feature to.
        plot_fit : bool, Default : True
            If *True*, plot the median of the fit and the fitted Gaussian.
            Otherwise, don't plot those two things. This allows creating plots
            of failed fits to see the context of the data.
        verbose : bool, Default : False
            If *True*, the function will print out additional information as it
            runs.

        """

        # Set up the figure.
        fig = plt.figure(figsize=(6.5, 4.5), dpi=80, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('Flux (photo-electrons)')
#        ax.xaxis.set_major_locator(ticker.FixedLocator(self.wavelengths
#                                                       .to(u.angstrom).value))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.set_xlim(left=self.baryArray[self.lowContinuumIndex - 3],
                    right=self.baryArray[self.highContinuumIndex + 3])
        # Set y-limits so a fit doesn't balloon the plot scale out.
        ax.set_ylim(top=self.continuumLevel * 1.08,
                    bottom=self.fluxMinimum * 0.95)

        # Plot the expected and measured wavelengths.
        ax.axvline(self.correctedWavelength.to(u.angstrom),
                   color='LightSteelBlue', linestyle=':',
                   label='RV-corrected λ={:.3f}'.format(
                           self.correctedWavelength.to(u.angstrom)))
        # Don't plot the median if this is a failed fit.
        if hasattr(self, 'median') and hasattr(self, 'velocityOffset'):
            ax.axvline(self.median.to(u.angstrom),
                       color='IndianRed',
                       label='Fit ({:.3f}, {:+.2f})'.
                       format(self.median.to(u.angstrom),
                              self.velocityOffset.to(u.m/u.s)),
                       linestyle='-')
        # Plot the actual data.
        ax.errorbar(self.baryArray[self.lowContinuumIndex:
                                   self.highContinuumIndex],
                    self.fluxArray[self.lowContinuumIndex:
                                   self.highContinuumIndex],
                    yerr=self.errorArray[self.lowContinuumIndex:
                                         self.highContinuumIndex],
                    color='SandyBrown', ecolor='Sienna',
                    label='Flux', barsabove=True)

        # Generate some x-values across the plot range.
        x = np.linspace(self.baryArray[self.lowContinuumIndex].value,
                        self.baryArray[self.highContinuumIndex].value, 1000)
        # Plot the initial guess for the gaussian.
        ax.plot(x, vcl.gaussian(x, *self.initial_guess),
                color='SlateGray', label='Initial guess',
                linestyle='--', alpha=0.8)
        # Plot the fitted gaussian, unless this is a failed fit attempt.
        if plot_fit:
            ax.plot(x, vcl.gaussian(x, *self.popt),
                    color='DarkGreen',
                    linestyle='-', label='Fitted Gaussian')

        ax.legend()
        # Save the resultant plot.
        fig.savefig(str(context_plot_path))
        if verbose:
            tqdm.write('Created wider context plot at {}'.format(
                    context_plot_path))

        # Now create a close-in version to focus on the fit.
        ax.set_xlim(left=self.baryArray[self.lowFitIndex - 1],
                    right=self.baryArray[self.highFitIndex])
        ax.set_ylim(top=self.fluxes.max() * 1.08,
                    bottom=self.fluxes.min() * 0.93)

        fig.savefig(str(close_up_plot_path))
        if verbose:
            tqdm.write('Created close up plot at {}'.format(
                    close_up_plot_path))
        plt.close(fig)
