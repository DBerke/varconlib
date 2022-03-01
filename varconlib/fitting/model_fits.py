#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:44:15 2019

@author: dberke

Code to define a class for a model fit to an absorption line.

"""

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm
import unyt as u

from varconlib.exceptions import PositiveAmplitudeError
from varconlib.fitting import gaussian, integrated_gaussian
from varconlib.miscellaneous import (shift_wavelength, velocity2wavelength,
                                     wavelength2index, wavelength2velocity)

# This line prevents the wavelength formatting from being in the form of
# scientific notation.
matplotlib.rcParams['axes.formatter.useoffset'] = False
# Don't use TeX for font rendering, as these are just diagnostic plots and it
# slows everything way down.
matplotlib.rcParams['text.usetex'] = False


class GaussianFit(object):
    """A class to fit an absorption line and store information about the fit.

    """

    def __init__(self, transition, observation, order, radial_velocity=None,
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
        order : int
            The order in the e2ds file to fit the transition in. Zero-indexed,
            so ranging from [0-71].

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

        # Store the transition.
        self.transition = transition
        # Grab some observation-specific information from the observation.
        self.dateObs = observation.dateObs
        self.BERV = observation.BERV
        self.airmass = observation.airmass
        self.exptime = observation.exptime
        self.calibrationFile = observation.calibrationFile
        self.calibrationSource = observation.calibrationSource

        self.order = int(order)

        # Store the plot paths.
        self.close_up_plot_path = close_up_plot_path
        self.context_plot_path = context_plot_path

        # Define some useful numbers and variables.

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
        nominal_wavelength = self.transition.wavelength.to(u.angstrom)
        self.correctedWavelength = shift_wavelength(nominal_wavelength,
                                                    radial_velocity)
        if verbose:
            tqdm.write('Given RV {:.2f}: line {:.3f} should be at {:.3f}'.
                       format(radial_velocity,
                              nominal_wavelength.to(u.angstrom),
                              self.correctedWavelength.to(u.angstrom)))

        self.baryArray = observation.barycentricArray[self.order]
        self.fluxArray = observation.photonFluxArray[self.order]
        self.errorArray = observation.errorArray[self.order]

        # Figure out the range in wavelength space to search around the nominal
        # wavelength for the flux minimum, as well as the range to take for
        # measuring the continuum.
        search_range = velocity2wavelength(search_range_vel,
                                           self.correctedWavelength)

        self.continuumRange = velocity2wavelength(continuum_range_vel,
                                                  self.correctedWavelength)

        low_search_index = wavelength2index(self.correctedWavelength -
                                            search_range,
                                            self.baryArray)

        high_search_index = wavelength2index(self.correctedWavelength +
                                             search_range,
                                             self.baryArray)

        self.lowContinuumIndex = wavelength2index(self.correctedWavelength
                                                  - self.continuumRange,
                                                  self.baryArray)
        self.highContinuumIndex = wavelength2index(self.correctedWavelength
                                                   + self.continuumRange,
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
            tqdm.write('Initial parameters are:\n{}\n{}\n{}\n{}'.format(
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
                self.popt, self.pcov = curve_fit(integrated_gaussian,
                                                 (pixel_edges_lower.value,
                                                  pixel_edges_upper.value),
                                                 self.fluxes,
                                                 sigma=self.errors,
                                                 absolute_sigma=True,
                                                 p0=self.initial_guess,
                                                 method='lm', maxfev=10000)
            else:
                self.popt, self.pcov = curve_fit(gaussian,
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
                         plot_fit=False, verbose=True)
            raise

        if verbose:
            print(self.popt)
            print(self.pcov)

        # Recover the fitted values for the parameters:
        self.amplitude = self.popt[0]
        self.mean = self.popt[1] * u.angstrom
        self.sigma = self.popt[2] * u.angstrom

        if self.amplitude > 0:
            err_msg = ('Fit for'
                       f' {self.transition.wavelength.to(u.angstrom)}'
                       ' has a positive amplitude.')
            tqdm.write(err_msg)
            self.plotFit(close_up_plot_path, context_plot_path,
                         plot_fit=True, verbose=verbose)
            raise PositiveAmplitudeError(err_msg)

        # Find 1-σ errors from the covariance matrix:
        self.perr = np.sqrt(np.diag(self.pcov))

        self.amplitudeErr = self.perr[0]
        self.meanErr = self.perr[1] * u.angstrom
        self.meanErrVel = abs(wavelength2velocity(self.mean,
                                                  self.mean +
                                                  self.meanErr))
        self.sigmaErr = self.perr[2] * u.angstrom

        if (self.chiSquaredNu > 1):
            self.meanErr *= np.sqrt(self.chiSquaredNu)
        if verbose:
            tqdm.write('χ^2_ν = {}'.format(self.chiSquaredNu))

        # Find the full width at half max.
        # 2.354820 ≈ 2 * sqrt(2 * ln(2)), the relationship of FWHM to the
        # standard deviation of a Gaussian.
        self.FWHM = 2.354820 * self.sigma
        self.FWHMErr = 2.354820 * self.sigmaErr
        self.velocityFWHM = wavelength2velocity(self.mean,
                                                self.mean +
                                                self.FWHM).to(u.km/u.s)
        self.velocityFWHMErr = wavelength2velocity(self.mean,
                                                   self.mean +
                                                   self.FWHMErr).to(u.km/u.s)

        # Compute the offset between the input wavelength and the wavelength
        # found in the fit.
        self.offset = self.correctedWavelength - self.mean
        self.offsetErr = self.meanErr
        self.velocityOffset = wavelength2velocity(self.correctedWavelength,
                                                  self.mean)

        self.velocityOffsetErr = wavelength2velocity(self.mean,
                                                     self.mean +
                                                     self.offsetErr)

        if verbose:
            print(self.continuumLevel)
            print(self.fluxMinimum)
            print(self.wavelengths)

    @property
    def chiSquared(self):
        if not hasattr(self, '_chiSquared'):
            residuals = self.fluxes - gaussian(self.wavelengths.value,
                                               *self.popt)
            self._chiSquared = sum((residuals / self.errors) ** 2)
        return self._chiSquared

    @property
    def chiSquaredNu(self):
        return self.chiSquared / 3  # ν = 7 (pixels) - 4 (params)

    @property
    def label(self):
        return self.transition.label + '_' + str(self.order)

    def getFitInformation(self):
        """Return a list of information about the fit which can be written as
        a CSV file.

        Returns
        -------
        list
            A list containing the following information about the fit:
            1. Observation date, in ISO format
            2. The amplitude of the fit (in photons)
            3. The error on the amplitude (in photons)
            4. The mean of the fit (in Å)
            5. The error on the mean (in Å)
            6. The error on the mean (in m/s in velocity space)
            7. The sigma of the fitted Gaussian (in Å)
            8. The error on the sigma (in Å)
            9. The offset from expected wavelength (in m/s)
            10. The error on the offset (in m/s)
            11. The FWHM (in velocity space)
            12. The error on the FWHM (in m/s)
            13. The chi-squared-nu value
            14. The order the fit was made on (starting at 0, so in [0, 71].
            15. The mean airmass of the observation.

        """

        return [self.dateObs.isoformat(timespec='milliseconds'),
                self.amplitude,
                self.amplitudeErr,
                self.mean.value,
                self.meanErr.value,
                self.meanErrVel.value,
                self.sigma.value,
                self.sigmaErr.value,
                self.velocityOffset.to(u.m/u.s).value,
                self.velocityOffsetErr.to(u.m/u.s).value,
                self.velocityFWHM.to(u.m/u.s).value,
                self.velocityFWHMErr.to(u.m/u.s).value,
                self.chiSquaredNu,
                self.order,
                self.airmass]

    def plotFit(self, close_up_plot_path=None,
                context_plot_path=None,
                plot_fit=True,
                verbose=False):
        """Plot a graph of this fit.

        This method will produce a 'close-up' plot of just the fitted region
        itself, in order to check out the fit has worked out, and a wider
        'context' plot of the area around the feature.

        Optional
        --------
        close_up_plot_path : string or `pathlib.Path`
            The file name to save a close-up plot of the fit to. If not given,
            will default to using the value providing when initializing the
            fit.
        context_plot_path : string or `pathlib.Path`
            The file name to save a wider context plot (±25 km/s) around the
            fitted feature to. If not given, will default to using the value
            provided when initializing the fit.
        plot_fit : bool, Default : True
            If *True*, plot the mean of the fit and the fitted Gaussian.
            Otherwise, don't plot those two things. This allows creating plots
            of failed fits to see the context of the data.
        verbose : bool, Default : False
            If *True*, the function will print out additional information as it
            runs.

        """

        edge_pixels = (509, 510, 1021, 1022, 1533, 1534, 2045, 2046,
                       2557, 2558, 3069, 3070, 3581, 3582)

        # If no plot paths are given, assume we want to use the ones given
        # when initializing the fit.
        if close_up_plot_path is None:
            close_up_plot_path = self.close_up_plot_path
        if context_plot_path is None:
            context_plot_path = self.context_plot_path

        # Set up the figure.
        fig = plt.figure(figsize=(7, 5), dpi=100, tight_layout=True)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.tick_params(axis='x', direction='in')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_ylim(bottom=-3, top=3)

        ax2.yaxis.set_major_locator(ticker.FixedLocator([-2, -1, 0, 1, 2]))

        for pixel in edge_pixels:
            ax1.axvline(x=self.baryArray[pixel-1], ymin=0, ymax=0.2,
                        color='LimeGreen',
                        linestyle='--')

        ax1.set_ylabel('Flux (photo-electrons)')
        ax2.set_xlabel('Wavelength ($\\AA$)')
        ax2.set_ylabel('Residuals\n($\\sigma$)')

        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>7.1e}'))
        plt.xticks(horizontalalignment='right')
        ax1.set_xlim(left=self.correctedWavelength - self.continuumRange,
                     right=self.correctedWavelength + self.continuumRange)
        # Set y-limits so a fit doesn't balloon the plot scale out.
        ax1.set_ylim(top=self.continuumLevel * 1.25,
                     bottom=self.fluxMinimum * 0.93)

        # Plot the expected and measured wavelengths.
        ax1.axvline(self.correctedWavelength.to(u.angstrom),
                    color='LightSteelBlue', linestyle=':', alpha=0.8,
                    label=r'RV-corrected $\lambda=${:.3f}'.format(
                           self.correctedWavelength.to(u.angstrom)))

        # Don't plot the mean if this is a failed fit.
        if hasattr(self, 'mean') and hasattr(self, 'velocityOffset'):
            ax1.axvline(self.mean.to(u.angstrom),
                        color='IndianRed', alpha=0.7,
                        label='Mean ({:.4f}, {:+.2f})'.
                        format(self.mean.to(u.angstrom),
                               self.velocityOffset.to(u.m/u.s)),
                        linestyle='-')

        # Plot the actual data.
        ax1.errorbar(self.baryArray[self.lowContinuumIndex - 1:
                                    self.highContinuumIndex + 1],
                     self.fluxArray[self.lowContinuumIndex - 1:
                                    self.highContinuumIndex + 1],
                     yerr=self.errorArray[self.lowContinuumIndex - 1:
                                          self.highContinuumIndex + 1],
                     color='SandyBrown', ecolor='Sienna',
                     label='Flux', barsabove=True)

        # Generate some x-values across the plot range.
        x = np.linspace(self.baryArray[self.lowContinuumIndex].value,
                        self.baryArray[self.highContinuumIndex].value, 1000)
        # Plot the initial guess for the gaussian.
        ax1.plot(x, gaussian(x, *self.initial_guess),
                 color='SlateGray', label='Initial guess',
                 linestyle='--', alpha=0.5)
        # Plot the fitted gaussian, unless this is a failed fit attempt.
        if plot_fit:
            ax1.plot(x, gaussian(x, *self.popt),
                     color='DarkGreen', alpha=0.5,
                     linestyle='-.',
                     label=r'Fit ($\chi^2_\nu=${:.3f}, $\sigma=${:.4f})'.
                     format(self.chiSquaredNu, self.sigma))

        # Replace underscore in label so LaTeX won't crash on it.
        ax1.legend(loc='upper center', framealpha=0.6, fontsize=9,
                   ncol=2,
                   title=self.label.replace('_', r'\_') if\
                       matplotlib.rcParams['text.usetex'] else self.label,
                   title_fontsize=10,
                   labelspacing=0.4)

        # Add in some guidelines.
        ax2.axhline(color='Gray', linestyle='-')
        ax2.axhline(y=1, color='SkyBlue', linestyle='--')
        ax2.axhline(y=-1, color='SkyBlue', linestyle='--')
        ax2.axhline(y=2, color='LightSteelBlue', linestyle=':')
        ax2.axhline(y=-2, color='LightSteelBlue', linestyle=':')

        # Plot the residuals on the lower axis.
        residuals = (self.fluxes - gaussian(self.wavelengths.value,
                                            *self.popt)) / self.errors

        ax2.plot(self.wavelengths, residuals, color='Navy', alpha=0.6,
                 linestyle='', marker='D', linewidth=1.5, markersize=5)

        # Save the resultant plot.

        fig.savefig(str(context_plot_path), format="png")
        if verbose:
            tqdm.write('Created wider context plot at {}'.format(
                    context_plot_path))

        # Now create a close-in version to focus on the fit.
        ax1.set_xlim(left=self.baryArray[self.lowFitIndex - 1],
                     right=self.baryArray[self.highFitIndex])
        ax1.set_ylim(top=self.fluxes.max() * 1.15,
                     bottom=self.fluxes.min() * 0.95)

        fig.savefig(str(close_up_plot_path), format="png")
        if verbose:
            tqdm.write('Created close up plot at {}'.format(
                    close_up_plot_path))
        plt.close(fig)
