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
import unyt as u
from tqdm import tqdm
import varconlib as vcl

config_file = Path('/Users/dberke/code/config/variables.cfg')
config = configparser.ConfigParser(interpolation=configparser.
                                   ExtendedInterpolation())
config.read(config_file)


class AbsorptionFeatureFitGaussian(object):
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
        tqdm.write('Given RV {:.2f}: line {:.4f} should be at {:.4f}'.
                   format(radial_velocity,
                          nominal_wavelength.to(u.nm),
                          self.correctedWavelength.to(u.nm)))
        # Find which order of the echelle spectrum the wavelength falls in.
        order = self.observation.findWavelength(self.correctedWavelength,
                                                mid_most=True)

        baryArray = self.observation.barycentricArray[order]
        fluxArray = self.observation.photonFluxArray[order]
        errorArray = self.observation.errorArray[order]

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
        low_continuum_index = vcl.wavelength2index(self.correctedWavelength -
                                                   continuum_range,
                                                   baryArray)
        high_continuum_index = vcl.wavelength2index(self.correctedWavelength +
                                                    continuum_range,
                                                    baryArray)
        central_index = low_search_index + \
            fluxArray[low_search_index:high_search_index].argmin()

        self.continuum_level = fluxArray[low_continuum_index:
                                         high_continuum_index].max()

        flux_minimum = fluxArray[central_index]

        low_fit_index = central_index - pixel_range
        high_fit_index = central_index + pixel_range + 1

        # Grab the wavelengths, fluxes, and errors from the region to be fit.
        self.wavelengths = baryArray[low_fit_index:high_fit_index]
        self.fluxes = fluxArray[low_fit_index:high_fit_index]
        self.errors = errorArray[low_fit_index:high_fit_index]

        line_depth = self.continuum_level - flux_minimum

        self.initial_guess = (line_depth * -1,
                              self.correctedWavelength.to(u.angstrom).value,
                              0.08)

        tqdm.write('Attempting to fit line at {:.4f} with initial guess:'.
                   format(self.correctedWavelength))
        if verbose:
            print('Initial parameters are:\n{}\n{}\n{}'.format(
                  *self.initial_guess))

        # Do the fitting:
        try:
            self.popt, self.pcov = curve_fit(vcl.gaussian,
                                             self.wavelengths.value,
                                             self.fluxes -
                                             self.continuum_level,
                                             sigma=self.errors,
                                             absolute_sigma=True,
                                             p0=self.initial_guess,
                                             method='lm')
        except (OptimizeWarning, RuntimeError):
            raise

        if verbose:
            print(self.popt)
            print(self.pcov)

        # Recover the fitted values for the parameters:
        self.amplitude = self.popt[0]
        self.centroid = self.popt[1] * u.angstrom
        self.standardDev = self.popt[2] * u.angstrom

        # Find 1-σ errors from the covariance matrix:
        self.perr = np.sqrt(np.diag(self.pcov))

        self.amplitudeErr = self.perr[0]
        self.centroidErr = self.perr[1] * u.angstrom
        self.standardDevErr = self.perr[2] * u.angstrom

        # Compute the χ^2 value:
        residuals = (self.fluxes - self.continuum_level) - \
            vcl.gaussian(self.wavelengths.value, *self.popt)

        self.chiSquared = sum((residuals / self.errors) ** 2)
        self.chiSquaredNu = self.chiSquared / 4  # ν = 7 (pixels) - 3 (params)
        if (self.chiSquaredNu > 1):
            self.centroidErr *= np.sqrt(self.chiSquaredNu)

        # Find the full width at half max:
        self.FWHM = 2 * np.sqrt(2 * np.log(2)) * self.standardDev
        self.FWHMErr = 2 * np.sqrt(2 * np.log(2)) * self.standardDevErr
        self.velocityFWHM = vcl.wavelength2velocity(self.centroid,
                                                    self.centroid +
                                                    self.FWHM / 2)
        # TODO: Error for FWHM velocity.

        # Compute the offset between the input wavelength and the wavelength
        # found in the fit.
        self.offset = self.centroid - self.correctedWavelength
        self.velocityOffset = vcl.wavelength2velocity(self.centroid,
                                                      self.correctedWavelength)
        # TODO: Error for velocity offset.

        if verbose:
            print(self.continuum_level)
            print(flux_minimum)
            print(self.wavelengths)

    def toVelocity(self, attribute):
        wavelength1 = attribute
        try:
            wavelength2 = getattr(self, attribute.__name__ + 'Err')
        except KeyError:
            raise KeyError('No appropriately named "Err" attribute found.')
        return vcl.get_velocity_offset(wavelength1, wavelength2)

    def plotFit(self, outfile=None):
        import matplotlib.pyplot as plt
        # Set up the figure.
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        # Begin plotting.
        ax.axvline(self.correctedWavelength.to(u.angstrom),
                   color='SeaGreen', linestyle='--',
                   label='RV-corrected transition λ',)
        ax.axvline(self.centroid.to(u.angstrom),
                   color='PaleVioletRed', label='Fitted centroid',
                   linestyle='-')
        ax.errorbar(self.wavelengths.to(u.angstrom),
                    self.fluxes-self.continuum_level,
                    yerr=self.errors,
                    color='SaddleBrown', marker='o', linestyle='',
                    ecolor='Peru', label='Data')
        ax.plot(self.wavelengths.to(u.angstrom),
                (vcl.gaussian(self.wavelengths.value, *self.initial_guess)),
                color='DeepSkyBlue', label='Initial guess',
                linestyle='--')
        x = np.linspace(self.wavelengths.to(u.angstrom).min().value,
                        self.wavelengths.to(u.angstrom).max().value, 40)
        ax.plot(x, (vcl.gaussian(x, *self.popt)), color='Olive',
                linestyle='-', label='Fitted Gaussian')

        ax.legend()
        # Save the resultant plot.
        if outfile is None:
            stars_dir = Path(config['PATHS']['stars_dir'])
            outfile = stars_dir /\
                'Transition_{:.3f}.png'.format(
                        self.transition.wavelength.to(u.angstrom).value)
        print('Created plot at {}'.format(outfile))
        fig.savefig(str(outfile))
        plt.close(fig)
