#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:44:15 2019

@author: dberke

Code to define a class for a model fit to an absorption line.

"""

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import unyt as u
from tqdm import tqdm
import varconlib as vcl


class AbsorptionFeatureFit(object):
    """A class to fit an absorption line and store information about the fit.

    """

    def __init__(self, transition, observation, verbose=False):
        """Construct a fit to an absorption feature using the given model
        function.

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
        self.verbose = verbose
        self.fitAbsorptionFeature()

    def fitAbsorptionFeature(self):
        """Find and fit an absorption feature created by the transition given
        during initialization.

        """

        # Define some useful numbers and variables

        # The ranges in velocity space to search around to find the nadir of
        # an absorption line.
        search_range_vel = 5 * u.km / u.s
        # The range in velocity space to consider to find the continuum.
        continuum_range_vel = 25 * u.km / u.s
        # The number of pixels either side of the feature nadir to use in the
        # fit.
        pixel_range = 3

        radial_velocity = self.observation._radialVelocity
        # Shift the wavelength being searched for to correct for the radial
        # velocity of the star.
        rv_fixed_wavelength = vcl.shift_wavelength(self.transition.wavelength,
                                                   radial_velocity)
        rv_fixed_wavelength.convert_to_units(u.nm)
        tqdm.write('Given radial velocity {}, line {} should be at {:.4f}'.
                   format(radial_velocity,
                          self.transition.wavelength,
                          rv_fixed_wavelength))
        order = self.observation.findWavelength(rv_fixed_wavelength)

        wavelengthArray = self.observation._barycentricArray[order]
        fluxArray = self.observation._photonFluxArray[order]
        errorArray = self.observation._errorArray[order]

        # Figure out the range in wavelength space to search around the nominal
        # wavelength for the absorption nadir, as well as the range to take for
        # measuring the continuum.
        search_range = vcl.get_wavelength_separation(search_range_vel,
                                                     rv_fixed_wavelength)
        continuum_range = vcl.get_wavelength_separation(continuum_range_vel,
                                                        rv_fixed_wavelength)

        low_search_index = vcl.wavelength2index(rv_fixed_wavelength -
                                                search_range,
                                                wavelengthArray)
        high_search_index = vcl.wavelength2index(rv_fixed_wavelength +
                                                 search_range,
                                                 wavelengthArray)
        low_continuum_index = vcl.wavelength2index(rv_fixed_wavelength -
                                                   continuum_range,
                                                   wavelengthArray)
        high_continuum_index = vcl.wavelength2index(rv_fixed_wavelength +
                                                    continuum_range,
                                                    wavelengthArray)
        central_index = low_search_index + \
            fluxArray[low_search_index:high_search_index].argmin()

        continuum_level = fluxArray[low_continuum_index:high_continuum_index]\
            .max()

        line_nadir = fluxArray[central_index]

        low_fit_index = central_index - pixel_range
        high_fit_index = central_index + pixel_range + 1

        wavelengths = wavelengthArray[low_fit_index:high_fit_index]
        fluxes = fluxArray[low_fit_index:high_fit_index]
        errors = errorArray[low_fit_index:high_fit_index]

        line_depth = continuum_level - line_nadir

        initial_guess = (line_depth * -1,
                         rv_fixed_wavelength.to(u.angstrom).value,
                         0.08)

        tqdm.write('Attempting to fit line at {:.4f} with initial guess:'.
                   format(rv_fixed_wavelength))
        print(initial_guess)
        try:
            popt, pcov = curve_fit(vcl.gaussian, wavelengths.value,
                                   fluxes - continuum_level,
                                   sigma=errors, absolute_sigma=True,
                                   p0=initial_guess, method='lm')
        except (OptimizeWarning, RuntimeError):
            pass
        print(continuum_level)
        print(line_nadir)
        print(wavelengths)
#        print(initial_guess)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(wavelengths, fluxes-continuum_level, yerr=errors,
                    color='blue', marker='o', linestyle='')
        ax.plot(wavelengths, (vcl.gaussian(wavelengths.value, *initial_guess)),
                color='Black')
        ax.plot(wavelengths, (vcl.gaussian(wavelengths.value, *popt)),
                color='Red', linestyle='--')
        outfile = Path('/Users/dberke/Pictures/debug_norm.png')
        print('Created plot at {}'.format(outfile))
        fig.savefig(str(outfile))
        plt.close(fig)
#            raise

        print(popt)
        print(pcov)

# find order where transition is
#   localize wavelength, find limits of WFE arrays
# fit absorption feature (normalize first?)
# store information about fit
