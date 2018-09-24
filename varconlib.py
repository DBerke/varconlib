#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:28:13 2018

@author: dberke
"""

# Module to contain functions potentially useful across multiple programs

import numpy as np
import datetime as dt
from astropy.io import fits
from astropy.constants import c
from scipy.optimize import curve_fit
from math import sqrt, log
from pathlib import Path
import matplotlib.pyplot as plt

# Some generic information useful in different scripts

pairlist = [('443.9589', '444.1128'), ('450.0151', '450.3467'),
            ('459.9405', '460.3290'), ('460.5846', '460.6877'),
            ('465.8889', '466.2840'), ('473.3122', '473.3780'),
            ('475.9448', '476.0601'), ('480.0073', '480.0747'),
            ('484.0896', '484.4496'), ('488.6794', '488.7696'),
            ('490.9102', '491.0754'), ('497.1304', '497.4489'),
            ('500.5115', '501.1420'), ('506.8562', '507.4086'),
            ('507.3492', '507.4086'), ('513.2898', '513.8813'),
            ('514.8912', '515.3619'), ('524.8510', '525.1670'),
            ('537.5203', '538.1069'), ('554.4686', '554.5475'),
            ('563.5510', '564.3000'), ('571.3716', '571.9418'),
            ('579.4679', '579.9464'), ('579.5521', '579.9779'),
            ('580.8335', '581.0828'), ('593.1823', '593.6299'),
            ('595.4367', '595.8344'), ('600.4673', '601.0220'),
            ('616.3002', '616.8146'), ('617.2213', '617.5042'),
            ('617.7065', '617.8498'), ('623.9045', '624.6193'),
            ('625.9833', '626.0427'), ('625.9833', '626.2831'),
            ('647.0980', '647.7413')]

elemdict = {('443.9589', '444.1128'): 26,
            ('450.0151', '450.3467'): 25,
            ('459.9405', '460.3290'): 26,
            ('460.5846', '460.6877'): 26,
            ('465.8889', '466.2840'): 26,
            ('473.3122', '473.3780'): 28,
            ('475.9448', '476.0601'): 22,
            ('480.0073', '480.0747'): 26,
            ('484.0896', '484.4496'): 26,
            ('488.6794', '488.7696'): 26,
            ('490.9102', '491.0754'): 26,
            ('497.1304', '497.4489'): 26,
            ('500.5115', '501.1420'): 28,
            ('506.8562', '507.4086'): 26,
            ('507.3492', '507.4086'): 26,
            ('513.2898', '513.8813'): 26,
            ('514.8912', '515.3619'): 22,
            ('524.8510', '525.1670'): 26,
            ('537.5203', '538.1069'): 26,
            ('554.4686', '554.5475'): 26,
            ('563.5510', '564.3000'): 26,
            ('571.3716', '571.9418'): 26,
            ('579.4679', '579.9464'): 14,
            ('579.5521', '579.9779'): 26,
            ('580.8335', '581.0828'): 26,
            ('593.1823', '593.6299'): 26,
            ('595.4367', '595.8344'): 26,
            ('600.4673', '601.0220'): 26,
            ('616.3002', '616.8146'): 20,
            ('617.2213', '617.5042'): 26,
            ('617.7065', '617.8498'): 28,
            ('623.9045', '624.6193'): 14,
            ('625.9833', '626.0427'): 22,
            ('625.9833', '626.2831'): 22,
            ('647.0980', '647.7413'): 26}

# Lines known to be compromised by telluric lines.
badlines = frozenset(['506.8562', '507.4086', '593.1823', '593.6299',
                      '595.4367', '595.8344', '600.4673', '601.0220',
                      '647.0980', '647.7413'])

# Spectral format files for HARPS blue and red CCDs.
blueCCDpath = Path('/Users/dberke/code/tables/HARPS_CCD_blue.csv')
redCCDpath = Path('/Users/dberke/code/tables/HARPS_CCD_red.csv')


# Functions
def map_spectral_order(order):
    """
    Converts from HARPS' internal 0-71 order numbers to those in the HARPS
    spectral format (89-114, 116-161).

    Parameters
    ----------
    order : int
        An order number in the range [0, 71]
    """
    if not type(order) is int:
        raise ValueError("order parameter must be an integer in [0, 71]")
    if 0 <= order <= 45:
        new_order = 161 - order
        return new_order
    elif 46 <= order <= 71:
        new_order = 160 - order
        return new_order
    else:
        raise ValueError("Given order not in range [0, 71].")


def readHARPSfile(FITSfile, obj=False, wavelenmin=False, date_obs=False,
                  spec_bin=False, med_snr=False, hdnum=False, radvel=False,
                  coeffs=False):
    """Read a HARPS FITS file and return a dictionary of information.

    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    obj : bool, Default: False
        If *True*, the output will contain the contents of the OBJECT FITS
        header card.
    wavelenmin : bool, Default: False
        If *True*, the output will contain the contents of the WAVELMIN FITS
        header card.
    date_obs : bool, Default: False
        If *True*, the output will contain the contents of the DATE-OBS FITS
        header card.
    spec_bin : bool, Default: False
        If *True*, the output will contain the contents of the SPEC_BIN FITS
        header card.
    med_snr : bool, Default: False
        If *True*, the output will contain the contents of the SNR FITS header
        card.
    hdnum : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        HDNUM FITS header card. (Added to unify object identifiers across all
        stars, some of which were occasionally identified by things other than
        HD number.)
    radvel : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        RADVEL FITS header card. (Added to unify the radial velocity for each
        star, as a small minority of stars had different radial velocity
        information in their HIERARCH ESO TEL TAFG RADVEL header cards.)
    coeffs : bool, Default: False
        If *True*, the output will contain the contents of the various
        *ESO DRS CAL TH COEFF LLX* header cards, where *X* ranges from 0 to
        287.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:

        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            The estimated error array (HARPS returns no error array by
            default).

        Optionally
        ==========
        obj : str
            The object name from the 'OBJECT' flag.
        wlmin : float
            The minimum wavelength.
        date_obs : datetime object
            The date the file was observed.
        spec_bin : float
            The wavelength bin size.
        med_snr : float
            The median SNR of the flux array.
        hd_num : str
            The HD identifier of the star in the format "HDxxxxxx".
        radvel : float
            The radial velocity of the star in km/s.
        If the `coeffs` keyword argument is *True*, there will be 288 entries
        of the form "ESO DRS CAL TH COEFF LLX": *value*, where X will range
        from 0 to 287.
    """

    result = {}
    with fits.open(FITSfile) as hdulist:
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        data = hdulist[1].data
        w = data.WAVE[0]
        gain = header0['GAIN']
        # Multiply by the gain to convert from ADUs to photoelectrons
        f = data.FLUX[0] * gain
        e = 1.e6 * np.absolute(f)
        # Construct an error array by taking the square root of each flux value
        try:
            # First assume no negative flux values and use Numpy array
            # magic to speed up the process.
            e = np.sqrt(f)
        except ValueError:
            # If that raises an error, do it element-by-element.
            for i in np.arange(0, len(f), 1):
                if (f[i] > 0.0):
                    e[i] = np.sqrt(f[i])
        result['w'] = w
        result['f'] = f
        result['e'] = e
        if obj:
            result['obj'] = header1['OBJECT']
        if wavelenmin:
            result['wavelmin'] = header0['WAVELMIN']
        if date_obs:
            result['date_obs'] = dt.datetime.strptime(header0['DATE-OBS'],
                                                      '%Y-%m-%dT%H:%M:%S.%f')
        if spec_bin:
            result['spec_bin'] = header0['SPEC_BIN']
        if med_snr:
            result['med_snr'] = header0['SNR']
        if hdnum:
            result['hdnum'] = header0['HDNUM']
        if radvel:
            result['radvel'] = header0['RADVEL']

        # If the coeffs keyword is given, returna all 288 wavelength solution
        # coefficients.
        if coeffs:
            for i in range(0, 288, 1):
                key_string = 'ESO DRS CAL TH COEFF LL{0}'.format(str(i))
                result[key_string] = header0[key_string]

    return result


def readESPRESSOfile(ESPfile):
    """Read an ESPRESSO file and return a dictionary of information

    ESPfile: a path to the ESPRESSO file to be read

    output: a dictionary containing the following information:
        obj: the name from the OBJECT card
        w: the wavelength array
        f: the flux array
        e: the error array
    """
    with fits.open(ESPfile) as hdulist:
        data = hdulist[1].data
        obj = hdulist[0].header['OBJECT']
        w = data['wavelength']
        f = data['flux']
        e = data['error']
    return {'obj': obj, 'w': w, 'f': f, 'e': e}


def air_indexEdlen53(l, t=15., p=760.):
    """Return the index of refraction of air at given temp, pressures, and wl (A)

    l: vacuum wavelength in Angstroms
    t: temperature (don't use)
    p: pressure (don't use)

    Formula is Edlen 1953, provided directly by ESO
    """
    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t)\
        * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n


def vac2airESO(ll):
    """Return a vacuum wavlength from an air wavelength (A) using Edlen 1953

    ll: air wavlength in Angstroms

    """
    llair = ll/air_indexEdlen53(ll)
    return llair


def air2vacESO(air_arr):
    """Take an array of air wls (A) and return an array of vacum wls

    air_arr: an array-like list of wavelengths in air (Angstroms)

    returns: an array of wavelengths in vacuum (Angstroms)
    """
    if not type(air_arr) == np.ndarray:
        air_arr = np.array(air_arr)

    tolerance = 1e-12

    vac = []

    for i in range(0, len(air_arr)):
        newwl = air_arr[i]
        oldwl = 0.
        while abs(oldwl - newwl) > tolerance:
            oldwl = newwl
            n = air_indexEdlen53(newwl)
            newwl = air_arr[i] * n

        vac.append(round(newwl, 2))
    vac_arr = np.array(vac)

    return vac_arr


def vac2airMorton00(wl_vac):
    """Take an input vacuum wavelength in Angstroms and return the air wavelength.

    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    from Morton (2000, ApJ. Suppl., 130, 403) (IAU standard)
    """
    s = 1e4 / wl_vac
    n = 1 + 0.0000834254 + (0.02406147 / (130 - s**2)) +\
        (0.00015998 / (38.9 - s**2))
    return wl_vac / n


def air2vacMortonIAU(wl_air):
    """Take an input air wavelength in Angstroms and return the vacuum wavelength.

    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    """
    s = 1e4 / wl_air
    n = 1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s**2))\
        + (0.0001599740894897 / (38.92568793293 - s**2))
    return wl_air * n


def index2wavelength(index, step, min_wl):
    """Return the wavelength associated with an index.

    index: index position of the spectrum list
    step: the step in wavelength per index, in nm
    min_wl: the minimum wavelength of the spectrum, in nm
    """
    return round((step * index + min_wl), 2)


def wavelength2index(wl_arr, wl, reverse=False):
    """Find the index in a list associated with a given wavelength

    wl_arr: an iterable object of wavelengths, *in increasing order*
    wl: the wavelength to search for
    reverse: a Boolean for if the wavelength array is listed from large to
             small. Will first re

    returns: the index for which the wavelength is closest to the given
    """
    length = len(wl_arr)
    for i in range(0, length, 1):
        # First find the index for which the value is greater than the given wl
        if wl_arr[i] >= wl:
            # Then check if it's closest to this index or the previous one
            # Note that the way it's set up it should always be
            # wl_arr[i-1] <= wl <= wl_arr[i]
            if wl_arr[i] - wl > wl - wl_arr[i-1]:
                return i-1
            else:
                return i

    print("Couldn't find the given wavelength: {}".format(wl))
    raise ValueError


def lineshift(line, radvel):
    """Find the new position of a line given the radial velocity of a star

    line: line position. Can be nm or Angstroms, will return in same units
    radvel: radial velocity in km/s

    returns: the new line position
    """
    return ((radvel * 1000 / c.value) * line) + line


def getwlseparation(v, wl):
    """Return wavelength separation for a given velocity separation at a point

    v: the velocity separation. Should be in m/s
    wl: the wavelength at which the function should be evaluated, since
        it's also a function of wavelength. Returned in whatever units it's
        given in.

    """
    return (v * wl) / c.value


def getvelseparation(wl1, wl2):
    """Return velocity separation for a pair of points in wavelength space

    wl1 & wl2: wavelengths to get the velocity separation between in m/s.
               Should be in meters.
    """
    return (wl2 - wl1) * c.value / ((wl1 + wl2) / 2)


def gaussian(x, a, b, c):
    """Return the value of a Gaussian function with parameters a, b, and c

    x: independent variable
    a: amplitude of Gaussian
    b: center of Gaussian
    c: standard deviation of Gaussian
    """
    return a * np.exp(-1 * ((x - b)**2 / (2 * c * c)))


def fitGaussian(xnorm, ynorm, enorm, centralwl, radvel, continuum, linebottom,
                fluxrange, verbose=False):
    """
    Fit a Gaussian to the given data

    Parameters
    ----------
    xnorm : array_like
        An array of x-values (wavelength), normalized from -0.03 to 0.03.
    ynorm : array_like
        An array of y-values (photon counts) normalized from 0 to 1.
    enorm : array_like
        An array of error values for the y-values, normalized the same way as
        for `ynorm`.
    centralwl : float
        The wavelength of the pixel with the lowest flux value in the
        absorption line.
    radvel : float
        The radial velocity of the source in km/s, to the nearest tenth.
    continuum : float
        The flux value of the highest pixel within 20 km/s of the pixel with
        wavelength given by `centralwl`.
    linebottom : float
        The flux of the lowest pixel in the feature (i.e., the pixel at
        `centralwl`).
    fluxrange : float
        The (unnormalized) flux range between the highest pixel in the
        wavelength range selected (± 3 pixels around `centralwl`) and the
        lowest (given by `linebottom`).
    verbose : bool. Default: False
        If *True*, the function will print out diagnostic info on the process.

    Returns
    -------
    dict
        Returns a dictionary containing information about and relevant to
        the fit found.
    """

    # Fit a Gaussian to the line center
    linedepth = continuum - linebottom
    neg_linedepth = -1 * linedepth
    gauss_params = (neg_linedepth, 0, 1e3)
    try:
        popt_gauss, pcov_gauss = curve_fit(gaussian, xnorm,
                                           ynorm-continuum+linebottom,
                                           p0=gauss_params, sigma=enorm,
                                           absolute_sigma=True)
    except RuntimeError:
        print(continuum)
        print(linebottom)
        print(linedepth)
        print(neg_linedepth)
        print(gauss_params)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(xnorm, ynorm, yerr=enorm,
                    color='blue', marker='o', linestyle='')
        ax.plot(xnorm, (gaussian(xnorm, *gauss_params)), color='Black')
        plt.show()
        raise

    # Get the errors in the fitted parameters from the covariance matrix
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    r_gauss = (ynorm - continuum + linebottom) - gaussian(xnorm, *popt_gauss)
    chisq_gauss = sum((r_gauss / enorm) ** 2)
    chisq_nu_gauss = chisq_gauss / 4  # nu = 7 - 3

    # Find center of Gaussian &
    # correct for fitting normalized data
    gausscenterwl = popt_gauss[1] / 1000 + centralwl
    wl_err_gauss = perr_gauss[1] / 1000

    if chisq_nu_gauss > 1:
        wl_err_gauss *= sqrt(chisq_nu_gauss)

    # Multiply by 1e-9 to get nm to m for getvelseparation which requires m
    vel_err_gauss = getvelseparation(gausscenterwl*1e-9,
                                     (gausscenterwl+wl_err_gauss)*1e-9)
    # Shift line to stellar rest frame
    gaussrestframeline = lineshift(gausscenterwl, -1*radvel)

    # Get the width (sigma) of the Gaussian
    gauss_sigma = abs(popt_gauss[2] / 1000)
    gauss_sigma_err = perr_gauss[2] / 1000

    # Get the full width at half maximum (approximately 2.355 * sigma)
    gauss_fwhm = 2 * sqrt(2 * log(2)) * gauss_sigma
    gauss_fwhm_err = 2 * sqrt(2 * log(2)) * gauss_sigma_err

    # Convert sigma and FWHM to velocity space
    sigma_vel = getvelseparation(gausscenterwl*1e-9,
                                 (gausscenterwl+gauss_sigma)*1e-9)
    sigma_vel_err = getvelseparation(gausscenterwl*1e-9,
                                     (gausscenterwl+gauss_sigma_err)*1e-9)

    fwhm_vel = getvelseparation(gausscenterwl*1e-9,
                                (gausscenterwl+gauss_fwhm)*1e-9)
    fwhm_vel_err = getvelseparation(gausscenterwl*1e-9,
                                    (gausscenterwl+gauss_fwhm_err)*1e-9)

    # Get the aplitude of the Gaussian
    amp = popt_gauss[0]
    amp_err = perr_gauss[0]

    if verbose:
        print('-----------')
        print("Continuum level = {}".format(continuum))
        print('Depth of line = {}'.format(continuum - linebottom))
        print('fluxrange = {}'.format(fluxrange))
        print("Covariance matrix for Gaussian:")
        print(pcov_gauss)
        print('popt_gauss = {}'.format(popt_gauss))
        print('perr_gauss = {}'.format(perr_gauss))
        print(u'χ^2 (Gaussian) = {:.7f}'.format(chisq_gauss))
        print(u'χ_ν^2 (Gaussian) = {:.7f}'.format(chisq_nu_gauss))
        print('Gaussian central wl: {:.6f} nm'.format(gausscenterwl))
        print("1 stddev Gaussian = {:.6e} nm".format(wl_err_gauss))
        print("1 stddev Gaussian velspace = {:.7f} m/s".format(vel_err_gauss))
        print('1 sigma = {:.6f} nm'.format(gauss_sigma))
        print('1 sigma velspace = {:.7f} m/s'.format(sigma_vel))
        print('FWHM = {:.6f}'.format(gauss_fwhm))
        print('FWHM velspace = {:.7f} m/s'.format(fwhm_vel))
        print('Gaussian amplitude = {:.6f} photons'.format(amp))
        print('Gaussian amp err = {:.6f} photons'.format(amp_err))
        print("Found line center at {:.6f} nm.".format(gausscenterwl))
        print("Corrected for rad vel: {:.6f} nm".format(gaussrestframeline))

    return {'restframe_line_gauss': gaussrestframeline,
            'vel_err_gauss': vel_err_gauss,
            'amplitude_gauss': amp,
            'amplitude_err_gauss': amp_err,
            'width_gauss': sigma_vel,
            'width_err_gauss': sigma_vel_err,
            'fwhm_gauss': fwhm_vel,
            'fwhm_gauss_err': fwhm_vel_err,
            'chisq_nu_gauss': chisq_nu_gauss,
            'gausscenterwl': gausscenterwl,
            'popt_gauss': popt_gauss}


def pix_order_to_wavelength(pixel, order, coeffs_dict):
    """
    Returns the wavelength measured on the given pixel in the given order.

    Parameters
    ----------
    pixel : int, Range: 0 to 4095
        The pixel in the dispersion direction where the wavelength will be
        measured.
    order : int, Range: 0 to 71
        The spectral order to measure the wavelength in.
    coeff_dict: dict
        A dictionary containing wavelength solution coefficients in the form
        *ESO DRS CAL TH COEFF LLX*, where *X* ranges from 0 to 287.

    Returns
    -------
    float
        The wavelength observed at the given pixel and order in nanometers.

    Notes
    -----
    The algorithm used is derived from Dumusque 2018 [1]_.

    References
    ----------
    [1] Dumusque, X. "Measuring precise radial velocities on individual
    spectral lines I. Validation of the method and application to mitigate
    stellar activity", Astronomy & Astrophysics, 2018
    """
    if not (0 <= pixel <= 4095):
        print('pixel = {}, must be between 0 and 4095.'.format(pixel))
        raise ValueError
    if not (0 <= order <= 71):
        print('order = {}, must be between 0 and 71.'.format(order))
        raise ValueError

    wavelength = 0.
    for k in range(0, 4, 1):
        dict_key = 'ESO DRS CAL TH COEFF LL{0}'.format((4 * order) + k)
        wavelength += coeffs_dict[dict_key] * (pixel ** k)

    return wavelength / 10.


def wavelength_to_pix(wavelength, coeffs_dict, red_spec_form,
                      blue_spec_form):
    """
    """
    formats = (blue_spec_form, red_spec_form)
    orders = []
    for spec_form in formats:
        matched = 0
        for minwl, maxwl, order in zip(spec_form['startwl'],
                                       spec_form['endwl'], spec_form['order']):
            if minwl <= wavelength <= maxwl:
                orders.append(order)
                matched += 1
            if matched == 2:
                break
    pixels = []
    for order in orders:
        # Convert nanometers to Angstroms here.
        pixels.append(invertWavelengthMap(wavelength * 10, order, coeffs_dict))
    return pixels


def invertWavelengthMap(wavelength, order, coeffs_dict):
    """
    Returns the x-pixel of the CCD where the given wavelength is observed.

    Parameters
    ----------
    wavelength : float
        The wavelength to find the pixel of observation for.
    order : int
        The spectral order to search for the wavelength in.
    coeff_dict : dict
        A dictionary containing the coefficients for the wavelength solutions
        from an observation.

    Returns
    -------
    int
        The pixel in the x-direction (along the dispersion) where the given
        wavelength was measured.
    """
    oldwl = 0.
    for k in range(0, 4096, 1):
        newwl = pix_order_to_wavelength(k, order, coeffs_dict)
        if newwl > wavelength:
            if newwl - wavelength > oldwl - wavelength:
                return k - 1
            else:
                return k

