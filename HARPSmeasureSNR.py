#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:21:25 2018

@author: dberke
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os.path
from glob import glob
from astropy.io import fits

def vac2air(wl_vac):
    """Take an input vacuum wavelength in nm and return the air wavelength.
    
    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    from Morton (2000, ApJ. Suppl., 130, 403) (IAU standard)
    """
    s = 1e3 / wl_vac
    n = 1 + 0.0000834254 + (0.02406147 / (130 - s**2)) +\
        (0.00015998 / (38.9 - s**2))
    return wl_vac / n


def air2vac(wl_air):
    """Take an input air wavelength in nm and return the vacuum wavelength.
    
    Formula taken from 'www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion'
    """
    s = 1e3 / wl_air
    n = 1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s**2))\
        + (0.0001599740894897 / (38.92568793293 - s**2))
    return wl_air * n


def index2wavelength(index, step, min_wl):
    """Return the wavelength associated with an index.
    
    index -- index position of the spectrum list
    step -- the step in wavelength per index, in nm
    min_wl -- the minimum wavelength of the spectrum, in nm
    """
    return round((step * index + min_wl), 2)


def wavelength2index(wl, step, min_wl):
    """Return the index of the given wavelength."""
    
    return int((wl - min_wl) / step)


def readHARPSfile(FITSfile):
    """Read a HARPS FITS file and return a dictionary of information."""
    
    with fits.open(FITSfile) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data
        obj = header['OBJECT']
        wavelmin = hdulist[0].header['WAVELMIN']
        date_obs = hdulist[0].header['DATE-OBS']
        spec_bin = hdulist[0].header['SPEC_BIN']
        med_SNR = hdulist[0].header['SNR']
        w = data.WAVE[0]
        f = data.FLUX[0]
        e = 1.e6 * np.absolute(f)
        for i in np.arange(0, len(f), 1):
            if (f[i] > 0.0):
                e[i] = np.sqrt(f[i])
    return {'obj':obj, 'w':w, 'f':f, 'e':e,
            'wlmin':wavelmin, 'date_obs':date_obs,
            'spec_bin':spec_bin, 'med_snr':med_SNR}


def measureSNR(spectrum, start_wl, search_window=0.6, sub_window=0.05,
               step=0.01):
    """
    Measure the SNR in an area in increments of step and return the highest.
    
    spectrum -- a spectrum object from readHARPSfile
    start_wl -- the wavelength to begin searching at, in nm
    search_window -- the width of the window to search over, in nm
    sub_window -- the width of the individual search sections, in nm
    step -- the step width to use between searchs, nm
    
    Return the sub-window with the highest SNR, and the SNR itself.
    """

    # Figure out the minimum wavelength and step size of the spectrum.
    wl_min = spectrum['wlmin']
    spec_step = spectrum['spec_bin']
    # Work out the position in the list from the wavelength.
    start_pos = wavelength2index(start_wl, spec_step, wl_min)
    # Figure out where to stop.
    end_wl = start_wl + search_window
    # Calculate that position in the list.
    end_pos = wavelength2index(end_wl, spec_step, wl_min)
    step_size = int(step / spec_step) # Calculate the indices-to-nm ratio
    half_interval = int(sub_window / spec_step / 2)

    greatest_SNR = 0
    position = 0
    snr_list = []
    wl_list = []
    for i in range(start_pos, end_pos, step_size):
        lower_bound = i - half_interval
        upper_bound = i + half_interval
        interval = np.array(spectrum['f'][lower_bound:upper_bound])
        mean = np.mean(interval)
        rms = np.sqrt(np.mean(np.square(interval - mean))) # Root-mean-square
        SNR = mean / rms
        snr_list.append(SNR)
        wl_list.append(spectrum['w'][i])
        if SNR > greatest_SNR:
            greatest_SNR = SNR
            position = i
            #bounds = [lower_bound, upper_bound]

    #min_bound = index2wavelength(bounds[0], spec_step, wl_min)
    #max_bound = index2wavelength(bounds[1], spec_step, wl_min)
    central_wl = index2wavelength(position, spec_step, wl_min)
    print("Highest SNR ({:.2f}) found at {}.".format(greatest_SNR,
          central_wl))
    return {'SNRmax':greatest_SNR, 'SNRmaxWL':central_wl,
            'SNRlist':np.array(snr_list), 'SNRlistWL':np.array(wl_list)}

def line_free(x, m, b):
    return m * x + b

def line_fixed(x, m):
    return m * x

# Start main body.
outPicDir = "/Users/dberke/Pictures/"

objects = ("HD68168", "HD126525", "HD138573")
start_wavelengths = (561, 620.24, 623.2) # Adjusted for offset.

# Uncomment this to run minimal mode.
#objects = ("HD68168",)


search_window = 0.6
sub_window = 0.05
step_size = 0.01

for obj in objects:
    search_dir = "/Volumes/External Storage/HARPS/" + obj + "/"
    files = sorted(glob("{}".format(search_dir+'*.fits')))
    print("Found {} files for {}.".format(len(files), obj))

    window1, window2, window3 = [], [], []
    pos1, pos2, pos3 = [], [], []
    windows = (window1, window2, window3)
    positions = (pos1, pos2, pos3)

    for start_wl, window, pos in zip(start_wavelengths, windows, positions):
        print("Working on window starting at {}...".format(start_wl))
        med_SNRs = []
        
        fig_window = plt.figure(figsize=(12, 10))
        fig_window.suptitle('Window: {:.2f} nm - {:.2f} nm'.format(start_wl,
                            start_wl+search_window))
        ax_window = fig_window.add_subplot(1, 1, 1)
        ax_window.set_xlabel('Wavelength (nm)')
        
        ax_window.set_xlim(left=start_wl,
                           right=start_wl+search_window)

        fig_SNR = plt.figure(figsize=(12, 8))
        fig_window.suptitle('Window: {:.2f} nm - {:.2f} nm'.format(start_wl,
                            start_wl+search_window))
        ax_SNR = fig_SNR.add_subplot(1, 1, 1)
        ax_SNR.set_xlabel('SNR (measured)')
        ax_SNR.set_ylabel('Median SNR from header')


        for fitsfile in files:
            outFileBase = "{}_{:.2f}-{:.2f}.png".\
                          format(obj, start_wl, start_wl+search_window)
            print(os.path.basename(fitsfile))
            spectrum = readHARPSfile(fitsfile)
            result = measureSNR(spectrum, start_wl, search_window,
                                          sub_window, step_size)
            window.append(result['SNRmax'])
            pos.append(result['SNRmaxWL'])
            med_SNRs.append(spectrum['med_snr'])

            # Plot the individual spectra.
            fig_spectrum = plt.figure(figsize=(13,8))
            ax_spectrum = fig_spectrum.add_subplot(1, 1, 1)
            ax_spectrum.set_xlim(left=start_wl,
                           right=start_wl+search_window)

            ax_spectrum.plot(spectrum['w']/10, spectrum['f']/result['SNRmax'],
                             color='black',
                             linestyle='solid', marker='')
            ax_spectrum.plot(result['SNRlistWL']/10, result['SNRlist'],
                             color='blue', linestyle='', marker='+')
            upper = result['SNRmaxWL'] + (sub_window / 2)
            lower = result['SNRmaxWL'] - (sub_window / 2)
            ax_spectrum.axvspan(xmin=lower, xmax=upper, color='green',
                                alpha=0.3)
            outSpecDir = os.path.join(outPicDir, obj)
            if not os.path.exists(outSpecDir):
                os.mkdir(outSpecDir)
            outSpecFile = '_'.join((spectrum['date_obs'], outFileBase))
            plt.savefig(os.path.join(outSpecDir, outSpecFile), format='png')
            plt.close(fig_spectrum)

            # Plot the spectrum and position of max SNR.
            ax_window.plot(spectrum['w']/10, spectrum['f'], color='black',
                           linestyle='solid', marker='', alpha=0.8)
#            ax_window.axvspan(xmin=lower, xmax=upper,
#                             color='green', alpha=0.2)
            ax_window.vlines(x=result['SNRmaxWL'], ymin=0, ymax=50000,
                            color='green', alpha=0.4)

            # Plot the measured SNR vs. the median SNR from the header.
            ax_SNR.plot(result['SNRmax'], spectrum['med_snr'], color='green',
                        linestyle='', marker='+')

        snr_arr = np.array(window, dtype='float32')
        med_arr = np.array(med_SNRs, dtype='float32')

        # Calculate the least-squares-fit line with intercept = 0.
        popt, pcov = scipy.optimize.curve_fit(line_fixed, snr_arr, med_arr)
        
        ax_SNR.plot(window, popt[0] * snr_arr,
                    color='blue', linestyle='solid',
                    label='${:.3f}\cdot x$'.format(popt[0]))
        fig_SNR.legend(loc=4) 

        # Save the figure showing all the spectra plotted.
        out_window_file = '_'.join(('SNR', outFileBase))
        fig_window.savefig(os.path.join(outPicDir, out_window_file),
                           format='png')
        plt.close(fig_window)
        
        # Save the figure showing all the SNRs vs. median SNRs.
        out_SNR_file = '_'.join(('Median_SNR', outFileBase))
        fig_SNR.savefig(os.path.join(outPicDir, out_SNR_file), format='png')
        plt.close(fig_SNR)


    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("SNR Window 1 (561.00 - 561.60 nm)")
    ax.set_ylabel("SNR (Windows 2 & 3)")
    fig.suptitle("{}, {} spectra".format(obj, len(files)))
    
    ax.plot(window1, window2, marker='o', color='green',
            linestyle='None', label='Window 2 (620.24 - 620.84)')
    ax.plot(window1, window3, marker='o', color='blue',
            linestyle='None', label='Window 3 (623.20 - 623.80)')
    
    fig.legend()
    ax.grid(which='major', axis='both')
    outfile = os.path.join(outPicDir, "SNR_{}.png".\
                format(obj))
    fig.savefig(outfile, format='png')
    plt.close(fig)
    
    
