#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:41:58 2018

@author: dberke
"""

# Script to automatically find given transitions line from a list and the known
# radial velocity of the star, then fit them and measure their positions.

import numpy as np
import varconlib as vcl
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
import pandas as pd
from scipy.optimize import curve_fit
from astropy.visualization import hist as astrohist
from pathlib import Path
from tqdm import tqdm
plt.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)


def getpairlist(listfile):
    """Parse a list of pairs of lines from a given input file.

    Parameters
    ----------
    listfile : Path object or str
        A path to a file containing a list of line pairs.

    Examples
    --------
    The format for the given file should be

    pair1a
    pair1b

    pair2a
    pair2b

    etc. Any columns after the first, which should contain the wavelength of
    the line, are ignored.

    Higher multiplets than two are allowed, where it is assumed that of a
    stucture such as:

    line A
    line B
    line C
    line D...

    the line pairs to be returned are (line A, line B), (line A, line C),
    (line A, line D), etc.
    """
    pairlist = []
    print(listfile)
    with open(listfile, 'r') as f:
        lines = f.readlines()

    temppair = []
    for line in lines:
        if '#' in line:
            pass
        else:
            if not line == '\n':
                temppair.append(line.split()[0])
            else:
                if len(temppair) == 2:
                    pairlist.append((temppair[0], temppair[1]))
                elif len(temppair) > 2:
                    for wl in temppair[1:]:
                        pairlist.append((temppair[0], wl))
                elif temppair == []:
                    # This should skip the first blank line beneath the header
                    # that's output by default in pair list files.
                    pass
                else:
                    raise RuntimeError("Single unpaired line.")
                temppair = []

    return pairlist


def plotstarseparations(mseps):
    """
    """

    fig_par = plt.figure(figsize=(8, 6))
    fig_gauss = plt.figure(figsize=(8, 6))
    for i in range(len(mseps[0])):
            ax_par = fig_par.add_subplot(5, 7, i+1)
            ax_gauss = fig_gauss.add_subplot(5, 7, i+1)
            parhistlist = []
            gausshistlist = []
            for seplist in mseps:
                parhistlist.append(seplist[i][0])
                gausshistlist.append(seplist[i][1])
            parhistlist = np.array(parhistlist)
            parhistlist -= np.median(parhistlist)
            gausshistlist = np.array(gausshistlist)
            gausshistlist -= np.median(gausshistlist)
            parmax = parhistlist.max()
            parmin = parhistlist.min()
#            print(min, max)
            if parmax > abs(parmin):
                parlim = parmax
            else:
                parlim = abs(parmin)
#            print(lim)
            gaussmax = parhistlist.max()
            gaussmin = parhistlist.min()
#            print(min, max)
            if gaussmax > abs(gaussmin):
                gausslim = gaussmax
            else:
                gausslim = abs(gaussmin)
#            ax.hist(parhistlist, range=(-1.05*lim, 1.05*lim))
            astrohist(parhistlist, ax=ax_par, bins=10,
                      range=(-1.05*parlim, 1.05*parlim))
            astrohist(gausshistlist, ax=ax_gauss, bins=10,
                      range=(-1.05*gausslim, 1.05*gausslim))
#    outfile
#    outfile = '/Users/dberke/Pictures/.png'.format(i)
#    fig.savefig(outfile, format='png')
    plt.tight_layout(pad=0.5)
    plt.show()


def plot_line_comparisons(mseps, linepairs):
    """
    """

    for i, linepair in zip(range(len(mseps[0])), linepairs):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(r'$\delta v$ ({} nm - {} nm) [m/s]'.
                      format(linepair[0], linepair[1]), fontsize=18)
        ax.set_ylabel('Observation number', fontsize=18)
        parlist = []
        parerr = []
        gausslist = []
        gausserr = []
        sparlist = []
        sparerr = []
        for seplist in mseps:
                parlist.append(seplist[i]['parveldiff'])
                parerr.append(seplist[i]['pardifferr'])
                gausslist.append(seplist[i]['gaussveldiff'])
                gausserr.append(seplist[i]['gaussdifferr'])
                sparlist.append(seplist[i]['sparveldiff'])
                sparerr.append(seplist[i]['spardifferr'])
        parlist = np.array(parlist)
        parlist -= np.median(parlist)
        gausslist = np.array(gausslist)
        gausslist -= np.median(gausslist)
        sparlist = np.array(sparlist)
        sparlist -= np.median(sparlist)
        par_rms = np.sqrt(np.mean(np.square(parlist)))
        gauss_rms = np.sqrt(np.mean(np.square(gausslist)))
        spar_rms = np.sqrt(np.mean(np.square(sparlist)))
        y_par = np.arange(0, len(parlist), 1)
        y_gauss = np.arange(0.3, len(parlist)+0.3, 1)
        y_spar = np.arange(0.6, len(parlist)+0.6, 1)
        ymin, ymax = -5, len(parlist)+5
        lim = max(abs(parlist.min()), abs(gausslist.min()),
                  parlist.max(), gausslist.max())
        if lim == 0:
            lim = 10
        ax.vlines(x=0, ymin=ymin, ymax=ymax+2.3, color='black',
                  linestyle='--', alpha=1, zorder=1)
#        ax.set_xlim(left=-1.05*lim, right=1.05*lim)
        ax.set_xlim(left=-100, right=100)
        ax.set_ylim(bottom=ymin, top=ymax)
        mean_rms = np.mean([par_rms, gauss_rms, spar_rms])
        mean_rms = np.mean(gauss_rms)
        ax.axvspan(-1*mean_rms, mean_rms, color='gray', alpha=0.3)
#        ax.errorbar(parlist, y_par, xerr=parerr,
#                    color='DodgerBlue', marker='o', elinewidth=2, linestyle='',
#                    capsize=2, capthick=2, markersize=4,
#                    label='Parabola fit (RMS: {:.2f}, mean err: {:.2f})'.
#                    format(par_rms, np.mean(parerr)),
#                    zorder=2)
        ax.errorbar(gausslist, y_gauss, xerr=gausserr,
                    color='ForestGreen', marker='o', elinewidth=2,
                    linestyle='', capsize=2, capthick=2, markersize=4,
                    label='Gaussian fit (RMS: {:.1f}, mean err: {:.1f})'.
                    format(gauss_rms, np.mean(gausserr)),
                    zorder=3)
        ax.errorbar(0, len(gausslist)+1.3, xerr=gauss_rms,
                    color='FireBrick', marker='', elinewidth=4,
                    capsize=0, label='RMS')
        ax.errorbar(0, len(gausslist)+2.3, xerr=np.mean(gausserr),
                    color='DodgerBlue', marker='', elinewidth=4,
                    capsize=0, label='Mean error')
#        ax.errorbar(sparlist, y_spar, xerr=sparerr,
#                    color='FireBrick', marker='o', elinewidth=2, linestyle='',
#                    capsize=2, capthick=2, markersize=4,
#                    label='Const. parabola fit (RMS: {:.1f}, mean err: {:.1f})'.
#                    format(spar_rms, np.mean(sparerr)),
#                    zorder=2)

        ax.xaxis.set_minor_locator(ticks.AutoMinorLocator(5))
        ax.legend(framealpha=0.4, fontsize=18)
        plt.show()
        outfile = '/Users/dberke/Pictures/HD146233/Linepair{}.png'.\
                  format(i+1)
        fig.savefig(outfile, format='png')
        plt.close(fig)


############
#pairlistfile = 'data/GoldStandardLineList_vac_working.txt'
pairlistfile = 'data/linelists/Lines_purple_0.15-0.9_800kms_0.2.txt'
pairlist = getpairlist(pairlistfile)

print('{} total line pairs to analyze.'.format(len(pairlist)))

#pairlist = (('443.9589', '444.1128'), ('450.0151', '450.3467'),
#            ('459.9405', '460.3290'), ('460.5846', '460.6877'),
#            ('465.8889', '466.2840'), ('473.3122', '473.3780'),
#            ('475.9448', '476.0601'), ('480.0073', '480.0747'),
#            ('484.0896', '484.4496'), ('488.6794', '488.7696'),
#            ('490.9102', '491.0754'), ('497.1304', '497.4489'),
#            ('500.5115', '501.1420'), ('506.8562', '507.4086'),
#            ('507.3492', '507.4086'), ('513.2898', '513.8813'),
#            ('514.8912', '515.3619'), ('524.8510', '525.1670'),
#            ('537.5203', '538.1069'), ('554.4686', '554.5475'),
#            ('563.5510', '564.3000'), ('571.3716', '571.9418'),
#            ('579.4679', '579.9464'), ('579.5521', '579.9779'),
#            ('580.8335', '581.0828'), ('593.1823', '593.6299'),
#            ('595.4367', '595.8344'), ('600.4673', '601.0220'),
#            ('616.3002', '616.8146'), ('617.2213', '617.5042'),
#            ('617.7065', '617.8498'), ('623.9045', '624.6193'),
#            ('625.9833', '626.0427'), ('625.9833', '626.2831'),
#            ('647.0980', '647.7413'))

columns = ('object',
           'date',
           'line1_nom_wl',
           'line2_nom_wl',
           'vel_diff_gauss',
           'diff_err_gauss',
           'line1_wl_gauss',
           'line1_wl_err_gauss',
           'line1_amp_gauss',
           'line1_amp_err_gauss',
           'line1_width_gauss',
           'line1_width_err_gauss',
           'line1_fwhm_gauss',
           'line1_fwhm_err_gauss',
           'line1_chisq_nu_gauss',
           'line1_gauss_vel_offset',
           'line1_continuum',
           'line1_norm_depth',
           'line2_wl_gauss',
           'line2_wl_err_gauss',
           'line2_amp_gauss',
           'line2_amp_err_gauss',
           'line2_width_gauss',
           'line2_width_err_gauss',
           'line2_fwhm_gauss',
           'line2_fwhm_err_gauss',
           'line2_chisq_nu_gauss',
           'line2_gauss_vel_offset',
           'line2_continuum',
           'line2_norm_depth')

#pairlist = [(537.5203, 538.1069)]
#pairlist = [(579.4679, 579.9464)]
#pairlist = [(507.3498, 507.4086)]

baseDir = Path("/Volumes/External Storage/HARPS/")
global unfittablelines

#files = glob(os.path.join(baseDir, '4Vesta/*.fits')) # Vesta (6 files)
#files = glob(os.path.join(baseDir, 'HD208704/*.fits')) # G1 (17 files)
#files = glob(os.path.join(baseDir, 'HD138573/*.fits')) # G5
filepath = baseDir / 'HD146233'  # 18 Sco, G2 (151 files)
#filepath = baseDir / 'HD126525'  # (132 files)
#filepath = baseDir / 'HD78660'  # 1 file
#filepath = baseDir / 'HD183658' # 12 files
#filepath = baseDir / 'HD45184' # 116 files
#filepath = baseDir / 'HD138573' # 31 files
#filepath = Path('/Users/dberke/HD146233')
files = [file for file in filepath.glob('ADP*.fits')]
#files = [Path('/Users/dberke/HD146233/ADP.2014-09-16T11:06:39.660.fits')]
# Used this file with SNR = 200.3 to get flux arrays for simulating scatter
# in line positions.
#files = [Path('/Volumes/External Storage/HARPS/HD146233/ADP.2014-09-23T11:04:00.547.fits')]

total_results = []

num_file = 1
for infile in files:
    tqdm.write('Processing file {} of {}.'.format(num_file, len(files)))
    tqdm.write('filepath = {}'.format(infile))
    unfittablelines = 0
    results = vcl.searchFITSfile(infile, pairlist, columns, plot=True,
                                 save_arrays=True)
    if results is not None:
        total_results.extend(results)
        tqdm.write('\nFound {} unfittable lines.'.format(unfittablelines))
    num_file += 1

lines = pd.DataFrame(total_results, columns=columns)

tqdm.write("#############")
tqdm.write("{} files analyzed total.".format(len(files)))

file_parent = files[0].parent
target = file_parent.stem + '.csv'
csvfilePath = file_parent / target
tqdm.write('Output written to {}'.format(csvfilePath))
lines.to_csv(csvfilePath, index=False, header=True, encoding='utf-8',
             date_format='%Y-%m-%dT%H:%M:%S.%f')
