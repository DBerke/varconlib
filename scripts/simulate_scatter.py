#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:54:21 2018

@author: dberke
"""

# Script to simulate the uncertainties of measuring wavelength position
# of absorption lines by injecting Gaussian noise into real flux arrays.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import varconlib as vcl
from glob import glob
from tqdm import tqdm
from varconlib import fitGaussian, get_vel_separation, injectGaussianNoise
from pathlib import Path


if __name__ == 'main':
    search_string = '/Users/dberke/code/tables/linearray*.csv'
    files = [Path(file) for file in glob(search_string)]
    lines = []
    medians = []
    standard_devs = []
    for file in tqdm(files):
        filestring = str(file)
        nom_wavelength = filestring[-12:-4]
        data = pd.read_csv(file, header=0, engine='c')
        wavelengths, vel_offsets = injectGaussianNoise(data,
                                                       nom_wavelength,
                                                       num_iter=10,
                                                       plot=False)
        lines.append(nom_wavelength)
        med_wl = np.median(wavelengths)
        wavelengths -= med_wl
        medians.append(np.median(wavelengths))
        standard_devs.append(np.std(vel_offsets))

    for line, med, std in zip(lines, medians, standard_devs):
        print('{0}: med. value: {1:.4f}, standard deviation {2:.8f}'.
              format(line, med, std))
