#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:51:36 2019

@author: dberke

Short script to simulate errors on Gaussian random variables.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_exposure(continuum=100, length=100):

    gain = 1.36
    electrons = np.random.normal(loc=continuum, scale=np.sqrt(continuum),
                                 size=length)
    ADUs = electrons / gain
#    actual_flux = ADUs
    actual_flux = electrons
    assumed_flux = ADUs
#    assumed_flux = electrons
    err = np.sqrt(assumed_flux)
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#
#    ax.errorbar(x=range(len(ADUs)), y=ADUs, yerr=err,
#                color='Blue', marker='+', linestyle='')
#    plt.show()
    return np.std(actual_flux), np.median(err)


stddevs, errors = [], []
for i in range(10, 300):
    std, err = simulate_exposure(continuum=i*1000, length=100)
    stddevs.append(std)
    errors.append(err)

stddevs = np.array(stddevs)
errors = np.array(errors)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'Flux $\sigma$')
ax.set_ylabel('Median error')

x = np.linspace(int(stddevs.min()), int(stddevs.max()), 2)

ax.scatter(stddevs, errors, color='Blue', marker='+')
ax.plot(x, x, color='Black', linestyle='-')

annotation = 'actual_flux = electrons\n' +\
             'assumed_flux = ADUs\n' +\
             'err = np.sqrt(assumed_flux)'
plt.annotate(annotation, xy=(0.04, 0.85), xycoords='axes fraction')
plt.show()
