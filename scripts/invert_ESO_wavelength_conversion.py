#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:33:47 2018

@author: dberke
"""

# Script to invert the ESO vacuum-to-air wavelengths conversion.

# Min and max wavelengths of HARPS:
# WAVELMIN = 378.177
# WAVELMAX = 691.326

import numpy as np
import matplotlib.pyplot as plt
import varconlib as vcl

def air_indexEdlen53(l, t=15., p=760.):
    """Return the index of refraction of air at given t and p
    
    Formula is Edlen 1953, provided directly by ESO
    """
    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t)\
    * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n

def vac2airESO(ll):
    """Return a vacuum wavlength from an air wavelength (A) using Edlen 1953
    
    """
    llair = ll/air_indexEdlen53(ll)
    return llair


def air2vacESO(wl_arr):
    """Take an array of air wavelengths (A) and return an array of vacuum wavelengths   
    """
    tolerance = 1e-11

    air2vacDict = {}
    air_wl = []
    vac_wl = []

    # 3700
    # 7000
    for i in range(0, len(wl_arr)):
        newwl = wl_arr[i]
        oldwl = 0.
        #print(i)
        #print(newwl)
        while abs(oldwl - newwl) > tolerance:
            #print(abs(oldwl - newwl))
            oldwl = newwl
            n = air_indexEdlen53(newwl)
            newwl = wl_arr[i] * n
            #print(n)
            #print(newwl)
    
        air2vacDict[i] = newwl
        vac_wl.append(newwl)
        air_wl.append(wl_arr[i])
        #print((i, newwl))
    print(air2vacDict)




    return (np.array(vac_wl), np.array(wl_arr))

air_arr = range(3000, 7000, 1)
air_arr2 = np.array(range(3000, 7000, 1))
vac, air = air2vacESO(air_arr)
vac2 = vcl.air2vacESO(air_arr)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(air_arr2, vac2-air_arr2, 'r--')
ax.plot(air_arr2, vac-air, 'b:')
plt.show()
