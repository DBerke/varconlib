#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:40:24 2018

@author: dberke
"""

import matplotlib.pyplot as plt
import obs2d
import obs1d
import conversions
import varconlib as vcl
from pathlib import Path
import unyt as u


file2d = Path('/Users/dberke/HD68168/data/reduced/'
              '2012-02-25/HARPS.2012-02-26T04:02:48.797_e2ds_A.fits')
file1d = Path('/Users/dberke/HD68168/data/reduced/'
              '2012-02-25/ADP.2014-09-26T16:55:03.633.fits')
#blaze_file = Path('/Users/dberke/HARPS/Calibration/2012-02-25/'
#                  'data/reduced/2012-02-25/'
#                  'HARPS.2012-02-25T22:07:11.413_blaze_A.fits')

H_alpha_vac = 6564.614 * u.angstrom
test_telluric_line = 6280.03 * u.angstrom


line_vac = H_alpha_vac
left_lim = line_vac.to(u.nm) - 0.25 * u.nm
right_lim = line_vac.to(u.nm) + 0.15 * u.nm

obs_2d = obs2d.HARPSFile2DScience(file2d)
obs_1d = obs1d.readHARPSfile1d(file1d)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
fig.suptitle('Comparison of ADP and e2ds wavelengths.')
ax.set_xlim(left=left_lim.value, right=right_lim.value)
#ax.set_ylim(top=90000, bottom=15000)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Photon count')
ax.axvline(x=line_vac.to(u.nm), color='Red', linestyle=':')
#           label=r'H$\alpha$ (vacuum)')
ax.axvline(x=(conversions.vac2airESO(line_vac)).to(u.nm),
           color='GoldenRod', linestyle='--')
#           label=r'H$\alpha$ (air)')

wl1d = obs_1d['w'] * u.angstrom
flux1d = obs_1d['f']
print(f'number of wavelength points (1D) = {len(wl1d)}')

left_index = vcl.wavelength2index(wl1d, left_lim - 0.5 * u.nm)
right_index = vcl.wavelength2index(wl1d, right_lim + 0.1 * u.nm)

# Plot the initial 1D wavelengths (air, barycentric-corrected)
ax.plot(wl1d.to(u.nm), flux1d, color='ForestGreen', linestyle='-',
        label='Original (air, barycenter corrected)', marker='o',
        markersize=2)

# Plot the wavelengths shifted for the star's radial velocity, 9.39 km/s.
shifted_wl1d = vcl.shift_wavelength(wl1d[left_index:right_index],
                                    -1 * 9.39 * u.km / u.s)
ax.plot(shifted_wl1d.to(u.nm), flux1d[left_index:right_index],
        color='LimeGreen',
        linestyle='--',
        label='Radial velocity corrected')

# Plot the vacuum-converted wavelengths.
vac_shifted_wl1d = conversions.air2vacESO(shifted_wl1d)
ax.plot(vac_shifted_wl1d.to(u.nm), flux1d[left_index:right_index],
        color='MediumSeaGreen',
        linestyle='-', marker='+',
        label='Air-to-vacuum converted')

orders_found = [obs_2d.findWavelength(lim, obs_2d.barycentricArray) for
                lim in (left_lim, right_lim)]
print(orders_found)
orders_to_plot = set()
for orders in orders_found:
    for order in orders:
        orders_to_plot.add(order)

print(orders_to_plot)
# Shift the 2d wavelenths for the BERV.
obs_2d.BERV_shifted_array = obs_2d.shiftWavelengthArray(
                                   obs_2d._wavelengthArray,
                                   obs_2d._BERV)
print(obs_2d.BERV_shifted_array[1][:5])

# Shift the 2d wavelengths for radial velocity.
obs_2d.radvel_shifted_array = obs_2d.shiftWavelengthArray(
        obs_2d.BERV_shifted_array, -1 * obs_2d._radialVelocity)

for order in orders_to_plot:
    ax.plot(obs_2d._wavelengthArray[order].to(u.nm),
            obs_2d._photonFluxArray[order], linestyle='-.',
            label=f'Order {order}',
            color='CornflowerBlue')

    ax.plot(obs_2d.BERV_shifted_array[order].to(u.nm),
            obs_2d._photonFluxArray[order], linestyle='-',
            label=f'Order {order} (BERV corrected)',
            color='DeepSkyBlue', marker='o',
            markersize=2)

    ax.plot(obs_2d.radvel_shifted_array[order].to(u.nm),
            obs_2d._photonFluxArray[order], linestyle='--',
            label=f'Order {order} (BERV, radvel corrected)',
            color='DodgerBlue')

    # Convert the 2d wavelengths to vacuum.
    print('Converting air wavelengths to vacuum using Edlen `53 formula.')
    vacuum_array = conversions.air2vacESO(obs_2d.radvel_shifted_array[order])

    ax.plot(vacuum_array.to(u.nm),
            obs_2d._photonFluxArray[order], linestyle='-', marker='+',
            label=f'Order {order} (BERV, radvel corrected, (vac))',
            color='DarkSlateBlue')

ax.legend(ncol=2)

plt.show()
