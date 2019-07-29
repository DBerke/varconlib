#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:39:33 2019

@author: dberke

A script to find the centers of the various orders in HARPS. It does this by
running a box over each order to find the point with maximum flux.
"""

import os
from path import Path
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import obs2d

blaze_file_2003 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2003-11-12/'
                       'HARPS.2003-11-12T20:58:26.443_blaze_A.fits')

blaze_file_2005 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2005-04-23/'
                       'HARPS.2005-04-23T23:04:49.747_blaze_A.fits')

blaze_file_2007 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2007-07-10/'
                       'HARPS.2007-07-10T20:00:52.926_blaze_A.fits')

blaze_file_2009 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2009-03-23/'
                       'HARPS.2009-03-23T20:57:53.786_blaze_A.fits')

blaze_file_2011 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2011-06-16/'
                       'HARPS.2011-06-16T21:13:31.451_blaze_A.fits')

blaze_file_2013 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2013-01-21/'
                       'HARPS.2013-01-21T20:42:38.643_blaze_A.fits')

blaze_file_2015 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2015-05-14/'
                       'HARPS.2015-05-14T19:54:02.470_blaze_A.fits')

blaze_file_2016 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2016-09-20/'
                       'HARPS.2016-09-20T20:24:57.174_blaze_A.fits')

blaze_file_2017 = Path('/Volumes/External Storage/HARPS/'
                       'blaze_files/data/reduced/2017-02-24/'
                       'HARPS.2017-02-24T18:44:52.641_blaze_A.fits')


blaze_files = (blaze_file_2003, blaze_file_2005, blaze_file_2007,
               blaze_file_2009, blaze_file_2011, blaze_file_2013,
               blaze_file_2015, blaze_file_2016, blaze_file_2017)

dates = ('2003', '2005', '2007', '2009', '2011', '2013', '2015', '2016',
         '2017')


for blaze_file_path, date, in zip(blaze_files, dates):
    blaze_obs = obs2d.HARPSFile2D(blaze_file_path)

    for box_radius in range(10, 50, 10):
        out_dir_path = Path('/Users/dberke/Pictures/2D_vs_1D_investigation/'
                            'order_centers/box_radius_{}/'.format(box_radius))
        if not out_dir_path.exists():
            os.mkdir(out_dir_path)

        start_point = box_radius
        end_point = 4095 - 2

        median_flux_list = []
        median_flux_pixels = []

        for order in trange(72, unit='orders'):
            order_flux_list = []
            order_pixel_list = []
            for i in range(start_point, end_point, 1):
                median_flux = np.median(blaze_obs._rawData[order]
                                        [i - box_radius:i + box_radius])
                order_pixel_list.append(i)
                order_flux_list.append(median_flux)

            median_flux_list.append(order_flux_list)
            median_flux_pixels.append(order_pixel_list)

        median_flux_array = np.array(median_flux_list)
        pixel_location_array = np.array(median_flux_pixels)

        max_points = []

        for order in trange(72):

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.grid()

            ax.plot(blaze_obs._rawData[order], marker='+', linestyle='',
                    label=f'Order {order}')

            ax.plot(pixel_location_array[order], median_flux_array[order],
                    marker='x', linestyle='',
                    label=f'Order {order} smoothed')

            max_point = median_flux_array[order].argmax() + start_point
            max_points.append(max_point)
        #    tqdm.write(str(max_point))
            ax.axvline(x=max_point, linestyle='--',
                       label=f'Max (order {order})')

            ax.set_xlim(left=max_point-75, right=max_point+75)
            ax.set_ylim(top=median_flux_array[order][max_point]+0.02,
                        bottom=median_flux_array[order][max_point]-0.02)
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Normalized flux')

            ax.legend()

            out_file_path = out_dir_path /\
                            '{0}/{2}/Order_{1}_({2}).png'.format(date,
                                                                 order,
                                                                 box_radius)
            if not out_file_path.parent.exists():
                os.mkdir(str(out_file_path.parent.parent))
                os.mkdir(str(out_file_path.parent))
            fig.savefig(out_file_path)
            plt.close(fig)
        #    plt.show()

        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.grid()
        ax2.plot(max_points, linestyle=':', marker='o', color='BurlyWood',
                 markerfacecolor='CornflowerBlue',
                 markeredgecolor='CornflowerBlue')

        ax2.set_xlabel('Order')
        ax2.set_ylabel('Central pixel')

    #    plt.show()
        centers_path = out_dir_path.parent.parent /\
                        'Centers_{}_({}).png'.format(date, box_radius)
        fig2.savefig(centers_path)
        plt.close(fig2)
