#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:36:31 2021

@author: dberke

A class to hold information about pairs with q-coefficients to see how the
fitting models for them react to changes in parameters.
"""

import csv

import h5py
import hickle
import numpy as np
import unyt as u

import varconlib as vcl

params_file = vcl.output_dir /\
    'fit_params/quadratic_pairs_4.0sigma_params.hdf5'
params_file_real = vcl.output_dir /\
    'fit_params/quadratic_pairs_4.0sigma_params_real.hdf5'

with h5py.File(params_file, 'r') as f:
    model_func = hickle.load(f, path='/fitting_function')
    coeffs_dict = hickle.load(f, path='/coeffs_dict')
print('Loaded fake params file')

with h5py.File(params_file_real, 'r') as f:
    coeffs_dict_real = hickle.load(f, path='/coeffs_dict')
print('Loaded real params file.')

solar = np.stack((5772 * u.K, 0., 4.44), axis=0)
sp1_temp = np.stack((100 * u.K, 0, 0), axis=0)
sp1_metallicity = np.stack((0 * u.K, 0.1, 0), axis=0)
sp1_logg = np.stack((0 * u.K, 0, 0.2), axis=0)

parameters = (sp1_temp, sp1_metallicity, sp1_logg)

pairs_dict = {'4652.593Cr1_4653.460Cr1_29': 137,
              '4652.593Cr1_4653.460Cr1_30': 138,
              '4759.449Ti1_4760.600Ti1_32': 186,
              '4759.449Ti1_4760.600Ti1_33': 187,
              '4799.873Ti2_4800.072Fe1_33': 244,
              '4799.873Ti2_4800.072Fe1_34': 245,
              '5138.510Ni1_5143.171Fe1_42': 510,
              '5187.346Ti2_5200.158Fe1_43': 568,
              '6123.910Ca1_6138.313Fe1_60': 681,
              '6123.910Ca1_6139.390Fe1_60': 682,
              '6138.313Fe1_6139.390Fe1_60': 705,
              '6153.320Fe1_6155.928Na1_61': 715,
              '6153.320Fe1_6162.452Na1_61': 717,
              '6153.320Fe1_6168.150Ca1_61': 720,
              '6155.928Na1_6162.452Na1_61': 722,
              '6162.452Na1_6168.150Ca1_61': 731,
              '6162.452Na1_6175.044Fe1_61': 732,
              '6168.150Ca1_6175.044Fe1_61': 744,
              '6192.900Ni1_6202.028Fe1_61': 757,
              '6242.372Fe1_6244.834V1_62': 773}

pair_shifts_dict = {'4652.593Cr1_4653.460Cr1_29': -0.00038216777192312534,
                    '4652.593Cr1_4653.460Cr1_30': -0.00038216777192312534,
                    '4759.449Ti1_4760.600Ti1_32': -0.08608703448473931,
                    '4759.449Ti1_4760.600Ti1_33': -0.08608703448473931,
                    '4799.873Ti2_4800.072Fe1_33': 3.9716942693164747,
                    '4799.873Ti2_4800.072Fe1_34': 3.9716942693164747,
                    '5138.510Ni1_5143.171Fe1_42': 9.739375086915475,
                    '5187.346Ti2_5200.158Fe1_43': 8.601587091731624,
                    '6123.910Ca1_6138.313Fe1_60': -15.341020624104381,
                    '6123.910Ca1_6139.390Fe1_60': -17.000134267733543,
                    '6138.313Fe1_6139.390Fe1_60': -1.659113643629162,
                    '6153.320Fe1_6155.928Na1_61': 1.8428579660801092,
                    '6153.320Fe1_6162.452Na1_61': 1.9425795825067826,
                    '6153.320Fe1_6168.150Ca1_61': 4.516673632824562,
                    '6155.928Na1_6162.452Na1_61': 0.09972161642667351,
                    '6162.452Na1_6168.150Ca1_61': 2.574094050317779,
                    '6162.452Na1_6175.044Fe1_61': -2.4490549556175454,
                    '6168.150Ca1_6175.044Fe1_61': -5.023149005935324,
                    '6192.900Ni1_6202.028Fe1_61': 14.62489137690288,
                    '6242.372Fe1_6244.834V1_62': -3.9286452205982956}


class PairModel():

    def __init__(self, pair_label):

        self.pair_label = pair_label

        self.seps_pre = np.zeros((3, 7))
        self.seps_real_pre = np.zeros((3, 7))
        self.seps_post = np.zeros((3, 7))
        self.seps_real_post = np.zeros((3, 7))

        for era in ('pre', 'post'):
            label = '_'.join((self.pair_label, era))

            for i, parameter in zip(range(0, 3), parameters):
                for j in range(-3, 4):
                    value_real = model_func(solar + j * parameter,
                                            *coeffs_dict_real[label])
                    if era == 'pre':
                        self.seps_real_pre[i, j+3] = value_real.value
                    else:
                        self.seps_real_post[i, j+3] = value_real.value

                    value = model_func(solar + j * parameter,
                                       *coeffs_dict[label])
                    if era == 'pre':
                        self.seps_pre[i, j+3] = value.value
                    else:
                        self.seps_post[i, j+3] = value.value

        solar_pre = self.seps_pre[0, 3]
        solar_pre_real = self.seps_real_pre[0, 3]
        solar_post = self.seps_post[0, 3]
        solar_post_real = self.seps_real_post[0, 3]

        self.diffs_pre = -1 * self.seps_pre + solar_pre
        self.diffs_real_pre = -1 * self.seps_real_pre + solar_pre_real
        self.diffs_post = -1 * self.seps_post + solar_post
        self.diffs_real_post = -1 * self.seps_real_post + solar_post_real

        self.model_diffs_pre = self.seps_real_pre - self.seps_pre
        self.model_diffs_post = self.seps_real_post - self.seps_post

        self.model_diffs2_pre = self.diffs_real_pre - self.diffs_pre
        self.model_diffs2_post = self.diffs_real_post - self.diffs_post

    def save_csv(self, csv_filepath):
        cols_header = ['-3', '-2', '-1', 'Solar', '+1', '+2', '+3']

        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Real diffs pre',
                             f'{pair_shifts_dict[self.pair_label]:.5f}'])
            writer.writerow(cols_header)
            writer.writerows(self.diffs_real_pre)
            writer.writerow(['Fake diffs pre'])
            writer.writerow(cols_header)
            writer.writerows(self.diffs_pre)
            writer.writerow(['Real diffs post'])
            writer.writerow(cols_header)
            writer.writerows(self.diffs_real_post)
            writer.writerow(['Fake diffs post'])
            writer.writerow(cols_header)
            writer.writerows(self.diffs_post)
            writer.writerow(['Model diffs pre (real - shifted)'])
            writer.writerows(self.model_diffs_pre)
            writer.writerow(['Model diffs post (real - shifted)'])
            writer.writerows(self.model_diffs_post)
            writer.writerow(['Model diffs2 pre (real - shifted)'])
            writer.writerow(cols_header)
            writer.writerows(self.model_diffs2_pre)
            writer.writerow(['Model diffs2 post (real - shifted)'])
            writer.writerow(cols_header)
            writer.writerows(self.model_diffs2_post)


if __name__ == '__main__':
    print('Running main body.')
    test = PairModel('5138.510Ni1_5143.171Fe1_42')
#    print(test.seps_post)
#    print(test.seps_real_post)
#    print(test.diffs_post)
#    print(test.diffs_real_post)
    print(test.model_diffs_post)
    print(test.model_diffs_pre)