#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:19:52 2020

@author: dberke

This script finds the remaining scatter in the pair-wise velocity separation
measurements beyond what can be accounted for by the statistical uncertainties.

"""

import argparse
import csv
from pathlib import Path
import pickle

import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm
import unyt as u

import varconlib as vcl
import varconlib.fitting as fitting


def main():
    """Run the main routine for this script."""

    data_dir = vcl.output_dir / 'pair_separation_files'
    csv_file = data_dir / 'pair_excess_scatters.csv'
    eras = ('pre', 'post')

    # Import the list of pairs to use.
    with open(vcl.final_pair_selection_file, 'r+b') as f:
        pairs_list = pickle.load(f)

    pair_results_dict = {}

    model_func = fitting.constant_model

    for pair in tqdm(pairs_list):
        for order_num in pair.ordersToMeasureIn:
            pair_label = "_".join([pair.label, str(order_num)])

            pair_results_list = [pair_label]

            pre_file = data_dir/f'pre/{pair_label}_pair_separations_pre.csv'
            post_file = data_dir/f'post/{pair_label}_pair_separations_post.csv'
            pre_values = pd.read_csv(pre_file)
            post_values = pd.read_csv(post_file)

            # Find the scatter in the pre-change values.
            seps_pre = ma.masked_invalid(pre_values['delta(v)_pair (m/s)'])
            y_pre = seps_pre[~seps_pre.mask]
            errs_pre = ma.masked_invalid(pre_values['err_stat_pair (m/s)'])
            errs_pre = errs_pre[~seps_pre.mask]
            x_pre = ma.array([i for i in
                              range(len(pre_values['delta(v)_pair (m/s)']))])
            x_pre = ma.array(x_pre[~seps_pre.mask])
            weighted_mean_pre = np.average(y_pre, weights=errs_pre**-2)

            results_pre = fitting.find_sys_scatter(model_func, x_pre, y_pre,
                                                   errs_pre,
                                                   weighted_mean_pre,
                                                   verbose=args.verbose)
            vprint('Terminated with sigma_sys ='
                   f' {results_pre["sys_err_list"][-1]:.5f} and chi^2 ='
                   f' {results_pre["chi_squared_list"][-1]:.5f}')
            pair_results_list.append(results_pre['sys_err_list'][-1])

            # Find the scatter in the post-change values.
            seps_post = ma.masked_invalid(post_values['delta(v)_pair (m/s)'])
            y_post = seps_post[~seps_post.mask]
            errs_post = ma.masked_invalid(post_values['err_stat_pair (m/s)'])
            errs_post = errs_post[~seps_post.mask]
            x_post = ma.array([i for i in
                               range(len(post_values['delta(v)_pair (m/s)']))])
            x_post = ma.array(x_post[~seps_post.mask])
            weighted_mean_post = np.average(y_post, weights=errs_post**-2)

            results_post = fitting.find_sys_scatter(model_func, x_post, y_post,
                                                    errs_post,
                                                    weighted_mean_post,
                                                    verbose=args.verbose)
            vprint('Terminated with sigma_sys ='
                   f' {results_post["sys_err_list"][-1]:.5f} and chi^2 ='
                   f' {results_post["chi_squared_list"][-1]:.5f}')
            pair_results_list.append(results_post['sys_err_list'][-1])

            # Find the scatter in the combined results.
            # y_tot = ma.append(y_pre, y_post)
            # x_tot = ma.append(x_pre, x_post)
            # errs_tot = ma.append(errs_pre, errs_post)
            # weighted_mean_tot = np.average(y_tot, weights=errs_tot**-2)

            # results_tot = fitting.find_sys_scatter(model_func, x_tot, y_tot,
            #                                        errs_tot,
            #                                        weighted_mean_tot,
            #                                        verbose=args.verbose)
            # vprint('Terminated with sigma_sys ='
            #        f' {results_tot["sys_err_list"][-1]:.5f} and chi^2 ='
            #        f' {results_tot["chi_squared_list"][-1]:.5f}')
            # pair_results_list.append(results_tot['sys_err_list'][-1])

            pair_results_dict[pair_label] = pair_results_list


    with open(csv_file, 'w', newline='') as f:
        datawriter = csv.writer(f)
        header = ('pair_label', 'sigma_sys_pre (m/s)', 'sigma_sys_post (m/s)')
        datawriter.writerow(header)
        for value in pair_results_dict.values():
            datawriter.writerow(value)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to find the'
                                     ' systematic scatter in pair separations')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out more information about the script.')

    args = parser.parse_args()

    vprint = vcl.verbose_print(args.verbose)

    main()
