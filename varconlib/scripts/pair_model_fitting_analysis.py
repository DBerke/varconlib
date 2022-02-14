#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:46:36 2021

@author: dberke


A script to investigate model fitting for the pair 5138.510Ni1_5143.171Fe1_42.
"""

import argparse
from copy import deepcopy

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy.ma as ma
import numpy as np
from scipy.optimize import curve_fit


star_temperatures = np.array([5929.0, 5675.0, 5604.0, 5712.0, 5741.0, 5628.0,
                              5693.0, 5922.0, 5745.0, 5895.0, 5779.0, 5826.0,
                              5730.0, 5999.0, 5651.0, 5765.0, 5598.0, 5487.0,
                              5923.0, 5468.0, 5920.0, 5773.0, 5884.0, 5789.0,
                              5859.0, 5701.0, 5931.0, 5778.0, 6000.0, 5863.0,
                              5656.0, 5619.0, 5966.0, 5468.0, 5981.0, 6031.0,
                              5740.0, 6004.0, 5630.0, 5844.0, 5443.0, 5426.0])

star_metallicities = np.array([0.03, 0.28, -0.33, 0.21, -0.16, 0.07, 0.14,
                               0.01, -0.04, -0.04, 0.19, 0.06, 0.03, -0.24,
                               0.08, -0.24, 0.38, -0.27, -0.44, 0.0, -0.08,
                               -0.09, -0.28, -0.12, -0.29, 0.02, -0.17, -0.25,
                               0.06, 0.04, 0.25, 0.05, 0.05, 0.08, 0.09, -0.21,
                               0.05, 0.3, -0.24, -0.15, 0.07, -0.22])

star_gravities = np.array([4.33, 4.26, 4.5, 4.35, 4.46, 4.42, 4.34, 4.33, 4.41,
                           4.29, 4.37, 4.42, 4.42, 4.42, 4.45, 4.41, 4.28,
                           4.56, 4.47, 4.49, 4.38, 4.39, 4.46, 4.3, 4.4, 4.26,
                           4.39, 4.51, 4.36, 4.42, 4.32, 4.41, 4.42, 4.54,
                           4.36, 4.42, 4.27, 4.38, 4.49, 4.48, 4.46, 4.56])

star_pair_separations_real = ma.array([272204.26838101516, 272237.60685862513,
                                       272211.07471684524, 272221.4955971642,
                                       272193.19898914715, 272220.2058714348,
                                       272211.8107139474, 272175.02941655647,
                                       272205.1720695164, 272186.4941378495,
                                       272203.1461032811, 272174.33177556266,
                                       272203.7990204861, 272184.268811788,
                                       272195.52463358646, 272210.9182709817,
                                       272233.49027260114, 272215.61716811033,
                                       272197.5195466602, 272231.4163262674,
                                       272176.4761360873, 272183.20243924024,
                                       272188.71579142456, 272214.87267004355,
                                       272209.68384867697, 272232.8721581025,
                                       272193.97635705245, 272172.21473271796,
                                       272165.14598151256, 272177.9825334518,
                                       272223.1717996521, 272214.0529265494,
                                       272166.33748991444, 272187.0900153659,
                                       272170.61089830054, 272153.89382796467,
                                       272197.1351158594, 272168.9253981088,
                                       272222.70431122114, 272172.08578705305,
                                       272243.5270543148, 272214.970264181])

errors_weighted_mean_real = np.array([15.040850464293646, 16.91094777712614,
                                      6.423126801674854, 6.971778041404439,
                                      2.89241715242658, 4.610997597655417,
                                      10.30210943784157, 5.6617820200614,
                                      19.31116685647467, 3.8521649914002043,
                                      6.590208910553629, 2.0363043050595238,
                                      8.295688960543332, 0.8256818057989733,
                                      1.6220559981283718, 7.56525028748937,
                                      0.4520603843247118, 6.9051616189324125,
                                      6.009812235524046, 7.343904513776376,
                                      4.5209558724356285, 11.333812164221879,
                                      2.3831128634330367, 2.1769060481366678,
                                      11.169536803322275, 10.020391378649038,
                                      21.63022307738677, 2.9890768928736278,
                                      6.871066897051186, 5.810595240789181,
                                      9.95328445028001, 12.297628220260876,
                                      10.407800408237565, 6.731620060908986,
                                      2.8112912994386137, 18.454760651958743,
                                      3.1290512793091416, 4.998923889250891,
                                      8.89542474125502, 8.036826422685648,
                                      8.556534811279496, 6.7455818905531855])

errors_on_mean_real = np.array([0.011749187054905715, 16.91094777712614,
                                0.0015325879649561336, 0.0070633233975215814,
                                0.002776501298120265, 0.0015065034250245602,
                                0.010536151996269182, 0.008026733768920602,
                                19.31116685647467, 0.0033612032610568767,
                                0.0045692103069436794, 0.002087120670809289,
                                0.0027444123904487657, 0.0008092834627063813,
                                0.0017553920042988908, 0.002178246329153683,
                                0.00038906232491389936, 0.002668309782383085,
                                0.0028944370563734677, 0.011026463123121293,
                                0.0033277692176816222, 0.005249645145160902,
                                0.0020733552011286908, 0.0020315897183279773,
                                0.002341271470127546, 0.002361884479910257,
                                21.63022307738677, 0.0025167783630422346,
                                0.004496957619218894, 0.003185523005395126,
                                0.00685798423414051, 12.297628220260876,
                                0.00980363372112904, 0.004474905316110108,
                                0.0027165699269610854, 0.01060084180823119,
                                0.003316163622231283, 0.0010181733647861557,
                                0.005835947602678184, 0.0026606732843280577,
                                0.002620892353466455, 0.004851509069486999])

star_pair_separations_fake = ma.array([272204.26838101516, 272237.60685862513,
                                       272211.07471684524, 272221.4955971642,
                                       272194.9028847911, 272220.2058714348,
                                       272211.8107139474, 272181.274306768,
                                       272214.91144316044, 272186.4941378495,
                                       272203.1461032811, 272174.33177556266,
                                       272203.7990204861, 272184.4701318523,
                                       272195.6606651667, 272210.9182709817,
                                       272233.49027260114, 272215.61716811033,
                                       272197.5195466602, 272231.4163262674,
                                       272176.4761360873, 272189.0794219535,
                                       272188.71579142456, 272217.27307519165,
                                       272209.68384867697, 272237.9929538773,
                                       272193.97635705245, 272173.9021520748,
                                       272165.14598151256, 272180.6038103853,
                                       272223.1717996521, 272214.0529265494,
                                       272166.33748991444, 272187.0900153659,
                                       272170.61089830054, 272153.89382796467,
                                       272197.1351158594, 272168.9253981088,
                                       272222.70431122114, 272172.08578705305,
                                       272243.5270543148, 272214.970264181])

errors_weighted_mean_fake = np.array([15.040850464293646, 16.91094777712614,
                                      6.423126801674854, 6.971778041404439,
                                      2.961419429552108, 4.610997597655417,
                                      10.30210943784157, 6.443500218163256,
                                      19.31116685647467, 3.8521649914002043,
                                      6.590208910553629, 2.0363043050595238,
                                      8.295688960543332, 0.8289504101920876,
                                      1.6809017683208793, 7.56525028748937,
                                      0.4520603843247118, 6.9051616189324125,
                                      6.009812235524046, 7.343904513776376,
                                      4.5209558724356285, 7.206254648065806,
                                      2.3831128634330367, 2.3154822885199895,
                                      11.169536803322275, 17.355994377230154,
                                      21.63022307738677, 3.1114996197106732,
                                      6.871066897051186, 8.31842764361902,
                                      9.95328445028001, 12.297628220260876,
                                      10.407800408237565, 6.731620060908986,
                                      2.8112912994386137, 18.454760651958743,
                                      3.1290512793091416, 4.998923889250891,
                                      8.89542474125502, 8.036826422685648,
                                      8.556534811279496, 6.7455818905531855])

errors_on_mean_fake = np.array([0.011749187054905715, 16.91094777712614,
                                0.0015325879649561336, 0.0070633233975215814,
                                0.002459771460459631, 0.0015065034250245602,
                                0.010536151996269182, 0.008118998128382744,
                                19.31116685647467, 0.0033612032610568767,
                                0.0045692103069436794, 0.002087120670809289,
                                0.0027444123904487657, 0.0008055710699209822,
                                0.0016535580529764681, 0.002178246329153683,
                                0.00038906232491389936, 0.002668309782383085,
                                0.0028944370563734677, 0.011026463123121293,
                                0.0033277692176816222, 0.005271268568470388,
                                0.0020733552011286908, 0.001841317219674378,
                                0.002341271470127546, 17.355994377230154,
                                21.63022307738677, 0.0021289230869698714,
                                0.004496957619218894, 0.0012805791393874768,
                                0.00685798423414051, 12.297628220260876,
                                0.00980363372112904, 0.004474905316110108,
                                0.0027165699269610854, 0.01060084180823119,
                                0.003316163622231283, 0.0010181733647861557,
                                0.005835947602678184, 0.0026606732843280577,
                                0.002620892353466455, 0.004851509069486999])

# This is the indices of stars to have their values shifted
# artificially. Only 6 solar twins had post-fiber-change observations
# for this pair.
indices_list = [8, 11, 12, 21, 25, 29, 36]


def constant_model(data, a):
    """
    Return a constant value.

    Parameters
    ----------
    data : array-like
        The independent variable data.
    a : float or int
        A constant value to return.

    Returns
    -------
    a : float or int
        The value input as `a`.

    """

    return a


def linear_model(data, a, b, c, d):
    """
    Return the value of a three-dimensional linear model

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a, b, c, d : float or int
        Values of the coefficient for the model. `a` is the zeroth-order
        (constant) value, while `b`, `c`, and `d` represent the first-order
        (linear) coefficients for the three dimensions.

    Returns
    -------
    float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2]


def quadratic_model(data, a, b, c, d, e, f, g):
    """
    Return the value of a three-dimensional function of second order.

    Parameters
    ----------
    data : array-like with dimensions (3, n)
        The idenpendent variable. Each column represents a collection of three
        values to be passed to the thee dimensions of the function.
    a : float or int
        The constant term for the function.
    b, c, d : float or int
        The values of the coefficients for the linear terms of the function.
    e, f, g : float or int
        The values of the coefficients for the quadratic terms of the function.

    Returns
    -------
     float
        The value of the function for the given data and coefficients.

    """

    return a + b * data[0] + c * data[1] + d * data[2] +\
        e * data[0] ** 2 + f * data[1] ** 2 + g * data[2] ** 2


def calc_chi_squared_nu(residuals, errors, n_params):
    """
    Return the value of the reduced chi-squared statistic for the given data.

    Parameters
    ----------
    residuals : array-like of floats or ints
        An array of values representing a set of measured deviations from a
        fitted model.
    errors : array-like of floats or ints
        An array of variances for each point, of the same length as
        `residuals`.
    n_params : int
        The number of fitted parameters in the model.

    Returns
    -------
    float
        The value of chi-squared per degree-of-freedom for the given data and
        number of fitted parameters.

    """

    chi_squared = np.sum(np.square(residuals / errors))
    dof = len(residuals) - n_params
    if dof <= 0:
        return np.nan
    else:
        return chi_squared / dof


def find_sys_scatter(model_func, x_data, y_data, err_array, beta0,
                     n_sigma=2.5, tolerance=0.001, verbose=False):
    """Find the systematic scatter in a dataset with a given model.

    Takes a model function `model_func`, and arrays of x, y, and uncertainties
    (which must have the same length) and an initial guess to the parameters of
    the function, and fits the model to the data. It then checks the reduced
    chi-squared value, and if it is greater than 1 (with a tolerance of 1e-3),
    it adds an additional amount in quadrature to the error array and refits
    the data, continuing until the chi-squared value is within the tolerance.

    Parameters
    ----------
    model_func : callable
        The function to fit the data with.
    x_data : array-like
        The array of x-values to fit.
    y_data : array-like
        The array of y-values to fit. Must have same length as `x_data`.
    err_array : array-like
        The error array for the y-values. Must have same length as `x_data` and
        `y_data`.
    beta0 : tuple
        A tuple of values to use as the initial guesses for the paremeters in
        the function given by `model_func`.
    n_sigma : float
        The number of sigma outside of which a data point is considered an
        outlier.
    tolerance : float, Default : 0.001
        The distance from one within which the chi-squared per degree of
        freedom must fall for the iteration to exit. (Note that if the
        chi-squared value is naturally less than one on the first iteration,
        the iteration will end even if the value is not closer to one than the
        tolerance.)
    verbose : bool, Default : False
        Whether to print out more diagnostic information on the process.

    Returns
    -------
    dict
        A dictionary containing the following keys:
            popt : tuple
                A tuple of the optimized values found by the fitting routine
                for the parameters.
            pcov : `np.array`
                The covariance matrix for the fit.
            residuals : `np.array`
                The value of `y_data` minus the model values at all given
                independent variables.
            sys_err_list : list of floats
                A list conainting the values of the systematic error at each
                iteration. The last value is the values which brings the
                chi-squared per degree of freedom for the data within the
                tolerance to one.
            chi_squared_list : list of floats
                A list containing the calculated chi-squared per degree of
                freedom for each step of the iteration.
            mask_list : list of lists
                A list containing the mask applied to the data at each
                iteration. Each entry will be a list of 1s and 0s.

    """

#    vprint = vcl.verbose_print(verbose)

    # Iterate to find what additional systematic error is needed
    # to get a chi^2 of ~1.
    chi_tol = tolerance
    diff = 1
    sys_err = 0
    iter_err_array = np.sqrt(np.square(err_array) +
                             np.square(sys_err))

    chi_squared_list = []
    sigma_sys_list = []
    mask_list = []
    sigma_sys_change_list = []

    x_data.mask = False
    y_data.mask = False
    err_array.mask = False

    orig_x_data = ma.copy(x_data)
    orig_y_data = ma.copy(y_data)
    orig_errs = ma.copy(err_array)

    last_mask = np.zeros_like(y_data)
    new_mask = np.ones_like(y_data)

    iterations = 0
    chi_squared_flips = 0

    if verbose:
        print('  #   sigma_sys      diff     chi^2      SSCA   #*   flips')

    while True:
        iterations += 1
        popt, pcov = curve_fit(model_func, x_data, y_data,
                               sigma=iter_err_array,
                               p0=beta0,
                               absolute_sigma=True,
                               method='lm', maxfev=10000)

        iter_model_values = model_func(x_data, *popt)

        iter_residuals = y_data - iter_model_values

        # Find the chi^2 value for this fit:
        chi_squared_nu = calc_chi_squared_nu(iter_residuals, iter_err_array,
                                             len(popt))

        try:
            last_chi_squared = chi_squared_list[-1]
        except IndexError:  # On the first iteration
            pass
        else:
            if chi_squared_nu > 1 and last_chi_squared < 1:
                chi_squared_flips += 1
            elif chi_squared_nu < 1 and last_chi_squared > 1:
                chi_squared_flips += 1
            else:
                pass

        sigma_sys_list.append(sys_err)
        chi_squared_list.append(chi_squared_nu)
        mask_list.append(last_mask)

        diff = abs(chi_squared_nu - 1)

        # Set the amount to change by using the latest chi^2 value.
        sigma_sys_change_amount = np.power(chi_squared_nu, 2/3)
        sigma_sys_change_list.append(sigma_sys_change_amount)

        if verbose:
            print(f'{iterations:>3}, '
                  f'{sys_err:>10.6f}, {diff:>8.4f}, {chi_squared_nu:>8.4f},'
                  f' {sigma_sys_change_amount:>8.4f},'
                  f' {iter_residuals.count():>3},  {chi_squared_flips}')
#        if verbose:
#            sleep_length = 0 if chi_squared_flips < 3 else 0.1
#            sleep(sleep_length)

        if chi_squared_nu > 1:
            if sys_err == 0:
                sys_err = np.sqrt(chi_squared_nu - 1) * np.nanmedian(err_array)
                # sys_err = np.sqrt(chi_squared_nu)
            else:
                sys_err = sys_err * sigma_sys_change_amount
        elif chi_squared_nu < 1:
            if sys_err == 0:
                # If the chi-squared value is naturally lower
                # than 1, don't change anything, just exit.
                break
            else:
                sys_err = sys_err * sigma_sys_change_amount

        # Construct new error array using all errors.
        iter_err_array = np.sqrt(np.square(orig_errs) +
                                 np.square(sys_err))
        new_mask = np.zeros_like(y_data)

        # Find residuals for all data, including that masked this iteration:
        full_model_values = model_func(orig_x_data, *popt)
        full_residuals = orig_y_data - full_model_values

        # Check for outliers at each point, and mark the mask appropriately.
        for i in range(len(iter_err_array)):
            if abs(full_residuals[i]) > (n_sigma * iter_err_array[i]):
                new_mask[i] = 1

        # Set up the mask on the x and y data and errors for the next iteration
        for array in (x_data, y_data, iter_err_array):
            if chi_squared_flips < 5:
                array.mask = new_mask
                last_mask = new_mask
            # If chi^2 flips between less than and greater than one too many
            # times, the routine is probably stuck in a loop adding and
            # removing a borderline point, so simply stop re-evaluating points
            # for inclusion.
            else:
                array.mask = last_mask

        # If chi^2 gets within the tolerance and the mask hasn't changed in the
        # last iteration, end the loop.
        if ((diff < chi_tol) and (np.all(last_mask == new_mask))):
            break

        # If the mask is still changing, but the sigma_sys value has
        # clearly converged to a value (by both the 10th and 100th most
        # recent entries being within the given tolerance of the most recent
        # entry), end the loop. Most runs terminate well under 100 steps so
        # this should only catch the problem cases.
        elif ((iterations > 100) and (diff < chi_tol) and
              ((abs(sigma_sys_list[-1] - sigma_sys_list[-10])) < chi_tol) and
              ((abs(sigma_sys_list[-1] - sigma_sys_list[-100])) < chi_tol)):
            break

        # If the chi^2 value is approaching 1 from the bottom, it may be the
        # case that it can never reach 1, even if sigma_sys goes to 0 (but it
        # will take forever to get there). In the case that chi^2_nu < 1,
        # and the values have clearly converged to a value within the tolerance
        # over the last 100 iterations, break the loop. This does leave the
        # derived sigma_sys value somewhat meaningless, but it should be small
        # enough in these cases as to be basically negligible.

        elif ((iterations > 100) and (chi_squared_nu < 1.) and
              ((abs(chi_squared_list[-1]) -
                chi_squared_list[-10]) < chi_tol) and
              ((abs(chi_squared_list[-1]) -
                chi_squared_list[-100]) < chi_tol)):
            # If sigma_sys is less than a millimeter per second, just set it
            # to zero.
            if sys_err < 0.0011:
                sigma_sys_list[-1] = 0
            break

        # If the iterations go on too long, it may be because it's converging
        # very slowly to zero sigma_sys, so give it a nudge if it's still large
        elif iterations == 500:
            if (sys_err > 0.001) and (sys_err < 0.01) and\
                    (sigma_sys_list[-1] < sigma_sys_list[-2]):
                sys_err = 0.001

        # If it's taking a really long time to converge, but sigma_sys is less
        # than a millimeter per second, just set it to zero and end the loop.
        elif iterations == 999:
            if sys_err < 0.0011:
                sigma_sys_list[-1] = 0
                break
            else:
                print(f'Final sys_err = {sys_err}')
                print(f'Final chi^2 = {chi_squared_nu}')
                print(f'diff = {diff}')
                print(np.all(last_mask == new_mask))
                for i, j in zip(last_mask, new_mask):
                    print(f'{i}  {j}')
                raise RuntimeError("Process didn't converge.")

    # ---------

    results_dict = {'popt': popt, 'pcov': pcov,
                    'residuals': iter_residuals,
                    'sys_err_list': sigma_sys_list,
                    'chi_squared_list': chi_squared_list,
                    'mask_list': mask_list}

    return results_dict


# Begin main code body.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '---synthetic', action='store_true',
                        help='Use synthetic data with a linear model.')
    parser.add_argument('-l', '--linear-model', action='store_true',
                        help='Use a linear model for fitting instead of a'
                        'quadratic one.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print more information about the'
                        ' fitting process.')
    args = parser.parse_args()

    # Stack the stellar parameters into vertical slices
    # for passing to model functions.
    # Create multiple copies since each fitting might end up excluding
    # different points.
    x_data_const_real = ma.array(np.stack((star_temperatures,
                                           star_metallicities,
                                           star_gravities), axis=0))
    x_data_const_fake = ma.array(np.stack((star_temperatures,
                                           star_metallicities,
                                           star_gravities), axis=0))
    x_data_quad_real = ma.array(np.stack((star_temperatures,
                                          star_metallicities,
                                          star_gravities), axis=0))
    x_data_quad_fake = ma.array(np.stack((star_temperatures,
                                          star_metallicities,
                                          star_gravities), axis=0))

    linear_params = [2.51415722e+05, -1.92405836e+00, -5.40185088e+01,
                     1.24225507e+04, 0, 0, 0]

    if args.synthetic:
        # Generate synthetic data using just the linear coefficients found for
        # the real, unshifted data for this pair.
        star_pair_synth_orig = ma.array(
                [quadratic_model(x_data_quad_real[:, i], *linear_params)
                    for i in range(len(x_data_quad_real[0]))])
        star_pair_synth_shift = deepcopy(star_pair_synth_orig)

        # Shift to apply to these stars, for this pair it's 9.739 m/s.
        fake_shift = 9.739
        for i in indices_list:
            star_pair_synth_shift[i] += fake_shift

    # Generate the error arrays for the real and fake data by taking the
    # maximum of the error on the weighted mean and error on the mean for each
    # star.
    err_array_real = ma.array(np.maximum(errors_weighted_mean_real,
                                         errors_on_mean_real))
    err_array_fake = ma.array(np.maximum(errors_weighted_mean_fake,
                                         errors_on_mean_fake))

    # Create a list of initial guesses (all 0's) for the fitting function:
    num_params = 4 if args.linear_model else 7
    params_list = [0 for i in range(num_params)]
    params_list[0] = float(np.mean(star_pair_separations_real))
    beta0_const = params_list[0]
    beta0 = tuple(params_list)

    # Set whether we're using real or sythetic data.
    if args.synthetic:
        y_data_orig = star_pair_synth_orig
        y_data_shift = star_pair_synth_shift

    else:
        y_data_orig = star_pair_separations_real
        y_data_shift = star_pair_separations_fake

    # Get the results for the real and shifted data.
    # First fit with a constant model to see how the data looks.
    if args.verbose:
        print('Fitting real data with constant model.')
    results_const_real = find_sys_scatter(constant_model,
                                          x_data_const_real,
                                          y_data_orig,
                                          err_array_real, beta0_const,
                                          n_sigma=4.0,
                                          tolerance=0.001,
                                          verbose=args.verbose)

    mask_real = results_const_real['mask_list'][-1]
    residuals_const_real = ma.array(results_const_real['residuals'],
                                    mask=mask_real)
    x_data_const_real.mask = mask_real
    err_array_real.mask = mask_real
    sys_err_real = results_const_real['sys_err_list'][-1]
    full_err_array_real = np.sqrt(ma.compressed(err_array_real) ** 2 +
                                  sys_err_real ** 2)
    coeffs_const_real = results_const_real['popt']
    if args.verbose:
        print()

    if args.verbose:
        print('Fitting fake data with constant model.')
    results_const_fake = find_sys_scatter(constant_model,
                                          x_data_const_fake,
                                          y_data_shift,
                                          err_array_fake, beta0_const,
                                          n_sigma=4.0,
                                          tolerance=0.001,
                                          verbose=args.verbose)

    mask_fake = results_const_fake['mask_list'][-1]
    residuals_const_fake = ma.array(results_const_fake['residuals'],
                                    mask=mask_fake)
    x_data_const_fake.mask = mask_fake
    err_array_fake.mask = mask_fake
    sys_err_fake = results_const_fake['sys_err_list'][-1]
    full_err_array_fake = np.sqrt(ma.compressed(err_array_fake) ** 2 +
                                  sys_err_fake ** 2)
    coeffs_const_fake = results_const_fake['popt']
    if args.verbose:
        print()

    # Now fit with the chosen model (linear or quadratic).
    if args.linear_model:
        chosen_model = linear_model
        model_type = 'linear'
    else:
        chosen_model = quadratic_model
        model_type = 'quadratic'
    if args.verbose:
        print(f'Fitting real data with {model_type} model.')
    results_quad_real = find_sys_scatter(chosen_model,
                                         x_data_quad_real,
                                         y_data_orig,
                                         err_array_real, beta0,
                                         n_sigma=4.0,
                                         tolerance=0.001,
                                         verbose=args.verbose)

    mask_real = results_quad_real['mask_list'][-1]
    residuals_quad_real = ma.array(results_quad_real['residuals'],
                                   mask=mask_real)
    x_data_quad_real.mask = mask_real
    err_array_real.mask = mask_real
    sys_err_real = results_quad_real['sys_err_list'][-1]
    full_err_array_real = np.sqrt(ma.compressed(err_array_real) ** 2 +
                                  sys_err_real ** 2)
    coeffs_real = results_quad_real['popt']
    print('The coefficients found for the real data are:')
    print(coeffs_real)
    print()

    if args.verbose:
        print(f'Fitting fake data with {model_type} model.')
    results_quad_fake = find_sys_scatter(chosen_model,
                                         x_data_quad_fake,
                                         y_data_shift,
                                         err_array_fake, beta0,
                                         n_sigma=4.0,
                                         tolerance=0.001,
                                         verbose=args.verbose)

    mask_fake = results_quad_fake['mask_list'][-1]
    residuals_quad_fake = ma.array(results_quad_fake['residuals'],
                                   mask=mask_fake)
    x_data_quad_fake.mask = mask_fake
    err_array_fake.mask = mask_fake
    sys_err_fake = results_quad_fake['sys_err_list'][-1]
    full_err_array_fake = np.sqrt(ma.compressed(err_array_fake) ** 2 +
                                  sys_err_fake ** 2)
    coeffs_fake = results_quad_fake['popt']
    print('The coefficients found for the shifted data are:')
    print(coeffs_fake)

    # Calculate the model offsets for solar parameters, ±1 SP1 step.
    solar = np.stack((5772, 0., 4.44), axis=0)
    sp1_temp = np.stack((100, 0, 0), axis=0)
    sp1_metallicity = np.stack((0, 0.1, 0), axis=0)
    sp1_logg = np.stack((0, 0, 0.1), axis=0)

    parameters = (sp1_temp, sp1_metallicity, sp1_logg)
    param_names = ('T_eff', '[Fe/H]', 'log(g)')
    unshifted_lists = []

    print()
    print('Absolute model values for unshifted data.')
    print('           -2        -1       solar      +1        +2')
    for i, parameter in enumerate(parameters):
        values = []
        for j in range(-2, 3):
            params = solar + j * parameter
            if (i == 2) and (j < 0):
                params -= parameter
            elif (i == 2) and (j > 0):
                params += parameter
            values.append(chosen_model(params,
                                       *coeffs_real))
        print(f'{param_names[i]:6}: {values[0]:.2f} {values[1]:.2f}'
              f' {values[2]:.2f} {values[3]:.2f} {values[4]:.2f}')
        unshifted_lists.append(values)
    print('Difference from solar:')
    solar_value = chosen_model(solar, *coeffs_real)
    for i, parameter in enumerate(parameters):
        values = []
        for j in range(-2, 3):
            params = solar + j * parameter
            if (i == 2) and (j < 0):
                params -= parameter
            elif (i == 2) and (j > 0):
                params += parameter
            values.append(solar_value - chosen_model(params,
                                                     *coeffs_real))
        print(f'{param_names[i]:6}: {values[0]:^9.2f} {values[1]:^9.2f}'
              f' {values[2]:^9.2f} {values[3]:^9.2f} {values[4]:^9.2f}')

    # Make an array out of unshifted values:
    unshifted_array = np.array(unshifted_lists)
#    print(unshifted_array)

    print()
    print('Difference for shifted data (unshifted - shifted).')
    print('           -2        -1       solar      +1        +2')
    for i, parameter in enumerate(parameters):
        values = []
        for j in range(-2, 3):
            params = solar + j * parameter
            if (i == 2) and (j < 0):
                params -= parameter
            elif (i == 2) and (j > 0):
                params += parameter
            values.append(unshifted_array[i, j+2] -
                          chosen_model(params,
                                       *coeffs_fake))
        print(f'{param_names[i]:6}: {values[0]:^ 9.2f} {values[1]:^ 9.2f}'
              f' {values[2]:^ 9.2f} {values[3]:^ 9.2f} {values[4]:^ 9.2f}')
    print('Difference from unshifted solar:')
    solar_value = chosen_model(solar, *coeffs_real)
    for i, parameter in enumerate(parameters):
        values = []
        for j in range(-2, 3):
            params = solar + j * parameter
            if (i == 2) and (j < 0):
                params -= parameter
            elif (i == 2) and (j > 0):
                params += parameter
            values.append(solar_value - chosen_model(solar + j * parameter,
                                                     *coeffs_fake))
        print(f'{param_names[i]:6}: {values[0]:^9.2f} {values[1]:^9.2f}'
              f' {values[2]:^9.2f} {values[3]:^9.2f} {values[4]:^9.2f}')

    # Now plot both sets of data as functions of all three parameters, with
    # the function sliced along each parameter overplotted.

    fig = plt.Figure(figsize=(10, 10), tight_layout=True)
    gs = GridSpec(ncols=3, nrows=4, figure=fig,
                  wspace=0)

    # Axes to plot with a zeroth-order constant function.
    temp_ax_const_real = fig.add_subplot(gs[0, 0])
    temp_ax_const_fake = fig.add_subplot(gs[2, 0], sharex=temp_ax_const_real,
                                         sharey=temp_ax_const_real)

    mtl_ax_const_real = fig.add_subplot(gs[0, 1], sharey=temp_ax_const_real)
    mtl_ax_const_fake = fig.add_subplot(gs[2, 1], sharex=mtl_ax_const_real,
                                        sharey=mtl_ax_const_real)

    logg_ax_const_real = fig.add_subplot(gs[0, 2], sharey=temp_ax_const_real)
    logg_ax_const_fake = fig.add_subplot(gs[2, 2], sharex=logg_ax_const_real,
                                         sharey=logg_ax_const_real)

    # Axes to plot with a quadratic function.
    temp_ax_quad_real = fig.add_subplot(gs[1, 0], sharex=temp_ax_const_real)
    temp_ax_quad_fake = fig.add_subplot(gs[3, 0], sharex=temp_ax_quad_real,
                                        sharey=temp_ax_quad_real)

    mtl_ax_quad_real = fig.add_subplot(gs[1, 1], sharex=mtl_ax_const_real,
                                       sharey=temp_ax_quad_real)
    mtl_ax_quad_fake = fig.add_subplot(gs[3, 1], sharex=mtl_ax_quad_real,
                                       sharey=mtl_ax_quad_real)

    logg_ax_quad_real = fig.add_subplot(gs[1, 2], sharex=logg_ax_const_real,
                                        sharey=temp_ax_quad_real)
    logg_ax_quad_fake = fig.add_subplot(gs[3, 2], sharex=logg_ax_quad_real,
                                        sharey=logg_ax_quad_real)

    mtl_ax_const_real.tick_params(labelleft=False)
    logg_ax_const_real.tick_params(labelleft=False)
    mtl_ax_quad_real.tick_params(labelleft=False)
    logg_ax_quad_real.tick_params(labelleft=False)
    mtl_ax_const_fake.tick_params(labelleft=False)
    logg_ax_const_fake.tick_params(labelleft=False)
    mtl_ax_quad_fake.tick_params(labelleft=False)
    logg_ax_quad_fake.tick_params(labelleft=False)

    temp_ax_quad_fake.set_xlabel('T$_{eff}$')
    mtl_ax_quad_fake.set_xlabel('[Fe/H]')
    logg_ax_quad_fake.set_xlabel('log(g)')

    temp_ax_const_real.set_ylabel('Offsets (real) (m/s)')
    temp_ax_quad_real.set_ylabel('Offsets (real) (m/s)')
    temp_ax_const_fake.set_ylabel('Offset (fake) (m/s)')
    temp_ax_quad_fake.set_ylabel('Offsets (fake) (m/s)')

    marker_edge_color = 'Black'
    color = 'Chocolate'
    elinecolor = 'Chocolate'
    function_line_color = 'ForestGreen'

    # Set up values to plot functions as slices along each parameter.
    x_temp = np.linspace(5400, 6080, 100)
    x_metallicity = np.linspace(-0.5, 0.5, 100)
    x_logg = np.linspace(4.25, 4.57, 100)
    x_values = (x_temp, x_metallicity, x_logg)

    solar_temp = np.full((100,), 5772)
    solar_metallicity = np.full((100,), 0)
    solar_logg = np.full((100,), 4.44)

    temp_plot_values = np.stack((x_temp, solar_metallicity, solar_logg),
                                axis=0)
    mtl_plot_values = np.stack((solar_temp, x_metallicity, solar_logg),
                               axis=0)
    logg_plot_values = np.stack((solar_temp, solar_metallicity, x_logg),
                                axis=0)
    plot_values = (temp_plot_values, mtl_plot_values, logg_plot_values)

    # Plot the results on each axis:
    # Plot the constant fits.
    for i, ax in enumerate((temp_ax_const_real, mtl_ax_const_real,
                            logg_ax_const_real)):
        ax.axhline(y=0, color='Gray', linestyle='-', zorder=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.errorbar(x=ma.compressed(x_data_const_real[i]),
                    y=ma.compressed(residuals_const_real),
                    yerr=ma.compressed(err_array_real),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='',
                    capsize=4, capthick=1.5,
                    zorder=10)
        ax.errorbar(x=ma.compressed(x_data_quad_real[i]),
                    y=ma.compressed(residuals_const_real),
                    yerr=ma.compressed(full_err_array_real),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='o',
                    capsize=3, capthick=2,
                    markeredgecolor=marker_edge_color, zorder=15)
        ax.plot(ma.compressed(x_data_quad_real[i][indices_list]),
                ma.compressed(residuals_const_real)[indices_list],
                color='Green',
                linestyle='', marker='o',
                markeredgecolor=marker_edge_color, zorder=16)
        # Plot the fitted fuction with solar values except for the parameter
        # plotted on this axis.
        ax.plot(x_values[i], chosen_model(plot_values[i],
                *coeffs_real)-coeffs_const_real[0],
                color=function_line_color, linestyle='-', zorder=20)

    for i, ax in enumerate((temp_ax_const_fake, mtl_ax_const_fake,
                            logg_ax_const_fake)):
        ax.axhline(y=0, color='Gray', linestyle='-', zorder=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.errorbar(x=ma.compressed(x_data_quad_fake[i]),
                    y=ma.compressed(residuals_const_fake),
                    yerr=ma.compressed(err_array_fake),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='',
                    capsize=4, capthick=1.5,
                    zorder=10)
        ax.errorbar(x=ma.compressed(x_data_quad_fake[i]),
                    y=ma.compressed(residuals_const_fake),
                    yerr=ma.compressed(full_err_array_fake),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='o',
                    capsize=3, capthick=2,
                    markeredgecolor=marker_edge_color, zorder=15)
        ax.plot(ma.compressed(x_data_quad_fake[i][indices_list]),
                ma.compressed(residuals_const_fake)[indices_list],
                color='Green',
                linestyle='', marker='o',
                markeredgecolor=marker_edge_color, zorder=16)
        # Plot the 1-dimensional model slice.
        ax.plot(x_values[i], chosen_model(plot_values[i],
                *coeffs_fake)-coeffs_const_fake[0],
                color=function_line_color, linestyle='-', zorder=20)

    # Plot the quadratic fits.
    for i, ax in enumerate((temp_ax_quad_real, mtl_ax_quad_real,
                            logg_ax_quad_real)):
        ax.axhline(y=0, color='Gray', linestyle='-', zorder=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.errorbar(x=ma.compressed(x_data_quad_real[i]),
                    y=ma.compressed(residuals_quad_real),
                    yerr=ma.compressed(err_array_real),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='',
                    capsize=4, capthick=1.5,
                    zorder=10)
        ax.errorbar(x=ma.compressed(x_data_quad_real[i]),
                    y=ma.compressed(residuals_quad_real),
                    yerr=ma.compressed(full_err_array_real),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='o',
                    capsize=3, capthick=2,
                    markeredgecolor=marker_edge_color, zorder=15)
        ax.plot(ma.compressed(x_data_quad_real[i][indices_list]),
                ma.compressed(residuals_quad_real)[indices_list],
                color='Green',
                linestyle='', marker='o',
                markeredgecolor=marker_edge_color, zorder=16)

    for i, ax in enumerate((temp_ax_quad_fake, mtl_ax_quad_fake,
                            logg_ax_quad_fake)):
        ax.axhline(y=0, color='Gray', linestyle='-', zorder=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.errorbar(x=ma.compressed(x_data_quad_fake[i]),
                    y=ma.compressed(residuals_quad_fake),
                    yerr=ma.compressed(err_array_fake),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='',
                    capsize=4, capthick=1.5,
                    zorder=10)
        ax.errorbar(x=ma.compressed(x_data_quad_fake[i]),
                    y=ma.compressed(residuals_quad_fake),
                    yerr=ma.compressed(full_err_array_fake),
                    color=color, ecolor=elinecolor,
                    linestyle='', marker='o',
                    capsize=3, capthick=2,
                    markeredgecolor=marker_edge_color, zorder=15)
        ax.plot(ma.compressed(x_data_quad_fake[i][indices_list]),
                ma.compressed(residuals_quad_fake)[indices_list],
                color='Green',
                linestyle='', marker='o',
                markeredgecolor=marker_edge_color, zorder=16)

    if args.synthetic:
        filename = f'Pair_modeling_analysis_synthetic_{model_type}.png'
    else:
        filename = f'Pair_modeling_analysis_{model_type}.png'

    fig.savefig(filename)
