import argparse
from pprint import pprint
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import unyt as u


parser = argparse.ArgumentParser()

parser.add_argument('file_name_no_nans', action='store',
                    help='The full path to the data file without NaNs to use.')
parser.add_argument('file_name_with_nans', action='store',
                    help='The full path to the data file with NaNs to use.')


args = parser.parse_args()


def find_sigma_sys(model_func, x_data, y_data, err_array, beta0):
    """Find the systematic scatter in a dataset with a given model.

    Takes a model function `model_func`, and arrays of x, y, and uncertainties
    (which must have the same length) and an initial guess to the parameters of
    the function, and fits the model to the data. It then checks the reduced
    chi-squared value, and if it is greater than 1 (with a tolerance of 1e-3),
    it adds an additional amount in quadrature to the error array and refits the
    data, continuing until the chi-squared value is within the tolerance.

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

    Returns
    -------
    tuple
        A tuple of containing a tuple of the optimized parameters of the same
        length as the number of parameters of `model_func` minus one, an
        `np.array` containing the covariance matrix, a `unyt.unyt_array`
        containing the residuals from the final fit, a `unyt.unyt_quantity`
        holding the systematic error found, and the final value of the reduced
        chi-squared value.

    """

    # Iterate to find what additional systematic error is needed
    # to get a chi^2 of ~1.
    chi_tol = 0.001
    diff = 1
    sys_err = 0
    num_iters = 0
    sigma_sys_change_amount = 0.25  # Range (0, 1)

    chi2 = []
    tsys = []

    while diff > chi_tol:

        num_iters += 1

        iter_err_array = np.sqrt(np.square(err_array) +
                                 np.square(sys_err))

        popt, pcov = curve_fit(model_func, x_data, y_data,
                               sigma=iter_err_array,
                               p0=beta0,
                               absolute_sigma=True,
                               method='lm', maxfev=10000)

        results = model_func(x_data, *popt)

        residuals = y_data - results

        # print(y_data)
        # print(residuals)
        # print(iter_err_array)

        # Find the chi^2 value for this distribution:
        chi_squared = np.sum((residuals / iter_err_array) ** 2)
        # print(chi_squared)
        dof = len(y_data) - len(popt)
        # print(f'DOF = {dof}')
        chi_squared_nu = chi_squared / dof
        # print(f'{sys_err} {chi_squared_nu}')

        tsys.append(sys_err)
        chi2.append(chi_squared_nu)

        diff = abs(chi_squared_nu - 1)
        if diff > 2:
            sigma_sys_change_amount = 0.75
        elif diff > 1:
            sigma_sys_change_amount = 0.5
        else:
            sigma_sys_change_amount = 0.25

        if chi_squared_nu > 1:
            if sys_err == 0:
                sys_err = np.sqrt(chi_squared_nu)
            else:
                sys_err *= (1 + sigma_sys_change_amount)
        elif chi_squared_nu < 1:
            if sys_err == 0:
                # If the chi-squared value is naturally lower
                # than 1, don't change anything, just exit.
                break
            else:
                sys_err *= (1 - sigma_sys_change_amount)

    return (popt, pcov, residuals, sys_err, chi_squared_nu, chi2, tsys)


def parabola(x, a, b, c):
    return a + b * x + c * x ** 2

def linear_model(data, a, b, c, d):

    return a + b * data[0] + c * data[1] + d * data[2]


# import data without NaNs from file
real_data = np.loadtxt(args.file_name_no_nans,
                       usecols=(1, 4, 5, 6, 7), dtype=float)
offsets = real_data[:, 0]
errors = real_data[:, 1]
temperatures = real_data[:, 2]
metallicities = real_data[:, 3]
magnitudes = real_data[:, 4]

non_nan_mean = np.mean(offsets)

# import data with NaNs from file
real_nan_data = np.loadtxt(args.file_name_with_nans,
                           usecols=(1, 4, 5, 6, 7), dtype=float)

nan_mean = np.nanmean(real_nan_data[:, 0])
print(f'mean = {non_nan_mean}, NaN mean = {nan_mean}')

masked_offsets = ma.masked_invalid(real_nan_data[:, 0])
# pprint(masked_offsets)
m_offsets = real_nan_data[:, 0][~masked_offsets.mask]
# pprint(m_offsets)
u_offsets = u.unyt_array(masked_offsets[~masked_offsets.mask],
                         units=u.m/u.s)
# pprint(u_offsets)

m_errors = real_nan_data[:, 1][~masked_offsets.mask]

m_temperatures = real_nan_data[:, 2][~masked_offsets.mask]

m_metallicities = real_nan_data[:, 3][~masked_offsets.mask]

m_magnitudes = real_nan_data[:, 4][~masked_offsets.mask]
# for array in (m_offsets, m_errors, m_temperatures, m_metallicities,
#               m_magnitudes):
#     print(array.shape)
# pprint(m_offsets)


x_stacked = np.stack((temperatures, metallicities, magnitudes), axis=0)
x_nan_stacked = np.stack((m_temperatures, m_metallicities, m_magnitudes),
                         axis=0)

# find Tsys iteratively
beta0 = (non_nan_mean, 0, 0, 0)
results = find_sigma_sys(linear_model, x_stacked, offsets, errors, beta0)
popt, pcov, residuals, Tsys, chi_squared_nu, chi2, tsys = results
print('--------')
print(popt)
print(f'T_sys = {tsys[-1]}')
print('--------')

n_beta0 = (nan_mean, 0, 0, 0)
n_results = find_sigma_sys(linear_model, x_nan_stacked, m_offsets,
                           m_errors, n_beta0)
n_popt, n_pcov, n_residuals, n_Tsys, n_chi_squared_nu, n_chi2, n_tsys =\
    n_results
for a, b in zip(n_chi2, n_tsys):
    print(a, b)

p, pc = curve_fit(linear_model, x_stacked, offsets,
                  sigma=errors,
                  p0=beta0,
                  absolute_sigma=True,
                  method='lm', maxfev=10000)

print('--------')
print(p)
print('--------')


plt.figure(figsize=(12, 8), tight_layout=True)

plt.subplot(331)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

# now we'll plot the model and the data points with *obervational* error bars on
# each
# plt.plot(xx1, y, 'k-', label="underlying model")

plt.errorbar(m_temperatures, m_offsets, yerr=m_errors, fmt='x',
             color='Green', label="NaN temperatures")
plt.errorbar(temperatures, offsets, yerr=errors, fmt='.',
             label="temperatures")



# also plot the fit found by the iterative fitting process
# y_fit = linear_model(np.stack((xx1, xx2, xx3), axis=0), *popt)
# plt.plot(xx1, y_fit, color='Red', label='iterative fit',
#          linestyle='-')

# plot the fitting function
# plt.plot(xx1, yy, 'b--', label="polynomial fit")
plt.xlabel(r"T_eff")
plt.ylabel(r"Offsets (m/s)")
plt.legend()

yscale = 1.5

plt.subplot(332)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

# plt.plot(xx2, y, 'k-', label="underlying model")
plt.errorbar(m_metallicities, m_offsets, yerr=m_errors, fmt='x',
             color='Green', label="NaN metallicities")
plt.errorbar(metallicities, offsets, yerr=errors, fmt='.',
             label="metallicities")


# also plot the fit found by the iterative fitting process
# plt.plot(xx2, y_fit, color='Red', label='iterative fit',
#          linestyle='-')

# plot the fitting function
# plt.plot(xx2, yy, 'b--', label="polynomial fit")
plt.xlabel("[Fe/H]")
plt.ylabel(r"Offsets (m/s)")
plt.legend()

yscale = 1.5

plt.subplot(333)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

# plt.plot(xx3, y, 'k-', label="underlying model")
plt.errorbar(m_magnitudes, m_offsets, yerr=m_errors, fmt='x',
             color='Green', label="NaN magnitudes")
plt.errorbar(magnitudes, offsets, yerr=errors, fmt='.',
             label="magnitudes")


# also plot the fit found by the iterative fitting process
# plt.plot(xx3, y_fit, color='Red', label='iterative fit',
#          linestyle='-')

# plot the fitting function
# plt.plot(xx3, yy, 'b--', label="polynomial fit")
plt.xlabel(r"M_V")
plt.ylabel(r"Offsets (m/s)")
plt.legend()

yscale = 1.5


plt.subplot(334)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

plt.axhline(0, color='Black')
# yfit = yy
# residuals = ye - yfit
plt.errorbar(m_temperatures, n_residuals, yerr=m_errors, fmt='x',
             color='Green', label="NaN residuals temperature")
plt.errorbar(temperatures, residuals, yerr=errors, fmt='.',
             label="residuals temperature")


plt.ylabel("residuals (data-model)")
plt.xlabel(r"T_eff")
plt.ylim(min(residuals)*yscale, max(residuals)*yscale)

# mask = np.abs(residuals) > 2.5 * yerrs
# print(mask)


# plt.errorbar(x[mask], residuals[mask], yerr=yerr[mask], fmt='.',
#              color='red',
#              label=r"bad points $(> 2.5 \sigma)$")

plt.legend()

plt.subplot(335)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

plt.axhline(0, color='Black')
plt.errorbar(m_metallicities, n_residuals, yerr=m_errors, fmt='x',
             color='Green', label="NaN residuals metallicity")
plt.errorbar(metallicities, residuals, yerr=errors, fmt='.',
             label="residuals metallicity")


plt.ylabel("residuals (data-model)")
plt.xlabel("[Fe/H]")
plt.ylim(min(residuals)*yscale, max(residuals)*yscale)

plt.legend()

plt.subplot(336)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

plt.axhline(0, color='Black')
plt.errorbar(m_magnitudes, n_residuals, yerr=m_errors, fmt='x',
             color='Green', label="NaN residuals magnitude")
plt.errorbar(magnitudes, residuals, yerr=errors, fmt='.',
             label="residuals magnitude")


plt.ylabel("residuals (data-model)")
plt.xlabel(r"M_V")
plt.ylim(min(residuals)*yscale, max(residuals)*yscale)

plt.legend()

# show histograms of the residuals to the model
# compare to the sources of scatter (intrinsic, observational)

plt.subplot(337)
scatter = np.around(np.std(residuals), 2)

residual_histogram = plt.hist(residuals,
                              orientation='horizontal',
                              label="Sample dispersion: "
                              + str(scatter),
                              histtype='step')

# extract the histogram values and the positions of the bins
res_hist = residual_histogram[0]
res_bins = residual_histogram[1]

# plot the residuals
plt.ylim(min(residuals)*yscale, max(residuals)*yscale)
plt.xlabel("N")
plt.ylabel("residuals (data-model)")

# set up a linear space on the y-axis for a fine Gaussian plot
# for each of the scatter terms
yy = np.linspace(min(residuals)*yscale, max(residuals)*yscale, 100)

# normalise each Gaussian to the maximum bin (a bit arbitrary, but helpful)
fac = np.max(res_hist)

# plot the three Gaussians

# xx = np.exp(-(yy**2)/(2*np.mean(y_err_observational)**2))*fac
# plt.plot(xx, yy, 'b-', label="Observational error : " +
         # str(np.mean(y_err_observational)))

# xx = np.exp(-(yy**2)/(2*y_intrinsic**2))*fac
# plt.plot(xx, yy, 'r-', label="Intrinsic scatter : "+str(y_intrinsic))

# both_sigma = np.sqrt(np.mean(y_err_observational)**2+y_intrinsic**2)
# xx = np.exp(-(yy**2)/(2*both_sigma**2))*fac
# plt.plot(xx, yy, 'g--', label="Sample scatter : "+str(both_sigma))

plt.legend()

plt.subplot(338)
plt.grid(which='major', axis='both', linestyle='--',
         color='Gray', alpha=0.7)

# generate a range of Tsys values to sample from 0 to twice the measured scatter
tsys_min = scatter*0
tsys_max = scatter*2

# number of points to sample chi-squared-nu at
N_csn = 1000
test_tsys = np.linspace(tsys_min, tsys_max, N_csn)
csn = np.zeros(len(test_tsys))

# calculate the reduced chi-squared at each value of Tsys
dof = len(offsets) - 4  # !!!
for i in range(len(test_tsys)):
    # add them in quadrature
    t_tot = np.sqrt(errors**2 + test_tsys[i]**2)
    chi_squared = np.sum((residuals/t_tot)**2)
    chi_squared_nu = chi_squared / dof
    csn[i] = chi_squared_nu

    # print(test_tsys[i], chi_squared_nu)

# now plot the values of Tsys vs. reduced chi-squared
plt.plot(test_tsys, csn, 'g-',
         label='reduced chi-squared')
plt.plot(tsys, chi2, color='Red', marker='+',
         linestyle='', alpha=0.9,
         label='iterative chi-squared\n'
         f'sigma_sys: {tsys[-1]:.2f}')
plt.plot(n_tsys, n_chi2, color='Blue', marker='x',
         linestyle='', alpha=0.9,
         label='iterative chi-squared (NaNs)\n'
         f'sigma_sys: {n_tsys[-1]:.2f}')
plt.ylim(0, 3)
plt.xlim(left=scatter*0.05, right=scatter*1.2)
plt.xlabel("Tsys")
plt.ylabel(r"$\chi^2_\nu$")
plt.axhline(1.0, color='Black')
plt.axvline(scatter, color='Blue')

plt.legend()

plt.show()
