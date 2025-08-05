#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         lsf_fitting.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  LSF fitting functions for the AMASE DRP.
'''

import numpy as np
from scipy.optimize import curve_fit
from .wavelength_calibration import detect_lines


def gaussian(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset


def lsf_gaussian_fitting(
        spectrum, spectrum_wls, target_wl, cutout_wl_half_width=1.,
):
    """ Fit a Gaussian to the LSF of a spectrum around a target wavelength. """
    # adjust the wavelength of the target line
    peaks, _, _, _ = detect_lines(spectrum, n_strongest_lines=20)
    residuals = np.abs(spectrum_wls[peaks] - target_wl)
    adjusted_target_wl = spectrum_wls[peaks[np.argmin(residuals)]]
    if np.abs(adjusted_target_wl - target_wl) < cutout_wl_half_width:
        target_wl = adjusted_target_wl
    else:
        pass
    target_wl = spectrum_wls[peaks[np.argmin(residuals)]]
    # cut out the region around the target wavelength
    cutout_wl_window = np.array([-cutout_wl_half_width, cutout_wl_half_width])
    cutout_wl_window += target_wl
    cond = spectrum_wls >= np.min(cutout_wl_window)
    cond &= spectrum_wls <= np.max(cutout_wl_window)
    cutout_spectrum_wls = spectrum_wls[cond]
    cutout_spectrum = spectrum[cond]
    del cond
    # fit a Gaussian to the cutout spectrum
    try:
        # initial guess
        ini_mu = target_wl
        ini_sigma = cutout_wl_half_width / 2.
        ini_offset = np.nanmedian(cutout_spectrum) * 0.2
        ini_a = np.nanmax(cutout_spectrum - ini_offset)
        initial_guess = [ini_a, ini_mu, ini_sigma, ini_offset]
        # fitting
        x_fit = cutout_spectrum_wls
        y_fit = cutout_spectrum
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess)
        # FWHM of the fitted Gaussian
        target_popt = popt.copy()
        target_fwhm = 2.355 * np.abs(target_popt[2])
    except:  # noqa: E722
        target_popt = np.array([np.nan, np.nan, np.nan, np.nan])
        target_fwhm = np.nan
    return target_fwhm, target_popt


def lsf_fitting(spectrum, spectrum_wls, target_wl):
    target_fwhm, _ = lsf_gaussian_fitting(
        spectrum, spectrum_wls, target_wl,
        cutout_wl_half_width=1.,
    )
    return target_fwhm
