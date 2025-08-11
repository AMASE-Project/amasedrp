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


def adjust_target_wavelength(spectrum, spectrum_wls, target_wl):
    """
    Adjust the target wavelength based on the detected peaks in the spectrum.
    (i.e., find the closest peak to the target wavelength)
    """
    peaks, _, _, _ = detect_lines(spectrum, n_strongest_lines=20)
    # if no peaks are detected, return NaN
    if len(peaks) == 0:
        return np.nan
    residuals = np.abs(spectrum_wls[peaks] - target_wl)
    # if residuals are empty, return NaN
    if len(residuals) == 0:
        return np.nan
    adjusted_target_wl = spectrum_wls[peaks[np.argmin(residuals)]]
    return adjusted_target_wl


def extract_spectrum_segment(
        spectrum, spectrum_wls, target_wl, cutout_wl_half_width=1.,
):
    """ Cut out a region of the spectrum around a target wavelength. """
    cutout_wl_window = np.array([-cutout_wl_half_width, cutout_wl_half_width])
    cutout_wl_window += target_wl
    cond = spectrum_wls >= np.min(cutout_wl_window)
    cond &= spectrum_wls <= np.max(cutout_wl_window)
    cutout_spectrum, cutout_spectrum_wls = spectrum[cond], spectrum_wls[cond]
    return cutout_spectrum, cutout_spectrum_wls


def gaussian(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset


def gaussian_fitting(cutout_spectrum, cutout_spectrum_wls, target_wl):
    # fit a Gaussian to the cutout spectrum
    try:
        # initial guess
        ini_mu = target_wl
        ini_sigma = np.ptp(cutout_spectrum_wls) / 5.
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


def lsf_gaussian_fitting(
        spectrum, spectrum_wls, target_wl,
        cutout_wl_half_width=1.,
        adjust_target_wl=False,
        ignore_double_peaks=True,
):
    """ Fit a Gaussian to the LSF of a spectrum around a target wavelength. """
    # default return
    target_fwhm = np.nan
    target_popt = np.array([np.nan, np.nan, np.nan, np.nan])
    target_func = lambda x: np.full(len(x), np.nan, dtype=float)  # noqa: E731
    cutout_spectrum_wls = np.array([])
    cutout_spectrum = np.array([])

    # adjust the wavelength of the target line
    if adjust_target_wl:
        target_wl = adjust_target_wavelength(spectrum, spectrum_wls, target_wl)

    # if the target wavelength is NaN, return NaN values
    if np.isnan(target_wl):
        return (
            target_fwhm,
            target_popt, target_func,
            cutout_spectrum_wls, cutout_spectrum,
        )

    # cut out the region around the target wavelength
    cutout_spectrum, cutout_spectrum_wls = extract_spectrum_segment(
        spectrum, spectrum_wls, target_wl, cutout_wl_half_width,
    )

    # if the cutout spectrum is all NaN, return NaN values
    if np.all(np.isnan(cutout_spectrum)):
        return (
            target_fwhm,
            target_popt, target_func,
            cutout_spectrum_wls, cutout_spectrum,
        )

    # if the cutout spectrum is empty, return NaN values
    if len(cutout_spectrum) == 0:
        return (
            target_fwhm,
            target_popt, target_func,
            cutout_spectrum_wls, cutout_spectrum,
        )

    # if there are two or more strong lines in the cutout spectrum,
    # then the corresponding PSF may be too complex to fit a Gaussian,
    # so we return NaN values
    if ignore_double_peaks:
        peaks, _, _, _ = detect_lines(cutout_spectrum, n_strongest_lines=2)
        if len(peaks) >= 2:
            if (
                np.min(cutout_spectrum[peaks]) / np.max(cutout_spectrum[peaks])
                > 0.3
            ):
                return (
                    target_fwhm,
                    target_popt, target_func,
                    cutout_spectrum_wls, cutout_spectrum,
                )

    # fit a Gaussian to the cutout spectrum
    target_fwhm, target_popt = gaussian_fitting(
        cutout_spectrum, cutout_spectrum_wls, target_wl,
    )

    # fitting function
    def target_func(x):
        return gaussian(x, *target_popt)

    return (
        target_fwhm,
        target_popt, target_func,
        cutout_spectrum_wls, cutout_spectrum,
    )


def lsf_fitting(spectrum, spectrum_wls, target_wl):
    target_fwhm, _, _, _, _ = lsf_gaussian_fitting(
        spectrum, spectrum_wls, target_wl,
        cutout_wl_half_width=1.,
    )
    return target_fwhm
