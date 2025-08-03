#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         wavelength_calibration.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Wavelength calibration functions for the AMASE DRP.
'''

import numpy as np
from itertools import combinations
from itertools import product
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from ...utils.parallel_processing import run as prun


def detect_lines(spectrum, n_strongest_lines=20, n_all_lines=100):
    """
    Detect all and strongest lines in the spectrum.
    """
    # smooth the spectrum & find peaks
    spectrum = gaussian_filter1d(spectrum, sigma=1)
    peaks, _ = find_peaks(spectrum, height=np.nanmedian(spectrum))
    ys = np.arange(len(spectrum))[peaks]
    heights = spectrum[peaks]
    # sort by height
    sorted_indices = np.argsort(heights)[::-1]
    ys = ys[sorted_indices]
    heights = heights[sorted_indices]
    del sorted_indices
    # only keep the 'n_all_lines' strongest lines
    n_all_lines = min(n_all_lines, len(ys))
    all_peak_ys = ys[:n_all_lines]
    all_peak_heights = heights[:n_all_lines]
    # only keep the 'n_strongest_lines' strongest lines
    n_strongest_lines = min(n_strongest_lines, len(ys))
    strong_peak_ys = ys[:n_strongest_lines]
    strong_peak_heights = heights[:n_strongest_lines]
    return strong_peak_ys, strong_peak_heights, all_peak_ys, all_peak_heights


def calculate_fitting_score(poss_poly, known_wls, all_peak_ys):
    """
    Suppose that the (strong-enough) known lines with "known_wls" should be
    included in the peaks detected with "all_peak_ys".
    This function is used to calculate a "score" for the fitting
    based on the residuals between the known wavelengths and the fitted
    wavelengths at the detected peak y coordinates.
    The smaller the score, the better the fitting.
    """
    all_peaks_detected_wls = poss_poly(all_peak_ys)
    residuals = np.abs(all_peaks_detected_wls[:, None] - known_wls[None, :])
    min_res = np.nanmin(residuals, axis=0)  # deviation: line & nearest peak
    score = np.sum(min_res ** 2)
    score /= len(known_wls)  # i.e., average error per line
    return score


def fitting(
        ys: list[float],
        wls: list[float],
        known_wls: list[float],
        all_peak_ys: list[float],
        deg: int = 3,
        poly_form=np.polynomial.Legendre,
):
    """ Fit the solution for given y coordinates and wavelengths, and
    use the known wavelengths and all detected peaks to calculate
    a "score" for the fitting. """
    try:
        coeffs = poly_form.fit(ys, wls, deg=deg).convert().coef
        poss_poly = poly_form(coeffs)
        # calculate a "score" for the fitting
        score = calculate_fitting_score(
            poss_poly, known_wls, all_peak_ys)
        output = np.append(coeffs, score)
    except:  # noqa: E722
        output = np.append(np.full(deg+1, np.nan, dtype=float), np.nan)
    return output


def wavelength_calibration(
        poss_wls: list[float],
        poss_ys: list[float],
        known_wls: list[float],
        all_peak_ys: list[float],
        min_deg: int = 3,
        poly_form=np.polynomial.Legendre,
        full_search: bool = True,
        parallel: bool = True,
        n_jobs: int = -1,
        backend: str = 'loky'
):
    """
    poss_wls: the most probable wavelengths to be observed
    (e.g., for the strongest lines)
    poss_ys: the possible y coordinates of the peaks for the strongest lines
    NOTE: deg + 1 <= len(poss_wls) <= len(poss_ys)
    NOTE: lines with poss_wls should be included in those with poss_ys, AMAP !!
    known_wls: the known wavelengths of some ("strong enough") lines
    all_peak_ys: the y coordinates of all detected peaks
    of the uncalibrated spectrum
    If full_search is True, then the return degree of the polynomial
    can be larger than "deg".
    """
    # sort
    poss_wls = np.sort(poss_wls)
    poss_ys = np.sort(poss_ys)
    known_wls = np.sort(known_wls)
    all_peak_ys = np.sort(all_peak_ys)
    deg = min_deg  # the minimum degree
    # full search: try to find the best solution. Could take a while.
    if full_search:
        inputs = []
        for n in range(deg+1, min(len(poss_ys), len(poss_wls))+1):
            # possible combination
            ys_poss_comb = np.array(list(combinations(poss_ys, n)))
            wls_poss_comb = np.array(list(combinations(poss_wls, n)))
            # all possible pairs of combinations
            inputs += list(product(
                ys_poss_comb, wls_poss_comb,
                [known_wls], [all_peak_ys],
                [int(n-1)], [poly_form]
            ))
    # the initial guess is good enough:
    # e.g., "poss_wls" and "poss_ys" exactly match each other,
    # corresponding to [deg+1] known lines.
    # e.g., at least [deg+1] lines with "poss_wls" are
    # included in those with "poss_ys"
    else:
        # possible combination
        ys_poss_comb = np.array(list(combinations(poss_ys, deg+1)))
        wls_poss_comb = np.array(list(combinations(poss_wls, deg+1)))
        # all possible pairs of combinations
        inputs = list(product(
            ys_poss_comb, wls_poss_comb,
            [known_wls], [all_peak_ys],
            [deg], [poly_form]
        ))
    # fit for all possible pairs of combinations
    outputs = prun(
        function=fitting,
        inputs=inputs,
        parallel=parallel, n_jobs=n_jobs, backend=backend,
    )
    # find the best coefficients
    scores = [outputs[i][-1] for i in range(len(outputs))]
    score = np.nanmin(scores)
    poss_coeffs = outputs[np.nanargmin(scores)][:-1]
    poss_poly = poly_form(poss_coeffs)
    return poss_poly, score
