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
from scipy.optimize import minimize
from ...utils.parallel_processing import run as prun


def detect_lines(spectrum, n_strongest_lines=20, n_all_lines=100):
    """
    Detect all and strongest lines in the spectrum.
    """
    # smooth the spectrum & find peaks
    smooth_spectrum = gaussian_filter1d(spectrum, sigma=1)
    # if spectrum is all NaN, return empty arrays
    if np.all(np.isnan(smooth_spectrum)):
        return np.array([]), np.array([]), np.array([]), np.array([])
    # find peaks in the spectrum
    peaks, _ = find_peaks(smooth_spectrum, height=np.nanmedian(spectrum))
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
    included in the peaks detected with "all_peak_ys", as much as possible.
    This function is used to calculate a "score" for the fitting
    based on the residuals between the known wavelengths and the fitted
    wavelengths at the detected peak y coordinates.
    The smaller the score, the better the fitting.
    """
    all_peak_wls = poss_poly(all_peak_ys)
    residuals = np.abs(all_peak_wls[:, None] - known_wls[None, :])
    # residuals between known lines and their nearest peaks
    min_res = np.nanmin(residuals, axis=0)  # [wl unit]
    # remove outliers
    cond = np.nanpercentile(min_res, 16) <= min_res
    cond &= min_res <= np.nanpercentile(min_res, 84)
    min_res = min_res[cond]
    del cond
    # calculate the score (kind of "RMSE")
    score = np.sqrt(np.sum(min_res ** 2)) / len(min_res)  # [wl unit]
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
    except:  # noqa: E722
        coeffs = np.full(deg+1, 0., dtype=float)
        coeffs[0] = -1.
        score = -1.
    output = np.append(coeffs, score)
    return output


def find_poss_wavelength_solution(
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
    known_wls: the known wavelengths of some ("strong enough") lines
    all_peak_ys: the y coordinates of all detected peaks
    of the uncalibrated spectrum
    If full_search is True, then the return degree of the polynomial
    can be larger than "deg".
    # NOTE: deg + 1 <= len(poss_wls) <= len(poss_ys)
    # NOTE:
    # (1) known_wls should be included in all_peak_ys
    # (2) poss_wls should be included in poss_ys
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
    if len(inputs):
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
    else:
        coeffs = np.full(deg+1, 0., dtype=float)
        coeffs[0] = -1.
        score = -1.
        poss_poly = poly_form(coeffs)
    return poss_poly, score


def match_lines_refine_poss_solution(
        guess_poss_poly,
        known_wls, all_peak_ys,
):
    """
    Refine the possible solution with the known wavelengths and
    all detected peaks.
    Match the known wavelengths to the detected peaks
    and calculate the for the fitting.
    """
    matched_ys = np.full(len(known_wls), np.nan, dtype=float)
    matched_residuals = np.full(len(known_wls), np.nan, dtype=float)
    all_peak_wls = guess_poss_poly(all_peak_ys)
    # match the known wavelengths to the detected peaks
    for i in range(len(known_wls)):
        # find the nearest peak
        j = np.argmin(np.abs(all_peak_wls - known_wls[i]))
        # update the matched y coordinate
        matched_ys[i] = all_peak_ys[j]
        matched_residuals[i] = np.abs(all_peak_wls[j] - known_wls[i])
    print(f"Matched residuals: {matched_residuals}")
    # re-fit the solution with the matched y coordinates
    output = fitting(
        ys=matched_ys, wls=known_wls,
        known_wls=known_wls, all_peak_ys=all_peak_ys,
        deg=guess_poss_poly.degree(), poly_form=type(guess_poss_poly)
    )
    score = output[-1]
    coeffs = output[:-1]
    poss_poly = type(guess_poss_poly)(coeffs)
    return poss_poly, score


def lstsq_refine_poss_solution(
        guess_poss_poly,
        known_wls, all_peak_ys,
):
    """
    Refine the possible solution with the known wavelengths and
    all detected peaks using least squares fitting.
    (i.e., poss_poly = a * guess_poss_poly + b
    where a and b are the coefficients to be determined.)
    """
    def objective(params):
        a, b = params

        # define the new polynomial as a * guess_poss_poly + b
        def poss_poly(x):
            return a * guess_poss_poly(x) + b

        # calculate the fitting score
        return calculate_fitting_score(poss_poly, known_wls, all_peak_ys)

    # initial guess for a and b
    initial_guess = [1.0, 0.0]

    # minimize the objective function
    result = minimize(objective, initial_guess)
    # extract the optimized parameters
    a_opt, b_opt = result.x

    # optimized result
    poss_poly = a_opt * guess_poss_poly + b_opt
    score = calculate_fitting_score(poss_poly, known_wls, all_peak_ys)
    return poss_poly, score


def refine_poss_solution(
        guess_poss_poly,
        known_wls, all_peak_ys,
):
    # match lines method may not work well
    return lstsq_refine_poss_solution(
        guess_poss_poly=guess_poss_poly,
        known_wls=known_wls,
        all_peak_ys=all_peak_ys
    )


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
        backend: str = 'loky',
        guess_poss_poly=None,
        auto_refine: bool = True,
):
    if guess_poss_poly is None:
        poss_poly, score = find_poss_wavelength_solution(
            poss_wls, poss_ys, known_wls, all_peak_ys,
            min_deg=min_deg, poly_form=poly_form,
            full_search=full_search, parallel=parallel,
            n_jobs=n_jobs, backend=backend,
        )
    else:
        poss_poly, score = refine_poss_solution(
            guess_poss_poly, known_wls, all_peak_ys,
        )
    if auto_refine:
        while True:
            # use poss_ys and poss_wls to refine the solution
            old_score = float(score)
            poss_poly, score = refine_poss_solution(
                poss_poly, poss_wls, poss_ys,
            )
            if np.isclose(old_score, score, atol=1e-8):
                break
        score = calculate_fitting_score(poss_poly, known_wls, all_peak_ys)
    return poss_poly, score


def inv_poss_poly(poss_poly, wl, y_min=0., y_max=9600., atol=1e-5):
    """ Inverse the polynomial to get y from wl. """
    y_mid = (y_min + y_max) / 2.
    while np.abs(y_max - y_min) > atol:
        wl_mid = poss_poly(y_mid)
        if wl_mid < wl:
            y_min = y_mid
        else:
            y_max = y_mid
        y_mid = (y_min + y_max) / 2.
    return y_mid
