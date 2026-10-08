#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         calibration.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for the calibration stage.

The wavelength solution is solved on one reference fiber and then propagated
to its neighbours.  Only the reference fiber needs a full search over every
line combination; the other fibers start from the solution of the fiber next
to them, which is far cheaper and keeps neighbouring solutions consistent.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..reduction.core.fiberframe import FiberFrame
from ..reduction.core.tracemask import TraceMask
from .core.linespreadfunction import LineSpreadFunction
from .core.wavelengthsolution import POLY_FAMILIES, WavelengthSolution
from .methods.lsf_fitting import lsf_fitting
from .methods.wavelength_calibration import detect_lines, wavelength_calibration

__all__ = [
    "apply_wavelength_solution",
    "fit_line_spread_function",
    "solve_wavelength_solution",
]


def solve_wavelength_solution(
    frame: FiberFrame,
    tracemask: TraceMask,
    known_wls: NDArray[np.floating],
    poss_wls: NDArray[np.floating],
    *,
    n_strongest_lines: int = 20,
    n_all_lines: int = 100,
    min_deg: int = 3,
    poly_kind: str = "legendre",
    full_search: bool = True,
    trace_length_fraction: float = 0.7,
    auto_refine: bool = True,
    min_calibrated_fraction: float | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
    backend: str = "loky",
) -> WavelengthSolution:
    """Solve the pixel-to-wavelength mapping of every fiber of an arc frame.

    The fiber with the longest trace is fitted with a full search over line
    combinations.  Its solution then propagates outward: every other fiber
    starts from the solution of the neighbor that was already solved.  A fiber
    whose trace is shorter than *trace_length_fraction* of the reference
    trace, or whose fit fails, inherits the neighbor solution and is marked
    with a score of ``-1``.

    Parameters
    ----------
    frame
        Extracted spectra of an arc-lamp exposure, with a fiber map attached.
        Rows flagged in ``frame.mask`` are treated as missing, not as zero
        flux.
    tracemask
        Trace model of the same exposure.
    known_wls
        Wavelengths of the reference lines used for scoring.
    poss_wls
        Wavelengths of the strongest lines, used to enumerate candidates.
    n_strongest_lines
        Number of strongest detected peaks offered as candidate line
        positions.
    n_all_lines
        Number of detected peaks used to score a candidate solution.
    min_deg
        Minimum polynomial degree.  With *full_search* the degree may grow.
    poly_kind
        Polynomial family, ``"legendre"`` or ``"polynomial"``.
    full_search
        If ``True``, search every line combination for the reference fiber.
        If ``False``, only use ``min_deg + 1`` strongest lines.
    trace_length_fraction
        A fiber shorter than this fraction of the reference trace inherits the
        neighbor solution instead of being fitted.
    auto_refine
        Iteratively refine each solution against the detected peaks.
    min_calibrated_fraction
        Smallest fraction of fibers that must end up with a solution.  ``None``
        (default) accepts any fraction, which is what a focus sweep needs: at
        strong defocus the arc lines broaden and blend, and most fibers legitimately
        fail.  An entry point that serves real observations must pass a value
        close to one, because a shortfall there points at a software or hardware
        fault; see ``AGENTS.md``.
    parallel
        Run the candidate combinations in parallel.
    n_jobs
        Number of parallel jobs; ``-1`` uses every core.
    backend
        joblib backend.

    Returns
    -------
    WavelengthSolution
        One polynomial per fiber.  ``scores`` holds the fitting score, and
        ``-1`` marks a fiber that inherited its neighbor solution.

    Raises
    ------
    ValueError
        If *frame* has no fiber map, if the fiber counts disagree, if
        *poly_kind* is unknown, if *known_wls* or *poss_wls* is empty, or if
        *min_calibrated_fraction* is outside ``[0, 1]``.
    RuntimeError
        If the reference fiber itself cannot be calibrated, in which case
        there is nothing to propagate, or if the calibrated fraction falls
        below *min_calibrated_fraction*.
    """
    if frame.fibermap is None:
        raise ValueError("frame must carry a fiber map to calibrate it.")
    if tracemask.n_fibers != frame.n_fibers:
        raise ValueError(
            f"tracemask has {tracemask.n_fibers} fibers, but frame has "
            f"{frame.n_fibers}."
        )
    if poly_kind not in POLY_FAMILIES:
        raise ValueError(
            f"poly_kind must be one of {sorted(POLY_FAMILIES)}, "
            f"got {poly_kind!r}"
        )
    if len(known_wls) == 0 or len(poss_wls) == 0:
        raise ValueError("known_wls and poss_wls must not be empty.")
    if min_calibrated_fraction is not None and not 0.0 <= min_calibrated_fraction <= 1.0:
        raise ValueError(
            f"min_calibrated_fraction must lie in [0, 1], got "
            f"{min_calibrated_fraction}"
        )

    poly_form = POLY_FAMILIES[poly_kind]
    fiber_ids = np.asarray(frame.fibermap["FIBERID"], dtype=int)
    n_fibers = frame.n_fibers

    spans = tracemask.domain[:, 1] - tracemask.domain[:, 0]
    reference = int(np.argmax(spans))

    def fit(index: int, guess: Any | None) -> tuple[Any, float]:
        """Fit the wavelength solution of one fiber.

        Rows flagged in the mask hold no flux, so they are turned into ``nan``
        to keep them out of the peak detection, as an untraced row would be.
        """
        spectrum = frame.flux[index].astype(float).copy()
        spectrum[frame.mask[index] != 0] = np.nan

        strong_ys, _, all_ys, _ = detect_lines(
            spectrum,
            n_strongest_lines=n_strongest_lines,
            n_all_lines=n_all_lines,
        )
        return wavelength_calibration(
            poss_wls=poss_wls,
            poss_ys=strong_ys,
            known_wls=known_wls,
            all_peak_ys=all_ys,
            min_deg=min_deg,
            poly_form=poly_form,
            full_search=full_search,
            parallel=parallel,
            n_jobs=n_jobs,
            backend=backend,
            guess_poss_poly=guess,
            auto_refine=auto_refine,
        )

    # Reference fiber: the only one that gets the full search.
    ref_poly, ref_score = fit(reference, guess=None)
    if not np.isfinite(ref_score) or ref_score < 0:
        raise RuntimeError(
            f"the reference fiber {fiber_ids[reference]} (index {reference}) "
            f"could not be calibrated (score {ref_score}); there is nothing "
            f"to propagate."
        )

    coeffs = np.zeros((n_fibers, ref_poly.degree() + 1), dtype=float)
    scores = np.full(n_fibers, -1.0)
    coeffs[reference, :] = ref_poly.coef
    scores[reference] = ref_score

    trace_length_threshold = trace_length_fraction * spans[reference]

    def propagate(index: int, neighbor: int) -> None:
        """Solve one fiber, starting from its already-solved neighbor."""
        if spans[index] < trace_length_threshold:
            coeffs[index, :] = coeffs[neighbor, :]
            return

        poly, score = fit(index, guess=poly_form(coeffs[neighbor, :]))
        if not np.isfinite(score) or score < 0:
            warnings.warn(
                f"fiber {fiber_ids[index]} (index {index}) could not be "
                f"calibrated (score {score}); it inherits the solution of "
                f"fiber {fiber_ids[neighbor]}.",
                RuntimeWarning,
                stacklevel=2,
            )
            coeffs[index, :] = coeffs[neighbor, :]
            return

        coeffs[index, :] = poly.coef
        scores[index] = score

    # Walk away from the reference fiber, in both directions.
    for index in range(reference - 1, -1, -1):
        propagate(index, neighbor=index + 1)
    for index in range(reference + 1, n_fibers):
        propagate(index, neighbor=index - 1)

    solution = WavelengthSolution(
        coeffs=coeffs,
        fiber_ids=fiber_ids,
        poly_kind=poly_kind,
        scores=scores,
    )

    if min_calibrated_fraction is not None:
        fraction = solution.calibrated_fraction
        if fraction < min_calibrated_fraction:
            raise RuntimeError(
                f"only {fraction:.1%} of the {solution.n_fibers} fibers could be "
                f"calibrated, below the required "
                f"{min_calibrated_fraction:.1%}. A focus sweep can fall this "
                f"low at strong defocus; a real observation cannot, so this "
                f"points at a software or hardware fault."
            )

    return solution


def apply_wavelength_solution(
    frame: FiberFrame,
    solution: WavelengthSolution,
) -> FiberFrame:
    """Attach a wavelength solution to extracted spectra.

    Parameters
    ----------
    frame
        Extracted spectra, typically of a science or flat exposure.
    solution
        Solution solved on an arc exposure taken with the same fiber setup.

    Returns
    -------
    FiberFrame
        A new frame whose ``wave`` is ``(n_fibers, n_wave)``.  When the input
        frame carries a fiber map, a ``WAVCAL_SCORE`` column is added to a copy
        of it.

    Raises
    ------
    ValueError
        If the solution holds a different number of fibers than the frame.
    """
    if solution.n_fibers != frame.n_fibers:
        raise ValueError(
            f"solution holds {solution.n_fibers} fibers, but frame has "
            f"{frame.n_fibers}."
        )

    rows = np.arange(frame.n_wave)
    wave = solution.eval(rows)

    fibermap = frame.fibermap
    if fibermap is not None:
        fibermap = fibermap.copy()
        fibermap["WAVCAL_SCORE"] = solution.scores

    return FiberFrame(
        wave=wave,
        flux=frame.flux,
        ivar=frame.ivar,
        mask=frame.mask,
        fibermap=fibermap,
        meta=frame.meta,
    )


def fit_line_spread_function(
    frame: FiberFrame,
    target_wls: NDArray[np.floating],
) -> LineSpreadFunction:
    """Measure the line-spread function width of arc lines.

    Every fiber is measured at every target wavelength with
    :func:`~amasedrp.calibration.methods.lsf_fitting.lsf_fitting`, which cuts a
    narrow window around the line, centres the cut-out on the nearest detected
    peak, and fits a Gaussian.

    Parameters
    ----------
    frame
        Extracted spectra with a wavelength solution applied, so that ``wave``
        is per fiber.  Rows flagged in ``mask`` are treated as missing rather
        than as zero flux.
    target_wls
        Wavelengths of the measured lines, shape ``(n_wls,)``.  Use lines that
        are isolated: a pair of strong lines inside the fitting window is
        rejected, and returns ``nan``.

    Returns
    -------
    LineSpreadFunction
        Widths of shape ``(n_wls, n_fibers)``.  ``nan`` marks a measurement
        that could not be fitted.

    Raises
    ------
    ValueError
        If *target_wls* is not a non-empty 1-D array, or if *frame* has no
        per-fiber wavelength solution.
    """
    target_wls = np.asarray(target_wls, dtype=float)
    if target_wls.ndim != 1:
        raise ValueError(
            f"target_wls must be 1-D, got shape {target_wls.shape}"
        )
    if target_wls.size == 0:
        raise ValueError("target_wls must not be empty.")
    if frame.wave.ndim != 2 or frame.wave.shape != frame.flux.shape:
        raise ValueError(
            "frame must carry a per-fiber wavelength solution; apply one with "
            "apply_wavelength_solution() first."
        )

    n_fibers = frame.n_fibers
    fwhm = np.full((target_wls.size, n_fibers), np.nan)

    for index in range(n_fibers):
        # Rows flagged in the mask hold no flux; as ``nan`` they stay out of the
        # peak detection, exactly as an untraced row would.
        spectrum = frame.flux[index].astype(float).copy()
        spectrum[frame.mask[index] != 0] = np.nan
        spectrum_wls = frame.wave[index]

        for j, target_wl in enumerate(target_wls):
            fwhm[j, index] = lsf_fitting(spectrum, spectrum_wls, target_wl)

    if frame.fibermap is not None:
        fiber_ids = np.asarray(frame.fibermap["FIBERID"], dtype=int)
    else:
        fiber_ids = np.arange(n_fibers, dtype=int)

    n_failed = int(np.count_nonzero(~np.isfinite(fwhm)))
    if n_failed:
        warnings.warn(
            f"{n_failed} of {fwhm.size} line-spread fits did not converge; "
            f"their width is nan.",
            RuntimeWarning,
            stacklevel=2,
        )

    return LineSpreadFunction(
        target_wls=target_wls,
        fwhm=fwhm,
        fiber_ids=fiber_ids,
    )
