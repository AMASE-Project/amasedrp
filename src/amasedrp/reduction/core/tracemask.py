#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         tracemask.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Trace fiber positions on the CCD and model them with
              polynomials.

A TraceMask stores the *model* of where each fiber lies on the detector.
The model is a set of polynomial coefficients per fiber, which can be
 cheaply evaluated at any row to give sub-pixel trace positions.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numba import jit
from numpy.polynomial.legendre import Legendre
from numpy.typing import NDArray

from .fibermap import FiberMap

__all__ = ["TraceMask"]


class TraceMask:
    """Polynomial model of fiber traces on a 2-D image.

    Rather than storing a dense 2-D array of trace positions, TraceMask
    stores Legendre polynomial coefficients per fiber.  This is memory
    efficient and gives sub-pixel precision when evaluated.

    Parameters
    ----------
    coeffs
        Array of shape ``(n_fibers, n_coeffs)`` containing polynomial
        coefficients.
    fiber_ids
        1-D array of fiber IDs, same length as the first axis of *coeffs*.
    poly_kind
        Kind of polynomial used (default ``"legendre"``).
    domain
        Row index range ``(row_min, row_max)`` over which the polynomials
        are defined.

    Examples
    --------
    >>> tracemask = TraceMask.from_fibermap(fibermap, image, poly_deg=10)
    >>> rows = np.arange(image.shape[0])
    >>> positions = tracemask.eval(rows)   # shape (n_fibers, n_rows)
    """

    def __init__(
        self,
        coeffs: NDArray[np.floating],
        fiber_ids: NDArray[np.integer],
        poly_kind: str = "legendre",
        domain: tuple[int, int] | None = None,
    ) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.fiber_ids = np.asarray(fiber_ids, dtype=int)
        self.poly_kind = poly_kind
        self.domain = domain

        if self.coeffs.shape[0] != len(self.fiber_ids):
            raise ValueError(
                "coeffs.shape[0] must match len(fiber_ids)."
            )

    # ------------------------------------------------------------------ #
    #  Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_fibermap(
        cls,
        fibermap: FiberMap,
        image: NDArray[np.floating],
        poly_deg: int = 10,
        max_shift: float = 1.0,
        cdisp_half_width: int = 3,
        threshold_fraction: float = 0.1,
    ) -> "TraceMask":
        """Trace fibers and fit a polynomial to each trace.

        Parameters
        ----------
        fibermap
            Output of :class:`FibersIdentifier` with ``APPROX_X`` positions.
        image
            The same fiber-flat image used for identification.
        poly_deg
            Degree of the Legendre polynomial fitted to each trace.
        max_shift
            Maximum allowed shift (pixels) between consecutive rows during
            barycenter tracing.
        cdisp_half_width
            Half-width of the aperture (in pixels) used for barycenter
            calculation.
        threshold_fraction
            Flux threshold relative to the global image maximum.  Rows with
            peak flux below this are skipped.

        Returns
        -------
        TraceMask
            Polynomial model of the fiber traces.
        """
        n_fibers = fibermap.n_fibers
        n_rows = image.shape[0]
        center_row = int(fibermap["CENTER_ROW"][0])

        # Allocate trace array: -1 means "not traced / invalid"
        traces = np.full((n_fibers, n_rows), -1.0, dtype=float)

        for i in range(n_fibers):
            approx_x = float(fibermap["APPROX_X"][i])
            trace = _trace_single_fiber(
                image=image,
                ini_row=center_row,
                ini_guess=approx_x,
                max_shift=max_shift,
                cdisp_half_width=cdisp_half_width,
                threshold_fraction=threshold_fraction,
            )
            traces[i, :] = trace

        # Fit Legendre polynomial to each trace
        coeffs = _fit_legendre_all(traces, poly_deg)
        fiber_ids = np.asarray(fibermap["FIBERID"], dtype=int)
        domain = (0, n_rows - 1)

        return cls(
            coeffs=coeffs,
            fiber_ids=fiber_ids,
            poly_kind="legendre",
            domain=domain,
        )

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #

    def eval(self, rows: NDArray[np.integer]) -> NDArray[np.floating]:
        """Evaluate trace positions at the given rows.

        Parameters
        ----------
        rows
            1-D array of row (spectral) indices.

        Returns
        -------
        ndarray
            Array of shape ``(n_fibers, len(rows))`` with cross-dispersion
            positions in pixels.
        """
        rows = np.asarray(rows, dtype=float)
        n_fibers = self.coeffs.shape[0]
        n_rows = len(rows)
        positions = np.empty((n_fibers, n_rows), dtype=float)

        for i in range(n_fibers):
            model = Legendre(self.coeffs[i], domain=self.domain)
            positions[i, :] = model(rows)

        return positions

    @property
    def n_fibers(self) -> int:
        """Number of fibers in the trace mask."""
        return self.coeffs.shape[0]


# ---------------------------------------------------------------------------
#  Internal helpers (module-level for Numba compatibility in the future)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _trace_single_fiber(
    image: NDArray[np.floating],
    ini_row: int,
    ini_guess: float,
    max_shift: float = 1.0,
    cdisp_half_width: int = 3,
    threshold_fraction: float = 0.1,
) -> NDArray[np.floating]:
    """Trace one fiber upward and downward from *ini_row*.

    Returns a 1-D array of length ``image.shape[0]`` where ``-1`` marks
    rows where the fiber could not be traced.
    """
    n_rows, n_cols = image.shape
    trace = np.full(n_rows, -1.0, dtype=float)

    # Initial row
    trace[ini_row] = _barycenter_at_row(
        image, ini_row, ini_guess, max_shift, cdisp_half_width, threshold_fraction
    )

    # Upward
    for r in range(ini_row - 1, -1, -1):
        guess = trace[r + 1]
        trace[r] = _barycenter_at_row(
            image, r, guess, max_shift, cdisp_half_width, threshold_fraction
        )
        if trace[r] < 0:
            break

    # Downward
    for r in range(ini_row + 1, n_rows):
        guess = trace[r - 1]
        trace[r] = _barycenter_at_row(
            image, r, guess, max_shift, cdisp_half_width, threshold_fraction
        )
        if trace[r] < 0:
            break

    return trace


@jit(nopython=True)
def _barycenter_at_row(
    image,
    row,
    guess_position,
    max_shift,
    cdisp_half_width,
    threshold_fraction,
):
    """Compute the barycenter of a fiber at a single row.

    Returns ``-1.0`` if the fiber cannot be measured (e.g. too faint or
    shifted too far from the guess).
    """
    if guess_position < 0:
        return -1.0

    n_cols = image.shape[1]
    col_center = int(round(guess_position))
    col_start = max(0, col_center - cdisp_half_width)
    col_end = min(n_cols, col_center + cdisp_half_width + 1)

    profile = image[row, col_start:col_end]
    n_pix = col_end - col_start
    col_range = np.empty(n_pix, dtype=np.float64)
    for i in range(n_pix):
        col_range[i] = col_start + i

    # Threshold check
    if np.nansum(profile) <= threshold_fraction * np.nanmax(image):
        return -1.0

    # Barycenter using only pixels above the median
    med = np.nanmedian(profile)
    mask = profile >= med
    if not mask.any():
        return -1.0

    bary = np.nansum(profile[mask] * col_range[mask]) / np.nansum(profile[mask])

    # Shift check
    if abs(bary - guess_position) > max_shift:
        return -1.0

    return float(bary)


def _fit_legendre_all(
    traces: NDArray[np.floating],
    deg: int,
) -> NDArray[np.floating]:
    """Fit a Legendre polynomial of degree *deg* to each fiber trace.

    Parameters
    ----------
    traces
        Array of shape ``(n_fibers, n_rows)``.  Entries ``< 0`` are treated
        as invalid and ignored.
    deg
        Polynomial degree.

    Returns
    -------
    ndarray
        Coefficients of shape ``(n_fibers, deg + 1)``.
    """
    n_fibers, n_rows = traces.shape
    coeffs = np.zeros((n_fibers, deg + 1), dtype=float)
    rows_all = np.arange(n_rows, dtype=float)

    for i in range(n_fibers):
        valid = traces[i, :] >= 0
        n_valid = valid.sum()

        if n_valid < deg + 1:
            warnings.warn(
                f"Fiber {i}: only {n_valid} valid points, "
                f"cannot fit degree-{deg} polynomial.  Returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            coeffs[i, 0] = -1.0
            continue

        data_x = rows_all[valid]
        data_y = traces[i, valid]
        domain = np.array([data_x.min(), data_x.max()])

        model = Legendre.fit(data_x, data_y, deg=deg, domain=domain)
        coeffs[i, :] = model.coef

    return coeffs
