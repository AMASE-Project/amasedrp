#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         fiber_tracing.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Fiber tracing algorithms.

Note: The core tracing logic (barycenter tracing + polynomial fitting)
lives in ``core/tracemask.py``.  This module is reserved for future
alternative tracing methods (e.g., cross-correlation, Gaussian centroid).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def trace_fibers_barycenter(
    image: NDArray[np.floating],
    approx_positions: NDArray[np.floating],
    center_row: int,
    max_shift: float = 1.0,
    cdisp_half_width: int = 3,
    threshold_fraction: float = 0.1,
) -> NDArray[np.floating]:
    """Trace fiber barycenters upward and downward from the center row.

    Parameters
    ----------
    image
        2-D fiber-flat image.
    approx_positions
        Initial cross-dispersion positions for each fiber.
    center_row
        Row from which to start tracing.
    max_shift
        Maximum allowed pixel shift between consecutive rows.
    cdisp_half_width
        Half-width of the aperture for barycenter calculation.
    threshold_fraction
        Flux threshold relative to the global image maximum.  Rows whose
        aperture sum stays below it are skipped.  The maximum is measured
        once, over the whole image.

    Returns
    -------
    ndarray
        Array of shape ``(n_fibers, n_rows)`` with trace positions.
        ``-1`` marks untraceable rows.

    Note
    ----
    This is a thin wrapper around the Numba-jitted helpers in
    ``core/tracemask.py``.  It exists so that alternative tracers can
    be swapped in without touching ``TraceMask``.
    """
    from ..core.tracemask import _trace_single_fiber

    n_fibers = len(approx_positions)
    n_rows = image.shape[0]
    traces = np.full((n_fibers, n_rows), -1.0, dtype=float)

    # Measured once: measuring it per row would scan the full image each time.
    threshold = threshold_fraction * float(np.nanmax(image))

    for i in range(n_fibers):
        traces[i, :] = _trace_single_fiber(
            image=image,
            ini_row=center_row,
            ini_guess=approx_positions[i],
            max_shift=max_shift,
            cdisp_half_width=cdisp_half_width,
            threshold=threshold,
        )

    return traces


def fit_traces_polynomial(
    traces: NDArray[np.floating],
    poly_deg: int = 10,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Fit a Legendre polynomial to each fiber trace.

    Parameters
    ----------
    traces
        Array of shape ``(n_fibers, n_rows)`` with trace positions.
        Negative values are treated as invalid.
    poly_deg
        Polynomial degree.

    Returns
    -------
    coeffs
        Array of shape ``(n_fibers, poly_deg + 1)``.
    domains
        Fitted row range of each fiber, shape ``(n_fibers, 2)``.
    """
    from ..core.tracemask import _fit_legendre_all

    return _fit_legendre_all(traces, poly_deg)
