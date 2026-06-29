#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         boxcar.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Boxcar (aperture) spectral extraction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["extract_boxcar"]

# Mask bit definitions
MASK_BAD_TRACE = np.uint32(1)
MASK_NO_PIXELS = np.uint32(2)
MASK_BAD_VARIANCE = np.uint32(4)


def extract_boxcar(
    image: NDArray[np.floating],
    trace_positions: NDArray[np.floating],
    aperture_radius: int = 3,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Boxcar extraction: sum flux within a fixed aperture around each trace.

    Parameters
    ----------
    image
        2-D calibrated science image, shape ``(n_rows, n_cols)``.
    trace_positions
        Array of shape ``(n_fibers, n_rows)`` with trace x-positions.
    aperture_radius
        Half-width of the extraction aperture in pixels.
    variance
        Optional variance image, same shape as *image*.
    mask
        Optional boolean bad-pixel mask, same shape as *image*.

    Returns
    -------
    flux
        Extracted flux, shape ``(n_fibers, n_rows)``.
    ivar
        Inverse variance, shape ``(n_fibers, n_rows)``.
    out_mask
        Bitmask, shape ``(n_fibers, n_rows)``.
    """
    n_rows, n_cols = image.shape
    n_fibers = trace_positions.shape[0]

    if trace_positions.shape != (n_fibers, n_rows):
        raise ValueError(
            f"trace_positions shape {trace_positions.shape} incompatible with "
            f"image shape {image.shape}"
        )

    flux = np.zeros((n_fibers, n_rows), dtype=np.float64)
    ivar = np.zeros((n_fibers, n_rows), dtype=np.float64)
    out_mask = np.zeros((n_fibers, n_rows), dtype=np.uint32)

    for i in range(n_fibers):
        for r in range(n_rows):
            center = trace_positions[i, r]
            if not np.isfinite(center):
                out_mask[i, r] |= MASK_BAD_TRACE
                continue

            col_center = int(round(center))
            col_start = max(0, col_center - aperture_radius)
            col_end = min(n_cols, col_center + aperture_radius + 1)

            if col_start >= col_end:
                out_mask[i, r] |= MASK_NO_PIXELS
                continue

            pixel_mask = np.ones(col_end - col_start, dtype=bool)
            if mask is not None:
                pixel_mask = ~mask[r, col_start:col_end]

            if not pixel_mask.any():
                out_mask[i, r] |= MASK_NO_PIXELS
                continue

            pix = image[r, col_start:col_end]
            flux_val = float(np.nansum(pix[pixel_mask]))
            flux[i, r] = flux_val

            if variance is not None:
                var_val = float(np.nansum(variance[r, col_start:col_end][pixel_mask]))
            else:
                var_val = max(abs(flux_val), 1.0)

            if var_val > 0 and np.isfinite(var_val):
                ivar[i, r] = 1.0 / var_val
            else:
                out_mask[i, r] |= MASK_BAD_VARIANCE
                ivar[i, r] = 0.0

    return flux, ivar, out_mask
