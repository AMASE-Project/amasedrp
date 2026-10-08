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

    # One aperture shape per fiber, evaluated for every row at once.  Looping
    # over rows in Python costs about 20 s on a 9600-row frame.
    offsets = np.arange(-aperture_radius, aperture_radius + 1)
    rows = np.arange(n_rows)

    for i in range(n_fibers):
        centers = trace_positions[i]
        traced = np.isfinite(centers)
        out_mask[i, ~traced] |= MASK_BAD_TRACE
        if not traced.any():
            continue

        row_idx = rows[traced][:, None]
        cols = (
            np.round(centers[traced]).astype(int)[:, None]
            + offsets[None, :]
        )
        inside = (cols >= 0) & (cols < n_cols)
        cols = np.clip(cols, 0, n_cols - 1)

        usable = inside
        if mask is not None:
            usable = usable & ~mask[row_idx, cols]

        # A row with no usable pixel is flagged and left at zero flux.
        no_pixels = ~usable.any(axis=1)
        out_mask[i, rows[traced][no_pixels]] |= MASK_NO_PIXELS

        pixel_values = np.where(usable, image[row_idx, cols], np.nan)
        flux_values = np.nansum(pixel_values, axis=1)
        flux[i, rows[traced]] = flux_values

        if variance is not None:
            variance_values = np.nansum(
                np.where(usable, variance[row_idx, cols], np.nan), axis=1
            )
        else:
            # Placeholder when the caller supplies no variance image.
            variance_values = np.maximum(np.abs(flux_values), 1.0)

        good = ~no_pixels & (variance_values > 0) & np.isfinite(variance_values)
        ivar[i, rows[traced][good]] = 1.0 / variance_values[good]
        out_mask[i, rows[traced][~good]] |= MASK_BAD_VARIANCE

    return flux, ivar, out_mask
