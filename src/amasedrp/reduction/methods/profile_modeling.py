#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         profile_modeling.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Build fiber profiles from a master flat-field image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.fiberprofile import FiberProfile
from amasedrp.reduction.core.tracemask import TraceMask

__all__ = ["build_fiber_profile"]


def build_fiber_profile(
    flat_image: NDArray[np.floating],
    tracemask: TraceMask,
    fibermap: FiberMap,
    half_width: int = 5,
) -> FiberProfile:
    """Build a normalized fiber profile from a master flat image.

    For each fiber and each row, a cross-dispersion slice centered on the
    trace position is extracted from *flat_image*, clipped to non-negative
    values, and normalized to sum to one.

    Parameters
    ----------
    flat_image
        2-D master flat-field image.
    tracemask
        Polynomial trace model.
    fibermap
        Fiber metadata (used for validation and stored in the profile).
    half_width
        Half-width of the extraction aperture in pixels.

    Returns
    -------
    FiberProfile
        Normalized cross-dispersion profile model.
    """
    n_rows = flat_image.shape[0]
    n_fibers = tracemask.n_fibers
    n_offsets = 2 * half_width + 1
    x_offsets = np.arange(-half_width, half_width + 1, dtype=float)

    profile = np.zeros((n_fibers, n_rows, n_offsets), dtype=np.float64)
    rows = np.arange(n_rows, dtype=int)
    trace_positions = tracemask.eval(rows)  # shape (n_fibers, n_rows)

    for i in range(n_fibers):
        for r in range(n_rows):
            center = trace_positions[i, r]
            if not np.isfinite(center):
                # Untraceable row: fall back to delta (handled by FiberProfile)
                continue

            col_center = int(round(center))
            col_start = max(0, col_center - half_width)
            col_end = min(flat_image.shape[1], col_center + half_width + 1)

            slice_data = flat_image[r, col_start:col_end].astype(float)

            # Subtract local floor (minimum in the slice) to reduce background
            floor = slice_data.min()
            if floor > 0:
                slice_data = slice_data - floor

            # Clip negative values
            slice_data = np.clip(slice_data, 0, None)

            # Map slice pixels back to offset indices
            offset_start = col_start - col_center + half_width
            offset_end = offset_start + len(slice_data)
            profile[i, r, offset_start:offset_end] = slice_data

    return FiberProfile(
        profile=profile,
        x_offsets=x_offsets,
        fibermap=fibermap,
    )
