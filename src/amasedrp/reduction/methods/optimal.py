#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         optimal.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Optimal spectral extraction (placeholder interface).

The full implementation will follow MaNGA-style row-by-row profile fitting
with iterative sigma-clipping rejection.  For now only the public API
signature is defined.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from amasedrp.reduction.core.fiberprofile import FiberProfile

__all__ = ["extract_optimal"]


def extract_optimal(
    image: NDArray[np.floating],
    trace_positions: NDArray[np.floating],
    fiber_profile: FiberProfile,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
    sigma_clip: float = 5.0,
    maxiter: int = 5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Optimal extraction using a flat-derived fiber profile.

    Parameters
    ----------
    image
        2-D calibrated science image.
    trace_positions
        Array of shape ``(n_fibers, n_rows)`` with trace x-positions.
    fiber_profile
        Normalized cross-dispersion profile model.
    variance
        Optional variance image.
    mask
        Optional boolean bad-pixel mask.
    sigma_clip
        Sigma-clipping threshold for iterative rejection.
    maxiter
        Maximum rejection iterations.

    Returns
    -------
    flux
        Extracted flux, shape ``(n_fibers, n_rows)``.
    ivar
        Inverse variance, shape ``(n_fibers, n_rows)``.
    out_mask
        Bitmask, shape ``(n_fibers, n_rows)``.

    Raises
    ------
    NotImplementedError
        This is a placeholder.
    """
    raise NotImplementedError("optimal extraction is not yet implemented.")
