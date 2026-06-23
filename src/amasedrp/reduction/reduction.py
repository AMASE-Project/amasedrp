#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         reduction.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for the data reduction pipeline (DRP).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .core.fiberidentifier import FibersIdentifier
from .core.fibermap import FiberMap
from .core.tracemask import TraceMask


def identify_and_trace_fibers(
    image: NDArray[np.floating],
    n_blocks_expected: int,
    n_fibers_per_block_expected: int,
    *,
    center_row: int | None = None,
    band_half_width: int = 100,
    poly_deg: int = 10,
    strict: bool = True,
    max_shift: float = 1.0,
    cdisp_half_width: int = 3,
    threshold_fraction: float = 0.1,
    block_valley_threshold_frac: float = 0.3,
    fiber_peak_height_frac: float = 0.5,
    smooth_sigma_factor: float = 10.0,
) -> tuple[FiberMap, TraceMask]:
    """One-shot fiber identification and tracing.

    This is the user-facing entry point for the fiber detection stage.
    It chains :class:`FibersIdentifier` (block + fiber detection) with
    :class:`TraceMask` (barycenter tracing + polynomial fitting).

    Parameters
    ----------
    image
        2-D pre-processed fiber-flat image.
    n_blocks_expected
        Expected number of fiber blocks.
    n_fibers_per_block_expected
        Expected number of fibers per block.
    center_row
        Row around which to extract the cross-dispersion profile.
        Defaults to ``image.shape[0] // 2``.
    band_half_width
        Half-height of the band collapsed to form the profile.
    poly_deg
        Degree of the Legendre polynomial fitted to each trace.
    strict
        If ``True``, raise on fiber/block count mismatch.
    max_shift
        Maximum allowed shift between consecutive rows during tracing.
    cdisp_half_width
        Half-width of the aperture for barycenter calculation.
    threshold_fraction
        Flux threshold relative to the global image maximum.
    block_valley_threshold_frac
        Valley depth threshold for block detection.
    fiber_peak_height_frac
        Peak height threshold for fiber detection.
    smooth_sigma_factor
        Gaussian smoothing scale for fiber peak finding.

    Returns
    -------
    fibermap
        Structured table with one row per detected fiber.
    tracemask
        Polynomial model of the fiber traces.

    Examples
    --------
    >>> fibermap, tracemask = identify_and_trace_fibers(
    ...     image=fflat,
    ...     n_blocks_expected=19,
    ...     n_fibers_per_block_expected=29,
    ... )
    """
    identifier = FibersIdentifier(
        image=image,
        n_blocks_expected=n_blocks_expected,
        n_fibers_per_block_expected=n_fibers_per_block_expected,
        strict=strict,
    )
    fibermap = identifier.identify(
        center_row=center_row,
        band_half_width=band_half_width,
        block_valley_threshold_frac=block_valley_threshold_frac,
        fiber_peak_height_frac=fiber_peak_height_frac,
        smooth_sigma_factor=smooth_sigma_factor,
    )

    tracemask = TraceMask.from_fibermap(
        fibermap=fibermap,
        image=image,
        poly_deg=poly_deg,
        max_shift=max_shift,
        cdisp_half_width=cdisp_half_width,
        threshold_fraction=threshold_fraction,
    )

    return fibermap, tracemask


def run_reduction():
    """Main function to run the full data reduction pipeline.

    TODO: Implement the full reduction pipeline.
    """
    raise NotImplementedError("Full reduction pipeline is not yet implemented.")


def run_quick_reduction():
    """Quick reduction pipeline for rapid data inspection.

    TODO: Implement the quick reduction pipeline.
    """
    raise NotImplementedError("Quick reduction pipeline is not yet implemented.")
