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
from .core.fiberframe import FiberFrame
from .core.fiberprofile import FiberProfile
from .core.tracemask import TraceMask
from .methods.boxcar import extract_boxcar
from .methods.optimal import extract_optimal
from .methods.profile_modeling import build_fiber_profile


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


def extract_spectra(
    image: NDArray[np.floating],
    tracemask: TraceMask,
    fibermap: FiberMap,
    method: str = "boxcar",
    fiber_profile: FiberProfile | None = None,
    wave: NDArray[np.floating] | None = None,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
    aperture_radius: int = 3,
    sigma_clip: float = 5.0,
    maxiter: int = 5,
    meta: dict[str, Any] | None = None,
) -> FiberFrame:
    """Extract 1-D spectra from a 2-D image using traced fiber positions.

    Parameters
    ----------
    image
        2-D calibrated science image.
    tracemask
        Polynomial trace model.
    fibermap
        Per-fiber metadata table.
    method
        Extraction method: ``"boxcar"`` or ``"optimal"``.
    fiber_profile
        Required when *method* is ``"optimal"``.
    wave
        Optional wavelength array.  Defaults to pixel indices ``[0, n_rows)``.
    variance
        Optional variance image.
    mask
        Optional boolean bad-pixel mask.
    aperture_radius
        Aperture half-width for boxcar extraction.
    sigma_clip
        Sigma-clipping threshold for optimal extraction.
    maxiter
        Maximum rejection iterations for optimal extraction.
    meta
        Optional metadata dictionary.

    Returns
    -------
    FiberFrame
        Extracted row-stacked spectra.

    Raises
    ------
    ValueError
        If *method* is unknown or ``"optimal"`` is requested without
        *fiber_profile*.
    """
    if method not in ("boxcar", "optimal"):
        raise ValueError(f"Unknown extraction method: {method!r}")

    if method == "optimal" and fiber_profile is None:
        raise ValueError('method="optimal" requires fiber_profile.')

    rows = np.arange(image.shape[0], dtype=int)
    trace_positions = tracemask.eval(rows)

    if method == "boxcar":
        flux, ivar, out_mask = extract_boxcar(
            image=image,
            trace_positions=trace_positions,
            aperture_radius=aperture_radius,
            variance=variance,
            mask=mask,
        )
    else:  # optimal
        flux, ivar, out_mask = extract_optimal(
            image=image,
            trace_positions=trace_positions,
            fiber_profile=fiber_profile,
            variance=variance,
            mask=mask,
            sigma_clip=sigma_clip,
            maxiter=maxiter,
        )

    if wave is None:
        wave = np.arange(image.shape[0], dtype=float)

    extraction_meta = {
        "METHOD": method,
        "APERTURE": aperture_radius,
        **(meta or {}),
    }
    if method == "optimal":
        extraction_meta["SIGMA_CLIP"] = sigma_clip
        extraction_meta["MAXITER"] = maxiter

    return FiberFrame(
        wave=wave,
        flux=flux,
        ivar=ivar,
        mask=out_mask,
        fibermap=fibermap,
        meta=extraction_meta,
    )


def run_quick_reduction(
    image: NDArray[np.floating],
    flat_image: NDArray[np.floating],
    *,
    n_blocks_expected: int = 19,
    n_fibers_per_block_expected: int = 29,
    aperture_radius: int = 3,
    poly_deg: int = 10,
    meta: dict[str, Any] | None = None,
    **identify_kwargs: Any,
) -> FiberFrame:
    """Quick-look reduction: identify fibers and extract with boxcar.

    Parameters
    ----------
    image
        2-D science image.
    flat_image
        2-D fiber-flat image used for identification and tracing.
    n_blocks_expected
        Expected number of fiber blocks.
    n_fibers_per_block_expected
        Expected number of fibers per block.
    aperture_radius
        Boxcar aperture half-width.
    poly_deg
        Trace polynomial degree.
    meta
        Optional metadata.
    **identify_kwargs
        Additional arguments forwarded to :func:`identify_and_trace_fibers`.

    Returns
    -------
    FiberFrame
        Boxcar-extracted spectra.
    """
    fibermap, tracemask = identify_and_trace_fibers(
        image=flat_image,
        n_blocks_expected=n_blocks_expected,
        n_fibers_per_block_expected=n_fibers_per_block_expected,
        poly_deg=poly_deg,
        **identify_kwargs,
    )

    return extract_spectra(
        image=image,
        tracemask=tracemask,
        fibermap=fibermap,
        method="boxcar",
        aperture_radius=aperture_radius,
        meta=meta,
    )


def run_reduction(
    image: NDArray[np.floating],
    flat_image: NDArray[np.floating],
    *,
    n_blocks_expected: int = 19,
    n_fibers_per_block_expected: int = 29,
    method: str = "optimal",
    aperture_radius: int = 3,
    profile_half_width: int = 5,
    poly_deg: int = 10,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
    meta: dict[str, Any] | None = None,
    **identify_kwargs: Any,
) -> FiberFrame:
    """Full reduction: identify fibers, build profile, and extract spectra.

    Parameters
    ----------
    image
        2-D science image.
    flat_image
        2-D fiber-flat image.
    n_blocks_expected
        Expected number of fiber blocks.
    n_fibers_per_block_expected
        Expected number of fibers per block.
    method
        ``"optimal"`` (default) or ``"boxcar"``.
    aperture_radius
        Boxcar aperture half-width (used for boxcar or fallback).
    profile_half_width
        Half-width for fiber profile extraction from flat.
    poly_deg
        Trace polynomial degree.
    variance
        Optional variance image.
    mask
        Optional bad-pixel mask.
    meta
        Optional metadata.
    **identify_kwargs
        Additional arguments forwarded to :func:`identify_and_trace_fibers`.

    Returns
    -------
    FiberFrame
        Extracted spectra.
    """
    fibermap, tracemask = identify_and_trace_fibers(
        image=flat_image,
        n_blocks_expected=n_blocks_expected,
        n_fibers_per_block_expected=n_fibers_per_block_expected,
        poly_deg=poly_deg,
        **identify_kwargs,
    )

    if method == "optimal":
        fiber_profile = build_fiber_profile(
            flat_image=flat_image,
            tracemask=tracemask,
            fibermap=fibermap,
            half_width=profile_half_width,
        )
        return extract_spectra(
            image=image,
            tracemask=tracemask,
            fibermap=fibermap,
            method="optimal",
            fiber_profile=fiber_profile,
            variance=variance,
            mask=mask,
            meta=meta,
        )

    if method == "boxcar":
        return extract_spectra(
            image=image,
            tracemask=tracemask,
            fibermap=fibermap,
            method="boxcar",
            aperture_radius=aperture_radius,
            variance=variance,
            mask=mask,
            meta=meta,
        )

    raise ValueError(f"Unknown reduction method: {method!r}")
