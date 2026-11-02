#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         reduction.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for the data reduction pipeline (DRP).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import amasecore
import numpy as np
from numpy.typing import NDArray

from .core.fiberidentifier import FibersIdentifier
from .core.fibermap import FiberMap
from .core.fiberframe import FiberFrame
from .core.tracemask import TraceMask
from .methods.boxcar import extract_boxcar
from .methods.optimal import extract_optimal

#: Bundle whose slit layout is assumed when the caller does not name one.
DEFAULT_BUNDLE = "canon"


@lru_cache(maxsize=None)
def _slit_layout(bundle: str = DEFAULT_BUNDLE) -> tuple[int, int]:
    """Return the slit layout of one bundle, from amasecore.

    Parameters
    ----------
    bundle
        Bundle name, as recorded in ``amasecore``'s ``instrument/slits.yaml``.

    Returns
    -------
    tuple of int
        Number of blocks per slit side, and fibers per block.

    Raises
    ------
    ValueError
        If amasecore holds no single layout for *bundle*.
    """
    table = amasecore.load_table(amasecore.data_path("instrument", "slits.yaml"))
    found = table[table["BUNDLE"] == bundle]
    if len(found) != 1:
        raise ValueError(
            f"amasecore holds no single slit layout for bundle {bundle!r}; "
            f"available: {[str(name) for name in table['BUNDLE']]}"
        )
    return int(found[0]["BLOCKS"]), int(found[0]["FIBERS_PER_BLOCK"])


def _resolve_layout(
    blocks: int | None, fibers_per_block: int | None
) -> tuple[int, int]:
    """Fill in whichever of the two layout numbers the caller left out.

    Parameters
    ----------
    blocks
        Expected number of blocks, or ``None`` to take it from amasecore.
    fibers_per_block
        Expected fibers per block, or ``None`` to take it from amasecore.

    Returns
    -------
    tuple of int
        The two numbers, both resolved.
    """
    if blocks is not None and fibers_per_block is not None:
        return blocks, fibers_per_block
    default_blocks, default_fibers = _slit_layout()
    return (
        default_blocks if blocks is None else blocks,
        default_fibers if fibers_per_block is None else fibers_per_block,
    )


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
    ...     n_blocks_expected=22,
    ...     n_fibers_per_block_expected=25,
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
    flat_image: NDArray[np.floating] | None = None,
    wave: NDArray[np.floating] | None = None,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
    aperture_radius: int = 3,
    gain: float = 1.0,
    read_noise: float = 1.0,
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
    flat_image
        Fiber-flat image.  Required when *method* is ``"optimal"``, because
        flat-relative extraction needs the flat itself.
    wave
        Optional wavelength array.  Defaults to pixel indices ``[0, n_rows)``.
    variance
        Optional variance image.  Used by boxcar extraction only; FOX builds
        its own noise model from the science counts.
    mask
        Optional boolean bad-pixel mask.
    aperture_radius
        Aperture half-width, for boxcar extraction and for the flat aperture
        of optimal extraction.
    gain
        Detector gain in electrons per ADU, for optimal extraction.
    read_noise
        Detector read noise in electrons, for optimal extraction.
    meta
        Optional metadata dictionary.

    Returns
    -------
    FiberFrame
        Extracted row-stacked spectra.

    Raises
    ------
    ValueError
        If *method* is unknown, if ``"optimal"`` is requested without
        *flat_image*, or if a *variance* image is passed to ``"optimal"``.
    """
    if method not in ("boxcar", "optimal"):
        raise ValueError(f"Unknown extraction method: {method!r}")

    if method == "optimal":
        if flat_image is None:
            raise ValueError('method="optimal" requires flat_image.')
        if variance is not None:
            raise ValueError(
                'method="optimal" builds its own noise model from the science '
                "counts; pass gain and read_noise instead of variance."
            )

    rows = np.arange(image.shape[0], dtype=int)
    trace_positions = tracemask.eval(rows)

    # Outside a fiber's fitted row range the polynomial extrapolates, so the
    # position carries no information.  Mark those rows invalid instead.
    outside = (rows[None, :] < tracemask.domain[:, 0:1]) | (
        rows[None, :] > tracemask.domain[:, 1:2]
    )
    trace_positions[outside] = np.nan

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
            flat_image=flat_image,
            trace_positions=trace_positions,
            aperture_radius=aperture_radius,
            gain=gain,
            read_noise=read_noise,
            mask=mask,
        )

    if wave is None:
        wave = np.arange(image.shape[0], dtype=float)

    extraction_meta = {
        "METHOD": method,
        "APERTURE": aperture_radius,
        **(meta or {}),
    }
    if method == "optimal":
        extraction_meta["GAIN"] = gain
        extraction_meta["RDNOISE"] = read_noise

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
    n_blocks_expected: int | None = None,
    n_fibers_per_block_expected: int | None = None,
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
        Expected number of fiber blocks.  ``None`` takes it from amasecore's
        slit layout for :data:`DEFAULT_BUNDLE`.
    n_fibers_per_block_expected
        Expected number of fibers per block, resolved the same way.
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
    n_blocks_expected, n_fibers_per_block_expected = _resolve_layout(
        n_blocks_expected, n_fibers_per_block_expected
    )
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
    n_blocks_expected: int | None = None,
    n_fibers_per_block_expected: int | None = None,
    method: str = "optimal",
    aperture_radius: int = 3,
    poly_deg: int = 10,
    gain: float = 1.0,
    read_noise: float = 1.0,
    variance: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
    meta: dict[str, Any] | None = None,
    **identify_kwargs: Any,
) -> FiberFrame:
    """Full reduction: identify fibers, trace them, and extract spectra.

    Parameters
    ----------
    image
        2-D science image.
    flat_image
        2-D fiber-flat image.  Used for identification, tracing, and the
        flat-relative weights of optimal extraction.
    n_blocks_expected
        Expected number of fiber blocks.  ``None`` takes it from amasecore's
        slit layout for :data:`DEFAULT_BUNDLE`.
    n_fibers_per_block_expected
        Expected number of fibers per block, resolved the same way.
    method
        ``"optimal"`` (default) or ``"boxcar"``.
    aperture_radius
        Aperture half-width.
    poly_deg
        Trace polynomial degree.
    gain
        Detector gain in electrons per ADU, for optimal extraction.
    read_noise
        Detector read noise in electrons, for optimal extraction.
    variance
        Optional variance image, for boxcar extraction.
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
    n_blocks_expected, n_fibers_per_block_expected = _resolve_layout(
        n_blocks_expected, n_fibers_per_block_expected
    )
    fibermap, tracemask = identify_and_trace_fibers(
        image=flat_image,
        n_blocks_expected=n_blocks_expected,
        n_fibers_per_block_expected=n_fibers_per_block_expected,
        poly_deg=poly_deg,
        **identify_kwargs,
    )

    if method == "optimal":
        return extract_spectra(
            image=image,
            tracemask=tracemask,
            fibermap=fibermap,
            method="optimal",
            flat_image=flat_image,
            mask=mask,
            aperture_radius=aperture_radius,
            gain=gain,
            read_noise=read_noise,
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
