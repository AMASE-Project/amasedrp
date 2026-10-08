#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         simulate.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Synthetic frames used to verify the pipeline.

The generators here make a frame whose contents are known exactly, so a test
or a tutorial can compare what the pipeline recovers against what went in.
They model the fiber geometry and the arc spectrum of the AMASE-P prototype
measured on the 2025-07 collimator-sweep frames.  They are not an instrument
model: they carry no detector cosmetics, no cosmic rays and no scattered
light.  Add a feature here only when a check needs it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["synthetic_fiber_flat", "synthetic_arc"]

# Fiber-flat geometry measured on the 9600 x 6422 sweep frames.
FIBER_SIGMA = 2.0
FIBER_SPACING = 8.0
BLOCK_GAP = 26.0
MARGIN = 7.0

# Dispersion of the blue channel, in Angstrom per row, measured on the same
# frames.  One resolution element spans about six rows.
BLUE_DISPERSION = 0.08221

# Gaussian FWHM in units of sigma.
_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def synthetic_fiber_flat(
    n_rows: int = 512,
    n_blocks: int = 3,
    n_fibers_per_block: int = 5,
    fiber_sigma: float = FIBER_SIGMA,
    fiber_spacing: float = FIBER_SPACING,
    block_gap: float = BLOCK_GAP,
    margin: float = MARGIN,
    peak: float = 1.0,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[NDArray[np.floating], list[float]]:
    """Build a synthetic fiber-flat image.

    Each fiber is a Gaussian in the cross-dispersion direction and constant
    along the dispersion direction.  Fibers are grouped into blocks.  A dark
    margin brackets the whole array, and a dark gap separates two blocks.

    The default geometry follows the instrument: fibers sit 8 px apart with a
    2 px Gaussian width, so the fibers of one block merge into a single bright
    strip, and neighbouring blocks are separated by a 26 px gap that reaches
    the background.  Block detection separates blocks by the depth of the
    valleys in the cross-dispersion profile, and that criterion only holds for
    this geometry.  A flat with sparser fibers, or with a gap wider than a
    block, does not represent the instrument and is rejected by the
    identifier.

    Parameters
    ----------
    n_rows
        Number of rows (dispersion direction).
    n_blocks
        Number of fiber blocks.
    n_fibers_per_block
        Number of fibers in each block.
    fiber_sigma
        Gaussian sigma of one fiber, in pixels.
    fiber_spacing
        Center-to-center spacing of neighbouring fibers, in pixels.
    block_gap
        Width of the dark gap between two blocks, in pixels.
    margin
        Width of the dark margin on each side of the fiber array, in pixels.
    peak
        Peak value of one fiber.
    noise_std
        Standard deviation of the additive Gaussian noise, as a fraction of
        *peak*.
    seed
        Seed for the noise generator.

    Returns
    -------
    image
        Array of shape ``(n_rows, n_cols)``.
    centers
        Cross-dispersion center of each fiber, ordered by block and then by
        position inside the block.

    Notes
    -----
    The fibers are straight, so the trace of a fiber is a constant.  A real
    fiber flat has a slight curvature; use
    :func:`~amasedrp.reduction.methods.fiber_tracing` on real frames to see it.

    Examples
    --------
    >>> flat, centers = synthetic_fiber_flat(n_blocks=3, n_fibers_per_block=5)
    >>> len(centers)
    15
    """
    half_width = int(4 * fiber_sigma)
    block_width = (n_fibers_per_block - 1) * fiber_spacing + 2 * half_width
    n_cols = int(2 * margin + n_blocks * block_width + (n_blocks - 1) * block_gap)

    image = np.zeros((n_rows, n_cols), dtype=np.float64)
    x = np.arange(n_cols)
    centers: list[float] = []

    for block in range(n_blocks):
        offset = margin + block * (block_width + block_gap)
        for fiber in range(n_fibers_per_block):
            center = offset + half_width + fiber * fiber_spacing
            centers.append(float(center))
            profile = peak * np.exp(-0.5 * ((x - center) / fiber_sigma) ** 2)
            image += profile[np.newaxis, :]

    image += np.random.default_rng(seed).normal(0, noise_std * peak, image.shape)
    return image, centers


def synthetic_arc(
    flat: NDArray[np.floating],
    lines: NDArray[np.floating],
    *,
    wavelength_zero: float = 4390.0,
    dispersion: float = BLUE_DISPERSION,
    line_fwhm: float = 0.49,
    line_flux: float = 1.0,
    continuum: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build a synthetic arc frame from a fiber flat.

    The arc spectrum is a set of Gaussian emission lines on a flat continuum.
    It is multiplied into *flat*, so the fibers carry the lamp and the dark
    gaps stay dark, as on a real arc exposure.  The lines run along the
    dispersion direction, which means every fiber sees the same spectrum.

    Parameters
    ----------
    flat
        Fiber-flat image, usually the output of :func:`synthetic_fiber_flat`.
        The arc frame takes its shape.
    lines
        Vacuum wavelengths of the lines to inject, in Angstrom.  Use a
        reference list such as
        :func:`~amasedrp.calibration.lines.thar_lines`.
    wavelength_zero
        Wavelength at row 0, in Angstrom.
    dispersion
        Angstrom per row.  The default is the measured blue-channel
        dispersion, so one resolution element spans about six rows.
    line_fwhm
        FWHM of every injected line, in Angstrom.  The default reproduces the
        line width measured on the sweep frames.
    line_flux
        Peak flux of every injected line, in units of *flat*.
    continuum
        Flux level between the lines.
    noise_std
        Standard deviation of the additive Gaussian noise, in the same units
        as *flat*.  Use 0 for a noiseless frame.
    seed
        Seed for the noise generator.

    Returns
    -------
    image
        Arc image of shape ``flat.shape``.
    wavelength
        True wavelength of every row, shape ``(n_rows,)``.  Compare the
        recovered solution against this array to measure the calibration
        error.

    Raises
    ------
    ValueError
        If *flat* is not two-dimensional, or if *dispersion* is zero.

    Notes
    -----
    Only the lines inside the row range are visible; a line outside it is
    silently absent, so check the coverage before scoring a calibration.

    Examples
    --------
    >>> flat, _ = synthetic_fiber_flat(n_rows=300)
    >>> image, wave = synthetic_arc(flat, np.array([4704.0, 4764.9]))
    >>> wave[0]
    np.float64(4390.0)
    """
    flat = np.asarray(flat, dtype=float)
    if flat.ndim != 2:
        raise ValueError(f"flat must be 2-D, got shape {flat.shape}")
    if dispersion == 0:
        raise ValueError("dispersion must not be zero.")

    n_rows = flat.shape[0]
    rows = np.arange(n_rows, dtype=float)
    wavelength = wavelength_zero + dispersion * rows

    spectrum = np.full(n_rows, continuum, dtype=float)
    sigma_rows = line_fwhm / (_FWHM_OVER_SIGMA * abs(dispersion))
    for line in np.atleast_1d(np.asarray(lines, dtype=float)):
        center = (line - wavelength_zero) / dispersion
        spectrum += line_flux * np.exp(-0.5 * ((rows - center) / sigma_rows) ** 2)

    image = flat * spectrum[:, np.newaxis]

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0.0, noise_std, image.shape)

    return image, wavelength
