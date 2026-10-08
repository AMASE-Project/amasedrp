#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         optimal.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Flat-relative optimal extraction (FOX).

The fiber flat carries the cross-dispersion profile of every fiber.  Weighting
the science frame by the flat, and dividing the weighted sum of the science by
the weighted sum of the flat, gives a flat-relative spectrum with the best
possible signal-to-noise ratio.  A smooth version of the boxcar-extracted flat
then removes the lamp spectrum and returns the spectrum to the same scale as a
plain boxcar sum.

The method follows Naylor (1998), "An optimal extraction algorithm for imaging
spectroscopy", and the AMASE prototype implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .boxcar import (
    MASK_BAD_TRACE,
    MASK_BAD_VARIANCE,
    MASK_NO_PIXELS,
    extract_boxcar,
)

__all__ = ["extract_optimal"]

# Number of bins used for the smooth trend of the boxcar-extracted flat.
_SPLINE_BINS = 100


def _smooth_flat_spectrum(
    spectrum: NDArray[np.floating],
    bins: int = _SPLINE_BINS,
) -> NDArray[np.floating] | None:
    """Fit a natural cubic spline to the smooth trend of a flat spectrum.

    The flat carries the lamp spectrum on top of the fiber profile.  A spline
    through the median of a few dozen bins removes the lamp shape without
    injecting the noise of the flat into the extracted spectrum.

    Parameters
    ----------
    spectrum
        1-D boxcar-extracted flat spectrum.
    bins
        Number of equal-width bins used to take the median trend.

    Returns
    -------
    ndarray or None
        The smoothed spectrum, sampled at every pixel.  ``None`` when fewer
        than two bins hold a positive median.
    """
    good = spectrum > 0
    if good.sum() < 2:
        return None

    data_x = np.arange(len(spectrum))[good]
    data_y = spectrum[good]
    edges = np.linspace(data_x.min(), data_x.max(), bins)

    centers: list[float] = []
    medians: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (data_x >= low) & (data_x < high)
        if not in_bin.any():
            continue
        median = float(np.median(data_y[in_bin]))
        if median <= 0:
            continue
        centers.append(0.5 * (low + high))
        medians.append(median)

    if len(centers) < 2:
        return None

    spline = CubicSpline(
        np.asarray(centers), np.asarray(medians), bc_type="natural"
    )
    return spline(np.arange(len(spectrum), dtype=float))


def extract_optimal(
    image: NDArray[np.floating],
    flat_image: NDArray[np.floating],
    trace_positions: NDArray[np.floating],
    aperture_radius: int = 3,
    gain: float = 1.0,
    read_noise: float = 1.0,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Extract spectra with flat-relative optimal extraction (FOX).

    The noise model is built from the science counts:

    .. math:: \\mathrm{var} = \\mathrm{gain} \\cdot \\mathrm{image}
              + \\mathrm{read\\_noise}^2

    Negative science and flat pixels come from an over-subtracted bias and
    carry no signal, so they are clipped to zero.

    Parameters
    ----------
    image
        2-D calibrated science image, shape ``(n_rows, n_cols)``.
    flat_image
        2-D fiber-flat image, same shape as *image*.  The flat must be taken
        with the same fiber configuration as *image*.
    trace_positions
        Array of shape ``(n_fibers, n_rows)`` with trace x-positions.
        Non-finite entries mark rows that were never traced.
    aperture_radius
        Half-width of the extraction aperture in pixels.  The same aperture is
        used for the flat.
    gain
        Detector gain in electrons per ADU.
    read_noise
        Detector read noise in electrons.
    mask
        Optional boolean bad-pixel mask, same shape as *image*.  Masked
        pixels are left out of the weighted sums.

    Returns
    -------
    flux
        Extracted flux, shape ``(n_fibers, n_rows)``, on the same scale as a
        boxcar sum of *image*.
    ivar
        Inverse variance, shape ``(n_fibers, n_rows)``.
    out_mask
        Bitmask, shape ``(n_fibers, n_rows)``.  Sets ``MASK_BAD_TRACE`` on
        rows with no finite trace, ``MASK_NO_PIXELS`` where the aperture
        holds no usable pixel, and ``MASK_BAD_VARIANCE`` where the flat
        spectrum offers no positive sample to convert the relative spectrum.

    Raises
    ------
    ValueError
        If *flat_image* does not match *image*, if *trace_positions* does not
        match *image*, if *gain* is not positive, or if *read_noise* is
        negative.

    Notes
    -----
    The prototype rounded the aperture edges with ``floor`` and ``ceil``, which
    makes the aperture width vary with the fractional trace position.  This
    implementation keeps the fixed-width aperture of
    :func:`~amasedrp.reduction.methods.boxcar.extract_boxcar` instead.
    """
    if image.shape != flat_image.shape:
        raise ValueError(
            f"flat_image shape {flat_image.shape} incompatible with "
            f"image shape {image.shape}"
        )

    n_rows, n_cols = image.shape
    n_fibers = trace_positions.shape[0]
    if trace_positions.shape != (n_fibers, n_rows):
        raise ValueError(
            f"trace_positions shape {trace_positions.shape} incompatible with "
            f"image shape {image.shape}"
        )
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    if read_noise < 0:
        raise ValueError(f"read_noise must not be negative, got {read_noise}")

    science = np.clip(image.astype(np.float64) * gain, 0.0, None)
    flat = np.clip(flat_image.astype(np.float64) * gain, 0.0, None)
    noise = science + read_noise ** 2

    weight = np.zeros_like(noise)
    np.divide(1.0, noise, out=weight, where=noise > 0)

    numerator = science * flat * weight
    denominator = flat * flat * weight

    offsets = np.arange(-aperture_radius, aperture_radius + 1)
    rows = np.arange(n_rows)

    relative = np.zeros((n_fibers, n_rows), dtype=np.float64)
    aperture_weight = np.zeros((n_fibers, n_rows), dtype=np.float64)
    out_mask = np.zeros((n_fibers, n_rows), dtype=np.uint32)

    for i in range(n_fibers):
        centers = trace_positions[i]
        traced = np.isfinite(centers)
        out_mask[i, ~traced] |= MASK_BAD_TRACE
        if not traced.any():
            continue

        row_idx = rows[traced][:, None]
        cols = np.round(centers[traced]).astype(int)[:, None] + offsets[None, :]
        in_bounds = (cols >= 0) & (cols < n_cols)
        cols = np.clip(cols, 0, n_cols - 1)

        usable = in_bounds
        if mask is not None:
            usable = usable & ~mask[row_idx, cols]

        num_sum = np.where(usable, numerator[row_idx, cols], 0.0).sum(axis=1)
        den_sum = np.where(usable, denominator[row_idx, cols], 0.0).sum(axis=1)

        good = den_sum > 0
        relative[i, rows[traced][good]] = num_sum[good] / den_sum[good]
        aperture_weight[i, rows[traced][good]] = den_sum[good]
        out_mask[i, rows[traced][~good]] |= MASK_NO_PIXELS

    flat_spectra, _, _ = extract_boxcar(
        flat, trace_positions, aperture_radius=aperture_radius, mask=mask
    )

    flux = np.zeros((n_fibers, n_rows), dtype=np.float64)
    ivar = np.zeros((n_fibers, n_rows), dtype=np.float64)
    unusable: list[int] = []

    for i in range(n_fibers):
        smoothed = _smooth_flat_spectrum(flat_spectra[i])
        if smoothed is None:
            unusable.append(i)
            out_mask[i, :] |= MASK_BAD_VARIANCE
            continue

        good = (aperture_weight[i] > 0) & (smoothed > 0)
        flux[i, good] = relative[i, good] * smoothed[good]
        ivar[i, good] = aperture_weight[i, good] / smoothed[good] ** 2

        # The aperture collected light, but the flat cannot convert it.
        unconvertible = (aperture_weight[i] > 0) & ~good
        out_mask[i, unconvertible] |= MASK_BAD_VARIANCE

    if unusable:
        warnings.warn(
            f"{len(unusable)} fiber(s) hold no positive flat spectrum "
            f"(first: fiber {unusable[0]}); their flux is left at zero.",
            RuntimeWarning,
            stacklevel=2,
        )

    return flux, ivar, out_mask
