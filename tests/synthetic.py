#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic fiber-flat images shared by the reduction tests.

The geometry follows the fiber flat measured on the 2025-07 collimator-sweep
frames: fibers sit ~8 px apart with a ~2 px Gaussian width, so the 29 fibers
of one block merge into a single bright strip, and neighbouring blocks are
separated by a narrow (~26 px) gap that reaches the background.

Block detection separates blocks by the depth of the valleys in the
cross-dispersion profile.  That criterion only works when the fibers inside a
block are close enough to keep the intra-block valleys bright.  A synthetic
flat with sparser fibers, or with a gap wider than a block, does not have that
property and does not represent the instrument.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["synthetic_fiber_flat"]

# Fiber-flat geometry measured on the 9600 x 6422 sweep frames.
FIBER_SIGMA = 2.0
FIBER_SPACING = 8.0
BLOCK_GAP = 26.0
MARGIN = 7.0


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

    Each fiber is a Gaussian in the cross-dispersion direction, and constant
    along the dispersion direction.  Fibers are grouped into blocks.  A dark
    margin brackets the whole array, and a dark gap separates two blocks.

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
