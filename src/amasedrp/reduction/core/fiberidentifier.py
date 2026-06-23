#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         fiberidentifier.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Identify fiber blocks and individual fibers from a fiber-flat
              image.

This module separates *identification* ("where are the fibers?") from
*tracing* ("how do they move along the dispersion direction?").  The latter
is handled by :class:`TraceMask`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .fibermap import FiberMap

__all__ = ["FibersIdentifier"]


class FibersIdentifier:
    """Identify fiber blocks and individual fibers in a fiber-flat image.

    The algorithm works in three steps:

    1. **Profile extraction** — median-collapse a band around the centre row
       to obtain a 1-D cross-dispersion profile.
    2. **Block detection** — find the valleys (dips) between blocks and
       validate the count against the expected number.
    3. **Fiber peak detection** — within each block, smooth the profile and
       locate individual fiber peaks.

    Parameters
    ----------
    image
        2-D fiber-flat image (spectral × spatial).
    n_blocks_expected
        Expected number of fiber blocks (e.g. from instrument metadata).
    n_fibers_per_block_expected
        Expected number of fibers per block.
    strict
        If ``True`` (default), raise :class:`ValueError` when the detected
        number of blocks or fibers does not match the expectation.

    Examples
    --------
    >>> identifier = FibersIdentifier(
    ...     image=fflat,
    ...     n_blocks_expected=19,
    ...     n_fibers_per_block_expected=29,
    ... )
    >>> fibermap = identifier.identify(center_row=1024, band_half_width=100)
    >>> fibermap.n_fibers
    551
    """

    def __init__(
        self,
        image: NDArray[np.floating],
        n_blocks_expected: int,
        n_fibers_per_block_expected: int,
        strict: bool = True,
    ) -> None:
        self.image = image
        self.n_blocks_expected = n_blocks_expected
        self.n_fibers_per_block_expected = n_fibers_per_block_expected
        self.strict = strict

        # Internal state populated by identify()
        self._profile: NDArray[np.floating] | None = None
        self._profile_xs: NDArray[np.integer] | None = None
        self._center_row: int | None = None
        self._blocks: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def identify(
        self,
        center_row: int | None = None,
        band_half_width: int = 100,
        block_valley_threshold_frac: float = 0.3,
        fiber_peak_height_frac: float = 0.5,
        smooth_sigma_factor: float = 10.0,
    ) -> FiberMap:
        """Run the full identification pipeline.

        Parameters
        ----------
        center_row
            Row around which to extract the cross-dispersion profile.
            Defaults to ``image.shape[0] // 2``.
        band_half_width
            Half-height of the band (in rows) collapsed to form the profile.
        block_valley_threshold_frac
            Valley depth threshold relative to the median profile.
        fiber_peak_height_frac
            Peak height threshold relative to the median block profile.
        smooth_sigma_factor
            Gaussian smoothing sigma = block_width / (n_fibers_expected *
            smooth_sigma_factor).

        Returns
        -------
        FiberMap
            Structured table with one row per detected fiber.
        """
        # Step 1: extract profile
        self._extract_profile(center_row, band_half_width)

        # Step 2: identify blocks
        self._identify_blocks(threshold_frac=block_valley_threshold_frac)

        # Step 3: identify fibers within blocks
        peak_xs, peak_block_ids = self._identify_fibers(
            peak_height_frac=fiber_peak_height_frac,
            smooth_sigma_factor=smooth_sigma_factor,
        )

        # Build FiberMap
        n_fibers = len(peak_xs)
        fiber_ids = np.arange(n_fibers, dtype=int)
        block_ids = np.asarray(peak_block_ids, dtype=int)
        approx_x = np.asarray(peak_xs, dtype=float)
        center_row_used = self._center_row

        return FiberMap.from_arrays(
            fiber_ids=fiber_ids,
            block_ids=block_ids,
            approx_x=approx_x,
            center_row=center_row_used,
        )

    # ------------------------------------------------------------------ #
    #  Step 1: profile extraction
    # ------------------------------------------------------------------ #

    def _extract_profile(
        self,
        center_row: int | None,
        band_half_width: int,
    ) -> None:
        if center_row is None:
            center_row = self.image.shape[0] // 2
        self._center_row = int(center_row)

        start = max(0, self._center_row - band_half_width)
        end = min(self.image.shape[0], self._center_row + band_half_width)
        band = self.image[start:end, :]

        self._profile = np.nanmedian(band, axis=0).astype(float)
        self._profile_xs = np.arange(len(self._profile), dtype=int)

    # ------------------------------------------------------------------ #
    #  Step 2: block identification
    # ------------------------------------------------------------------ #

    def _identify_blocks(self, threshold_frac: float) -> None:
        profile = self._profile

        block_width_estimate = len(profile) / self.n_blocks_expected
        # sigma = block_width/6 suppresses noise while keeping edges sharp
        smoothed = gaussian_filter1d(profile, sigma=block_width_estimate / 6)

        threshold = np.nanmax(smoothed) * threshold_frac
        above = smoothed > threshold
        transitions = np.where(np.diff(above.astype(int)) != 0)[0] + 1

        if above[0]:
            transitions = np.concatenate([[0], transitions])
        if above[-1]:
            transitions = np.concatenate([transitions, [len(profile) - 1]])

        n_found = len(transitions) // 2
        if n_found != self.n_blocks_expected:
            msg = (
                f"Expected {self.n_blocks_expected} blocks, "
                f"but found {n_found}."
            )
            if self.strict:
                raise ValueError(msg)

        # Build block metadata
        self._blocks = []
        for i in range(n_found):
            self._blocks.append(
                {
                    "block_id": i,
                    "edge_left": int(transitions[2 * i]),
                    "edge_right": int(transitions[2 * i + 1]),
                    "center": float(np.mean(transitions[2 * i : 2 * i + 2])),
                }
            )

    # ------------------------------------------------------------------ #
    #  Step 3: fiber identification within blocks
    # ------------------------------------------------------------------ #

    def _identify_fibers(
        self,
        peak_height_frac: float,
        smooth_sigma_factor: float,
    ) -> tuple[list[float], list[int]]:
        """Return (peak_x_positions, block_ids_for_each_peak)."""
        profile = self._profile
        xs = self._profile_xs

        peak_xs: list[float] = []
        peak_block_ids: list[int] = []

        for block in self._blocks:
            bid = block["block_id"]
            lo, hi = block["edge_left"], block["edge_right"]
            block_xs = xs[lo:hi]
            block_profile = profile[lo:hi]

            if len(block_xs) == 0:
                continue

            # Smooth
            sigma = (
                np.ptp(block_xs)
                / self.n_fibers_per_block_expected
                / smooth_sigma_factor
            )
            smoothed = gaussian_filter1d(block_profile, sigma=sigma)

            # Find peaks
            height = np.nanmedian(smoothed) * peak_height_frac
            distance = max(1, int(np.ptp(block_xs) / self.n_fibers_per_block_expected * 0.5))
            peaks, _ = find_peaks(smoothed, height=height, distance=distance)

            n_found = len(peaks)
            if n_found != self.n_fibers_per_block_expected and self.strict:
                raise ValueError(
                    f"Block {bid}: expected {self.n_fibers_per_block_expected} "
                    f"fibers, found {n_found}."
                )

            for p in peaks:
                peak_xs.append(float(block_xs[p]))
                peak_block_ids.append(bid)

        return peak_xs, peak_block_ids
