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

import warnings
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
        If ``True`` (default), raise :class:`ValueError` when the number of
        blocks, or the number of fibers inside a block, does not match the
        expectation.  If ``False``, warn instead, keep the fibers that were
        found, and set ``VALID = False`` on the fibers of the affected
        block.

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
            Valley depth threshold relative to the median profile.  A valley
            qualifies as a block edge only if it drops below this fraction
            of the median.
        fiber_peak_height_frac
            Peak height threshold relative to the median block profile.
        smooth_sigma_factor
            Gaussian smoothing sigma = block_width / (n_fibers_expected *
            smooth_sigma_factor).

        Returns
        -------
        FiberMap
            Structured table with one row per detected fiber.  Fibers of a
            block whose fiber count differs from
            *n_fibers_per_block_expected* have ``VALID = False`` whenever
            ``strict`` is ``False``.
        """
        # Step 1: extract profile
        self._extract_profile(center_row, band_half_width)

        # Step 2: identify blocks
        self._identify_blocks(threshold_frac=block_valley_threshold_frac)

        # Step 3: identify fibers within blocks
        peak_xs, peak_block_ids, peak_valid = self._identify_fibers(
            peak_height_frac=fiber_peak_height_frac,
            smooth_sigma_factor=smooth_sigma_factor,
        )

        # Build FiberMap
        n_fibers = len(peak_xs)
        fiber_ids = np.arange(n_fibers, dtype=int)
        block_ids = np.asarray(peak_block_ids, dtype=int)
        approx_x = np.asarray(peak_xs, dtype=float)
        valid = np.asarray(peak_valid, dtype=bool)
        center_row_used = self._center_row

        return FiberMap.from_arrays(
            fiber_ids=fiber_ids,
            block_ids=block_ids,
            approx_x=approx_x,
            center_row=center_row_used,
            valid=valid,
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
        xs = self._profile_xs

        # Blocks are separated by valleys in the cross-dispersion profile.
        # The valleys sit roughly half a block width apart, so a minimum
        # separation rejects the valleys that belong to individual fibers.
        # The profile is not smoothed here: smoothing fills the gaps that
        # separate the blocks.
        min_separation = len(profile) / self.n_blocks_expected / 2.0
        valleys, _ = find_peaks(-profile, distance=min_separation)
        edges = xs[valleys]

        # Keep only the valleys that fall deep enough below the median.
        threshold = np.nanmedian(profile) * threshold_frac
        edges = edges[profile[edges] <= threshold]

        # Keep only the edges whose neighbouring gap holds actual fibers.
        keep: list[int] = []
        for i in range(len(edges) - 1):
            between = np.nanmedian(profile[edges[i]:edges[i + 1]])
            if between >= threshold:
                keep += [i, i + 1]
        edges = edges[np.unique(keep)]

        n_found = len(edges) - 1
        if n_found != self.n_blocks_expected:
            msg = (
                f"Expected {self.n_blocks_expected} blocks, "
                f"but found {n_found}."
            )
            if self.strict:
                raise ValueError(msg)

        # Build block metadata
        self._blocks = []
        for i in range(max(n_found, 0)):
            self._blocks.append(
                {
                    "block_id": i,
                    "edge_left": int(edges[i]),
                    "edge_right": int(edges[i + 1]),
                    "center": float(np.mean(edges[i : i + 2])),
                }
            )

    # ------------------------------------------------------------------ #
    #  Step 3: fiber identification within blocks
    # ------------------------------------------------------------------ #

    def _identify_fibers(
        self,
        peak_height_frac: float,
        smooth_sigma_factor: float,
    ) -> tuple[list[float], list[int], list[bool]]:
        """Locate fiber peaks inside every block.

        Parameters
        ----------
        peak_height_frac
            Peak height threshold, relative to the median of the block.
        smooth_sigma_factor
            Gaussian smoothing sigma = block_width / (n_fibers_expected *
            smooth_sigma_factor).

        Returns
        -------
        peak_xs
            Cross-dispersion position of each detected fiber peak.
        peak_block_ids
            Block ID of each detected fiber peak.
        peak_valid
            ``True`` when the block holds the expected number of fibers.
            Every fiber of a block whose count differs is marked ``False``.

        Raises
        ------
        ValueError
            If ``strict`` is ``True`` and a block holds a number of fibers
            other than *n_fibers_per_block_expected*.
        """
        profile = self._profile
        xs = self._profile_xs

        peak_xs: list[float] = []
        peak_block_ids: list[int] = []
        peak_valid: list[bool] = []

        for block in self._blocks:
            bid = block["block_id"]
            lo, hi = block["edge_left"], block["edge_right"]
            block_xs = xs[lo:hi]
            block_profile = profile[lo:hi]

            peaks = np.array([], dtype=int)
            if len(block_xs) > 0:
                sigma = (
                    np.ptp(block_xs)
                    / self.n_fibers_per_block_expected
                    / smooth_sigma_factor
                )
                smoothed = gaussian_filter1d(block_profile, sigma=sigma)
                height = np.nanmedian(smoothed) * peak_height_frac
                distance = max(
                    1,
                    int(
                        np.ptp(block_xs)
                        / self.n_fibers_per_block_expected
                        * 0.5
                    ),
                )
                peaks, _ = find_peaks(
                    smoothed, height=height, distance=distance
                )

            n_found = len(peaks)
            block_valid = n_found == self.n_fibers_per_block_expected
            if not block_valid:
                msg = (
                    f"Block {bid}: expected "
                    f"{self.n_fibers_per_block_expected} fibers, "
                    f"found {n_found}."
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

            for p in peaks:
                peak_xs.append(float(block_xs[p]))
                peak_block_ids.append(bid)
                peak_valid.append(block_valid)

        return peak_xs, peak_block_ids, peak_valid
