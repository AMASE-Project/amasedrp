#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for profile modeling: building fiber profiles from master flats."""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.tracemask import TraceMask
from amasedrp.reduction.methods.profile_modeling import build_fiber_profile


def _synthetic_flat_with_curved_fibers(
    n_rows: int = 256,
    n_fibers: int = 5,
    fiber_sigma: float = 2.5,
    fiber_spacing: int = 15,
) -> tuple[np.ndarray, FiberMap, TraceMask]:
    """Create a synthetic flat image with slightly curved fibers."""
    n_cols = n_fibers * fiber_spacing + 50
    image = np.zeros((n_rows, n_cols), dtype=np.float64)
    x = np.arange(n_cols)
    rng = np.random.default_rng(42)

    centers = []
    for f in range(n_fibers):
        base_center = 25 + f * fiber_spacing
        # Add slight curvature: quadratic drift across rows
        row_offsets = 0.5 * ((np.arange(n_rows) - n_rows / 2) / n_rows) ** 2 * 4
        centers.append(float(base_center + row_offsets[n_rows // 2]))
        for r in range(n_rows):
            center = base_center + row_offsets[r]
            profile = np.exp(-0.5 * ((x - center) / fiber_sigma) ** 2)
            image[r, :] += profile

    image += rng.normal(0, 0.01, image.shape)

    fm = FiberMap.from_arrays(
        fiber_ids=np.arange(n_fibers),
        block_ids=np.zeros(n_fibers, dtype=int),
        approx_x=np.array(centers),
        center_row=n_rows // 2,
    )
    tracemask = TraceMask.from_fibermap(
        fibermap=fm, image=image, poly_deg=3, max_shift=2.0, cdisp_half_width=3,
    )
    return image, fm, tracemask


class TestBuildFiberProfile:
    """Tests for build_fiber_profile."""

    def test_returns_fiberprofile(self):
        """S1: build_fiber_profile returns a FiberProfile."""
        image, fm, tm = _synthetic_flat_with_curved_fibers()
        fp = build_fiber_profile(image, tm, fm, half_width=3)
        from amasedrp.reduction.core.fiberprofile import FiberProfile
        assert isinstance(fp, FiberProfile)

    def test_profile_shape(self):
        """S1: Profile shape is (n_fibers, n_rows, 2*half_width+1)."""
        image, fm, tm = _synthetic_flat_with_curved_fibers(n_rows=256, n_fibers=5)
        half_width = 4
        fp = build_fiber_profile(image, tm, fm, half_width=half_width)
        assert fp.n_fibers == 5
        assert fp.n_rows == 256
        assert fp.n_offsets == 2 * half_width + 1

    def test_profile_normalizes_to_one(self):
        """S1: Each profile row sums to 1."""
        image, fm, tm = _synthetic_flat_with_curved_fibers()
        fp = build_fiber_profile(image, tm, fm, half_width=3)
        np.testing.assert_allclose(
            fp.profile.sum(axis=-1), 1.0, atol=1e-12,
        )

    def test_profile_peaks_near_center(self):
        """S1: Profile peaks near the center offset for bright fibers."""
        image, fm, tm = _synthetic_flat_with_curved_fibers()
        fp = build_fiber_profile(image, tm, fm, half_width=3)
        center_idx = fp.n_offsets // 2
        # The peak should be at or near the center offset (within 1 pixel)
        for i in range(fp.n_fibers):
            for r in range(fp.n_rows):
                prof = fp.at(i, r)
                peak_idx = int(np.argmax(prof))
                assert abs(peak_idx - center_idx) <= 1

    def test_handles_edge_traces(self):
        """S2: Edge traces do not cause indexing errors."""
        image, fm, tm = _synthetic_flat_with_curved_fibers()
        # Use a large half_width to force edge clipping
        fp = build_fiber_profile(image, tm, fm, half_width=10)
        assert fp.n_offsets == 21
        # All rows should still normalize to 1
        np.testing.assert_allclose(
            fp.profile.sum(axis=-1), 1.0, atol=1e-12,
        )
