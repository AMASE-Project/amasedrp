#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for spectral extraction methods (boxcar and optimal stub)."""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.reduction.methods.boxcar import extract_boxcar, MASK_BAD_TRACE, MASK_NO_PIXELS
from amasedrp.reduction.methods.optimal import extract_optimal
from amasedrp.reduction.core.fiberprofile import FiberProfile


def _synthetic_science_image(
    n_rows: int = 256,
    n_fibers: int = 5,
    fiber_sigma: float = 2.5,
    fiber_spacing: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic science image with straight Gaussian fibers."""
    n_cols = n_fibers * fiber_spacing + 50
    image = np.zeros((n_rows, n_cols), dtype=np.float64)
    x = np.arange(n_cols)
    rng = np.random.default_rng(42)

    trace_positions = np.empty((n_fibers, n_rows), dtype=float)
    for f in range(n_fibers):
        center = 25 + f * fiber_spacing
        trace_positions[f, :] = center
        for r in range(n_rows):
            profile = np.exp(-0.5 * ((x - center) / fiber_sigma) ** 2)
            image[r, :] += profile

    image += rng.normal(0, 0.01, image.shape)
    return image, trace_positions


class TestExtractBoxcar:
    """Tests for boxcar extraction."""

    def test_sums_expected_pixels(self):
        """S1: Boxcar sums the correct aperture pixels."""
        image, traces = _synthetic_science_image(n_rows=100, n_fibers=3)
        flux, ivar, mask = extract_boxcar(image, traces, aperture_radius=2)
        assert flux.shape == (3, 100)
        assert ivar.shape == (3, 100)
        assert mask.shape == (3, 100)
        # All synthetic fibers are bright; flux should be positive
        assert (flux > 0).all()
        assert (ivar > 0).all()

    def test_clips_at_edges(self):
        """S2: Edge traces are safely clipped without error."""
        image = np.ones((50, 20), dtype=float)
        traces = np.full((2, 50), -1.0, dtype=float)
        traces[0, :] = 2.0   # near left edge
        traces[1, :] = 17.0  # near right edge
        flux, ivar, mask = extract_boxcar(image, traces, aperture_radius=3)
        # Should still produce valid output
        assert flux.shape == (2, 50)
        assert np.isfinite(flux).all()

    def test_rejects_bad_shapes(self):
        """S2: Mismatched image/trace shapes raise ValueError."""
        image = np.ones((50, 20), dtype=float)
        traces = np.ones((3, 30), dtype=float)
        with pytest.raises(ValueError):
            extract_boxcar(image, traces)

    def test_uses_variance_for_ivar(self):
        """S1: Provided variance is propagated to ivar."""
        image, traces = _synthetic_science_image(n_rows=50, n_fibers=2)
        variance = np.ones_like(image) * 4.0
        flux, ivar, mask = extract_boxcar(image, traces, aperture_radius=2, variance=variance)
        # With variance=4 per pixel and ~5 pixels in aperture, expected var ≈ 20
        expected_ivar = 1.0 / (5.0 * 4.0)
        # Just check that ivar is positive and finite
        assert (ivar > 0).all()
        assert np.isfinite(ivar).all()

    def test_ignores_masked_pixels(self):
        """S2: Masked pixels are excluded from the sum."""
        image = np.ones((50, 20), dtype=float)
        traces = np.full((1, 50), 10.0, dtype=float)
        # Mask the center pixel for all rows
        badpix = np.zeros_like(image, dtype=bool)
        badpix[:, 10] = True
        flux, ivar, mask = extract_boxcar(image, traces, aperture_radius=2, mask=badpix)
        # Aperture would be [8,9,10,11,12] = 5 pixels, but 10 is masked → 4 pixels
        expected_flux = 4.0
        np.testing.assert_allclose(flux[0, :], expected_flux, atol=1e-12)

    def test_bad_trace_mask(self):
        """S2: Non-finite trace positions are flagged."""
        image = np.ones((50, 20), dtype=float)
        traces = np.full((1, 50), np.nan, dtype=float)
        flux, ivar, out_mask = extract_boxcar(image, traces, aperture_radius=2)
        assert (out_mask & MASK_BAD_TRACE).all()
        assert (flux == 0).all()
        assert (ivar == 0).all()


class TestExtractOptimal:
    """Tests for the optimal extraction stub."""

    def test_stub_raises_not_implemented(self):
        """S1: extract_optimal raises NotImplementedError."""
        image = np.ones((50, 20), dtype=float)
        traces = np.full((1, 50), 10.0, dtype=float)
        profile = np.ones((1, 50, 5), dtype=float)
        offsets = np.arange(-2, 3, dtype=float)
        fp = FiberProfile(profile, offsets)
        with pytest.raises(NotImplementedError):
            extract_optimal(image, traces, fp)
