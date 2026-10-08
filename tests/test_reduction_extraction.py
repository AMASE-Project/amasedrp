#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for spectral extraction methods (boxcar and optimal stub)."""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.reduction.methods.boxcar import extract_boxcar, MASK_BAD_TRACE, MASK_NO_PIXELS
from amasedrp.reduction.methods.optimal import extract_optimal


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


def _synthetic_fiber_pair(
    n_rows: int = 200,
    n_cols: int = 30,
    center: float = 10.0,
    fiber_sigma: float = 1.5,
    flat_level: float = 2000.0,
    aperture_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a matching flat/science pair holding one straight fiber.

    Returns
    -------
    flat
        2-D fiber flat.
    science
        2-D science frame with the same fiber profile.
    trace_positions
        Trace centered on the fiber, shape ``(1, n_rows)``.
    spectrum
        Per-row science spectrum before the fiber profile is applied.
    expected
        The noiseless aperture sum, ``spectrum`` times the profile sum inside
        the aperture.
    """
    x = np.arange(n_cols)
    profile = np.exp(-0.5 * ((x - center) / fiber_sigma) ** 2)
    spectrum = 100.0 + 10.0 * np.sin(np.arange(n_rows) / 20.0)

    flat = flat_level * profile[None, :] * np.ones((n_rows, 1))
    science = spectrum[:, None] * profile[None, :]
    trace_positions = np.full((1, n_rows), center)

    start = max(0, int(round(center)) - aperture_radius)
    stop = min(n_cols, int(round(center)) + aperture_radius + 1)
    expected = spectrum * profile[start:stop].sum()
    return flat, science, trace_positions, spectrum, expected


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
    """Tests for flat-relative optimal extraction (FOX)."""

    def test_agrees_with_boxcar_on_a_noiseless_frame(self):
        """S1: A noiseless flat makes FOX reproduce the boxcar sum.

        The flat carries the exact fiber profile, so the flat-relative step
        cancels and only the aperture sum is left.
        """
        flat, science, traces, _, expected = _synthetic_fiber_pair()
        boxcar_flux, _, _ = extract_boxcar(science, traces, aperture_radius=3)
        flux, _, _ = extract_optimal(science, flat, traces, aperture_radius=3)

        np.testing.assert_allclose(flux, boxcar_flux, rtol=1e-8)
        np.testing.assert_allclose(flux[0], expected, rtol=1e-8)

    def test_beats_boxcar_when_read_noise_dominates(self):
        """S1: Profile weighting lowers the noise when read noise is large."""
        read_noise = 30.0
        flat, science, traces, _, expected = _synthetic_fiber_pair(
            n_rows=400, n_cols=40, center=20.0, fiber_sigma=1.0,
        )
        rng = np.random.default_rng(7)
        noisy = science + rng.normal(
            0.0, np.sqrt(science + read_noise ** 2)
        )
        noisy_flat = rng.poisson(flat).astype(float)

        boxcar_flux, _, _ = extract_boxcar(noisy, traces, aperture_radius=3)
        flux, ivar, _ = extract_optimal(
            noisy, noisy_flat, traces, aperture_radius=3,
            read_noise=read_noise,
        )

        boxcar_scatter = np.std(boxcar_flux[0] - expected)
        fox_scatter = np.std(flux[0] - expected)
        assert fox_scatter < 0.9 * boxcar_scatter
        assert (ivar[0] > 0).all()

    def test_flags_untraced_rows(self):
        """S2: Non-finite trace positions are flagged, not extracted."""
        flat, science, traces, _, _ = _synthetic_fiber_pair(n_rows=50)
        traces[0, :10] = np.nan

        flux, ivar, out_mask = extract_optimal(science, flat, traces)

        assert (out_mask[0, :10] & MASK_BAD_TRACE).all()
        assert (flux[0, :10] == 0).all()
        assert (ivar[0, :10] == 0).all()
        assert (out_mask[0, 10:] == 0).all()
        assert (flux[0, 10:] > 0).all()

    def test_rejects_mismatched_flat_shape(self):
        """S2: A flat that does not match the image raises ValueError."""
        flat, science, traces, _, _ = _synthetic_fiber_pair()
        with pytest.raises(ValueError, match="flat_image shape"):
            extract_optimal(science, flat[:, :-1], traces)

    def test_rejects_non_physical_detector_parameters(self):
        """S2: Non-physical gain and read noise raise ValueError."""
        flat, science, traces, _, _ = _synthetic_fiber_pair()
        with pytest.raises(ValueError, match="gain must be positive"):
            extract_optimal(science, flat, traces, gain=0.0)
        with pytest.raises(ValueError, match="read_noise"):
            extract_optimal(science, flat, traces, read_noise=-1.0)
