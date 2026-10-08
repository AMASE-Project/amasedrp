#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the synthetic frames used by the tutorials and the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.simulate import synthetic_arc, synthetic_fiber_flat


class TestSyntheticFiberFlat:
    def test_every_returned_center_is_a_local_maximum(self):
        """S1: the returned centers are where the image is brightest."""
        image, centers = synthetic_fiber_flat(
            n_rows=64, n_blocks=3, n_fibers_per_block=5, noise_std=0.0
        )
        profile = image[32]
        for center in centers:
            column = int(round(center))
            local = profile[column - 2:column + 3]
            assert profile[column] == pytest.approx(local.max(), rel=1e-6)

    def test_blocks_are_separated_by_a_dark_gap(self):
        """S1: the profile between two blocks reaches the background."""
        image, centers = synthetic_fiber_flat(n_rows=64, noise_std=0.0)
        profile = image[32]
        between = int(round((centers[4] + centers[5]) / 2))
        assert profile[between] < 0.01 * profile.max()

    def test_shape_follows_the_requested_counts(self):
        """S1: n_blocks and n_fibers_per_block set the array shape."""
        image, centers = synthetic_fiber_flat(
            n_rows=100, n_blocks=2, n_fibers_per_block=4
        )
        assert image.shape[0] == 100
        assert len(centers) == 8


class TestSyntheticArc:
    def test_returns_the_exact_row_to_wavelength_mapping(self):
        """S1: the returned wavelength array is linear in the row index."""
        flat, _ = synthetic_fiber_flat(n_rows=200)
        _, wavelength = synthetic_arc(
            flat, np.array([4700.0]), wavelength_zero=4600.0, dispersion=0.5
        )
        assert wavelength.shape == (200,)
        np.testing.assert_allclose(wavelength[0], 4600.0, atol=1e-9)
        np.testing.assert_allclose(np.diff(wavelength), 0.5, atol=1e-12)

    @pytest.mark.parametrize("line", [4700.0, 4750.0])
    def test_puts_a_line_at_the_row_its_wavelength_maps_to(self, line):
        """S1: a line peaks at the row that (wl - zero) / dispersion gives."""
        flat, centers = synthetic_fiber_flat(n_rows=400, noise_std=0.0)
        image, _ = synthetic_arc(
            flat, np.array([line]),
            wavelength_zero=4600.0, dispersion=0.5, line_fwhm=0.8,
        )
        spectrum = image[:, int(round(centers[7]))]
        expected = int(round((line - 4600.0) / 0.5))
        assert abs(int(np.argmax(spectrum)) - expected) <= 2

    def test_line_width_scales_with_the_requested_fwhm(self):
        """S1: line_fwhm is in Angstrom, so a wider line spans more rows."""
        flat, centers = synthetic_fiber_flat(n_rows=2000, noise_std=0.0)
        column = int(round(centers[7]))
        widths = []
        for fwhm in (0.5, 2.0):
            image, _ = synthetic_arc(
                flat, np.array([4650.0]), wavelength_zero=4600.0,
                dispersion=0.05, line_fwhm=fwhm,
            )
            spectrum = image[:, column]
            widths.append(int(np.count_nonzero(spectrum > 0.5 * spectrum.max())))
        # four times wider in Angstrom must be four times wider in rows
        assert widths[1] == pytest.approx(4 * widths[0], rel=0.15)

    def test_keeps_the_dark_gaps_dark(self):
        """S1: the arc is multiplied into the flat, so gaps stay at zero."""
        flat, centers = synthetic_fiber_flat(n_rows=100, noise_std=0.0)
        image, _ = synthetic_arc(flat, np.array([4700.0]))
        between = int(round((centers[4] + centers[5]) / 2))
        assert np.all(image[:, between] == 0.0)

    def test_matches_the_flat_shape(self):
        """S1: the arc takes the shape of the flat."""
        flat, _ = synthetic_fiber_flat(n_rows=300, n_blocks=2)
        image, _ = synthetic_arc(flat, np.array([4700.0]))
        assert image.shape == flat.shape

    def test_rejects_a_flat_that_is_not_two_dimensional(self):
        """S2: a 1-D flat raises ValueError."""
        with pytest.raises(ValueError, match="must be 2-D"):
            synthetic_arc(np.zeros(10), np.array([4700.0]))

    def test_rejects_a_zero_dispersion(self):
        """S2: a zero dispersion raises ValueError."""
        flat, _ = synthetic_fiber_flat(n_rows=50)
        with pytest.raises(ValueError, match="dispersion"):
            synthetic_arc(flat, np.array([4700.0]), dispersion=0.0)

    def test_noise_is_reproducible_from_the_seed(self):
        """S2: the same seed gives the same frame."""
        flat, _ = synthetic_fiber_flat(n_rows=50, noise_std=0.0)
        first, _ = synthetic_arc(flat, np.array([4700.0]), noise_std=0.1, seed=7)
        second, _ = synthetic_arc(flat, np.array([4700.0]), noise_std=0.1, seed=7)
        np.testing.assert_array_equal(first, second)
