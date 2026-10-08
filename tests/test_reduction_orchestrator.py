#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TDD test for the reduction orchestrator.

This test validates ``identify_and_trace_fibers`` — the user-facing
one-shot function that chains block detection → fiber detection →
tracing → polynomial fitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.reduction import identify_and_trace_fibers, extract_spectra, run_quick_reduction, run_reduction
from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.tracemask import TraceMask
from amasedrp.reduction.core.fiberframe import FiberFrame

# Shared synthetic fiber flat; see tests/synthetic.py for the geometry.
from synthetic import synthetic_fiber_flat as _synthetic_fiber_flat


class TestIdentifyAndTraceFibers:
    """Tests for the ``identify_and_trace_fibers`` orchestrator."""

    def test_returns_correct_types(self):
        """S1: Function returns (FiberMap, TraceMask)."""
        image, _ = _synthetic_fiber_flat()
        fibermap, tracemask = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        assert isinstance(fibermap, FiberMap)
        assert isinstance(tracemask, TraceMask)

    def test_fiber_count_matches_expectation(self):
        """S1: Detected fiber count equals n_blocks * n_fibers_per_block."""
        image, _ = _synthetic_fiber_flat()
        fibermap, tracemask = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        assert fibermap.n_fibers == 15
        assert tracemask.n_fibers == 15

    def test_trace_eval_shape(self):
        """S1: TraceMask.eval covers all rows of the image."""
        image, _ = _synthetic_fiber_flat(n_rows=300)
        fibermap, tracemask = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        rows = np.arange(image.shape[0])
        positions = tracemask.eval(rows)
        assert positions.shape == (15, 300)
        assert np.isfinite(positions).all()

    def test_custom_center_row(self):
        """S1: Custom center_row is respected."""
        image, _ = _synthetic_fiber_flat(n_rows=400)
        fibermap, _ = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
            center_row=200,
        )
        assert fibermap["CENTER_ROW"][0] == 200

    def test_strict_mode_rejects_bad_input(self):
        """S2: Strict mode raises ValueError on mismatch."""
        image, _ = _synthetic_fiber_flat(n_blocks=2, n_fibers_per_block=5)
        with pytest.raises(ValueError):
            identify_and_trace_fibers(
                image=image,
                n_blocks_expected=3,  # wrong — image only has 2 blocks
                n_fibers_per_block_expected=5,
            )


# ---------------------------------------------------------------------------
# extract_spectra tests
# ---------------------------------------------------------------------------

class TestExtractSpectra:
    """Tests for the extract_spectra orchestrator."""

    def _make_flat_and_science(self, n_rows=256, n_blocks=2, n_fibers_per_block=5):
        flat, _ = _synthetic_fiber_flat(n_rows=n_rows, n_blocks=n_blocks, n_fibers_per_block=n_fibers_per_block)
        # Science = flat + small noise
        science = flat + np.random.default_rng(123).normal(0, 0.01, flat.shape)
        return flat, science

    def test_boxcar_returns_fiberframe(self):
        """S1: extract_spectra with boxcar returns a FiberFrame."""
        flat, science = self._make_flat_and_science()
        fibermap, tracemask = identify_and_trace_fibers(
            image=flat, n_blocks_expected=2, n_fibers_per_block_expected=5,
        )
        frame = extract_spectra(
            image=science, tracemask=tracemask, fibermap=fibermap, method="boxcar",
        )
        assert isinstance(frame, FiberFrame)
        assert frame.n_fibers == 10
        assert frame.flux.ndim == 2

    def test_optimal_returns_fiberframe(self):
        """S1: extract_spectra with optimal returns a FiberFrame."""
        flat, science = self._make_flat_and_science()
        fibermap, tracemask = identify_and_trace_fibers(
            image=flat, n_blocks_expected=2, n_fibers_per_block_expected=5,
            strict=False,
        )
        frame = extract_spectra(
            image=science, tracemask=tracemask, fibermap=fibermap,
            method="optimal", flat_image=flat,
        )
        assert isinstance(frame, FiberFrame)
        assert frame.n_fibers == 10
        assert frame.flux.ndim == 2
        assert frame.meta["METHOD"] == "optimal"

    def test_optimal_requires_flat(self):
        """S2: Calling optimal without flat_image raises ValueError."""
        flat, science = self._make_flat_and_science()
        fibermap, tracemask = identify_and_trace_fibers(
            image=flat, n_blocks_expected=2, n_fibers_per_block_expected=5,
            strict=False,
        )
        with pytest.raises(ValueError, match="requires flat_image"):
            extract_spectra(
                image=science, tracemask=tracemask, fibermap=fibermap,
                method="optimal",
            )

    def test_optimal_rejects_variance_image(self):
        """S2: FOX builds its own noise model, so variance is refused."""
        flat, science = self._make_flat_and_science()
        fibermap, tracemask = identify_and_trace_fibers(
            image=flat, n_blocks_expected=2, n_fibers_per_block_expected=5,
            strict=False,
        )
        with pytest.raises(ValueError, match="own noise model"):
            extract_spectra(
                image=science, tracemask=tracemask, fibermap=fibermap,
                method="optimal", flat_image=flat,
                variance=np.ones_like(science),
            )

    def test_rejects_unknown_method(self):
        """S2: Unknown method raises ValueError."""
        flat, science = self._make_flat_and_science()
        fibermap, tracemask = identify_and_trace_fibers(
            image=flat, n_blocks_expected=2, n_fibers_per_block_expected=5,
        )
        with pytest.raises(ValueError):
            extract_spectra(
                image=science, tracemask=tracemask, fibermap=fibermap, method="magic",
            )


# ---------------------------------------------------------------------------
# run_quick_reduction tests
# ---------------------------------------------------------------------------

class TestRunQuickReduction:
    """Tests for the quick-look reduction pipeline."""

    def test_returns_fiberframe(self):
        """S1: run_quick_reduction returns a FiberFrame."""
        flat, _ = _synthetic_fiber_flat(n_rows=256, n_blocks=2, n_fibers_per_block=5)
        science = flat + np.random.default_rng(123).normal(0, 0.01, flat.shape)
        frame = run_quick_reduction(
            image=science,
            flat_image=flat,
            n_blocks_expected=2,
            n_fibers_per_block_expected=5,
        )
        assert isinstance(frame, FiberFrame)
        assert frame.n_fibers == 10
        assert frame.flux.ndim == 2


# ---------------------------------------------------------------------------
# run_reduction tests
# ---------------------------------------------------------------------------

class TestRunReduction:
    """Tests for the full reduction pipeline."""

    def test_boxcar_method_returns_fiberframe(self):
        """S1: run_reduction with boxcar returns a FiberFrame."""
        flat, _ = _synthetic_fiber_flat(n_rows=256, n_blocks=2, n_fibers_per_block=5)
        science = flat + np.random.default_rng(123).normal(0, 0.01, flat.shape)
        frame = run_reduction(
            image=science,
            flat_image=flat,
            n_blocks_expected=2,
            n_fibers_per_block_expected=5,
            method="boxcar",
        )
        assert isinstance(frame, FiberFrame)
        assert frame.n_fibers == 10

    def test_optimal_method_returns_fiberframe(self):
        """S1: run_reduction with optimal returns a FiberFrame."""
        flat, _ = _synthetic_fiber_flat(n_rows=256, n_blocks=2, n_fibers_per_block=5)
        science = flat + np.random.default_rng(123).normal(0, 0.01, flat.shape)
        frame = run_reduction(
            image=science,
            flat_image=flat,
            n_blocks_expected=2,
            n_fibers_per_block_expected=5,
            method="optimal",
            strict=False,
        )
        assert isinstance(frame, FiberFrame)
        assert frame.n_fibers == 10
        assert frame.meta["METHOD"] == "optimal"

    def test_rejects_unknown_method(self):
        """S2: Unknown method raises ValueError."""
        flat, _ = _synthetic_fiber_flat(n_rows=256, n_blocks=2, n_fibers_per_block=5)
        science = flat + np.random.default_rng(123).normal(0, 0.01, flat.shape)
        with pytest.raises(ValueError):
            run_reduction(
                image=science,
                flat_image=flat,
                n_blocks_expected=2,
                n_fibers_per_block_expected=5,
                method="magic",
            )
