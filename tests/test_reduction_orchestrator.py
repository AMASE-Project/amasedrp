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

from amasedrp.reduction import identify_and_trace_fibers
from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.tracemask import TraceMask


def _synthetic_fiber_flat(
    n_rows: int = 512,
    n_blocks: int = 3,
    n_fibers_per_block: int = 5,
) -> np.ndarray:
    """Create a synthetic fiber-flat image."""
    fiber_sigma = 2.5
    fiber_spacing = 15
    block_gap = 100
    fiber_profile_width = int(6 * fiber_sigma)
    block_width = n_fibers_per_block * fiber_spacing + fiber_profile_width
    n_cols = n_blocks * block_width + (n_blocks - 1) * block_gap

    image = np.zeros((n_rows, n_cols), dtype=np.float64)
    x = np.arange(n_cols)
    col_offset = fiber_profile_width // 2

    for b in range(n_blocks):
        for f in range(n_fibers_per_block):
            center = col_offset + f * fiber_spacing
            profile = np.exp(-0.5 * ((x - center) / fiber_sigma) ** 2)
            image += profile[np.newaxis, :]
        col_offset += block_width + block_gap

    image += np.random.default_rng(42).normal(0, 0.02, image.shape)
    return image


class TestIdentifyAndTraceFibers:
    """Tests for the ``identify_and_trace_fibers`` orchestrator."""

    def test_returns_correct_types(self):
        """S1: Function returns (FiberMap, TraceMask)."""
        image = _synthetic_fiber_flat()
        fibermap, tracemask = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        assert isinstance(fibermap, FiberMap)
        assert isinstance(tracemask, TraceMask)

    def test_fiber_count_matches_expectation(self):
        """S1: Detected fiber count equals n_blocks * n_fibers_per_block."""
        image = _synthetic_fiber_flat()
        fibermap, tracemask = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        assert fibermap.n_fibers == 15
        assert tracemask.n_fibers == 15

    def test_trace_eval_shape(self):
        """S1: TraceMask.eval covers all rows of the image."""
        image = _synthetic_fiber_flat(n_rows=300)
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
        image = _synthetic_fiber_flat(n_rows=400)
        fibermap, _ = identify_and_trace_fibers(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
            center_row=200,
        )
        assert fibermap["CENTER_ROW"][0] == 200

    def test_strict_mode_rejects_bad_input(self):
        """S2: Strict mode raises ValueError on mismatch."""
        image = _synthetic_fiber_flat(n_blocks=2, n_fibers_per_block=5)
        with pytest.raises(ValueError):
            identify_and_trace_fibers(
                image=image,
                n_blocks_expected=3,  # wrong — image only has 2 blocks
                n_fibers_per_block_expected=5,
            )
