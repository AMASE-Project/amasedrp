#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `GEOMSRC`, the card that says what a product's geometry rests on.

The value comes from amasecore's own ``status`` field rather than from a
constant here, so that a change of data changes the products without anyone
remembering to edit code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from amasedrp.reduction import reduction
from amasedrp.reduction.reduction import _resolve_layout, _slit_layout
from amasedrp.reduction.reduction import run_quick_reduction, run_reduction
from amasedrp.simulate import synthetic_fiber_flat

# The simulator's geometry, small enough to reduce in a test.
BLOCKS = 2
FIBERS = 5


class TestTheLayoutCarriesItsProvenance:
    """amasecore declares the status, so the reader must pass it on."""

    def test_the_shipped_layout_is_a_design(self) -> None:
        assert _slit_layout() == (22, 25, "design")

    def test_defaults_take_both_the_numbers_and_the_status(self) -> None:
        assert _resolve_layout(None, None) == (22, 25, "design")

    def test_a_caller_supplied_layout_has_no_status(self) -> None:
        assert _resolve_layout(19, 29) == (19, 29, None)

    def test_one_number_given_still_borrows_the_status(self) -> None:
        assert _resolve_layout(19, None) == (19, 25, "design")


def _reduce(**kwargs):
    """Reduce a small synthetic frame and return the FiberFrame."""
    flat, _ = synthetic_fiber_flat(n_rows=128, n_blocks=BLOCKS, n_fibers_per_block=FIBERS)
    science = flat + np.random.default_rng(7).normal(0, 0.01, flat.shape)
    return run_quick_reduction(image=science, flat_image=flat, **kwargs)


class TestTheProductRecordsIt:
    """A product built from amasecore's layout says so."""

    def test_the_frame_carries_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            reduction, "_slit_layout", lambda bundle=reduction.DEFAULT_BUNDLE: (BLOCKS, FIBERS, "design")
        )
        frame = _reduce()
        assert frame.meta["GEOMSRC"] == "design"

    def test_a_written_product_carries_the_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            reduction, "_slit_layout", lambda bundle=reduction.DEFAULT_BUNDLE: (BLOCKS, FIBERS, "design")
        )
        out = tmp_path / "frame.fits"
        _reduce().to_fits(out)
        assert fits.getheader(out)["GEOMSRC"] == "design"

    def test_the_full_reduction_records_it_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            reduction, "_slit_layout", lambda bundle=reduction.DEFAULT_BUNDLE: (BLOCKS, FIBERS, "design")
        )
        flat, _ = synthetic_fiber_flat(n_rows=128, n_blocks=BLOCKS, n_fibers_per_block=FIBERS)
        science = flat + np.random.default_rng(7).normal(0, 0.01, flat.shape)
        frame = run_reduction(image=science, flat_image=flat, method="boxcar")
        assert frame.meta["GEOMSRC"] == "design"

    def test_a_caller_supplied_layout_claims_nothing(self) -> None:
        frame = _reduce(
            n_blocks_expected=BLOCKS, n_fibers_per_block_expected=FIBERS
        )
        assert "GEOMSRC" not in frame.meta

    def test_the_callers_metadata_is_not_modified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            reduction, "_slit_layout", lambda bundle=reduction.DEFAULT_BUNDLE: (BLOCKS, FIBERS, "design")
        )
        given = {"EXPTIME": 60.0}
        frame = _reduce(meta=given)
        assert given == {"EXPTIME": 60.0}, "the caller's dict must not be written into"
        assert frame.meta["EXPTIME"] == pytest.approx(60.0)
        assert frame.meta["GEOMSRC"] == "design"
