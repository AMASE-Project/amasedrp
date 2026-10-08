#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RED-phase pytest suite for reduction/core modules.

These tests are written against the *target* API. They will fail until
fibermap.py, fiberidentifier.py, and tracemask.py are implemented.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

# ---------------------------------------------------------------------------
# Imports to test (will fail until implemented)
# ---------------------------------------------------------------------------
from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.fiberidentifier import FibersIdentifier
from amasedrp.reduction.core.tracemask import TraceMask
from amasedrp.reduction.core.fiberframe import FiberFrame

# Shared synthetic fiber flat; see tests/synthetic.py for the geometry.
from synthetic import synthetic_fiber_flat as _synthetic_fiber_flat


# ---------------------------------------------------------------------------
# FiberMap tests
# ---------------------------------------------------------------------------

class TestFiberMap:
    """Tests for the FiberMap data model."""

    def test_create_from_arrays(self):
        """S1: FiberMap can be created from arrays and exposes columns."""
        n_fibers = 15
        fiber_ids = np.arange(n_fibers)
        block_ids = np.repeat([0, 1, 2], 5)
        approx_x = np.linspace(100, 400, n_fibers)
        center_row = 1024

        fm = FiberMap.from_arrays(
            fiber_ids=fiber_ids,
            block_ids=block_ids,
            approx_x=approx_x,
            center_row=center_row,
        )
        assert len(fm) == n_fibers
        assert list(fm.colnames) == [
            "FIBERID", "BLOCKID", "BLOCK_LOCAL_ID", "APPROX_X",
            "CENTER_ROW", "VALID",
        ]
        assert fm["FIBERID"][0] == 0
        assert fm["BLOCKID"][5] == 1
        assert fm["VALID"].all()

    def test_block_local_id_auto(self):
        """S1: BLOCK_LOCAL_ID is auto-assigned per block."""
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(6),
            block_ids=np.array([0, 0, 0, 1, 1, 1]),
            approx_x=np.arange(6),
            center_row=512,
        )
        assert list(fm["BLOCK_LOCAL_ID"]) == [0, 1, 2, 0, 1, 2]

    def test_get_block(self):
        """S1: Can extract a sub-table for one block."""
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(6),
            block_ids=np.array([0, 0, 0, 1, 1, 1]),
            approx_x=np.arange(6),
            center_row=512,
        )
        block0 = fm.get_block(0)
        assert len(block0) == 3
        assert all(block0["BLOCKID"] == 0)

    def test_mark_invalid(self):
        """S2: Individual fibers can be marked invalid."""
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(3),
            block_ids=np.array([0, 0, 0]),
            approx_x=np.arange(3),
            center_row=512,
        )
        fm.mark_invalid([1])
        assert fm["VALID"][0] == True
        assert fm["VALID"][1] == False
        assert fm["VALID"][2] == True

    def test_n_fibers_property(self):
        """S1: n_fibers property returns row count."""
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(10),
            block_ids=np.zeros(10, dtype=int),
            approx_x=np.arange(10),
            center_row=512,
        )
        assert fm.n_fibers == 10


# ---------------------------------------------------------------------------
# FibersIdentifier tests
# ---------------------------------------------------------------------------

class TestFibersIdentifier:
    """Tests for FibersIdentifier block+fiber detection."""

    def test_identify_blocks_and_fibers(self):
        """S1: Identify correct number of blocks and fibers."""
        image, _ = _synthetic_fiber_flat(
            n_rows=512, n_blocks=3, n_fibers_per_block=5,
        )
        identifier = FibersIdentifier(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        fibermap = identifier.identify(center_row=256, band_half_width=50)

        assert fibermap.n_fibers == 15
        assert len(np.unique(fibermap["BLOCKID"])) == 3

    def test_strict_mode_raises_on_fiber_count_mismatch(self):
        """S2: Strict mode raises when a block holds too few fibers."""
        # Three blocks of 4 fibers, but 5 per block are expected.
        image, _ = _synthetic_fiber_flat(n_fibers_per_block=4)
        identifier = FibersIdentifier(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
            strict=True,
        )
        with pytest.raises(ValueError, match="expected 5 fibers, found 4"):
            identifier.identify()

    def test_non_strict_warns_and_marks_block_invalid(self):
        """S2: Non-strict mode warns and flags the affected fibers."""
        # Real fiber flats always miss a few fibers, so the run must not stop;
        # the caller needs the QA flag instead.
        image, _ = _synthetic_fiber_flat(n_fibers_per_block=4)
        identifier = FibersIdentifier(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
            strict=False,
        )
        with pytest.warns(RuntimeWarning) as record:
            fibermap = identifier.identify()

        # One warning per block, naming the block and the count.
        assert len(record) == 3
        assert "Block 0: expected 5 fibers, found 4." == str(
            record[0].message
        )

        # The fibers that were found are kept, and every block is flagged.
        assert fibermap.n_fibers == 12
        assert len(np.unique(fibermap["BLOCKID"])) == 3
        assert not fibermap["VALID"].any()

    def test_center_row_defaults_to_middle(self):
        """S1: Default center_row is image.shape[0] // 2."""
        image, _ = _synthetic_fiber_flat(n_rows=300)
        identifier = FibersIdentifier(
            image=image, n_blocks_expected=3, n_fibers_per_block_expected=5,
        )
        fibermap = identifier.identify()
        assert fibermap["CENTER_ROW"][0] == 150

    def test_fibermap_has_valid_column(self):
        """S1: Output FiberMap has VALID column set to True."""
        image, _ = _synthetic_fiber_flat()
        identifier = FibersIdentifier(
            image=image, n_blocks_expected=3, n_fibers_per_block_expected=5,
        )
        fibermap = identifier.identify()
        assert "VALID" in fibermap.colnames
        assert fibermap["VALID"].dtype == bool
        assert fibermap["VALID"].all()


# ---------------------------------------------------------------------------
# TraceMask tests
# ---------------------------------------------------------------------------

class TestTraceMask:
    """Tests for TraceMask tracing and polynomial fitting."""

    def test_trace_and_fit(self):
        """S1: Trace fibers and fit polynomial; eval reproduces trace."""
        image, centers = _synthetic_fiber_flat(
            n_rows=512, n_blocks=3, n_fibers_per_block=5,
        )
        # Build a simple FiberMap manually using true centers
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(15),
            block_ids=np.repeat([0, 1, 2], 5),
            approx_x=np.array(centers),
            center_row=256,
        )

        tracemask = TraceMask.from_fibermap(
            fibermap=fm,
            image=image,
            poly_deg=3,
            max_shift=2.0,
            cdisp_half_width=3,
        )

        # TraceMask should have 15 fibers
        assert tracemask.n_fibers == 15
        # Evaluating at the center row should give positions close to approx_x
        rows = np.array([256])
        positions = tracemask.eval(rows)
        assert positions.shape == (15, 1)
        np.testing.assert_allclose(
            positions[:, 0], fm["APPROX_X"], atol=1.0,
        )

    def test_eval_shape(self):
        """S1: eval() returns (n_fibers, n_rows) array."""
        image, centers = _synthetic_fiber_flat(n_rows=512)
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(15),
            block_ids=np.repeat([0, 1, 2], 5),
            approx_x=np.array(centers),
            center_row=256,
        )
        tracemask = TraceMask.from_fibermap(
            fibermap=fm, image=image, poly_deg=3,
        )
        rows = np.arange(512)
        positions = tracemask.eval(rows)
        assert positions.shape == (15, 512)

    def test_eval_all_rows_finite(self):
        """S3: For straight synthetic fibers, all eval positions are finite."""
        image, centers = _synthetic_fiber_flat(n_rows=512)
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(15),
            block_ids=np.repeat([0, 1, 2], 5),
            approx_x=np.array(centers),
            center_row=256,
        )
        tracemask = TraceMask.from_fibermap(
            fibermap=fm, image=image, poly_deg=3,
        )
        rows = np.arange(512)
        positions = tracemask.eval(rows)
        assert np.isfinite(positions).all()

    def test_fiber_id_ordering(self):
        """S1: TraceMask fiber_ids match FiberMap ordering."""
        image, centers = _synthetic_fiber_flat(n_rows=512)
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(15),
            block_ids=np.repeat([0, 1, 2], 5),
            approx_x=np.array(centers),
            center_row=256,
        )
        tracemask = TraceMask.from_fibermap(
            fibermap=fm, image=image, poly_deg=3,
        )
        np.testing.assert_array_equal(tracemask.fiber_ids, fm["FIBERID"])

    def test_partial_trace_keeps_its_own_domain(self):
        """S2: A partly traced fiber is evaluated on its own fit domain.

        A fiber is fitted over the rows where it was actually traced, so the
        fitted row range has to travel with the coefficients.  Evaluating the
        same coefficients over the full image row range shifts the trace by
        several pixels.
        """
        n_rows, n_cols = 1000, 360
        image = np.zeros((n_rows, n_cols), dtype=np.float64)
        x = np.arange(n_cols)

        # One curved fiber that only exists on rows 100..200.
        rows_valid = np.arange(100, 201)
        true_center = np.full(n_rows, np.nan)
        true_center[rows_valid] = (
            300.0 + 0.05 * rows_valid + 1e-4 * rows_valid ** 2
        )
        for row in rows_valid:
            image[row, :] = np.exp(-0.5 * ((x - true_center[row]) / 2.0) ** 2)

        fm = FiberMap.from_arrays(
            fiber_ids=np.array([0]),
            block_ids=np.array([0]),
            approx_x=np.array([float(true_center[150])]),
            center_row=150,
        )
        tracemask = TraceMask.from_fibermap(
            fibermap=fm,
            image=image,
            poly_deg=4,
            max_shift=2.0,
            cdisp_half_width=3,
        )

        np.testing.assert_allclose(tracemask.domain[0], [100.0, 200.0])
        positions = tracemask.eval(rows_valid)[0]
        np.testing.assert_allclose(
            positions, true_center[rows_valid], atol=0.5,
        )


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end test: identifier → fibermap → tracemask."""

    def test_full_pipeline(self):
        """S1: identifier produces fibermap, tracemask traces and fits."""
        image, centers = _synthetic_fiber_flat(
            n_rows=512, n_blocks=3, n_fibers_per_block=5,
        )
        identifier = FibersIdentifier(
            image=image,
            n_blocks_expected=3,
            n_fibers_per_block_expected=5,
        )
        fibermap = identifier.identify(center_row=256, band_half_width=50)

        tracemask = TraceMask.from_fibermap(
            fibermap=fibermap,
            image=image,
            poly_deg=3,
        )

        assert fibermap.n_fibers == 15
        assert tracemask.n_fibers == 15
        rows = np.arange(512)
        positions = tracemask.eval(rows)
        assert positions.shape == (15, 512)
        # For straight synthetic fibers, eval at center_row should match
        # the detected approx_x positions within a few pixels
        center_positions = positions[:, 256]
        np.testing.assert_allclose(
            center_positions, fibermap["APPROX_X"], atol=2.0,
        )


# ---------------------------------------------------------------------------
# FiberFrame tests
# ---------------------------------------------------------------------------

class TestFiberFrame:
    """Tests for the FiberFrame extracted-spectra container."""

    def test_create_with_1d_wave(self):
        """S1: FiberFrame accepts 1-D shared wave grid."""
        wave = np.arange(1000, dtype=float)
        flux = np.ones((15, 1000), dtype=float)
        frame = FiberFrame(wave=wave, flux=flux)
        assert frame.n_fibers == 15
        assert frame.n_wave == 1000
        assert frame.shape == (15, 1000)

    def test_create_with_2d_wave(self):
        """S1: FiberFrame accepts 2-D per-fiber wave grid."""
        wave = np.tile(np.arange(1000, dtype=float), (15, 1))
        flux = np.ones((15, 1000), dtype=float)
        frame = FiberFrame(wave=wave, flux=flux)
        assert frame.n_fibers == 15
        assert frame.n_wave == 1000

    def test_defaults_ivar_and_mask(self):
        """S1: Default ivar is ones, default mask is zeros."""
        wave = np.arange(100, dtype=float)
        flux = np.ones((5, 100), dtype=float)
        frame = FiberFrame(wave=wave, flux=flux)
        np.testing.assert_array_equal(frame.ivar, np.ones_like(flux))
        np.testing.assert_array_equal(frame.mask, np.zeros_like(flux, dtype=np.uint32))

    def test_rejects_mismatched_flux_wave(self):
        """S2: Mismatched flux/wave shapes raise ValueError."""
        wave = np.arange(50, dtype=float)
        flux = np.ones((5, 100), dtype=float)
        with pytest.raises(ValueError):
            FiberFrame(wave=wave, flux=flux)

    def test_rejects_mismatched_ivar(self):
        """S2: Mismatched ivar shape raises ValueError."""
        wave = np.arange(100, dtype=float)
        flux = np.ones((5, 100), dtype=float)
        ivar = np.ones((5, 50), dtype=float)
        with pytest.raises(ValueError):
            FiberFrame(wave=wave, flux=flux, ivar=ivar)

    def test_rejects_mismatched_fibermap(self):
        """S2: fibermap row count must match n_fibers."""
        wave = np.arange(100, dtype=float)
        flux = np.ones((5, 100), dtype=float)
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(3),
            block_ids=np.zeros(3, dtype=int),
            approx_x=np.arange(3),
            center_row=512,
        )
        with pytest.raises(ValueError):
            FiberFrame(wave=wave, flux=flux, fibermap=fm)

    def test_fits_roundtrip(self, tmp_path):
        """S1: FITS write + read preserves data."""
        wave = np.linspace(4000, 7000, 500, dtype=float)
        flux = np.random.default_rng(42).random((10, 500))
        ivar = np.ones_like(flux)
        mask = np.zeros(flux.shape, dtype=np.uint32)
        fm = FiberMap.from_arrays(
            fiber_ids=np.arange(10),
            block_ids=np.repeat([0, 1], 5),
            approx_x=np.arange(10),
            center_row=512,
        )
        frame = FiberFrame(
            wave=wave,
            flux=flux,
            ivar=ivar,
            mask=mask,
            fibermap=fm,
            meta={"PIPELINE": "amasedrp"},
        )
        path = tmp_path / "frame.fits"
        frame.to_fits(path)
        restored = FiberFrame.from_fits(path)

        np.testing.assert_array_equal(restored.wave, wave)
        np.testing.assert_array_equal(restored.flux, flux)
        np.testing.assert_array_equal(restored.ivar, ivar)
        np.testing.assert_array_equal(restored.mask, mask)
        assert restored.fibermap is not None
        assert restored.fibermap.n_fibers == 10
        assert restored.meta.get("PIPELINE") == "amasedrp"

