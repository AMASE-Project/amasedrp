#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the calibration stage: wavelength solution and its application."""

from __future__ import annotations

import numpy as np
import pytest

from amasedrp.calibration import (
    LineSpreadFunction,
    WavelengthSolution,
    apply_wavelength_solution,
    fit_line_spread_function,
    solve_wavelength_solution,
    thar_lines,
)
from amasedrp.reduction.core.fiberframe import FiberFrame
from amasedrp.reduction.core.fibermap import FiberMap
from amasedrp.reduction.core.tracemask import TraceMask

# A synthetic arc: wavelength = WAVE_ZERO + WAVE_PER_ROW * row.
WAVE_ZERO = 4400.0
WAVE_PER_ROW = 0.5
# Irregular spacing, so that shifting the assignment by one line does not
# produce a fit that is just as good.  Twelve known lines are needed: the
# scoring trims outliers, so with only a handful of lines a wrong assignment
# can score better than the right one.
KNOWN_WLS = np.array([
    4500.0, 4538.0, 4601.0, 4640.0, 4705.0, 4749.0,
    4782.0, 4820.0, 4873.0, 4905.0, 4940.0, 4972.0,
])
POSS_WLS = np.array([4601.0, 4705.0, 4782.0, 4873.0, 4972.0])


def _synthetic_arc(
    n_rows: int = 1200,
    n_fibers: int = 3,
    noise_std: float = 0.0,
    flux_scale: float = 1.0,
) -> tuple[FiberFrame, TraceMask]:
    """Build an arc frame whose lines sit at known rows.

    Returns
    -------
    frame
        Extracted arc spectra with a fiber map attached.
    tracemask
        Trace model whose domains cover rows 10 to ``n_rows - 10``.
    """
    rows = np.arange(n_rows)
    wavelength = WAVE_ZERO + WAVE_PER_ROW * rows
    arc = np.zeros(n_rows, dtype=float)
    for known in KNOWN_WLS:
        center = (known - WAVE_ZERO) / WAVE_PER_ROW
        arc += np.exp(-0.5 * ((rows - center) / 2.0) ** 2)

    rng = np.random.default_rng(3)
    flux = np.tile(arc * flux_scale, (n_fibers, 1))
    if noise_std > 0:
        flux = flux + rng.normal(0.0, noise_std, flux.shape)

    fibermap = FiberMap.from_arrays(
        fiber_ids=np.arange(n_fibers),
        block_ids=np.zeros(n_fibers, dtype=int),
        approx_x=np.arange(n_fibers, dtype=float) * 10.0,
        center_row=n_rows // 2,
    )
    frame = FiberFrame(wave=np.arange(n_rows, dtype=float), flux=flux, fibermap=fibermap)

    tracemask = TraceMask(
        coeffs=np.zeros((n_fibers, 2)),
        fiber_ids=np.arange(n_fibers),
        domain=np.tile([10.0, n_rows - 10.0], (n_fibers, 1)),
    )
    return frame, tracemask


class TestWavelengthSolution:
    """Tests for the WavelengthSolution container."""

    def test_eval_linear_solution(self):
        """S1: eval() returns wavelength = c0 + c1 * row."""
        solution = WavelengthSolution(
            coeffs=np.array([[5000.0, 0.5]]),
            fiber_ids=np.array([0]),
            scores=np.array([0.01]),
        )
        np.testing.assert_allclose(solution.eval(np.array([0, 2, 4])), [[5000.0, 5001.0, 5002.0]])
        assert solution.n_fibers == 1
        assert solution.degree == 1
        assert solution.valid.all()

    def test_rejects_unknown_polynomial_family(self):
        """S2: An unknown poly_kind raises ValueError."""
        with pytest.raises(ValueError, match="poly_kind"):
            WavelengthSolution(np.array([[1.0]]), np.array([0]), poly_kind="chebyshev")

    def test_rejects_mismatched_fiber_ids(self):
        """S2: fiber_ids must match the coefficient rows."""
        with pytest.raises(ValueError, match="fiber_ids"):
            WavelengthSolution(np.array([[1.0], [2.0]]), np.array([0]))

    def test_negative_score_marks_invalid(self):
        """S2: A negative score marks a fiber that inherited a solution."""
        solution = WavelengthSolution(
            coeffs=np.array([[1.0], [2.0]]),
            fiber_ids=np.array([0, 1]),
            scores=np.array([0.5, -1.0]),
        )
        assert list(solution.valid) == [True, False]

    def test_calibrated_fraction_counts_the_valid_fibers(self):
        """S1: calibrated_fraction is the share of fibers with a solution."""
        solution = WavelengthSolution(
            coeffs=np.zeros((4, 2)),
            fiber_ids=np.arange(4),
            scores=np.array([0.1, 0.2, -1.0, 0.3]),
        )
        assert solution.calibrated_fraction == pytest.approx(0.75)

    def test_wavelength_at_inverts_eval(self):
        """S1: wavelength_at() inverts eval()."""
        solution = WavelengthSolution(
            coeffs=np.array([[WAVE_ZERO, WAVE_PER_ROW]]),
            fiber_ids=np.array([0]),
        )
        row = solution.wavelength_at(0, 4600.0, y_max=1200.0)
        assert row == pytest.approx(400.0, abs=1e-3)

class TestSolveWavelengthSolution:
    """Tests for solve_wavelength_solution."""

    def _solve(self, frame, tracemask, **kwargs):
        options = dict(
            known_wls=KNOWN_WLS,
            poss_wls=POSS_WLS,
            min_deg=3,
            full_search=False,
            parallel=False,
        )
        options.update(kwargs)
        return solve_wavelength_solution(frame, tracemask, **options)

    def test_recovers_a_linear_mapping(self):
        """S1: The solved polynomial reproduces the true wavelength mapping."""
        frame, tracemask = _synthetic_arc()
        solution = self._solve(frame, tracemask)

        assert solution.n_fibers == 3
        assert (solution.scores >= 0).all()
        assert np.max(solution.scores) < 0.1

        rows = np.arange(frame.n_wave)
        true_wavelength = WAVE_ZERO + WAVE_PER_ROW * rows
        solved = solution.eval(rows)
        assert solved.shape == (frame.n_fibers, frame.n_wave)
        np.testing.assert_allclose(solved[0], true_wavelength, atol=0.2)

    def test_short_trace_inherits_neighbour_solution(self):
        """S2: A short trace inherits and is marked with score -1."""
        frame, tracemask = _synthetic_arc()
        tracemask.domain[1] = [600.0, 640.0]  # far below 0.7 x reference span
        solution = self._solve(frame, tracemask)

        assert solution.scores[1] == -1.0
        np.testing.assert_allclose(solution.coeffs[1], solution.coeffs[0])
        assert (solution.scores[[0, 2]] >= 0).all()

    def test_all_blank_reference_raises(self):
        """S2: An arc with no lines raises, because nothing can propagate."""
        frame, tracemask = _synthetic_arc(flux_scale=0.0)
        with pytest.raises(RuntimeError, match="reference fiber"):
            self._solve(frame, tracemask)

    def test_flagged_rows_are_treated_as_missing(self):
        """S2: Rows flagged in the mask are not fed to peak detection."""
        frame, tracemask = _synthetic_arc()
        frame.mask[:, :] = 1  # every row flagged
        with pytest.raises(RuntimeError, match="reference fiber"):
            self._solve(frame, tracemask)

    def test_rejects_empty_line_lists(self):
        """S2: Empty reference lines raise ValueError."""
        frame, tracemask = _synthetic_arc()
        with pytest.raises(ValueError, match="must not be empty"):
            self._solve(frame, tracemask, known_wls=np.array([]))

    def test_rejects_unknown_polynomial_family(self):
        """S2: An unknown poly_kind raises ValueError."""
        frame, tracemask = _synthetic_arc()
        with pytest.raises(ValueError, match="poly_kind"):
            self._solve(frame, tracemask, poly_kind="chebyshev")

    def test_rejects_fiber_count_mismatch(self):
        """S2: A trace model for a different fiber count raises ValueError."""
        frame, tracemask = _synthetic_arc()
        short = TraceMask(
            coeffs=np.zeros((2, 2)),
            fiber_ids=np.array([0, 1]),
            domain=np.tile([10.0, 1190.0], (2, 1)),
        )
        with pytest.raises(ValueError, match="tracemask has"):
            self._solve(frame, short)

    def test_rejects_frame_without_fiber_map(self):
        """S2: Calibration without a fiber map raises ValueError."""
        frame, tracemask = _synthetic_arc()
        frame.fibermap = None
        with pytest.raises(ValueError, match="fiber map"):
            self._solve(frame, tracemask)

    def test_polynomial_family_is_honoured(self):
        """S1: A plain-polynomial solution fits as well as a Legendre one."""
        frame, tracemask = _synthetic_arc()
        solution = self._solve(frame, tracemask, poly_kind="polynomial")

        assert solution.poly_kind == "polynomial"
        rows = np.arange(frame.n_wave)
        np.testing.assert_allclose(
            solution.eval(rows)[0],
            WAVE_ZERO + WAVE_PER_ROW * rows,
            atol=0.2,
        )

    def test_fully_calibrated_frame_reports_a_fraction_of_one(self):
        """S1: Every fiber of the synthetic arc gets its own solution."""
        frame, tracemask = _synthetic_arc()
        solution = self._solve(frame, tracemask)
        assert solution.calibrated_fraction == pytest.approx(1.0)

    def test_minimum_fraction_accepts_a_good_frame(self):
        """S1: A threshold below the achieved fraction passes."""
        frame, tracemask = _synthetic_arc()
        solution = self._solve(frame, tracemask, min_calibrated_fraction=0.9)
        assert solution.calibrated_fraction == pytest.approx(1.0)

    def test_minimum_fraction_rejects_a_partial_solution(self):
        """S2: Falling below the threshold raises for an observing run.

        A focus sweep leaves the threshold unset, because strong defocus
        legitimately loses most fibers.  A real observation must not.
        """
        frame, tracemask = _synthetic_arc()
        tracemask.domain[1] = [600.0, 640.0]  # fiber 1 inherits, so 2 of 3
        with pytest.raises(RuntimeError, match="below the required"):
            self._solve(frame, tracemask, min_calibrated_fraction=0.9)
        # the same frame is accepted without a threshold
        assert self._solve(frame, tracemask).calibrated_fraction == pytest.approx(2 / 3)

    def test_rejects_an_out_of_range_minimum_fraction(self):
        """S2: A fraction outside [0, 1] raises ValueError."""
        frame, tracemask = _synthetic_arc()
        with pytest.raises(ValueError, match="must lie in"):
            self._solve(frame, tracemask, min_calibrated_fraction=1.5)


class TestApplyWavelengthSolution:
    """Tests for apply_wavelength_solution."""

    def _calibrated_pair(self):
        frame, tracemask = _synthetic_arc()
        solution = solve_wavelength_solution(
            frame, tracemask, known_wls=KNOWN_WLS, poss_wls=POSS_WLS,
            min_deg=3, full_search=False, parallel=False,
        )
        return frame, solution

    def test_fills_two_dimensional_wave(self):
        """S1: The returned frame carries a per-fiber wavelength array."""
        frame, solution = self._calibrated_pair()
        calibrated = apply_wavelength_solution(frame, solution)

        assert calibrated.wave.shape == frame.flux.shape
        assert calibrated.flux is frame.flux
        rows = np.arange(frame.n_wave)
        np.testing.assert_allclose(
            calibrated.wave[0],
            WAVE_ZERO + WAVE_PER_ROW * rows,
            atol=0.2,
        )
        # every fiber of the synthetic arc shares the same mapping
        np.testing.assert_allclose(
            calibrated.wave,
            np.broadcast_to(calibrated.wave[0], calibrated.wave.shape),
        )

    def test_records_the_score_in_the_fiber_map(self):
        """S1: WAVCAL_SCORE lands in a copy of the fiber map."""
        frame, solution = self._calibrated_pair()
        calibrated = apply_wavelength_solution(frame, solution)

        assert "WAVCAL_SCORE" in calibrated.fibermap.colnames
        np.testing.assert_array_equal(calibrated.fibermap["WAVCAL_SCORE"], solution.scores)
        # the caller's fiber map is left alone
        assert "WAVCAL_SCORE" not in frame.fibermap.colnames

    def test_rejects_solution_with_wrong_fiber_count(self):
        """S2: A solution for another fiber count raises ValueError."""
        frame, solution = self._calibrated_pair()
        other = WavelengthSolution(
            coeffs=np.zeros((5, 2)), fiber_ids=np.arange(5),
        )
        with pytest.raises(ValueError, match="solution holds"):
            apply_wavelength_solution(frame, other)


class TestLineSpreadFunction:
    """Tests for the LineSpreadFunction container."""

    def test_nan_marks_an_invalid_measurement(self):
        """S2: A line that could not be fitted is not valid."""
        lsf = LineSpreadFunction(
            target_wls=np.array([4657.9]),
            fwhm=np.array([[3.0, np.nan]]),
            fiber_ids=np.arange(2),
        )
        assert list(lsf.valid[0]) == [True, False]
        assert lsf.n_wls == 1
        assert lsf.n_fibers == 2

    def test_resolution_is_wavelength_over_fwhm(self):
        """S1: R = wavelength / fwhm."""
        lsf = LineSpreadFunction(
            target_wls=np.array([5000.0]),
            fwhm=np.array([[2.5]]),
            fiber_ids=np.array([0]),
        )
        np.testing.assert_allclose(lsf.resolution, [[2000.0]])

    def test_rejects_mismatched_wavelength_count(self):
        """S2: fwhm rows must match target_wls."""
        with pytest.raises(ValueError, match="rows"):
            LineSpreadFunction(
                target_wls=np.array([1.0, 2.0]),
                fwhm=np.ones((3, 4)),
                fiber_ids=np.arange(4),
            )

    def test_rejects_mismatched_fiber_ids(self):
        """S2: fiber_ids must match the fwhm columns."""
        with pytest.raises(ValueError, match="fiber_ids"):
            LineSpreadFunction(
                target_wls=np.array([1.0]),
                fwhm=np.ones((1, 4)),
                fiber_ids=np.arange(3),
            )


class TestFitLineSpreadFunction:
    """Tests for fit_line_spread_function."""

    def _calibrated_frame(self):
        """An arc frame with a solved and applied wavelength solution."""
        frame, tracemask = _synthetic_arc()
        solution = solve_wavelength_solution(
            frame, tracemask, known_wls=KNOWN_WLS, poss_wls=POSS_WLS,
            min_deg=3, full_search=False, parallel=False,
        )
        return apply_wavelength_solution(frame, solution)

    def test_recovers_the_line_width(self):
        """S1: A Gaussian line of 2 px sigma is 2.355 A wide at 0.5 A/px."""
        frame = self._calibrated_frame()
        lsf = fit_line_spread_function(
            frame, target_wls=np.array([4705.0, 4830.0]),
        )

        expected = 2.355 * 2.0 * WAVE_PER_ROW
        assert lsf.fwhm.shape == (2, frame.n_fibers)
        assert lsf.valid.all()
        np.testing.assert_allclose(lsf.fwhm, expected, rtol=0.05)

    def test_resolution_matches_the_widths(self):
        """S1: Resolution follows from the measured widths."""
        frame = self._calibrated_frame()
        lsf = fit_line_spread_function(frame, target_wls=np.array([4705.0]))

        np.testing.assert_allclose(
            lsf.resolution, lsf.target_wls[:, None] / lsf.fwhm,
        )

    def test_flagged_rows_leave_every_line_unfitted(self):
        """S2: A fully flagged frame yields no measurement, not a baseline."""
        frame = self._calibrated_frame()
        frame.mask[:, :] = 1

        with pytest.warns(RuntimeWarning, match="did not converge"):
            lsf = fit_line_spread_function(frame, target_wls=np.array([4705.0]))

        assert not lsf.valid.any()

    def test_rejects_a_frame_without_a_wavelength_solution(self):
        """S2: A pixel-index frame raises ValueError."""
        frame, _ = _synthetic_arc()
        with pytest.raises(ValueError, match="per-fiber wavelength solution"):
            fit_line_spread_function(frame, target_wls=np.array([4705.0]))

    def test_rejects_empty_target_wavelengths(self):
        """S2: No target lines raises ValueError."""
        frame = self._calibrated_frame()
        with pytest.raises(ValueError, match="must not be empty"):
            fit_line_spread_function(frame, target_wls=np.array([]))

    def test_rejects_two_dimensional_target_wavelengths(self):
        """S2: A 2-D target list raises ValueError."""
        frame = self._calibrated_frame()
        with pytest.raises(ValueError, match="must be 1-D"):
            fit_line_spread_function(frame, target_wls=np.array([[4705.0]]))


class TestTharLines:
    """Tests for the ThAr line selection."""

    @pytest.mark.parametrize("channel", ["blue", "red", "BLUE", " red "])
    def test_known_channels(self, channel):
        """S1: Both channels are available, case and padding insensitive."""
        lines = thar_lines(channel)
        assert len(lines.known_wls) > len(lines.poss_wls) >= 4
        assert len(lines.lsf_wls) >= 1
        # every LSF line must be a known line
        assert set(lines.lsf_wls) <= set(lines.known_wls)

    def test_lines_are_sorted_and_unique(self):
        """S1: Every list is sorted and free of duplicates."""
        for channel in ("blue", "red"):
            lines = thar_lines(channel)
            for array in (lines.known_wls, lines.poss_wls, lines.lsf_wls):
                assert np.all(np.diff(array) > 0)

    def test_poss_lines_are_known_lines(self):
        """S1: Candidate lines are a subset of the reference lines."""
        for channel in ("blue", "red"):
            lines = thar_lines(channel)
            assert set(lines.poss_wls) <= set(lines.known_wls)

    def test_rejects_unknown_channel(self):
        """S2: An unknown channel raises ValueError."""
        with pytest.raises(ValueError, match="unknown channel"):
            thar_lines("green")
