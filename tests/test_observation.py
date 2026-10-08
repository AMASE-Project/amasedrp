#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for frame selection by FITS header values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from amasedrp.observation import (
    LAMP_ARC,
    LAMP_FLAT,
    load_frame,
    reduced_dir,
    select_frames,
)


def _write_frame(
    directory: Path,
    name: str,
    channel: str | None = "BLUE",
    lamp: str | None = LAMP_ARC,
    col_foc: object = -1000,
    data: np.ndarray | None = None,
) -> Path:
    """Write a small FITS frame carrying the given identifying keywords."""
    header = fits.Header()
    if channel is not None:
        header["CHANNEL"] = channel
    if lamp is not None:
        header["LAMP"] = lamp
    if col_foc is not None:
        header["COL-FOC"] = col_foc
    if data is None:
        data = np.zeros((4, 4), dtype=np.uint16)
    path = directory / name
    fits.PrimaryHDU(data, header=header).writeto(path)
    return path


def _write_unparsable_col_foc(directory: Path, name: str) -> Path:
    """Write a frame whose COL-FOC card astropy cannot parse.

    The card is written by hand because astropy refuses to build such a card.
    This reproduces the condition of ``sweep_original/R135.fits``.
    """
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "CHANNEL = 'BLUE    '",
        "LAMP    = 'TH-AR   '",
        "COL-FOC = 1.5.3",
        "END",
    ]
    path = directory / name
    path.write_bytes(
        "".join(card.ljust(80) for card in cards).ljust(2880).encode("ascii")
    )
    return path


@pytest.fixture
def sweep_like(tmp_path: Path) -> Path:
    """A directory laid out like the 2025-07 collimator sweep."""
    _write_frame(tmp_path, "B114.fits", channel="BLUE", lamp=LAMP_ARC, col_foc=-5000)
    _write_frame(tmp_path, "B115.fits", channel="BLUE", lamp=LAMP_ARC, col_foc=-4000)
    _write_frame(tmp_path, "B136.fits", channel="BLUE", lamp=LAMP_FLAT, col_foc=-1000)
    _write_frame(tmp_path, "R136.fits", channel="RED", lamp=LAMP_ARC, col_foc=-1000)
    _write_frame(tmp_path, "R154.fits", channel="RED", lamp=LAMP_FLAT, col_foc=-1000)
    return tmp_path


def _names(paths: list[Path]) -> list[str]:
    return [path.name for path in paths]


class TestSelectFrames:
    """Tests for select_frames."""

    def test_selects_by_channel(self, sweep_like: Path):
        """S1: A channel filter returns that channel only."""
        assert _names(select_frames(sweep_like, channel="blue")) == [
            "B114.fits", "B115.fits", "B136.fits",
        ]
        assert _names(select_frames(sweep_like, channel="red")) == [
            "R136.fits", "R154.fits",
        ]

    def test_selects_by_lamp(self, sweep_like: Path):
        """S1: A lamp filter separates arcs from flats in both channels."""
        assert _names(select_frames(sweep_like, lamp=LAMP_ARC)) == [
            "B114.fits", "B115.fits", "R136.fits",
        ]
        assert _names(select_frames(sweep_like, lamp=LAMP_FLAT)) == [
            "B136.fits", "R154.fits",
        ]

    def test_selects_by_channel_and_lamp(self, sweep_like: Path):
        """S1: Criteria combine, so one arc/flat pair per channel is found."""
        assert _names(
            select_frames(sweep_like, channel="blue", lamp=LAMP_FLAT)
        ) == ["B136.fits"]

    def test_selects_by_collimator_position(self, sweep_like: Path):
        """S1: The position filter matches the numeric value."""
        assert _names(select_frames(sweep_like, col_foc=-4000)) == ["B115.fits"]
        assert _names(
            select_frames(sweep_like, channel="red", col_foc=-1000)
        ) == ["R136.fits", "R154.fits"]

    def test_selects_a_single_pair(self, sweep_like: Path):
        """S1: All three criteria together give the arc/flat pair of a position."""
        assert _names(select_frames(
            sweep_like, channel="blue", lamp=LAMP_ARC, col_foc=-4000,
        )) == ["B115.fits"]

    def test_no_criteria_returns_everything(self, sweep_like: Path):
        """S1: Without criteria, every FITS frame is returned."""
        assert len(select_frames(sweep_like)) == 5

    @pytest.mark.parametrize(
        "channel", ["blue", "BLUE", "Blue", " blue ", "blue\t"],
    )
    def test_channel_matching_ignores_case_and_padding(self, sweep_like, channel):
        """S1: Text matching ignores case and surrounding whitespace."""
        assert "B114.fits" in _names(select_frames(sweep_like, channel=channel))

    def test_result_is_sorted(self, sweep_like: Path):
        """S1: Results come back sorted by name."""
        names = _names(select_frames(sweep_like))
        assert names == sorted(names)

    def test_frame_without_lamp_card_is_not_matched(self, tmp_path: Path):
        """S2: A frame with no LAMP card never matches a lamp filter."""
        _write_frame(tmp_path, "nolamp.fits", channel="BLUE", lamp=None)
        assert select_frames(tmp_path, lamp=LAMP_ARC) == []
        # it is still reachable when no lamp filter is asked for
        assert _names(select_frames(tmp_path, channel="blue")) == ["nolamp.fits"]

    def test_non_numeric_position_warns_and_is_skipped(self, tmp_path: Path):
        """S2: A text COL-FOC warns and is skipped when filtering on position."""
        _write_frame(tmp_path, "badpos.fits", channel="BLUE", col_foc="garbage")
        with pytest.warns(RuntimeWarning, match="COL-FOC"):
            assert select_frames(tmp_path, col_foc=-1000) == []
        # without a position filter the frame is still usable
        assert _names(select_frames(tmp_path, channel="blue")) == ["badpos.fits"]

    def test_unparsable_position_card_warns_and_is_skipped(self, tmp_path: Path):
        """S2: A corrupt COL-FOC card warns and is skipped on a position filter."""
        _write_unparsable_col_foc(tmp_path, "corrupt.fits")
        with pytest.warns(RuntimeWarning, match="COL-FOC"):
            assert select_frames(tmp_path, col_foc=-1000) == []
        # the rest of the header is readable, so the frame is still selectable
        assert _names(select_frames(tmp_path, channel="blue")) == ["corrupt.fits"]

    def test_non_fits_file_warns_and_is_skipped(self, tmp_path: Path):
        """S2: A file that is not FITS warns and is skipped."""
        (tmp_path / "notes.fits").write_text("not a FITS file")
        _write_frame(tmp_path, "B114.fits")
        with pytest.warns(RuntimeWarning, match="header unreadable"):
            assert _names(select_frames(tmp_path)) == ["B114.fits"]

    def test_rejects_a_non_directory(self, tmp_path: Path):
        """S2: A path that is not a directory raises ValueError."""
        with pytest.raises(ValueError, match="is not a directory"):
            select_frames(tmp_path / "missing")


class TestLoadFrame:
    """Tests for putting a frame in the pipeline orientation.

    The camera writes the dispersion axis along axis 1; the pipeline keeps it
    along axis 0.  No header keyword records which orientation a file holds.
    """

    def test_accepts_the_pipeline_orientation(self, tmp_path: Path):
        """S1: A frame already in the pipeline orientation is left alone."""
        path = _write_frame(
            tmp_path, "flat.fits", data=np.arange(24).reshape(6, 4),
        )
        image = load_frame(path, shape=(6, 4))

        assert image.data.shape == (6, 4)
        np.testing.assert_array_equal(image.data, np.arange(24).reshape(6, 4))

    def test_transposes_the_raw_orientation(self, tmp_path: Path):
        """S2: A frame stored rotated is transposed into the convention."""
        raw = np.arange(24).reshape(4, 6)
        path = _write_frame(tmp_path, "bias.fits", data=raw)

        image = load_frame(path, shape=(6, 4))

        assert image.data.shape == (6, 4)
        np.testing.assert_array_equal(image.data, raw.T)

    def test_without_a_shape_the_frame_is_left_alone(self, tmp_path: Path):
        """S1: No expected shape means no rotation."""
        path = _write_frame(tmp_path, "bias.fits", data=np.zeros((4, 6)))
        assert load_frame(path).data.shape == (4, 6)

    def test_rejects_an_unrelated_shape(self, tmp_path: Path):
        """S2: A shape that is neither orientation raises ValueError."""
        path = _write_frame(tmp_path, "odd.fits", data=np.zeros((3, 5)))
        with pytest.raises(ValueError, match="neither"):
            load_frame(path, shape=(6, 4))


class TestReducedDir:
    """Tests for the output directory of a run."""

    def test_defaults_beside_the_raw_frames(self, tmp_path: Path):
        """S1: Without an override, products go in a reduced subdirectory."""
        observation = tmp_path / "20250705-collimator_sweep"
        observation.mkdir()

        root = reduced_dir(observation)

        assert root == observation / "reduced"
        assert root.is_dir()

    def test_honours_an_explicit_directory(self, tmp_path: Path):
        """S1: An explicit output directory is used as given."""
        root = reduced_dir(tmp_path / "observation", output_dir=tmp_path / "elsewhere")

        assert root == tmp_path / "elsewhere"
        assert root.is_dir()

    def test_creates_missing_parents(self, tmp_path: Path):
        """S1: A nested output directory is created."""
        root = reduced_dir(tmp_path / "a", output_dir=tmp_path / "b" / "c" / "d")
        assert root.is_dir()

    def test_is_idempotent(self, tmp_path: Path):
        """S1: Calling twice returns the same existing directory."""
        first = reduced_dir(tmp_path)
        second = reduced_dir(tmp_path)

        assert first == second
        assert first.is_dir()

    def test_leaves_raw_frames_untouched(self, tmp_path: Path):
        """S2: The raw frame directory itself is never the output directory."""
        observation = tmp_path / "observation"
        raw = observation / "sweep"
        raw.mkdir(parents=True)
        raw_frame = _write_frame(raw, "B136.fits")
        before = raw_frame.read_bytes()

        root = reduced_dir(observation)

        assert raw not in root.parents
        assert raw_frame.read_bytes() == before

    def test_rejects_a_file_as_output_path(self, tmp_path: Path):
        """S2: An output path that is a file raises ValueError."""
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            reduced_dir(tmp_path, output_dir=blocker)
