#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for master-frame combination, and for the ADU/electron bookkeeping.

``Image`` stores one electron array and derives ADU from it.  While both were
stored, a calibration step rewrote one and ``write_to_fits`` read the other, so
a bias subtraction was lost on disk without a warning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from amasedrp.preprocessing import Image
from amasedrp.preprocessing.methods.bias import subtract_bias
from amasedrp.preprocessing.methods.master import combine_master


def _mock_image(
    value: float,
    shape: tuple[int, int] = (6, 6),
    gain: float = 56.0,
    filename: str | None = None,
) -> Image:
    """Create an ``Image`` of a constant ADU level."""
    return Image(
        data=np.full(shape, value, dtype=np.float32),
        header=fits.Header({"GAIN": gain, "EXPTIME": 0.0}),
        filename=filename,
    )


class TestCombineMaster:
    """Tests for combine_master."""

    def test_median_rejects_a_rogue_frame(self):
        """S1: One warm frame does not move the master."""
        frames = [_mock_image(v) for v in (1200.0, 1204.0, 1196.0, 1200.0, 99999.0)]
        master = combine_master(frames)

        assert isinstance(master, Image)
        assert master.ADU[0, 0] == pytest.approx(1200.0)

    def test_keeps_the_header_of_the_first_frame(self):
        """S1: The master carries the header of the first input frame."""
        frames = [_mock_image(1200.0), _mock_image(1300.0)]
        master = combine_master(frames)

        assert master.header["GAIN"] == 56.0
        assert master.filename is None

    def test_records_its_inputs_in_the_history(self):
        """S1: The HISTORY cards name the combined frames."""
        frames = [
            _mock_image(1200.0, filename="/data/Bias-Blue-0001.fits"),
            _mock_image(1204.0, filename="/data/Bias-Blue-0002.fits"),
        ]
        master = combine_master(frames)

        history = list(master.header["HISTORY"])
        assert any("combined from 2 frames" in line for line in history)
        assert any("Bias-Blue-0001.fits" in line for line in history)
        assert any("Bias-Blue-0002.fits" in line for line in history)

    def test_single_frame_is_returned_as_the_master(self):
        """S1: Combining one frame gives that frame back."""
        master = combine_master([_mock_image(1200.0)])
        assert master.ADU[0, 0] == pytest.approx(1200.0)

    def test_rejects_an_empty_sequence(self):
        """S2: No frames raises ValueError."""
        with pytest.raises(ValueError, match="at least one frame"):
            combine_master([])

    def test_rejects_mismatched_shapes(self):
        """S2: Frames of different shapes raise ValueError."""
        with pytest.raises(ValueError, match="shapes differ"):
            combine_master([_mock_image(1.0, shape=(6, 6)), _mock_image(1.0, shape=(4, 4))])

    def test_rejects_a_frame_without_data(self):
        """S2: A frame holding no data raises ValueError."""
        empty = _mock_image(1.0)
        empty.data = None
        with pytest.raises(ValueError, match="holds no data"):
            combine_master([_mock_image(1.0), empty])


class TestImageUnitBookkeeping:
    """Tests for the ADU, electron and gain bookkeeping of ``Image``."""

    def test_gain_prefers_the_electronic_gain(self):
        """S2: ``EGAIN`` is the physical gain, ``GAIN`` only the camera setting."""
        image = Image(
            data=np.full((4, 4), 2000.0, dtype=np.float32),
            header=fits.Header({"GAIN": 56, "EGAIN": 1.0}),
        )

        assert image.gain == pytest.approx(1.0)
        assert image.data[0, 0] == pytest.approx(2000.0)

    def test_gain_falls_back_to_the_camera_setting(self):
        """S2: Without EGAIN, GAIN is used."""
        image = Image(
            data=np.ones((4, 4), dtype=np.float32),
            header=fits.Header({"GAIN": 56}),
        )
        assert image.gain == pytest.approx(56.0)

    def test_gain_absent_falls_back_to_the_default(self):
        """S2: Without either keyword, the default gain is assumed."""
        image = Image(data=np.ones((4, 4), dtype=np.float32), header=fits.Header())

        assert image.gain is None
        assert image.data[0, 0] == pytest.approx(1.0)

    def test_from_fits_rejects_a_file_without_data(self, tmp_path: Path):
        """S2: A file whose primary HDU holds no data raises a clear error.

        Falling back to the first HDU that holds data would turn a multi-
        extension product, such as a FiberFrame, into a wrong single array.
        """
        path = tmp_path / "not_an_image.fits"
        fits.HDUList([
            fits.PrimaryHDU(), fits.ImageHDU(np.ones((2, 2))),
        ]).writeto(path)

        with pytest.raises(ValueError, match="holds no data"):
            Image.from_fits(str(path))

    def test_adu_is_derived_from_the_electron_array(self):
        """S1: ADU follows data and the gain."""
        image = _mock_image(2000.0, gain=56.0)

        assert image.data[0, 0] == pytest.approx(2000.0 * 56.0)
        assert image.ADU[0, 0] == pytest.approx(2000.0)

    def test_rewriting_data_moves_adu_with_it(self):
        """S2: A calibration step that rewrites data is reflected in ADU."""
        image = _mock_image(2000.0, gain=56.0)
        image.data = image.data - 1200.0 * 56.0

        assert image.ADU[0, 0] == pytest.approx(800.0)

    def test_copy_keeps_the_electron_scale(self):
        """S1: copy() does not apply the gain twice."""
        image = _mock_image(2000.0, gain=56.0)
        clone = image.copy()

        assert clone.data[0, 0] == pytest.approx(image.data[0, 0])
        assert clone.ADU[0, 0] == pytest.approx(2000.0)

    def test_bias_subtraction_survives_the_write(self, tmp_path: Path):
        """S2: A subtracted frame keeps the subtraction on disk.

        ``subtract_bias`` rewrites the electron array while ``write_to_fits``
        writes ADU.  While those were two stored arrays, the file held the
        unsubtracted level and nothing warned.
        """
        header = fits.Header({"GAIN": 56, "EXPTIME": 0.0})
        raw_path = tmp_path / "raw.fits"
        bias_path = tmp_path / "bias.fits"
        out_path = tmp_path / "cal.fits"
        fits.PrimaryHDU(
            np.full((4, 4), 2000, dtype=np.uint16), header=header
        ).writeto(raw_path)
        fits.PrimaryHDU(
            np.full((4, 4), 1200, dtype=np.uint16), header=header
        ).writeto(bias_path)

        calibrated = subtract_bias(
            Image.from_fits(str(raw_path)), Image.from_fits(str(bias_path))
        )
        calibrated.write_to_fits(str(out_path))

        assert fits.getdata(out_path)[0, 0] == pytest.approx(800.0)

    def test_write_round_trip_preserves_adu(self, tmp_path: Path):
        """S1: Reading and writing a plain frame preserves its ADU values."""
        path = tmp_path / "frame.fits"
        out_path = tmp_path / "copy.fits"
        fits.PrimaryHDU(
            np.full((4, 4), 500, dtype=np.uint16),
            header=fits.Header({"GAIN": 3}),
        ).writeto(path)

        Image.from_fits(str(path)).write_to_fits(str(out_path))

        np.testing.assert_allclose(fits.getdata(out_path), 500.0)
