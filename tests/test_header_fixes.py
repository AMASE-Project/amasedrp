#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the header-fix overlay applied when a frame is read.

The rules these tests exercise are the real ones shipped in ``amasecore``, and
the frames they describe are the ones the 2025-07 sweep actually holds: red arcs
and red flats both read ``IMAGETYP='LIGHT'`` because the headers were typed by
hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from amasedrp.observation import load_frame
from amasedrp.preprocessing.core.image import HDRFIX_KEY, Image, night_from_header

NIGHT = "20250704"


def _write_frame(directory: Path, name: str, **cards: object) -> Path:
    """Write a small FITS frame carrying the given header cards."""
    header = fits.Header()
    for key, value in cards.items():
        header[key.replace("_", "-")] = value
    path = directory / name
    fits.PrimaryHDU(data=np.ones((4, 6), dtype=np.float32), header=header).writeto(path)
    return path


class TestNightFromHeader:
    """The night is the UTC date of DATE-OBS, not the directory name."""

    def test_a_typical_stamp_gives_the_date(self) -> None:
        header = fits.Header({"DATE-OBS": "2025-07-04T09:50:12.345"})
        assert night_from_header(header) == NIGHT

    @pytest.mark.parametrize("raw", [None, "", "junk", "2025-7-4"])
    def test_an_unreadable_stamp_gives_none(self, raw: object) -> None:
        header = fits.Header()
        if raw is not None:
            header["DATE-OBS"] = raw
        assert night_from_header(header) is None


class TestRulesAreApplied:
    """A frame whose night has rules gets them, and the change is recorded."""

    def test_a_red_arc_is_stamped_and_retyped(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "R132.fits", DATE_OBS="2025-07-04T10:20:00", LAMP="TH-AR",
            IMAGETYP="LIGHT",
        )
        image = Image.from_fits(str(path))

        assert image.header["IMAGETYP"] == "ARC"
        assert image.header[HDRFIX_KEY] == NIGHT
        assert any("IMAGETYP" in line for line in image.header["HISTORY"])

    def test_a_red_flat_is_retyped(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "R155.fits", DATE_OBS="2025-07-04T10:40:00", LAMP="QTH",
            IMAGETYP="LIGHT",
        )
        assert Image.from_fits(str(path)).header["IMAGETYP"] == "FLAT"

    def test_a_blue_flat_loses_its_other_spelling(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "B136.fits", DATE_OBS="2025-07-04T10:40:00", LAMP="QTH",
            IMAGETYP="Flat Field",
        )
        assert Image.from_fits(str(path)).header["IMAGETYP"] == "FLAT"

    def test_a_frame_no_rule_matches_keeps_its_type(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "S001.fits", DATE_OBS="2025-07-04T11:00:00", LAMP="QTH",
            IMAGETYP="LIGHT",
        )
        image = Image.from_fits(str(path))

        assert image.header["IMAGETYP"] == "LIGHT"
        assert image.header[HDRFIX_KEY] == NIGHT
        assert any("no rule matched" in line for line in image.header["HISTORY"])

    def test_the_repair_reaches_a_written_product(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "R132.fits", DATE_OBS="2025-07-04T10:20:00", LAMP="TH-AR",
            IMAGETYP="LIGHT",
        )
        out = tmp_path / "product.fits"
        Image.from_fits(str(path)).write_to_fits(str(out))
        assert fits.getheader(out)["IMAGETYP"] == "ARC"

    def test_the_load_frame_funnel_behaves_the_same(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "R132.fits", DATE_OBS="2025-07-04T10:20:00", LAMP="TH-AR",
            IMAGETYP="LIGHT",
        )
        assert load_frame(path).header["IMAGETYP"] == "ARC"


class TestNothingToApply:
    """The two ways a frame can come back unrepaired are told apart."""

    def test_a_night_without_rules_says_none(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "R150.fits", DATE_OBS="2025-12-10T04:00:00", LAMP="TH-AR",
            IMAGETYP="LIGHT",
        )
        image = Image.from_fits(str(path))

        assert image.header[HDRFIX_KEY] == "none"
        assert image.header["IMAGETYP"] == "LIGHT", "nothing may be guessed"

    def test_a_frame_without_a_date_says_unknown(self, tmp_path: Path) -> None:
        path = _write_frame(tmp_path, "R150.fits", LAMP="TH-AR", IMAGETYP="LIGHT")
        image = Image.from_fits(str(path))

        assert image.header[HDRFIX_KEY] == "unknown"
        assert image.header["IMAGETYP"] == "LIGHT"


class TestAProductKeepsItsOwnRecord:
    """A frame that already names its rules is a product and is left alone."""

    def test_an_existing_stamp_is_not_overwritten(self, tmp_path: Path) -> None:
        path = _write_frame(
            tmp_path, "master.fits", DATE_OBS="2025-12-10T04:00:00", LAMP="TH-AR",
            IMAGETYP="LIGHT", HDRFIX=NIGHT,
        )
        image = Image.from_fits(str(path))

        assert image.header[HDRFIX_KEY] == NIGHT
        assert image.header["IMAGETYP"] == "LIGHT"
        assert "HISTORY" not in image.header
