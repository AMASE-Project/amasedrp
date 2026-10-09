#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the product tree layout."""

from __future__ import annotations

import importlib.metadata

import pytest

from amasedrp.products import LEVELS, product_dir, product_name


class TestProductDir:
    """S1: One module builds every product path."""

    def test_default_version_comes_from_the_installation(self, tmp_path):
        """S1: Omitting drpver nests the products under the installed version."""
        version = importlib.metadata.version("amasedrp")
        root = product_dir(tmp_path, "20250705", obsid="p0001")

        assert root == tmp_path / version / "20250705" / "p0001"
        assert root.is_dir()

    def test_explicit_version_without_an_obsid(self, tmp_path):
        """S1: An explicit version, and a night-level directory."""
        root = product_dir(tmp_path, "20250705", drpver="9.9.9")

        assert root == tmp_path / "9.9.9" / "20250705"

    def test_qa_subdir(self, tmp_path):
        """S1: QA is a subdirectory of the observation, not a sibling file."""
        qa = product_dir(
            tmp_path, "20250705", obsid="p0001", subdir="qa", drpver="1.0"
        )

        assert qa == tmp_path / "1.0" / "20250705" / "p0001" / "qa"
        assert qa.is_dir()

    def test_night_level_calibration_dir(self, tmp_path):
        """S1: Nightly calibration products sit beside the observations."""
        cal = product_dir(tmp_path, "20250705", subdir="calibration", drpver="1.0")

        assert cal == tmp_path / "1.0" / "20250705" / "calibration"
        assert cal.is_dir()

    def test_creates_missing_parents(self, tmp_path):
        """S1: Nested missing directories are created."""
        root = product_dir(tmp_path / "archive" / "products", "20250705", drpver="1.0")

        assert root.is_dir()

    def test_is_idempotent(self, tmp_path):
        """S1: Calling twice returns the same existing directory."""
        first = product_dir(tmp_path, "20250705", drpver="1.0")
        second = product_dir(tmp_path, "20250705", drpver="1.0")

        assert first == second

    def test_create_false_leaves_the_disk_alone(self, tmp_path):
        """S1: create=False only computes the path."""
        root = product_dir(tmp_path, "20250705", drpver="1.0", create=False)

        assert not root.exists()

    def test_raises_when_the_path_is_a_file(self, tmp_path):
        """S2: A file where the directory belongs is an error."""
        (tmp_path / "1.0" / "20250705").parent.mkdir()
        (tmp_path / "1.0" / "20250705").write_text("not a directory")

        with pytest.raises(ValueError, match="not a directory"):
            product_dir(tmp_path, "20250705", drpver="1.0")

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "../escape"])
    def test_rejects_a_component_that_is_not_one_path_element(self, tmp_path, bad):
        """S2: A value holding a separator cannot silently add a level."""
        with pytest.raises(ValueError):
            product_dir(tmp_path, "20250705", obsid=bad, drpver="1.0")

    def test_rejects_a_bad_night(self, tmp_path):
        """S2: The night is validated too."""
        with pytest.raises(ValueError):
            product_dir(tmp_path, "../escape", drpver="1.0")

    def test_reports_a_missing_installation(self, tmp_path, monkeypatch):
        """S2: An unknown version fails loudly instead of naming a directory."""

        def _missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _missing)

        with pytest.raises(RuntimeError, match="not installed"):
            product_dir(tmp_path, "20250705")


class TestProductName:
    """S1: One module builds every product file name."""

    def test_full_name(self):
        """S1: Level, channel and exposure are joined in that order."""
        assert product_name("L2", channel="blue", exposure="0001") == "L2-blue-0001.fits"

    def test_level_only(self):
        """S1: A combined product carries no channel and no exposure."""
        assert product_name("L3") == "L3.fits"

    def test_custom_extension(self):
        """S1: The extension is the caller's choice."""
        name = product_name("L1", channel="red", exposure="0007", ext=".fits.gz")

        assert name == "L1-red-0007.fits.gz"

    def test_every_level_is_accepted(self):
        """S1: L0 through L3 all name a product."""
        for level in LEVELS:
            assert product_name(level) == f"{level}.fits"

    def test_rejects_an_unknown_level(self):
        """S2: A level outside the AMASE set is an error."""
        with pytest.raises(ValueError, match="level must be"):
            product_name("L4")

    def test_rejects_a_channel_that_is_not_one_path_element(self):
        """S2: A channel holding a separator is an error."""
        with pytest.raises(ValueError, match="channel"):
            product_name("L2", channel="a/b")

    def test_rejects_an_empty_exposure(self):
        """S2: An empty exposure is an error."""
        with pytest.raises(ValueError, match="exposure"):
            product_name("L2", exposure="")
