"""Tests for the amasecore version stamp on products.

A product that does not name the data version it came from cannot be
reproduced, so the stamp is checked at every place the pipeline writes FITS.
"""

from __future__ import annotations

import amasecore
import numpy as np
import pytest
from astropy.io import fits

from amasedrp.preprocessing.core.image import Image
from amasedrp.products import CORE_VERSION_KEY, core_version, stamp_core_version
from amasedrp.reduction.core.fiberframe import FiberFrame


def test_the_core_version_is_the_installed_data_version():
    assert core_version() == amasecore.version()
    assert isinstance(core_version(), str)


def test_stamping_writes_the_key():
    header = fits.Header()
    stamp_core_version(header)
    assert header[CORE_VERSION_KEY] == amasecore.version()


def test_the_key_fits_in_a_fits_card():
    """FITS allows eight characters, so a longer name would not be a card."""
    assert len(CORE_VERSION_KEY) <= 8


def test_an_image_product_names_its_data_version(tmp_path):
    image = Image(data=np.zeros((4, 4)), header=fits.Header())
    path = tmp_path / "image.fits"
    image.write_to_fits(str(path))
    assert fits.getheader(path)[CORE_VERSION_KEY] == amasecore.version()


def test_a_fiberframe_product_names_its_data_version(tmp_path):
    frame = FiberFrame(wave=np.arange(5.0), flux=np.ones((2, 5)))
    path = tmp_path / "frame.fits"
    frame.to_fits(path)
    assert fits.getheader(path)[CORE_VERSION_KEY] == amasecore.version()


def test_reading_a_fiberframe_back_does_not_move_the_stamp_into_meta(tmp_path):
    frame = FiberFrame(wave=np.arange(5.0), flux=np.ones((2, 5)), meta={"EXPTIME": 1.0})
    path = tmp_path / "frame.fits"
    frame.to_fits(path)
    reread = FiberFrame.from_fits(path)
    assert reread.meta["EXPTIME"] == pytest.approx(1.0)
    assert CORE_VERSION_KEY not in reread.meta
