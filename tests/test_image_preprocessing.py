#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive pytest suite for image_preprocessing.py and the Image class.

These tests are written against the *target* API (RED phase).  They will fail
until the implementation is updated to support:

* ``Image.write_to_fits(filename, update_header=...)``
* ``image_preprocessing(input_path, bias_path, dark_path, pixflat_path,
    output_path, update_header=...)``
* structured logging instead of ``print`` statements
* provenance keywords in output headers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits

from amasedrp.preprocessing import Image, image_preprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_mock_image(
    data: np.ndarray,
    header_dict: dict[str, Any] | None = None,
    filename: str | None = None,
) -> Image:
    """Create an ``Image`` instance with a populated ``fits.Header``.

    Args:
        data: 2-D numpy array to use as image data.
        header_dict: Optional mapping of FITS keyword -> value.
        filename: Optional filename to attach to the image.

    Returns:
        A fully initialised ``Image`` object.
    """
    header = fits.Header()
    if header_dict:
        for key, value in header_dict.items():
            header[key] = value
    return Image(data=data, header=header, filename=filename)


# ---------------------------------------------------------------------------
# 1. Image class additions
# ---------------------------------------------------------------------------

class TestImageProperties:
    """Tests for ``Image`` properties and I/O extensions."""

    def test_rdnoise_present(self) -> None:
        """RDNOISE in header -> rdnoise returns the correct float."""
        img = _create_mock_image(
            data=np.ones((10, 10), dtype=np.float32),
            header_dict={"RDNOISE": 3.5},
        )
        assert img.rdnoise == 3.5
        assert isinstance(img.rdnoise, float)

    def test_rdnoise_missing(self) -> None:
        """RDNOISE absent -> rdnoise is None."""
        img = _create_mock_image(
            data=np.ones((10, 10), dtype=np.float32),
        )
        assert img.rdnoise is None

    def test_write_to_fits_with_update_header(self, tmp_path: Path) -> None:
        """write_to_fits accepts update_header and persists the keywords."""
        img = _create_mock_image(
            data=np.ones((10, 10), dtype=np.float32),
            header_dict={"EXPTIME": 120.0},
        )
        out_path = tmp_path / "test_output.fits"
        img.write_to_fits(str(out_path), update_header={"OBJECT": "TEST"})

        assert out_path.exists()
        with fits.open(out_path) as hdul:
            header = hdul[0].header
            assert header["OBJECT"] == "TEST"
            assert header["EXPTIME"] == 120.0


# ---------------------------------------------------------------------------
# 2. Happy Path — Full Calibration Pipeline
# ---------------------------------------------------------------------------

class TestFullCalibrationPipeline:
    """End-to-end calibration under ideal conditions."""

    def test_full_calibration_pipeline(self, tmp_path: Path) -> None:
        """All three steps produce a correctly calibrated FITS file."""
        # Synthetic data where the "true" science signal is 100 everywhere.
        # bias = 10, dark current = 2 / sec, exptime = 30 sec
        # pixflat = uniform illumination (50) * pixel response (1.0)
        # science = 100 * 1.0 + 2*30 + 10 = 170
        shape = (20, 20)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.full(shape, 170.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.full(shape, 10.0, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.full(shape, 70.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 120.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(pixflat_path))

        result = image_preprocessing(
            input_path=str(sci_path),
            bias_path=str(bias_path),
            dark_path=str(dark_path),
            pixflat_path=str(pixflat_path),
            output_path=str(out_path),
            steps=("bias", "dark", "pixflat"),
        )

        assert isinstance(result, Image)
        assert result.data.dtype == np.float32

        # Calibration math:
        # darkcurr = (70 - 10) / 30 = 2.0
        # pixflat  = (120 - 10) - 2.0 * 30 = 50.0
        # pixresp  = 50.0 / 50.0 = 1.0
        # S1 = 170 - 10 = 160
        # S2 = 160 - 2.0 * 30 = 100
        # S3 = 100 / 1.0 = 100
        np.testing.assert_allclose(result.data, 100.0, rtol=1e-5)

        # Header provenance
        assert result.header.get("CALIBRAT") is not None
        assert result.header.get("MBIAS") is not None
        assert result.header.get("MDARK") is not None
        assert result.header.get("MPIXFLT") is not None
        assert any("bias" in str(h).lower() for h in result.header.get("HISTORY", []))
        assert any("dark" in str(h).lower() for h in result.header.get("HISTORY", []))
        assert any("flat" in str(h).lower() for h in result.header.get("HISTORY", []))


# ---------------------------------------------------------------------------
# 3. Edge Cases — Invalid Inputs & Partial Steps
# ---------------------------------------------------------------------------

class TestInvalidInputsAndPartialSteps:
    """Robustness checks for malformed inputs and selective calibration."""

    def test_shape_mismatch_raises_valueerror(self, tmp_path: Path) -> None:
        """Images with incompatible shapes must raise ValueError."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones((20, 20), dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones((10, 10), dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones((20, 20), dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.ones((20, 20), dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
            )

    def test_zero_median_pixflat_raises_valueerror(self, tmp_path: Path) -> None:
        """A pixel flat field with zero median must raise ValueError (division by zero)."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.zeros(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.zeros(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.zeros(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
            )

    def test_missing_exptime_raises_valueerror(self, tmp_path: Path) -> None:
        """Missing or invalid EXPTIME must raise ValueError."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32)
        ).write_to_fits(str(sci_path))  # no EXPTIME
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
            )

    def test_partial_steps_bias_only(self, tmp_path: Path) -> None:
        """steps=("bias",) should apply only bias subtraction."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        science_data = np.full(shape, 100.0, dtype=np.float32)
        bias_data = np.full(shape, 10.0, dtype=np.float32)

        _create_mock_image(
            science_data, {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            bias_data, {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.zeros(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 50.0, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        result = image_preprocessing(
            input_path=str(sci_path),
            bias_path=str(bias_path),
            dark_path=str(dark_path),
            pixflat_path=str(pixflat_path),
            output_path=str(out_path),
            steps=("bias",),
        )

        np.testing.assert_allclose(result.data, 90.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Regression — Integer Overflow Prevention
# ---------------------------------------------------------------------------

class TestIntegerOverflowPrevention:
    """Ensure uint16 inputs with high values do not wrap during calibration."""

    def test_no_integer_overflow(self, tmp_path: Path) -> None:
        """uint16 data near saturation must be promoted to float32 safely."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        # High uint16 values that would wrap if subtracted in uint16
        _create_mock_image(
            np.full(shape, 65000, dtype=np.uint16), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.full(shape, 50000, dtype=np.uint16), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.full(shape, 50000, dtype=np.uint16), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 60000, dtype=np.uint16), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        result = image_preprocessing(
            input_path=str(sci_path),
            bias_path=str(bias_path),
            dark_path=str(dark_path),
            pixflat_path=str(pixflat_path),
            output_path=str(out_path),
        )

        assert result.data.dtype == np.float32
        # 65000 - 50000 = 15000 (would wrap to 48536 in uint16)
        expected = 15000.0
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Integration — End-to-End Orchestrator
# ---------------------------------------------------------------------------

class TestImagePreprocessingOrchestrator:
    """Tests for the high-level ``image_preprocessing`` orchestrator."""

    def test_orchestrator_writes_valid_fits_with_header(
        self, tmp_path: Path
    ) -> None:
        """Full file-based pipeline produces a valid FITS with updated header."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        # Write temporary master frames
        _create_mock_image(
            np.full(shape, 100.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.full(shape, 10.0, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.full(shape, 70.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 120.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(pixflat_path))

        image_preprocessing(
            input_path=str(sci_path),
            bias_path=str(bias_path),
            dark_path=str(dark_path),
            pixflat_path=str(pixflat_path),
            output_path=str(out_path),
            update_header={"OBJECT": "QA"},
        )

        assert out_path.exists()
        with fits.open(out_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            assert header["OBJECT"] == "QA"
            assert header.get("CALIBRAT") is not None
            assert data is not None
            assert np.issubdtype(data.dtype, np.float32)


# ---------------------------------------------------------------------------
# 6. Step Order Enforcement
# ---------------------------------------------------------------------------

class TestStepOrderEnforcement:
    """Steps must be applied in fixed logical order regardless of tuple order."""

    def test_unordered_steps_applied_in_fixed_order(self, tmp_path: Path) -> None:
        """steps=("pixflat", "dark", "bias") must still yield correct result."""
        shape = (20, 20)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.full(shape, 170.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.full(shape, 10.0, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.full(shape, 70.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 120.0, dtype=np.float32), {"EXPTIME": 30.0}
        ).write_to_fits(str(pixflat_path))

        result = image_preprocessing(
            input_path=str(sci_path),
            bias_path=str(bias_path),
            dark_path=str(dark_path),
            pixflat_path=str(pixflat_path),
            output_path=str(out_path),
            steps=("pixflat", "dark", "bias"),
        )

        np.testing.assert_allclose(result.data, 100.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Step Dependency Validation
# ---------------------------------------------------------------------------

class TestStepDependencyValidation:
    """Invalid step combinations must raise ValueError."""

    def test_dark_without_bias_raises_valueerror(self, tmp_path: Path) -> None:
        """steps=("dark",) must raise ValueError because dark requires bias."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
                steps=("dark",),
            )

    def test_pixflat_without_bias_raises_valueerror(self, tmp_path: Path) -> None:
        """steps=("pixflat",) must raise ValueError because pixflat requires bias+dark."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
                steps=("pixflat",),
            )

    def test_pixflat_without_dark_raises_valueerror(self, tmp_path: Path) -> None:
        """steps=("bias", "pixflat") must raise ValueError because pixflat requires dark."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
                steps=("bias", "pixflat"),
            )


# ---------------------------------------------------------------------------
# 8. Missing Dark EXPTIME for Pixflat Step
# ---------------------------------------------------------------------------

class TestMissingDarkExptimeForPixflat:
    """master_dark exptime must be validated when pixflat step is requested."""

    def test_missing_dark_exptime_for_pixflat_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        """pixflat step requires master_dark exptime; missing it raises ValueError."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32)
        ).write_to_fits(str(dark_path))  # no EXPTIME
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with pytest.raises(ValueError):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
                steps=("bias", "dark", "pixflat"),
            )


# ---------------------------------------------------------------------------
# 9. Logging Test
# ---------------------------------------------------------------------------

class TestLogging:
    """Ensure calibration emits structured log messages instead of raw prints."""

    def test_logging_instead_of_print(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """caplog should capture 'bias', 'dark', and 'pixflat' log records."""
        shape = (10, 10)
        sci_path = tmp_path / "science.fits"
        bias_path = tmp_path / "bias.fits"
        dark_path = tmp_path / "dark.fits"
        pixflat_path = tmp_path / "pixflat.fits"
        out_path = tmp_path / "output.fits"

        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(sci_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 1.0}
        ).write_to_fits(str(bias_path))
        _create_mock_image(
            np.ones(shape, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(dark_path))
        _create_mock_image(
            np.full(shape, 2.0, dtype=np.float32), {"EXPTIME": 10.0}
        ).write_to_fits(str(pixflat_path))

        with caplog.at_level(logging.INFO):
            image_preprocessing(
                input_path=str(sci_path),
                bias_path=str(bias_path),
                dark_path=str(dark_path),
                pixflat_path=str(pixflat_path),
                output_path=str(out_path),
            )

        log_text = " ".join(record.message for record in caplog.records)
        assert "bias" in log_text.lower()
        assert "dark" in log_text.lower()
        assert "pixflat" in log_text.lower()
