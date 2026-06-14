#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manual QA script for image_preprocessing end-to-end pipeline.

Generates synthetic FITS frames (science with a cosmic ray spike, bias,
dark, flat), runs the full ``image_preprocessing()`` orchestrator, and
asserts correctness of the output data and header.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

# Ensure the package is importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amasedrp.imageprocessing import Image, image_preprocessing


def _inject_cosmic_ray(data: np.ndarray, x: int, y: int, amplitude: float) -> np.ndarray:
    """Inject a bright spike simulating a cosmic ray hit."""
    data = data.copy()
    data[y, x] = amplitude
    return data


def _generate_mock_fits(
    path: str,
    shape: tuple[int, int],
    base_value: float,
    exptime: float | None,
    cr_spike: bool = False,
) -> None:
    """Write a synthetic FITS file to *path*."""
    data = np.full(shape, base_value, dtype=np.float32)
    if cr_spike:
        data = _inject_cosmic_ray(data, x=50, y=50, amplitude=50000.0)

    header = fits.Header()
    if exptime is not None:
        header["EXPTIME"] = exptime
    header["IMAGETYP"] = "object"

    img = Image(data=data, header=header, filename=path)
    img.write_to_fits(path)


def main() -> int:
    shape = (100, 100)

    # Physics model:
    #   bias = 10
    #   dark current = 2 e-/s
    #   dark = 10 + 2*30 = 70
    #   flat = 50 + 70 = 120  (uniform illumination 50 * response 1.0)
    #   science = 100 + 70 = 170 (signal 100 * response 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        sci_path = os.path.join(tmp, "science.fits")
        bias_path = os.path.join(tmp, "bias.fits")
        dark_path = os.path.join(tmp, "dark.fits")
        flat_path = os.path.join(tmp, "flat.fits")
        out_path = os.path.join(tmp, "output.fits")

        _generate_mock_fits(bias_path, shape, base_value=10.0, exptime=1.0)
        _generate_mock_fits(dark_path, shape, base_value=70.0, exptime=30.0)
        _generate_mock_fits(flat_path, shape, base_value=120.0, exptime=30.0)
        _generate_mock_fits(sci_path, shape, base_value=170.0, exptime=30.0, cr_spike=True)

        print("Running image_preprocessing orchestrator...")
        output = image_preprocessing(
            input_path=sci_path,
            bias_path=bias_path,
            dark_path=dark_path,
            flat_path=flat_path,
            output_path=out_path,
            remove_cosmic_rays=True,
            update_header={"OBJECT": "QA_TEST", "TELESCOP": "Mock"},
        )

        # Assertions
        assert os.path.exists(out_path), f"Output file not found: {out_path}"
        print(f"[OK] Output file exists: {out_path}")

        with fits.open(out_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data

        assert np.dtype(data.dtype).kind == "f" and np.dtype(data.dtype).itemsize == 4, (
            f"Expected 32-bit float, got {data.dtype}"
        )
        print(f"[OK] Output dtype is 32-bit float ({data.dtype})")

        # Expected calibrated value = 100.0
        np.testing.assert_allclose(data, 100.0, rtol=1e-5)
        print(f"[OK] Calibrated values correct (expected 100.0)")

        # Check cosmic ray spike is attenuated
        assert data[50, 50] < 1000.0, f"CR spike not removed: {data[50, 50]}"
        print(f"[OK] Cosmic ray spike attenuated: {data[50, 50]:.2f} < 1000")

        # Header provenance
        assert header.get("CALIBRAT") is not None, "Missing CALIBRAT keyword"
        assert header.get("MBIAS") is not None, "Missing MBIAS keyword"
        assert header.get("MDARK") is not None, "Missing MDARK keyword"
        assert header.get("MFLAT") is not None, "Missing MFLAT keyword"
        assert header.get("OBJECT") == "QA_TEST", f"OBJECT mismatch: {header.get('OBJECT')}"
        assert header.get("TELESCOP") == "Mock", f"TELESCOP mismatch: {header.get('TELESCOP')}"
        print(f"[OK] Header provenance keywords present")

        # HISTORY entries
        history = [str(h) for h in header.get("HISTORY", [])]
        assert any("bias" in h.lower() for h in history), "Missing bias HISTORY"
        assert any("dark" in h.lower() for h in history), "Missing dark HISTORY"
        assert any("flat" in h.lower() for h in history), "Missing flat HISTORY"
        print(f"[OK] HISTORY entries present")

        print("\n============================================")
        print("QA PASSED — All assertions passed")
        print("============================================")
        return 0


if __name__ == "__main__":
    sys.exit(main())
