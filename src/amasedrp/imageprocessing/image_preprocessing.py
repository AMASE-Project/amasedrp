#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         image_preprocessing.py
@Time:         2026/04/09 16:59:59
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for image preprocessing,
               including e.g., bias subtraction, dark subtraction,
               pixel flat-field correction, etc.
"""

import logging
from typing import Any

from ..utils.logging import configure_logging
from .core.image import Image
from .methods import bias, cosmic, dark, flat

logger: logging.Logger = logging.getLogger(__name__)

_VALID_STEPS = frozenset(("bias", "dark", "pixflat"))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_image(img: Image, name: str) -> None:
    if not isinstance(img, Image):
        raise ValueError(f"{name} must be an Image instance, got {type(img).__name__}")
    if img.data is None:
        raise ValueError(f"{name} has no data (data is None)")


def _validate_shape(reference: Image, *others: tuple[str, Image]) -> None:
    ref_shape = reference.data.shape
    for name, img in others:
        if img.data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: input_image has shape {ref_shape}, "
                f"but {name} has shape {img.data.shape}"
            )


def _validate_exptime(img: Image, name: str, context: str) -> None:
    exptime: float | None = img.exptime
    if exptime is None:
        raise ValueError(f"{name} has no EXPTIME keyword, required for {context}")
    if exptime <= 0:
        raise ValueError(
            f"{name} EXPTIME={exptime} must be positive, required for {context}"
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _warn_if_already_calibrated(header: Any) -> None:
    if header.get("CALIBRAT") and header.get("CALIBRAT") is True:
        logger.warning(
            "Input image header indicates it has already been calibrated (CALIBRAT=True)."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _image_calibration(
    input_image: Image,
    master_bias_image: Image,
    master_dark_image: Image,
    master_pixflat_image: Image,
    steps: tuple[str, ...] = ("bias", "dark", "pixflat"),
) -> Image:
    """Calibrate the inout (science) image with bias, dark, and pixel flat-field corrections.

    The calibration follows standard detector physics:

    * bias    = bias_level + readout_noise
    * dark    = dark_current * t_exp + bias
    * pixflat = illumination * pixel_response + dark_current * t_exp + bias
    * science = signal * pixel_response + dark_current * t_exp + bias

    Steps applied (in fixed logical order by request)::

        dark_curr = (dark - bias) / dark_exptime
        pixflat    = (pixflat - bias) - dark_curr * pixflat_exptime
        pix_resp   = pixflat / median(pixflat)

        S1 = science - bias
        S2 = S1 - dark_curr * science_exptime
        S3 = S2 / pix_resp

    Args:
        input_image: The science ``Image`` to calibrate.
        master_bias_image: Master bias frame.
        master_dark_image: Master dark frame.
        master_pixflat_image: Master pixel flat field.
        steps: Calibration steps to apply.  Valid values are ``"bias"``,
            ``"dark"``, ``"pixflat"``.  Steps are applied in the fixed logical
            order regardless of the tuple order.

    Returns:
        A new calibrated ``Image`` whose header carries provenance keywords
        (``CALIBRAT``, ``MBIAS``, ``MDARK``, ``MPIXFLT``, and ``HISTORY``
        entries for each applied step).

    Raises:
        ValueError: If any input is not an ``Image``, data is missing,
            shapes do not match, required EXPTIME values are absent or
            non-positive, a requested *step* is invalid, or the median of
            the pixel flat field is zero/NaN.
    """
    if not steps:
        logger.warning(
            "No calibration steps requested; image will be returned unchanged."
        )
        return input_image.copy()

    _warn_if_already_calibrated(input_image.header)

    # -- validate steps -------------------------------------------------------
    invalid: list[str] = [s for s in steps if s not in _VALID_STEPS]
    if invalid:
        raise ValueError(
            f"Invalid step(s) {invalid}. Valid steps are: {tuple(_VALID_STEPS)}"
        )

    # -- build the list of images that are *actually* required ----------------
    # Bias is always needed because even dark/flat frames contain bias.
    required: list[tuple[str, Image]] = [
        ("input_image", input_image),
        ("master_bias_image", master_bias_image),
    ]
    if "dark" in steps or "pixflat" in steps:
        required.append(("master_dark_image", master_dark_image))
    if "pixflat" in steps:
        required.append(("master_pixflat_image", master_pixflat_image))

    # -- validate type, data presence, and shape consistency ------------------
    for name, img in required:
        _validate_image(img, name)
    _validate_shape(input_image, *required[1:])  # skip the reference itself

    # -- validate exposure times for steps that need them ----------------------
    if "dark" in steps:
        _validate_exptime(input_image, "input_image", "dark subtraction")
        _validate_exptime(master_dark_image, "master_dark_image", "dark subtraction")
    if "pixflat" in steps:
        _validate_exptime(
            master_pixflat_image, "master_pixflat_image", "pixel flat-field correction"
        )

    # -- prepare working copy -------------------------------------------------
    output_image: Image = input_image.copy()
    # Cast to float32 proactively — prevents integer overflow during
    # arithmetic on unsigned-integer master frames.
    output_image.data = output_image.data.astype("float32")

    # -- apply calibration steps -----------------------------------------------
    step_dispatch: dict[str, Any] = {
        "bias": lambda img: bias.subtract_bias(img, master_bias_image),
        "dark": lambda img: dark.subtract_dark(
            img, master_dark_image, master_bias_image
        ),
        "pixflat": lambda img: flat.apply_pixel_flat(
            img, master_pixflat_image, master_bias_image, master_dark_image
        ),
    }

    for step in steps:
        logger.info(f"Applying {step} calibration...")
        output_image = step_dispatch[step](output_image)
        logger.info(f"{step} calibration applied.")

    # -- header provenance ---------------------------------------------------
    output_image.header["CALIBRAT"] = True
    output_image.header["MBIAS"] = master_bias_image.filename or "unknown"
    output_image.header["MDARK"] = master_dark_image.filename or "unknown"
    output_image.header["MPIXFLT"] = master_pixflat_image.filename or "unknown"

    return output_image


def image_preprocessing(
    input_path: str,
    bias_path: str,
    dark_path: str,
    pixflat_path: str,
    output_path: str,
    steps: tuple[str, ...] = ("bias", "dark", "pixflat"),
    update_header: dict[str, Any] | None = None,
    log_file: str | None = None,
    cosmic_removal: bool = False,
    **cr_kwargs: Any,
) -> Image:
    """Image preprocessing: calibration (incl. bias, dark, pixel flat) + cosmic ray removal.

    High-level orchestrator that reads master calibration frames and a
    science frame from disk, runs ``_image_calibration``, writes the
    result to a FITS file, and returns the calibrated ``Image`` object.

    Args:
        input_path: Path to the science FITS file.
        bias_path: Path to the master bias FITS file.
        dark_path: Path to the master dark FITS file.
        pixflat_path: Path to the master pixel flat field FITS file.
        output_path: Destination path for the calibrated FITS file.
        steps: Calibration steps to apply.
        update_header: Optional dictionary of FITS header keywords to
            write into the output file (in addition to the provenance
            keywords set by ``_image_calibration``).
        log_file: Optional path to a log file. If provided, the
            preprocessing module logger is configured to write *INFO*
            and *WARNING* messages to this file.

    Returns:
        The calibrated ``Image`` object.
    """
    # -- configure logging ------------------------------------------------------
    if log_file is not None:
        configure_logging(log_file)

    # -- read input and master frames ------------------------------------------------
    input_image: Image = Image.from_fits(input_path)
    master_bias: Image = Image.from_fits(bias_path)
    master_dark: Image = Image.from_fits(dark_path)
    master_pixflat: Image = Image.from_fits(pixflat_path)

    # -- apply calibration steps -----------------------------------------------
    output: Image = _image_calibration(
        input_image=input_image,
        master_bias_image=master_bias,
        master_dark_image=master_dark,
        master_pixflat_image=master_pixflat,
        steps=steps,
    )

    # cosmic ray removal
    # NOTE: check whether it should be always applied after calibration
    if cosmic_removal:
        output = cosmic.remove_cosmic_rays(output, **cr_kwargs)

    output.write_to_fits(output_path, update_header=update_header)
    return output
