#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         image_preprocessing.py
@Time:         2026/04/09 16:59:59
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for image preprocessing,
               including e.g., bias subtraction, dark subtraction,
               pixel flat-field correction, cosmic ray removal, etc.
"""

import logging
from typing import Any

import numpy as np

from ..utils.logging import configure_logging
from .classes.image import Image

logger: logging.Logger = logging.getLogger(__name__)

_VALID_STEPS: tuple[str, str, str] = ("bias", "dark", "pixflat")


def _validate_inputs(
    input_image: Image,
    master_bias_image: Image,
    master_dark_image: Image,
    master_pixflat_image: Image,
    steps: tuple[str, ...],
) -> None:
    """Validate calibration input images.

    Checks that all inputs are ``Image`` instances with non-None data,
    that all data arrays have matching shapes, and that EXPTIME values
    required by the requested calibration steps are present and positive.

    Args:
        input_image: The raw science image or any other image to be calibrated.
        master_bias_image: Master bias frame.
        master_dark_image: Master dark frame.
        master_pixflat_image: Master pixel flat field.
        steps: Requested calibration steps.

    Raises:
        ValueError: If any validation check fails.
    """
    # Type checks
    for name, img in [
        ("input_image", input_image),
        ("master_bias_image", master_bias_image),
        ("master_dark_image", master_dark_image),
        ("master_pixflat_image", master_pixflat_image),
    ]:
        if not isinstance(img, Image):
            raise ValueError(
                f"{name} must be an Image instance, got {type(img).__name__}"
            )
        if img.data is None:
            raise ValueError(f"{name} has no data (data is None)")

    # Shape checks — all calibration frames must match the science frame
    ref_shape = input_image.data.shape
    for name, img in [
        ("master_bias_image", master_bias_image),
        ("master_dark_image", master_dark_image),
        ("master_pixflat_image", master_pixflat_image),
    ]:
        if img.data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: input_image has shape {ref_shape}, "
                f"but {name} has shape {img.data.shape}"
            )

    # EXPTIME checks (only for steps that actually need it)
    if "dark" in steps:
        for name, img, needed_for in [
            ("master_dark_image", master_dark_image, "dark subtraction"),
            ("input_image", input_image, "dark subtraction"),
        ]:
            exptime = img.exptime
            if exptime is None:
                raise ValueError(
                    f"{name} has no EXPTIME keyword, required for {needed_for}"
                )
            if exptime <= 0:
                raise ValueError(
                    f"{name} EXPTIME={exptime} must be positive, "
                    f"required for {needed_for}"
                )

    if "pixflat" in steps:
        for name, img, needed_for in [
            ("master_pixflat_image", master_pixflat_image, "pixel flat-field correction"),
        ]:
            exptime = img.exptime
            if exptime is None:
                raise ValueError(
                    f"{name} has no EXPTIME keyword, required for {needed_for}"
                )
            if exptime <= 0:
                raise ValueError(
                    f"{name} EXPTIME={exptime} must be positive, "
                    f"required for {needed_for}"
                )


def image_calibration(
    input_image: Image,
    master_bias_image: Image,
    master_dark_image: Image,
    master_pixflat_image: Image,
    steps: tuple[str, ...] = ("bias", "dark", "pixflat"),
    remove_cosmic_rays: bool = False,
    cr_kwargs: dict[str, Any] | None = None,
) -> Image:
    """Calibrate a science image with bias, dark, and pixel flat-field corrections.

    The calibration follows standard detector physics:

    * bias    = bias_level + readout_noise
    * dark    = dark_current * t_exp + bias
    * pixflat = illumination * pixel_response + dark_current * t_exp + bias
    * science = signal * pixel_response + dark_current * t_exp + bias

    Steps applied (in fixed order by request)::

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
        remove_cosmic_rays: If ``True``, detect and remove cosmic rays from
            the *input* image **before** any calibration steps.
        cr_kwargs: Optional keyword arguments forwarded to
            ``Image.detect_cosmic_rays()``.  Ignored when
            *remove_cosmic_rays* is ``False``.

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
    if input_image.header.get("CALIBRAT"):
        logger.warning(
            "Input image already has CALIBRAT keyword; it may have been calibrated."
        )
    if not steps:
        logger.warning(
            "No calibration steps requested; image will be returned unchanged."
        )

    # -- validate steps -------------------------------------------------------
    for step in steps:
        if step not in _VALID_STEPS:
            raise ValueError(f"Invalid step '{step}'. Valid steps are: {_VALID_STEPS}")

    # -- validate inputs ------------------------------------------------------
    _validate_inputs(
        input_image,
        master_bias_image,
        master_dark_image,
        master_pixflat_image,
        steps,
    )

    # -- create output copy ---------------------------------------------------
    output_image = input_image.copy()

    # -- cosmic ray removal (before any calibration) -------------------------
    if cr_kwargs and not remove_cosmic_rays:
        logger.warning(
            "cr_kwargs provided but remove_cosmic_rays is False; "
            "cr_kwargs will be ignored."
        )

    if remove_cosmic_rays:
        logger.info("Applying cosmic ray removal...")
        cr_kw = cr_kwargs or {}
        _crmask, cleanarr = input_image.detect_cosmic_rays(**cr_kw)
        output_image.data = cleanarr.astype(np.float32)
        logger.info("Cosmic ray removal applied.")
    else:
        # Cast to float32 proactively — prevents integer overflow during
        # arithmetic on unsigned-integer master frames.
        output_image.data = output_image.data.astype(np.float32)

    # -- pre-compute calibration data -----------------------------------------
    # All master arrays are explicitly cast to float32 before arithmetic
    # to avoid accidental integer wrapping.
    bias_data: np.ndarray = master_bias_image.data.astype(np.float32)
    dark_data: np.ndarray = master_dark_image.data.astype(np.float32)
    pixflat_data: np.ndarray = master_pixflat_image.data.astype(np.float32)

    darkcurr: np.ndarray | None = None
    pixresp: np.ndarray | None = None

    # darkcurr is needed by the "dark" step (subtraction) and by
    # the "pixflat" step (through pixflat/pixresp calculation).
    if "dark" in steps or "pixflat" in steps:
        dark_exptime = master_dark_image.exptime
        assert dark_exptime is not None and dark_exptime > 0  # validated
        darkcurr = (dark_data - bias_data) / dark_exptime

    if "pixflat" in steps:
        pixflat_exptime = master_pixflat_image.exptime
        assert pixflat_exptime is not None and pixflat_exptime > 0  # validated
        assert darkcurr is not None  # guard — computed above
        pixflat: np.ndarray = (pixflat_data - bias_data) - darkcurr * pixflat_exptime
        median_pixflat: float = float(np.nanmedian(pixflat))
        if median_pixflat == 0.0 or np.isnan(median_pixflat):
            raise ValueError(
                f"Median of pixel flat field is {median_pixflat}; "
                "cannot compute pixel flat-field correction (division by zero)."
            )
        pixresp = pixflat / median_pixflat

    # -- apply steps in fixed logical order ----------------------------------
    for step in steps:
        if step == "bias":
            logger.info("Applying bias subtraction...")
            output_image.data -= bias_data
            logger.info("Bias subtraction applied.")

        elif step == "dark":
            logger.info("Applying dark subtraction...")
            input_exptime = input_image.exptime
            assert input_exptime is not None and input_exptime > 0  # validated
            assert darkcurr is not None  # guard — computed above
            output_image.data -= darkcurr * input_exptime
            logger.info("Dark subtraction applied.")

        elif step == "pixflat":
            logger.info("Applying pixel flat-field correction...")
            assert pixresp is not None  # guard — computed above
            output_image.data /= pixresp
            logger.info("Pixel flat-field correction applied.")

    # -- header provenance ---------------------------------------------------
    output_image.header["CALIBRAT"] = True
    output_image.header["MBIAS"] = (
        master_bias_image.filename if master_bias_image.filename else "unknown"
    )
    output_image.header["MDARK"] = (
        master_dark_image.filename if master_dark_image.filename else "unknown"
    )
    output_image.header["MPIXFLT"] = (
        master_pixflat_image.filename if master_pixflat_image.filename else "unknown"
    )
    for step in steps:
        output_image.header.add_history(f"{step} calibration applied")

    return output_image


def image_preprocessing(
    input_path: str,
    bias_path: str,
    dark_path: str,
    pixflat_path: str,
    output_path: str,
    steps: tuple[str, ...] = ("bias", "dark", "pixflat"),
    remove_cosmic_rays: bool = False,
    cr_kwargs: dict[str, Any] | None = None,
    update_header: dict[str, Any] | None = None,
    log_file: str | None = None,
) -> Image:
    """Read, calibrate, and persist a science image.

    High-level orchestrator that reads master calibration frames and a
    science frame from disk, runs ``image_calibration``, writes the
    result to a FITS file, and returns the calibrated ``Image`` object.

    Args:
        input_path: Path to the science FITS file.
        bias_path: Path to the master bias FITS file.
        dark_path: Path to the master dark FITS file.
        pixflat_path: Path to the master pixel flat FITS file.
        output_path: Destination path for the calibrated FITS file.
        steps: Calibration steps to apply.
        remove_cosmic_rays: Whether to apply cosmic ray removal.
        cr_kwargs: Optional keyword arguments for cosmic ray detection.
        update_header: Optional dictionary of FITS header keywords to
            write into the output file (in addition to the provenance
            keywords set by ``image_calibration``).
        log_file: Optional path to a log file. If provided, the
            preprocessing module logger is configured to write *INFO*
            and *WARNING* messages to this file.

    Returns:
        The calibrated ``Image`` object.
    """
    if log_file is not None:
        configure_logging(log_file)

    if cr_kwargs and not remove_cosmic_rays:
        logger.warning(
            "cr_kwargs provided but remove_cosmic_rays is False; "
            "cr_kwargs will be ignored."
        )

    input_image = Image.from_fits(input_path)
    master_bias = Image.from_fits(bias_path)
    master_dark = Image.from_fits(dark_path)
    master_pixflat = Image.from_fits(pixflat_path)

    output = image_calibration(
        input_image=input_image,
        master_bias_image=master_bias,
        master_dark_image=master_dark,
        master_pixflat_image=master_pixflat,
        steps=steps,
        remove_cosmic_rays=remove_cosmic_rays,
        cr_kwargs=cr_kwargs,
    )

    output.write_to_fits(output_path, update_header=update_header)
    return output
