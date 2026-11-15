#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         image.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Class for handling 2D CMOS image data.
"""

import copy
import os
import warnings
from typing import Any, Self

import numpy as np
from astropy.io import fits

import amasecore

from ...products import stamp_core_version

DEFAULT_GAIN = 1.0  # Default gain value in electrons/ADU (NOTE: check what it is for our CMOS camera)

#: Keyword recording which night's header-fix rules were applied, if any.
#: ``none`` means the night has no rules, ``unknown`` that the night itself
#: could not be read.  Products carry the value they were made with, so a
#: frame that already holds this card is left alone.
HDRFIX_KEY = "HDRFIX"


def night_from_header(header: fits.Header) -> str | None:
    """Return the observing night recorded in a header, as ``YYYYMMDD``.

    Parameters
    ----------
    header
        A FITS header.

    Returns
    -------
    str or None
        The UTC date of ``DATE-OBS`` without separators, or ``None`` when the
        card is absent or does not start with a readable date.

    Notes
    -----
    The night is the UTC date of ``DATE-OBS``, the one time stamp that travels
    with the frames.  It is not the directory name: the 2025-07 sweep sits in a
    directory called ``20250705`` and its frames say ``2025-07-04``.
    """
    raw = header.get("DATE-OBS")
    if not isinstance(raw, str) or len(raw) < 10:
        return None
    digits = raw[:10].replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        return None
    return digits


def _apply_night_fixes(header: fits.Header, filename: str) -> None:
    """Repair a raw frame's header with amasecore's rules for its night.

    Parameters
    ----------
    header
        The header of the frame, changed in place.
    filename
        Path of the frame.  Only its basename is used, matched against each
        rule's glob.

    Notes
    -----
    The repair is recorded rather than silently applied.  ``HDRFIX`` names the
    night whose rules were used, and each card that changed gets a ``HISTORY``
    line, which is what the archive shows a reader years later.

    A frame that already carries ``HDRFIX`` is a product and is left alone:
    its card says what it was made from, and re-reading it must not rewrite
    that.  Raw frames are never modified on disk, so they are always stamped
    from the rules that exist now.
    """
    if HDRFIX_KEY in header:
        return

    night = night_from_header(header)
    if night is None:
        header[HDRFIX_KEY] = "unknown"
        header.add_history("amasecore hdrfix: DATE-OBS holds no readable night")
        return

    path = amasecore.hdrfix_path(night)
    if not path.exists():
        header[HDRFIX_KEY] = "none"
        header.add_history(f"amasecore hdrfix: no rules for night {night}")
        return

    changes = amasecore.apply_hdrfix(header, amasecore.load_hdrfix(night), filename)
    header[HDRFIX_KEY] = night
    if not changes:
        header.add_history(f"amasecore hdrfix {night}: no rule matched this frame")
    for keyword, old, new in changes:
        header.add_history(f"amasecore hdrfix {night}: {keyword} {old!r} -> {new!r}")


class Image:
    """
    Class to handle 2D CMOS image data.
    """

    def __init__(
        self,
        data: np.ndarray,
        header: fits.Header,
        filename: str | None = None,
        unit: str = "adu",
    ) -> None:
        if unit not in ("adu", "electron"):
            raise ValueError(f"unit must be 'adu' or 'electron', got {unit!r}")
        self.filename: str | None = filename
        self.header: fits.Header = header
        gain: float = self.gain if self.gain is not None else DEFAULT_GAIN
        # The electron array is the single stored array.  ``ADU`` is derived
        # from it, so a calibration step cannot leave the two out of step.
        if unit == "adu":
            self.data: np.ndarray = data.astype(np.float32) * gain
        else:  # unit == "electron"
            self.data: np.ndarray = data.astype(np.float32)
        self.header["BUNIT"] = unit
        if np.any(self.data < 0.0):
            warnings.warn(
                "Image data contains negative values, which may indicate an issue with the data or the gain value."
            )  # NOTE: Need to check if this is expected for our CMOS camera (e.g., due to bias subtraction) or if it indicates a problem.

    ########################################################################
    # I/O
    ########################################################################

    @classmethod
    def from_fits(cls, filename: str) -> Self:
        """Create an Image object by reading a FITS file.

        The ``BUNIT`` header keyword is used to determine the physical unit of
        the data array.  If absent, the data is assumed to be in ADU.

        Parameters
        ----------
        filename
            Path to the input FITS file.

        Returns
        -------
        Image
            The image read from ``filename``.
        """
        abs_filename = os.path.abspath(os.path.expanduser(filename))
        with fits.open(abs_filename) as hdul:
            hdu: fits.PrimaryHDU | Any = hdul[0]
            data: np.ndarray | None = hdu.data
            header: fits.Header = hdu.header
        if data is None:
            # Reading the first HDU that holds data would turn a multi-extension
            # product, such as a FiberFrame, into a wrong single array.
            raise ValueError(
                f"{abs_filename} holds no data in its primary HDU; this is not "
                f"an image file."
            )
        unit = header.get("BUNIT", "adu").lower()  # type: ignore[assignment]
        if unit not in ("adu", "electron"):
            unit = "adu"
        _apply_night_fixes(header, abs_filename)
        return cls(data=data, header=header, filename=abs_filename, unit=unit)

    def write_to_fits(
        self, filename: str, update_header: dict[str, Any] | None = None
    ) -> None:
        """Write the image to a FITS file.

        The raw ``ADU`` array is written to disk, and the ``BUNIT`` header
        keyword is set to ``"adu"`` to indicate the physical unit.  ``ADU`` is
        derived from the stored electron array by :attr:`ADU`.

        Parameters
        ----------
        filename
            Path to the output FITS file.
        update_header
            Optional dictionary of header keywords to update before writing.
            A copy of the current header is modified, so the original
            ``self.header`` is not mutated.
        """
        header = self.header.copy() if update_header is not None else self.header
        if update_header is not None:
            header.update(update_header)
        header["BUNIT"] = "adu"
        stamp_core_version(header)
        hdu = fits.PrimaryHDU(self.ADU, header=header)
        hdu.writeto(filename, overwrite=True)

    ########################################################################
    # image properties (e.g., exposure time, gain, readout noise, etc.)
    ########################################################################

    @property
    def ADU(self) -> np.ndarray:
        """The raw ADU array, derived from the stored electron array.

        ``data`` holds electrons and is the only stored array.  This property
        divides it by the gain, so a calibration step that rewrites ``data``
        is reflected here, and therefore in :meth:`write_to_fits`.

        Returns
        -------
        ndarray
            A newly allocated array, in ADU.
        """
        gain: float = self.gain if self.gain is not None else DEFAULT_GAIN
        return self.data / gain

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.data.shape if self.data is not None else None

    @property
    def exptime(self) -> float | None:
        value = self.header.get("EXPTIME", default=None)
        if value is None or not isinstance(value, (int, float)):
            return None
        return float(value)

    @property
    def gain(self) -> float | None:
        """Electronic gain, in electrons per ADU.

        ``EGAIN`` carries the physical gain of this camera.  ``GAIN``, which
        the acquisition software also writes, is the camera gain setting, so it
        only serves as a fallback.
        """
        for keyword in ("EGAIN", "GAIN"):
            value = self.header.get(keyword, default=None)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @property
    def rdnoise(self) -> float | None:
        """Readout noise of the detector, in electrons."""
        # NOTE: what is our keyword for readout noise? check it. make sure it's in electrons, not ADU.
        value = self.header.get("RDNOISE", default=None)
        if value is None or not isinstance(value, (int, float)):
            return None
        return float(value)

    @property
    def imgtype(self) -> str | None:
        value = self.header.get("IMAGETYP", default=None)
        if value is None or not isinstance(value, str):
            return None
        return value

    @property
    def history(self) -> list[str] | None:
        value = self.header.get("HISTORY", default=None)
        if value is None:
            return None
        return list(value)

    #########################################################################
    # utilities
    #########################################################################

    def copy(self) -> "Image":
        """Create a deep copy of the Image object.

        Returns
        -------
        Image
            A copy with deep-copied data and header.
        """
        return Image(
            data=copy.deepcopy(self.data),
            header=copy.deepcopy(self.header),
            filename=self.filename,
            unit="electron",
        )

    def cutout(self, x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
        """Extract a cutout from the image data.

        Parameters
        ----------
        x_start, x_end
            Column range of the cutout, ``x_start`` inclusive and ``x_end``
            exclusive.
        y_start, y_end
            Row range of the cutout, ``y_start`` inclusive and ``y_end``
            exclusive.

        Returns
        -------
        numpy.ndarray
            The cutout, with shape ``(y_end - y_start, x_end - x_start)``.

        Raises
        ------
        ValueError
            If the image has no data, or the cutout is out of bounds.
        """
        if self.data is None:
            raise ValueError("Image data is not available.")
        if (
            x_start < 0
            or y_start < 0
            or x_end > self.data.shape[1]
            or y_end > self.data.shape[0]
        ):
            raise ValueError("Cutout coordinates are out of bounds.")
        return self.data[y_start:y_end, x_start:x_end]
