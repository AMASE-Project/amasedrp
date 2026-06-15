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
from typing import Any, Self

import numpy as np
from astropy.io import fits

DEFAULT_GAIN = 1.0  # Default gain value in electrons/ADU (NOTE: check what it is for our CMOS camera)


class Image:
    """
    Class to handle 2D CMOS image data.
    """

    def __init__(
        self, data: np.ndarray, header: fits.Header, filename: str | None = None
    ) -> None:
        self.filename: str | None = filename
        self.header: fits.Header = header
        self.ADU: np.ndarray = data.astype(np.float32)  # raw data in ADU
        gain: float = self.gain if self.gain is not None else DEFAULT_GAIN
        self.electrons: np.ndarray = self.ADU * gain
        self.data: np.ndarray = self.electrons  # data for processing, in electrons
        # NOTE: the error & mask systems are still under development

    ########################################################################
    # I/O
    ########################################################################

    @classmethod
    def from_fits(cls, filename: str) -> Self:
        """Create an Image object by reading a FITS file."""
        abs_filename = os.path.abspath(os.path.expanduser(filename))
        with fits.open(abs_filename) as hdul:
            hdu: fits.PrimaryHDU | Any = hdul[0]
            data: np.ndarray = hdu.data
            header: fits.Header = hdu.header
        return cls(data=data, header=header, filename=abs_filename)

    def write_to_fits(
        self, filename: str, update_header: dict[str, Any] | None = None
    ) -> None:
        """Write the image to a FITS file.

        Args:
            filename: Path to the output FITS file.
            update_header: Optional dictionary of header keywords to update
                before writing. A copy of the current header is modified so
                the original ``self.header`` is not mutated.
        """
        header = self.header.copy() if update_header is not None else self.header
        if update_header is not None:
            header.update(update_header)
        hdu = fits.PrimaryHDU(self.ADU, header=header)
        hdu.writeto(filename, overwrite=True)

    ########################################################################
    # image properties (e.g., exposure time, gain, readout noise, etc.)
    ########################################################################

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
        value = self.header.get("GAIN", default=None)
        if value is None or not isinstance(value, (int, float)):
            return None
        return float(value)

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

    #########################################################################
    # utilities
    #########################################################################

    def copy(self) -> "Image":
        """Create a deep copy of the Image object."""
        return Image(
            data=copy.deepcopy(self.ADU),
            header=copy.deepcopy(self.header),
            filename=self.filename,
        )

    def cutout(self, x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
        """Extract a cutout from the image data."""
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
