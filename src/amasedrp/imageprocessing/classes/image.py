#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         image.py
@Time:         2026/04/09 15:48:04
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Class for handling 2D CMOS image data.
"""

import copy
import os
from typing import Any, Self

import numpy as np
from astropy.io import fits
from astroscrappy import (
    detect_cosmics,  # type: ignore[import] Cython extension, not in type stubs
)


class Image:
    """
    Class to handle 2D CMOS image data.
    """

    def __init__(
        self, data: np.ndarray, header: fits.Header, filename: str | None = None
    ) -> None:
        self.data: np.ndarray = data
        self.header: fits.Header = header
        self.filename: str | None = filename

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

    # NOTE: When to use this method vs. `from_fits()`?
    # NOTE: Maybe we can only keep `from_fits()`?
    def read_from_fits(self, filename: str) -> None:
        """Read the image from a FITS file."""
        with fits.open(filename) as hdul:
            hdu: fits.PrimaryHDU | Any = hdul[0]
            data: np.ndarray = hdu.data
            header: fits.Header = hdu.header
        self.data = data
        self.header = header
        self.filename = filename

    def write_to_fits(self, filename: str, update_header: dict[str, Any] | None = None) -> None:
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
        hdu = fits.PrimaryHDU(self.data, header=header)
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
    def readout_noise(self) -> float | None:
        """Readout noise of the detector, in electrons.

        The value is read from the ``RDNOISE`` keyword in the FITS header.

        Returns:
            float: The readout noise if the ``RDNOISE`` keyword is present and valid.
            None: If the keyword is missing or has an invalid value.
        """
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
            data=copy.deepcopy(self.data),
            header=copy.deepcopy(self.header),
            filename=self.filename,
        )

    def cutout(self, x_start: int, y_start: int, x_end: int, y_end: int) -> np.ndarray:
        """Extract a cutout from the image."""
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

    ########################################################################
    # image processing
    ########################################################################

    def detect_cosmic_rays(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect cosmic rays in the image using the L.A.Cosmic algorithm,
        based on Laplacian edge detection (van Dokkum 2001).

        This method is designed to be efficient and fast, utilizing
        the C/Cython implementation of `astroscrappy.detect_cosmics()`.

        References:
        - van Dokkum (2001):
            https://iopscience.iop.org/article/10.1086/323894
        - astroscrappy GitHub:
            https://github.com/astropy/astroscrappy
        - astroscrappy Docs:
            https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html
        """
        crmask, cleanarr = detect_cosmics(self.data, **kwargs)
        return crmask, cleanarr
