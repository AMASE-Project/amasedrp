#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         image.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Classes and functions used for image preprocessing.
'''

import copy
from astropy.io import fits
from typing import Any, Optional
from numpy.typing import NDArray


class Image():
    """
    Class to handle image data and conduct image preprocessing operations.
    """

    def __init__(
        self,
        data: NDArray[Any] = None,
        mask: NDArray[Any] = None,
        header: Optional[dict] = None,
        type: Optional[str] = None,
    ):
        """Initialize the Image class.

        Parameters
        ----------
        data : NDArray[Any], optional
            Image data array.
        mask : NDArray[Any], optional
            Image mask array.
        header : dict, optional
            Image header.
        type : str, optional
            Type of the image (e.g., 'bias', 'dark', 'flat', 'science').
        """
        self.data = data
        self.mask = mask
        self.header = header
        self.type = type

    def copy(self) -> "Image":
        """Create a deep copy of the Image object."""
        return Image(
            data=copy.deepcopy(self.data),
            mask=copy.deepcopy(self.mask),
            header=copy.deepcopy(self.header),
            type=self.type
        )

    def readFitsData(self, filename: str) -> None:
        """Read the image data from a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.
        """
        # read the FITS file
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        # update the class attributes
        self.filename = filename
        self.data = data
        self.header = header
        self.type = header['IMGTYPE']  # TODO: We should discuss what keyword to use to describe the image type (i.e., bias, dark, flat, science).  # noqa

    def writeFitsData(self, filename: str) -> None:
        """Write the image data to a FITS file.

        Parameters
        ----------
        filename : str
            Path to save the FITS file.
        """
        # write the FITS file
        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(filename, overwrite=True)

    @property
    def dimensions(self) -> Optional[tuple]:
        """Get the dimensions of the image data.

        Returns
        -------
        tuple
            Dimensions of the image data.
        """
        if self.data is not None:
            return self.data.shape
        else:
            return None

    @property
    def exptime(self) -> Optional[float]:
        """Get the exposure time of the image.

        Returns
        -------
        float
            Exposure time of the image.
        """
        if self.header is not None:
            return self.header['EXPTIME']  # TODO: Which keyword to use
        else:
            return None

    # TODO: Finish the following method
    def rejectCosmicRays(self):
        pass
