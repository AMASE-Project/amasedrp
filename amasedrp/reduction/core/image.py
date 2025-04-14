#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         image.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Classes and functions used for image preprocessing.
'''

import copy
from typing import Any, Optional
from numpy.typing import NDArray
from astropy.io import fits
from astroscrappy import detect_cosmics


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
            return self.header['EXPTIME']
        else:
            return None

    def rejectCosmicRays(
            self,
            sigclip: float = 4.5,
            sigfrac: float = 0.3,
            niter: int = 4,
            overwrite: bool = True,
            verbose: bool = False,
            ) -> Optional[tuple]:
        """
        Detect and reject cosmic rays using the L.A.Cosmic algorithm,
        based on Laplacian edge detection (van Dokkum 2001).

        This method is designed to be efficient and fast, utilizing
        the C/Cython implementation of `astroscrappy.detect_cosmics()`.

        Parameters
        ----------
        sigclip : float, optional
            Laplacian-to-noise limit for cosmic ray detection.
            Lower values will flag more pixels as cosmic rays. By default 4.5
        sigfrac : float, optional
            Fractional detection limit for neighboring pixels.
            For cosmic ray neighbor pixels, a lapacian-to-noise detection limit
            of sigfrac * sigclip will be used. By default 0.3
        niter : int, optional
            Number of iterations of the LA Cosmic algorithm to perform.
            By default 4
        overwrite : bool, optional
            If True, overwrite `self.data` with cleaned data and update masks.
            If False, return the cleaned results and cosmic ray mask.
            By default True.
        verbose : bool, optional
            Print detailed processing information to the screen or not.
            By default False.

        Returns
        -------
        Optional[tuple]:
            A tuple (mask, clean) if overwrite is False, otherwise None.
            `mask` is a boolean array indicating detected cosmic rays, and
            `clean` is the cleaned image.

        References:
        -------
        - van Dokkum (2001):
            https://iopscience.iop.org/article/10.1086/323894
        - astroscrappy GitHub:
            https://github.com/astropy/astroscrappy
        - astroscrappy Docs:
            https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html
        """
        # TODO: add/adjust/test more default parameters for this function.
        mask, clean = detect_cosmics(
            self.data,
            inmask=None,
            inbkg=None,
            invar=None,
            sigclip=sigclip,
            sigfrac=sigfrac,
            objlim=5.0,
            gain=1.0,
            readnoise=6.5,
            satlevel=65536.0,
            niter=niter,
            sepmed=True,
            cleantype='meanmask',
            fsmode='median',
            psfmodel='gauss',
            psffwhm=2.5,
            psfsize=7,
            psfk=None,
            psfbeta=4.765,
            verbose=verbose,
        )

        # overwrite the original data with the cleaned data and update the mask
        if overwrite:
            self.data = clean
            self.mask_cosmic_rays = mask
            # propagate the mask
            self.mask = self.mask | mask if self.mask is not None else mask
        else:
            return mask, clean
