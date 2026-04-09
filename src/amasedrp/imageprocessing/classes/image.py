#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         image.py
@Time:         2026/04/09 15:48:04
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Class for handling 2D CMOS image data.
'''

import os
from astropy.io import fits
from astroscrappy import detect_cosmics


class Image:
    """
    Class to handle 2D CMOS image data.
    """

    def __init__(self, data, header):
        self.data = data
        self.header = header

    ########################################################################
    # I/O
    ########################################################################

    @classmethod
    def from_fits(cls, filename):
        """Create an Image object by reading a FITS file."""
        filename = os.path.abspath(os.path.expanduser(filename))
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        image = cls(data=data, header=header)
        image.filename = filename
        return image

    def readFromFits(self, filename):
        """Read the image from a FITS file."""
        with fits.open(filename) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header
        self.filename = filename

    def writeToFits(self, filename):
        """Write the image to a FITS file."""
        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(filename, overwrite=True)

    ########################################################################
    # image properties (e.g., exposure time, gain, readout noise, etc.)
    ########################################################################

    @property
    def shape(self):
        return self.data.shape if self.data is not None else None

    @property
    def exptime(self):
        return self.header.get('EXPTIME', default=None)

    @property
    def gain(self):
        return self.header.get('GAIN', default=None)

    @property
    def imgtype(self):
        return self.header.get('IMAGETYP', default=None)

    ########################################################################
    # image processing
    ########################################################################

    def detectCosmicRays(self, **kwargs):
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
