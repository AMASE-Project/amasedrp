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
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import Legendre
from astropy.io import fits
from astroscrappy import detect_cosmics
from scipy.signal import find_peaks
from numba import jit, prange


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

    def keepOnlyTrimSection(self) -> None:
        """Keep only the trimmed section of the image data."""
        # check if 'TRIMSEC' keyword exists in the header
        if 'TRIMSEC' not in self.header:
            raise KeyError("The 'TRIMSEC' keyword is not found in the header.")
        # parse the TRIMSEC keyword
        trimsec = self.header['TRIMSEC'].strip('[]')
        col_range, row_range = trimsec.split(',')
        col_start, col_end = map(int, col_range.split(':'))
        row_start, row_end = map(int, row_range.split(':'))
        # covert to 0-based indexing
        col_start, col_end = col_start - 1, col_end - 1
        row_start, row_end = row_start - 1, row_end - 1
        # keep only the trimmed section of the image data
        self.data = self.data[row_start:row_end + 1, col_start:col_end + 1]
        # update the header
        del self.header['TRIMSEC']
        # TODO: anything else to update in the header?

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

    def identifyFibers(
            self,
            disp_band_half_width: int = 50,
            threshold_fraction: float = 0.5) -> tuple:
        """Fiber identification and tracing are both based on the continuum
        lamp image (#TODO: check this).
        The approach largely follows the methodology used in the DESI pipeline.
        For details refer to their paper (Guy et al. 2023).

        The fiber identification process is as follows:
        1.  central band (a number of central rows) from the image are selected
            (assuming the x-axis is cross-dispersion direction and the y-axis
            is dispersion direction), and the median cross-dispersion profile
            of these rows is calculated.s
        2.  isolated peaks are identified in this profile, to estimate the
            total number of fibers and their approximate x-axis positions.

        Parameters
        ----------
        disp_band_half_width : int, optional
            Half width of the central band for fiber identification.
            By default 50.
        threshold_fraction : float, optional
            Only peaks with a height greater than this fraction of the maximum
            intensity of the profile are considered.
            By default 0.5.

        Returns
        -------
        n_fibers : int
            The number of fibers identified in the image.
        fiber_approx_positions : NDArray[np.integer]
            The approximate x-axis positions of the fibers.
            This is a 1D array of length n_fibers.
        """
        center_row = self.dimensions[0] // 2
        start_row = int(center_row - disp_band_half_width)
        end_row = int(center_row + disp_band_half_width)
        band = self.data[start_row:end_row, :]
        profile = np.nanmedian(band, axis=0)
        peaks, _ = find_peaks(
            profile, height=threshold_fraction*np.nanmax(profile))
        n_fibers = len(peaks)
        fiber_approx_positions = peaks
        return n_fibers, fiber_approx_positions

    def traceFibers(
            self,
            idensity_fibers_disp_band_half_width: int = 50,
            idensity_fibers_threshold_fraction: float = 0.5,
            tracing_max_shift: float = 1.,
            tracing_cdisp_half_width: int = 3,
            tracing_threshold_fraction: float = 0.1,
            legendre_fitting: bool = True,
            legendre_fitting_deg: int = 10) -> NDArray[Any]:
        """
        # TODO: add more details about the fiber tracing process.
        """
        # identify the fibers and estimate their approximate positions
        n_fibers, fiber_approx_positions = self.identifyFibers(
            disp_band_half_width=idensity_fibers_disp_band_half_width,
            threshold_fraction=idensity_fibers_threshold_fraction)
        # trace the barycenter positions of all fibers
        barycenter_traces = _trace_fibers_barycenter_positions(
            image_data=self.data,
            n_fibers=n_fibers,
            fiber_approx_positions=fiber_approx_positions,
            tracing_max_shift=tracing_max_shift,
            tracing_cdisp_half_width=tracing_cdisp_half_width,
            tracing_threshold_fraction=tracing_threshold_fraction)
        # fiber traces
        traces = []
        for idx in range(n_fibers):
            trace = {}
            trace['FiberID'] = f'{idx:03d}'
            trace['Barycenter'] = barycenter_traces[idx]
            if legendre_fitting:
                trace['LegendreFittingModel'] \
                    = _legendre_fitting_barycenter_trace(
                        trace['Barycenter'], deg=legendre_fitting_deg)
            traces.append(trace)
        traces = np.array(traces)
        return traces


@jit(nopython=True)
def _calculate_fiber_barycenter_position(
        image_data: NDArray[np.floating],
        row: int,
        guess_position: float,
        max_shift: float = 1.,
        cdisp_half_width: int = 3,
        threshold_fraction: float = 0.1) -> float:
    barycenter = -1.
    if guess_position >= 0:
        n_cols = image_data.shape[1]
        col_start = guess_position - cdisp_half_width
        col_end = guess_position + cdisp_half_width + 1
        col_start = round(max(col_start, 0))
        col_end = round(min(col_end, n_cols - 1))
        profile = image_data[row, col_start:col_end]
        if np.nansum(profile) > (
            threshold_fraction * np.nanmax(image_data)
        ):
            col_range = np.arange(col_start, col_end, 1)
            barycenter = (
                np.nansum(profile * col_range) / np.nansum(profile)
            )
            if np.abs(barycenter - guess_position) > max_shift:
                barycenter = -1.
    return barycenter


@jit(nopython=True)
def _trace_fiber_barycenter_positions(
        image_data: NDArray[np.floating],
        ini_guess_position: float,
        max_shift: float = 1.,
        cdisp_half_width: int = 3,
        threshold_fraction: float = 0.1) -> list:
    n_rows = image_data.shape[0]
    center_row = n_rows // 2
    trace = np.full(n_rows, ini_guess_position, dtype=float)
    # center row
    trace[center_row] = _calculate_fiber_barycenter_position(
        image_data=image_data,
        row=center_row,
        guess_position=ini_guess_position,
        max_shift=max_shift,
        cdisp_half_width=cdisp_half_width,
        threshold_fraction=threshold_fraction,
    )
    # upward (from center row to top row)
    for i in range(center_row - 1, -1, -1):
        trace[i] = _calculate_fiber_barycenter_position(
            image_data=image_data,
            row=i,
            guess_position=trace[i + 1],
            max_shift=max_shift,
            cdisp_half_width=cdisp_half_width,
            threshold_fraction=threshold_fraction,
        )
    # downward (from center row to bottom row)
    for i in range(center_row + 1, n_rows, 1):
        trace[i] = _calculate_fiber_barycenter_position(
            image_data=image_data,
            row=i,
            guess_position=trace[i - 1],
            max_shift=max_shift,
            cdisp_half_width=cdisp_half_width,
            threshold_fraction=threshold_fraction,
        )
    return trace


@jit(nopython=True, parallel=True)
def _trace_fibers_barycenter_positions(
        image_data: NDArray[np.floating],
        n_fibers: int,
        fiber_approx_positions: NDArray[np.integer],
        tracing_max_shift: float = 1.,
        tracing_cdisp_half_width: int = 3,
        tracing_threshold_fraction: float = 0.1) -> NDArray[np.floating]:
    n_rows = image_data.shape[0]
    traces = np.full((n_fibers, n_rows), -1., dtype=float)
    for i in prange(n_fibers):
        traces[i, :] = _trace_fiber_barycenter_positions(
            image_data=image_data,
            ini_guess_position=fiber_approx_positions[i],
            max_shift=tracing_max_shift,
            cdisp_half_width=tracing_cdisp_half_width,
            threshold_fraction=tracing_threshold_fraction)
    return traces


def _legendre_fitting_barycenter_trace(barycenter_trace, deg=10):
    n_rows = len(barycenter_trace)
    mask = barycenter_trace >= 0.
    data_x = np.arange(n_rows)[mask]
    data_y = barycenter_trace[mask]
    model = Legendre.fit(
        data_x, data_y, deg=deg, domain=[np.nanmin(data_x), np.nanmax(data_x)])
    return model
