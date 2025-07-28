#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         fibers.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Classes and functions for identifying fiber blocks,
               detecting fibers, and performing fiber tracing.
'''

from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import Legendre
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from numba import jit, prange


class Fibers():
    """
    Class for handling fibers and performing fiber tracing.
    """

    def __init__(
        self,
        fflat_img_data: NDArray[Any] = None,
        n_blocks_expected: int = 19,
        n_fibers_per_block_expected: int = 29,
    ):
        self.fflat_img_data = fflat_img_data  # x: cross-dispersion
        self.n_blocks_expected = n_blocks_expected
        self.n_fibers_per_block_expected = n_fibers_per_block_expected

    def identifyBlocks(
            self,
            disp_band_center_row: Optional[int] = None,
            disp_band_half_width: int = 100,
    ) -> None:
        # extract a cross-dispersion profile for identifying fiber blocks
        if disp_band_center_row is not None:
            center_row = disp_band_center_row
        else:
            center_row = self.fflat_img_data.shape[0] // 2
        start_row = int(center_row - disp_band_half_width)
        end_row = int(center_row + disp_band_half_width)
        band = self.fflat_img_data[start_row:end_row, :]
        profile = np.nanmedian(band, axis=0)
        profile_xs = np.arange(len(profile))
        # identification: fiber blocks
        # find the "valleys" of the profile (i.e., edges of each block)
        distence = (len(profile) / self.n_blocks_expected) / 2.
        valleys, _ = find_peaks(-profile, distance=distence)
        blocks_edge = profile_xs[valleys]
        del distence
        # only the true valleys (with values low enough)
        threshold = np.nanmedian(profile) * 0.3
        cond = profile[blocks_edge] <= threshold
        blocks_edge = blocks_edge[cond]
        del cond
        # identify the block edges (median value between is high enough)
        list_ = np.array([], dtype=int)
        for i in range(len(blocks_edge) - 1):
            v = np.nanmedian(profile[blocks_edge[i]:blocks_edge[i+1]])
            if v >= threshold:
                list_ = np.append(list_, [i, i+1])
            del i, v
        blocks_edge = blocks_edge[np.unique(list_)]
        del list_
        # ensure the number of blocks is as expected
        if (len(blocks_edge) - 1) != self.n_blocks_expected:
            raise ValueError(
                f"Expected {self.n_blocks_expected} blocks, "
                f"but found {len(blocks_edge) - 1}."
            )
        # store the results
        self.n_blocks = len(blocks_edge) - 1
        self.blocks_edge = blocks_edge
        self.blocks_center = np.array(
            [np.nanmean(blocks_edge[i:i+2])
             for i in range(len(blocks_edge) - 1)]
        )
        self.cross_disp_profile_xs = profile_xs
        self.cross_disp_profile = profile
        self.disp_band_center_row = center_row

    def identifyFibers(
        self,
        disp_band_center_row: Optional[int] = None,
        disp_band_half_width: int = 100,
    ) -> None:
        """
        Fiber identification and tracing are both based on fiber flat image.
        The approach largely follows the methodology used in the DESI pipeline.
        (But with a little bit )
        """
        # fiber blocks identification
        self.identifyBlocks(
            disp_band_center_row=disp_band_center_row,
            disp_band_half_width=disp_band_half_width,
        )
        # fibers identification
        peak_xs = np.array([], dtype=int)
        for i in range(self.n_blocks):
            # identify fibers in each block
            cutout_profile_xs = self.cross_disp_profile_xs[
                self.blocks_edge[i]:self.blocks_edge[i+1]]
            cutout_profile = self.cross_disp_profile[
                self.blocks_edge[i]:self.blocks_edge[i+1]]
            # soomth the profile before peak finding
            sigma = (
                np.ptp(cutout_profile_xs)
                / self.n_fibers_per_block_expected
                / 10.
            )
            cutout_profile = gaussian_filter1d(cutout_profile, sigma=sigma)
            height = np.nanmedian(cutout_profile) * 0.5
            distance = int(
                np.ptp(cutout_profile_xs)
                / self.n_fibers_per_block_expected
                * 0.5
            )
            peaks, _ = find_peaks(
                cutout_profile, height=height, distance=distance)
            peak_xs = np.append(peak_xs, cutout_profile_xs[peaks])
            del peaks, height, distance
        # store the results
        self.n_fibers = len(peak_xs)
        self.approx_xs = peak_xs

    def traceFibers(
        self,
        identify_disp_band_center_row: Optional[int] = None,
        identify_disp_band_half_width: int = 100,
        tracing_max_shift: float = 1.,
        tracing_cdisp_half_width: int = 3,
        tracing_threshold_fraction: float = 0.1,
        legendre_fitting: bool = True,
        legendre_fitting_deg: int = 10,
    ) -> None:
        # fibers identification
        if hasattr(self, 'approx_xs'):
            pass
        else:
            self.identifyFibers(
                disp_band_center_row=identify_disp_band_center_row,
                disp_band_half_width=identify_disp_band_half_width,
            )
        # fiber tracing
        # trace the barycenter positions of all fibers
        barycenter_traces = _trace_fibers_barycenter_positions(
            image_data=self.fflat_img_data,
            n_fibers=self.n_fibers,
            fibers_ini_row=self.disp_band_center_row,
            fibers_approx_position=self.approx_xs,
            tracing_max_shift=tracing_max_shift,
            tracing_cdisp_half_width=tracing_cdisp_half_width,
            tracing_threshold_fraction=tracing_threshold_fraction)
        # fiber traces
        traces = []
        for idx in range(self.n_fibers):
            trace = {}
            trace['FiberID'] = f'{idx:03d}'
            trace['Barycenter'] = barycenter_traces[idx]
            if legendre_fitting:
                try:
                    trace['LegendreFittingModel'] \
                        = _legendre_fitting_barycenter_trace(
                            trace['Barycenter'], deg=legendre_fitting_deg)
                except ValueError:
                    trace['LegendreFittingModel'] = None
                    print(
                        f"Warning: Legendre fitting failed for fiber {idx:03d}."  # noqa: E501 NOTE: Optimize this part!!!
                    )
            traces.append(trace)
        # store the results
        self.traces = np.array(traces)


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
        ini_row: int,
        ini_guess_position: float,
        max_shift: float = 1.,  # NOTE: max_shift=1. may be too small for some cases??  # noqa: E501
        cdisp_half_width: int = 3,
        threshold_fraction: float = 0.1) -> list:
    n_rows = image_data.shape[0]
    trace = np.full(n_rows, -1., dtype=float)
    # initial row
    trace[ini_row] = _calculate_fiber_barycenter_position(
        image_data=image_data,
        row=ini_row,
        guess_position=ini_guess_position,
        max_shift=max_shift,
        cdisp_half_width=cdisp_half_width,
        threshold_fraction=threshold_fraction,
    )
    # upward (from initial row to top row)
    for i in range(ini_row - 1, -1, -1):
        trace[i] = _calculate_fiber_barycenter_position(
            image_data=image_data,
            row=i,
            guess_position=trace[i + 1],
            max_shift=max_shift,
            cdisp_half_width=cdisp_half_width,
            threshold_fraction=threshold_fraction,
        )
        # stop tracing if no valid position found
        if trace[i] < 0.:
            break
    # downward (from initial row to bottom row)
    for i in range(ini_row + 1, n_rows, 1):
        trace[i] = _calculate_fiber_barycenter_position(
            image_data=image_data,
            row=i,
            guess_position=trace[i - 1],
            max_shift=max_shift,
            cdisp_half_width=cdisp_half_width,
            threshold_fraction=threshold_fraction,
        )
        # stop tracing if no valid position found
        if trace[i] < 0.:
            break
    return trace


@jit(nopython=True, parallel=True)
def _trace_fibers_barycenter_positions(
        image_data: NDArray[np.floating],
        n_fibers: int,
        fibers_ini_row: int,
        fibers_approx_position: NDArray[np.integer],
        tracing_max_shift: float = 1.,
        tracing_cdisp_half_width: int = 3,
        tracing_threshold_fraction: float = 0.1) -> NDArray[np.floating]:
    n_rows = image_data.shape[0]
    traces = np.full((n_fibers, n_rows), -1., dtype=float)
    for i in prange(n_fibers):
        traces[i, :] = _trace_fiber_barycenter_positions(
            image_data=image_data,
            ini_row=fibers_ini_row,
            ini_guess_position=fibers_approx_position[i],
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
