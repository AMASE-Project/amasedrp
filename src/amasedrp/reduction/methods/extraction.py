#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         extraction.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Spectral extraction methods.

Placeholder module for boxcar, optimal, and spectro-perfectionism
extraction algorithms.  Will be implemented in a future iteration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_boxcar(
    image: NDArray[np.floating],
    trace_positions: NDArray[np.floating],
    aperture_radius: int = 3,
) -> NDArray[np.floating]:
    """Boxcar extraction: sum flux within a fixed aperture around each trace.

    Parameters
    ----------
    image
        2-D calibrated science image.
    trace_positions
        Array of shape ``(n_fibers, n_rows)`` with trace x-positions.
    aperture_radius
        Half-width of the extraction aperture in pixels.

    Returns
    -------
    ndarray
        Extracted 1-D spectra of shape ``(n_fibers, n_rows)``.

    Raises
    ------
    NotImplementedError
        This is a placeholder.
    """
    raise NotImplementedError("boxcar extraction is not yet implemented.")


def extract_optimal(
    image: NDArray[np.floating],
    trace_positions: NDArray[np.floating],
    psf_model: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Optimal extraction using a PSF model.

    Parameters
    ----------
    image
        2-D calibrated science image.
    trace_positions
        Array of shape ``(n_fibers, n_rows)`` with trace x-positions.
    psf_model
        PSF profile array.

    Returns
    -------
    ndarray
        Extracted 1-D spectra of shape ``(n_fibers, n_rows)``.

    Raises
    ------
    NotImplementedError
        This is a placeholder.
    """
    raise NotImplementedError("optimal extraction is not yet implemented.")
