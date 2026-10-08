#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         linespreadfunction.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Measured line-spread function widths, per fiber and wavelength.

The line-spread function itself is not stored: what the calibration measures is
its width, as a full width at half maximum, for a few isolated arc lines.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["LineSpreadFunction"]


class LineSpreadFunction:
    """Full width at half maximum of arc lines, per fiber and line.

    Parameters
    ----------
    target_wls
        Wavelength of each measured line, shape ``(n_wls,)``.
    fwhm
        Full width at half maximum, shape ``(n_wls, n_fibers)``.  ``nan`` marks
        a line that could not be fitted.
    fiber_ids
        Fiber ID of each column of *fwhm*.

    Raises
    ------
    ValueError
        If the shapes of *target_wls*, *fwhm* and *fiber_ids* disagree.

    Examples
    --------
    >>> lsf = LineSpreadFunction(
    ...     target_wls=np.array([4657.9, 4764.9]),
    ...     fwhm=np.full((2, 3), 3.0),
    ...     fiber_ids=np.arange(3),
    ... )
    >>> lsf.resolution.shape
    (2, 3)
    """

    def __init__(
        self,
        target_wls: NDArray[np.floating],
        fwhm: NDArray[np.floating],
        fiber_ids: NDArray[np.integer],
    ) -> None:
        self.target_wls = np.asarray(target_wls, dtype=float)
        if self.target_wls.ndim != 1:
            raise ValueError(
                f"target_wls must be 1-D, got shape {self.target_wls.shape}"
            )

        self.fwhm = np.asarray(fwhm, dtype=float)
        if self.fwhm.ndim != 2:
            raise ValueError(f"fwhm must be 2-D, got shape {self.fwhm.shape}")

        n_wls, n_fibers = self.fwhm.shape
        if n_wls != self.target_wls.size:
            raise ValueError(
                f"fwhm has {n_wls} rows, but target_wls holds "
                f"{self.target_wls.size} wavelengths."
            )

        self.fiber_ids = np.asarray(fiber_ids, dtype=int)
        if self.fiber_ids.shape != (n_fibers,):
            raise ValueError(
                f"fiber_ids shape {self.fiber_ids.shape} must be ({n_fibers},)"
            )

    # ------------------------------------------------------------------ #
    #  Derived quantities
    # ------------------------------------------------------------------ #

    @property
    def resolution(self) -> NDArray[np.floating]:
        """Resolving power ``R = wavelength / fwhm``, of shape ``(n_wls, n_fibers)``."""
        return self.target_wls[:, None] / self.fwhm

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Mask of measurements that produced a finite width."""
        return np.isfinite(self.fwhm)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def n_wls(self) -> int:
        """Number of measured lines."""
        return self.target_wls.size

    @property
    def n_fibers(self) -> int:
        """Number of fibers."""
        return self.fwhm.shape[1]
