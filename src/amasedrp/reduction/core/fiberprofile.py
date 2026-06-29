#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         fiberprofile.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Normalized cross-dispersion PSF model per fiber.

FiberProfile stores the spatial profile of each fiber measured from a
master flat-field image.  It is required for optimal extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from .fibermap import FiberMap

__all__ = ["FiberProfile"]


class FiberProfile:
    """Normalized cross-dispersion profile model.

    Parameters
    ----------
    profile
        Array of shape ``(n_fibers, n_rows, n_offsets)``.
    x_offsets
        1-D array of cross-dispersion offsets (pixels) corresponding to
        the last axis of *profile*.
    fibermap
        Optional per-fiber metadata table.
    meta
        Optional metadata dictionary.

    Examples
    --------
    >>> profile = np.ones((10, 512, 7))
    >>> offsets = np.arange(-3, 4, dtype=float)
    >>> fp = FiberProfile(profile, offsets)
    """

    def __init__(
        self,
        profile: NDArray[np.floating],
        x_offsets: NDArray[np.floating],
        fibermap: FiberMap | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.profile = np.asarray(profile, dtype=np.float64)
        if self.profile.ndim != 3:
            raise ValueError(f"profile must be 3-D, got shape {self.profile.shape}")

        self.x_offsets = np.asarray(x_offsets, dtype=np.float64)
        if self.x_offsets.ndim != 1:
            raise ValueError(f"x_offsets must be 1-D, got shape {self.x_offsets.shape}")

        if self.profile.shape[-1] != len(self.x_offsets):
            raise ValueError(
                f"profile last axis ({self.profile.shape[-1]}) must match "
                f"len(x_offsets) ({len(self.x_offsets)})"
            )

        n_fibers = self.profile.shape[0]
        if fibermap is not None and len(fibermap) != n_fibers:
            raise ValueError(
                f"fibermap has {len(fibermap)} rows, but profile has {n_fibers} fibers"
            )
        self.fibermap = fibermap

        self.meta = dict(meta) if meta is not None else {}

        # Normalize rows with positive finite sums to sum-to-one.
        self._normalize()

    def _normalize(self) -> None:
        """Ensure each finite profile row sums to 1."""
        profile = self.profile
        sums = np.nansum(profile, axis=-1, keepdims=True)
        positive = (sums > 0) & np.isfinite(sums)
        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.where(positive, profile / sums, profile)
        # For rows with zero/non-finite sums, fall back to centered delta.
        zero_rows = ~positive.squeeze(-1)
        if zero_rows.any():
            n_offsets = profile.shape[-1]
            center = n_offsets // 2
            profile[zero_rows, :] = 0.0
            profile[zero_rows, center] = 1.0
        self.profile = profile

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def n_fibers(self) -> int:
        """Number of fibers."""
        return self.profile.shape[0]

    @property
    def n_rows(self) -> int:
        """Number of spectral rows."""
        return self.profile.shape[1]

    @property
    def n_offsets(self) -> int:
        """Number of cross-dispersion offset samples."""
        return self.profile.shape[2]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def at(self, fiber_index: int, row: int) -> NDArray[np.floating]:
        """Return the profile slice for one fiber at one row.

        Parameters
        ----------
        fiber_index
            0-indexed fiber.
        row
            0-indexed spectral row.

        Returns
        -------
        ndarray
            1-D array of shape ``(n_offsets,)``.
        """
        return self.profile[fiber_index, row, :]

    # ------------------------------------------------------------------ #
    #  I/O
    # ------------------------------------------------------------------ #

    def to_fits(self, path: str | Path, overwrite: bool = False) -> None:
        """Write the FiberProfile to a FITS file.

        HDU layout:
        - Primary: metadata header
        - PROFILE: 3-D profile array
        - XOFFSETS: 1-D offset array
        - FIBERMAP: binary table (if present)
        """
        path = Path(path)
        hdul = fits.HDUList()

        header = fits.Header()
        for key, value in self.meta.items():
            if isinstance(value, (str, int, float, bool)):
                header[key] = value
        header["N_FIBERS"] = self.n_fibers
        header["N_ROWS"] = self.n_rows
        header["N_OFFSETS"] = self.n_offsets
        hdul.append(fits.PrimaryHDU(header=header))

        hdul.append(fits.ImageHDU(data=self.profile, name="PROFILE"))
        hdul.append(fits.ImageHDU(data=self.x_offsets, name="XOFFSETS"))

        if self.fibermap is not None:
            hdul.append(fits.BinTableHDU(self.fibermap, name="FIBERMAP"))

        hdul.writeto(path, overwrite=overwrite)

    @classmethod
    def from_fits(cls, path: str | Path) -> Self:
        """Read a FiberProfile from a FITS file."""
        path = Path(path)
        with fits.open(path) as hdul:
            header = hdul[0].header
            meta = {k: v for k, v in header.items() if k not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND") and not k.startswith("NAXIS")}

            profile = hdul["PROFILE"].data
            x_offsets = hdul["XOFFSETS"].data

            fibermap = None
            if "FIBERMAP" in hdul:
                fibermap = FiberMap(hdul["FIBERMAP"].data)

        return cls(
            profile=profile,
            x_offsets=x_offsets,
            fibermap=fibermap,
            meta=meta,
        )
