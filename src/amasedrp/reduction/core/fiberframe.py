#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         fiberframe.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Extracted 2-D row-stacked spectra container.

FiberFrame holds the canonical intermediate data product passed from
``reduction/`` to ``calibration/``.  Each row is one fiber's 1-D spectrum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from .fibermap import FiberMap

__all__ = ["FiberFrame"]

#: Primary-header keywords that describe the file rather than the reduction.
#: ``to_fits`` writes them, and ``from_fits`` leaves them out of ``meta`` so
#: that ``meta`` survives a round trip unchanged.
_STRUCTURAL_KEYS = frozenset(
    {
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "COMMENT",
        "HISTORY",
        "N_FIBERS",
        "N_WAVE",
    }
)

#: Python types that a FITS header keyword round-trips unchanged.
_HEADER_SCALARS = (str, bool, int, float)


def _header_value(key: str, value: Any) -> Any:
    """Return *value* as a FITS header scalar, or ``None`` to write nothing.

    Parameters
    ----------
    key
        Metadata key, used in the error message.
    value
        Value to convert.  ``None`` writes no keyword.  A numpy scalar is
        converted to its Python equivalent.

    Returns
    -------
    Any
        A scalar of type ``str``, ``bool``, ``int`` or ``float``, or ``None``.

    Raises
    ------
    TypeError
        If *value* has no FITS header representation.  Metadata is provenance,
        so a value that cannot be stored is an error, not something that
        disappears without a word.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, _HEADER_SCALARS):
        return value
    raise TypeError(
        f"meta[{key!r}] is {type(value).__name__}, which has no FITS header "
        f"representation; pass a scalar, or put per-fiber arrays in the fibermap."
    )


class FiberFrame:
    """Container for extracted row-stacked spectra.

    Parameters
    ----------
    wave
        Wavelength array.  Either 1-D ``(n_wave,)`` shared by all fibers,
        or 2-D ``(n_fibers, n_wave)`` per-fiber.
    flux
        Extracted flux, shape ``(n_fibers, n_wave)``.
    ivar
        Inverse variance.  Defaults to ones.
    mask
        Bitmask.  Defaults to zeros.
    fibermap
        Per-fiber metadata table.
    meta
        Scalar metadata, written to the primary FITS header by :meth:`to_fits`.
        A value of ``None`` writes no keyword.  Any other value without a FITS
        header representation raises :exc:`TypeError` instead of disappearing.

    Examples
    --------
    >>> frame = FiberFrame(
    ...     wave=np.arange(1000, dtype=float),
    ...     flux=np.ones((15, 1000)),
    ... )
    >>> frame.n_fibers
    15
    """

    def __init__(
        self,
        wave: NDArray[np.floating],
        flux: NDArray[np.floating],
        ivar: NDArray[np.floating] | None = None,
        mask: NDArray[np.integer] | None = None,
        fibermap: FiberMap | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.flux = np.asarray(flux, dtype=np.float64)
        if self.flux.ndim != 2:
            raise ValueError(f"flux must be 2-D, got shape {self.flux.shape}")

        self.wave = np.asarray(wave, dtype=np.float64)
        if self.wave.ndim not in (1, 2):
            raise ValueError(f"wave must be 1-D or 2-D, got shape {self.wave.shape}")

        n_fibers, n_wave = self.flux.shape

        if self.wave.ndim == 1 and self.wave.shape[0] != n_wave:
            raise ValueError(
                f"wave shape {self.wave.shape} incompatible with flux shape {self.flux.shape}"
            )
        if self.wave.ndim == 2 and self.wave.shape != (n_fibers, n_wave):
            raise ValueError(
                f"wave shape {self.wave.shape} incompatible with flux shape {self.flux.shape}"
            )

        if ivar is None:
            self.ivar = np.ones_like(self.flux, dtype=np.float64)
        else:
            self.ivar = np.asarray(ivar, dtype=np.float64)
            if self.ivar.shape != (n_fibers, n_wave):
                raise ValueError(
                    f"ivar shape {self.ivar.shape} incompatible with flux shape {self.flux.shape}"
                )

        if mask is None:
            self.mask = np.zeros((n_fibers, n_wave), dtype=np.uint32)
        else:
            self.mask = np.asarray(mask, dtype=np.uint32)
            if self.mask.shape != (n_fibers, n_wave):
                raise ValueError(
                    f"mask shape {self.mask.shape} incompatible with flux shape {self.flux.shape}"
                )

        if fibermap is not None and len(fibermap) != n_fibers:
            raise ValueError(
                f"fibermap has {len(fibermap)} rows, but flux has {n_fibers} fibers"
            )
        self.fibermap = fibermap

        self.meta = dict(meta) if meta is not None else {}

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def n_fibers(self) -> int:
        """Number of fibers (rows)."""
        return self.flux.shape[0]

    @property
    def n_wave(self) -> int:
        """Number of wavelength samples (columns)."""
        return self.flux.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_fibers, n_wave)``."""
        return self.flux.shape

    # ------------------------------------------------------------------ #
    #  I/O
    # ------------------------------------------------------------------ #

    def to_fits(self, path: str | Path, overwrite: bool = False) -> None:
        """Write the FiberFrame to a FITS file.

        HDU layout:
        - Primary: metadata header
        - WAVE: wavelength array
        - FLUX: extracted flux
        - IVAR: inverse variance
        - MASK: bitmask
        - FIBERMAP: binary table (if present)

        Parameters
        ----------
        path
            Output file.
        overwrite
            Replace an existing file.

        Raises
        ------
        TypeError
            If a ``meta`` value has no FITS header representation.  See
            :func:`_header_value`.
        """
        path = Path(path)
        hdul = fits.HDUList()

        # Primary HDU: metadata only (no data)
        header = fits.Header()
        for key, value in self.meta.items():
            scalar = _header_value(key, value)
            if scalar is not None:
                header[key] = scalar
        header["N_FIBERS"] = self.n_fibers
        header["N_WAVE"] = self.n_wave
        hdul.append(fits.PrimaryHDU(header=header))

        # Data HDUs
        hdul.append(fits.ImageHDU(data=self.wave, name="WAVE"))
        hdul.append(fits.ImageHDU(data=self.flux, name="FLUX"))
        hdul.append(fits.ImageHDU(data=self.ivar, name="IVAR"))
        hdul.append(fits.ImageHDU(data=self.mask, name="MASK"))

        if self.fibermap is not None:
            hdul.append(fits.BinTableHDU(self.fibermap, name="FIBERMAP"))

        hdul.writeto(path, overwrite=overwrite)

    @classmethod
    def from_fits(cls, path: str | Path) -> Self:
        """Read a FiberFrame from a FITS file.

        Parameters
        ----------
        path
            File to read, written by :meth:`to_fits`.

        Returns
        -------
        FiberFrame
            The frame.  ``meta`` holds the primary-header keywords except the
            structural ones, so ``meta`` of a written frame reads back
            unchanged.
        """
        path = Path(path)
        with fits.open(path) as hdul:
            header = hdul[0].header
            meta = {
                k: v
                for k, v in header.items()
                if k not in _STRUCTURAL_KEYS and not k.startswith("NAXIS")
            }

            wave = hdul["WAVE"].data
            flux = hdul["FLUX"].data
            ivar = hdul["IVAR"].data
            mask = hdul["MASK"].data

            fibermap = None
            if "FIBERMAP" in hdul:
                fibermap = FiberMap(hdul["FIBERMAP"].data)

        return cls(
            wave=wave,
            flux=flux,
            ivar=ivar,
            mask=mask,
            fibermap=fibermap,
            meta=meta,
        )
