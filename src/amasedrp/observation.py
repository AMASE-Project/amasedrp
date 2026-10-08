#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         observation.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Select frames of an observation by their header values.

This module is the one place that maps a physical quantity to a FITS header
keyword.  The AMASE metadata scheme is not settled, so keyword names and values
are not a stable contract; see ``AGENTS.md``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.verify import VerifyError

from .preprocessing.core.image import Image

__all__ = [
    "LAMP_ARC",
    "LAMP_FLAT",
    "collimator_position",
    "load_frame",
    "reduced_dir",
    "select_frames",
]

_KEYWORD_CHANNEL = "CHANNEL"
_KEYWORD_LAMP = "LAMP"
_KEYWORD_COL_FOC = "COL-FOC"

# Lamp header values.  ``IMAGETYP`` cannot replace them: in the red channel of
# the 2025-07 collimator sweep it reads ``'LIGHT'`` for arcs and flats alike, so
# a selection by ``IMAGETYP`` returns both and reports no error.
LAMP_ARC = "TH-AR"
LAMP_FLAT = "QTH"

# Tolerance for the collimator position, which is written as an integer number
# of steps by some tools and as a float by others.
_COL_FOC_ATOL = 1e-6


def select_frames(
    directory: str | Path,
    channel: str | None = None,
    lamp: str | None = None,
    col_foc: float | None = None,
) -> list[Path]:
    """Select the frames of a directory that match the given header values.

    Every provided criterion must match; a criterion left as ``None`` is not
    applied.  Text values are matched without regard to case or padding.

    Parameters
    ----------
    directory
        Directory to scan for ``*.fits`` files.
    channel
        Spectrograph channel, for example ``"blue"``.
    lamp
        Lamp, usually :data:`LAMP_ARC` or :data:`LAMP_FLAT`.
    col_foc
        Collimator focus position, in steps.

    Returns
    -------
    list of pathlib.Path
        Matching paths, sorted by name.

    Raises
    ------
    ValueError
        If *directory* is not a directory.

    Notes
    -----
    A file whose header cannot be read, or whose identifying card cannot be
    parsed, is skipped with a warning.  Both happen in real data: five frames
    of the 2025-07 sweep had no ``LAMP`` card, and two had an unparsable
    ``COL-FOC`` card.  A corrupt ``COL-FOC`` card only matters while filtering
    on *col_foc*; such a frame stays selectable by channel and lamp.

    Examples
    --------
    >>> arcs = select_frames("sweep", channel="blue", lamp=LAMP_ARC)
    >>> len(arcs)
    12
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory.")

    wanted_channel = None if channel is None else channel.strip().upper()
    wanted_lamp = None if lamp is None else lamp.strip().upper()

    selected: list[Path] = []
    for path in sorted(directory.glob("*.fits")):
        try:
            header = fits.getheader(path)
        except (OSError, VerifyError) as exc:
            warnings.warn(
                f"{path.name}: header unreadable ({exc}); frame skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        try:
            frame_channel = header.get(_KEYWORD_CHANNEL)
            frame_lamp = header.get(_KEYWORD_LAMP)
        except VerifyError as exc:
            warnings.warn(
                f"{path.name}: {exc}; frame skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if wanted_channel is not None:
            if str(frame_channel).strip().upper() != wanted_channel:
                continue
        if wanted_lamp is not None:
            if str(frame_lamp).strip().upper() != wanted_lamp:
                continue

        # Only read the position when it is asked for, so that a frame with a
        # corrupt COL-FOC card is still selectable by channel and lamp.
        if col_foc is not None:
            try:
                frame_col_foc = header.get(_KEYWORD_COL_FOC)
            except VerifyError as exc:
                warnings.warn(
                    f"{path.name}: cannot read {_KEYWORD_COL_FOC} ({exc}); "
                    f"frame skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if frame_col_foc is None:
                continue
            try:
                distance = abs(float(frame_col_foc) - float(col_foc))
            except (TypeError, ValueError):
                warnings.warn(
                    f"{path.name}: cannot read {_KEYWORD_COL_FOC} "
                    f"({frame_col_foc!r}); frame skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if distance > _COL_FOC_ATOL:
                continue

        selected.append(path)

    return selected


def load_frame(
    path: str | Path,
    shape: tuple[int, int] | None = None,
) -> Image:
    """Read a frame and put it in the pipeline orientation.

    The camera writes frames with the dispersion axis along axis 1.  The
    pipeline keeps the dispersion axis along axis 0, so that ``FiberFrame.wave``
    and the trace positions share one convention.  No header keyword records
    which orientation a file holds, so the caller states the expected shape and
    this function is the one place that rotates a frame.

    Parameters
    ----------
    path
        FITS file to read.
    shape
        Expected ``(n_rows, n_cols)`` in the pipeline orientation.  The frame
        is transposed when the file holds the transpose of *shape*.  ``None``
        accepts the file as stored.

    Returns
    -------
    Image
        The frame, with its dispersion axis along axis 0.

    Raises
    ------
    ValueError
        If the stored shape is neither *shape* nor its transpose.

    Examples
    --------
    >>> flat = load_frame("sweep/B136.fits")
    >>> bias = load_frame("bias/Blue/Bias-Blue-0001.fits", shape=flat.data.shape)
    """
    image = Image.from_fits(str(path))
    if shape is None:
        return image

    stored = image.data.shape
    if stored == shape:
        return image
    if stored == shape[::-1]:
        image.data = image.data.T
        return image
    raise ValueError(
        f"{Path(path).name} holds shape {stored}, which is neither {shape} "
        f"nor its transpose; the frame cannot be put in the pipeline "
        f"orientation."
    )


def collimator_position(path: str | Path) -> float:
    """Return the collimator focus position recorded in a frame header.

    Parameters
    ----------
    path
        FITS file.

    Returns
    -------
    float
        Collimator position, in steps.

    Raises
    ------
    ValueError
        If the header is unreadable, if the frame carries no ``COL-FOC`` card,
        or if the card does not hold a number.
    """
    try:
        value = fits.getheader(path).get(_KEYWORD_COL_FOC)
    except (OSError, VerifyError) as exc:
        raise ValueError(f"{Path(path).name}: header unreadable ({exc}).") from exc
    if value is None:
        raise ValueError(f"{Path(path).name} carries no {_KEYWORD_COL_FOC} card.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{Path(path).name}: cannot read {_KEYWORD_COL_FOC} ({value!r})."
        ) from exc


def reduced_dir(
    observation_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Return the directory that holds the reduced products of an observation.

    Every product of a run is written under the returned directory, and nothing
    outside it is touched, so raw frames are never overwritten.

    Parameters
    ----------
    observation_dir
        Directory of the observation, which holds the raw frame directories.
    output_dir
        Where to write the products.  ``None`` puts them in a ``reduced``
        subdirectory of *observation_dir*, beside the raw frames.

    Returns
    -------
    pathlib.Path
        The output directory, created when it does not exist.

    Raises
    ------
    ValueError
        If the output path exists and is not a directory.

    Examples
    --------
    >>> reduced_dir("~/data/Observations/20250705-collimator_sweep")
    PosixPath('.../20250705-collimator_sweep/reduced')
    """
    if output_dir is None:
        root = Path(observation_dir).expanduser() / "reduced"
    else:
        root = Path(output_dir).expanduser()

    if root.exists() and not root.is_dir():
        raise ValueError(f"{root} exists and is not a directory.")
    root.mkdir(parents=True, exist_ok=True)
    return root
