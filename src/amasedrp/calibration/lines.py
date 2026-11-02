#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         lines.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  ThAr reference lines used by the calibration stage, per channel.

The numbers live in ``amasecore``, in ``reference/thar_channels.yaml``, because
they are survey metadata and not pipeline code.  They are calibration inputs
rather than a lamp atlas: only the lines that fall inside a channel's coverage
and are strong enough to be measured are listed there.

This module keeps the API the calibration stage uses.  The table is read once
per process and cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import amasecore
import numpy as np
from numpy.typing import NDArray

__all__ = ["ChannelLines", "CHANNELS", "thar_lines"]

CHANNELS = ("blue", "red")

_LINES_PATH = amasecore.data_path("reference", "thar_channels.yaml")


class ChannelLines(NamedTuple):
    """Reference lines of one spectrograph channel.

    Attributes
    ----------
    known_wls
        Wavelengths of the lines that are strong enough to be matched.  Used
        to score a candidate wavelength solution.
    poss_wls
        Wavelengths of the few strongest lines.  Used to enumerate candidate
        line-to-peak assignments.
    lsf_wls
        Wavelengths of the lines that are isolated enough for a line-spread
        function fit.
    """

    known_wls: NDArray[np.floating]
    poss_wls: NDArray[np.floating]
    lsf_wls: NDArray[np.floating]


@lru_cache(maxsize=1)
def _table():
    """Return the ThAr selection table, read once per process.

    Returns
    -------
    astropy.table.Table
        The rows of ``reference/thar_channels.yaml``.
    """
    return amasecore.load_table(_LINES_PATH)


def thar_lines(channel: str) -> ChannelLines:
    """Return the ThAr line selection of a channel.

    Parameters
    ----------
    channel
        Spectrograph channel, one of :data:`CHANNELS`.  Matched without
        regard to case.

    Returns
    -------
    ChannelLines
        The known, possible and LSF wavelengths, in Angstrom.

    Raises
    ------
    ValueError
        If *channel* is not a known channel.

    Examples
    --------
    >>> blue = thar_lines("blue")
    >>> len(blue.known_wls)
    19
    """
    name = channel.strip().lower()
    if name not in CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r}; expected one of {list(CHANNELS)}"
        )

    table = _table()
    mine = table[table["CHANNEL"] == name]

    def select(group: str) -> NDArray[np.floating]:
        selected = mine[mine["GROUP"] == group]["WAVELENGTH"]
        return np.array([float(wavelength) for wavelength in selected])

    return ChannelLines(
        known_wls=select("known"),
        poss_wls=select("possible"),
        lsf_wls=select("lsf"),
    )
