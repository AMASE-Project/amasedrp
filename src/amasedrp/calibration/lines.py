#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         lines.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  ThAr reference lines used by the calibration stage, per channel.

The lists are selections for the AMASE-P prototype spectrograph, taken from the
2025-07 collimator-sweep reductions.  They are calibration inputs, not a lamp
atlas: only the lines that fall inside a channel's coverage and are strong
enough to be measured are listed here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["ChannelLines", "CHANNELS", "thar_lines"]

CHANNELS = ("blue", "red")


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


# Blue channel, wavelength coverage roughly 4390 - 5150 A.
_BLUE = ChannelLines(
    known_wls=np.array([
        4545.0519, 4579.3495, 4589.8978, 4609.5673,
        4657.9012, 4673.6609, 4703.9898, 4723.4382, 4726.8683,
        4735.9058, 4764.8646, 4806.0205, 4808.1337, 4879.8635,
        4894.9551, 4945.4587, 4965.0795, 5017.1628, 5044.7195,
    ]),
    poss_wls=np.array([
        4657.9012, 4703.9898, 4726.8683, 4764.8646,
        4806.0205, 4879.8635, 4894.9551,
    ]),
    lsf_wls=np.array([4657.9012, 4764.8646, 4965.0795]),
)

# Red channel, wavelength coverage roughly 6150 - 7000 A.
_RED = ChannelLines(
    known_wls=np.array([
        6169.82, 6172.28, 6182.62, 6203.49, 6342.86, 6384.72, 6411.90,
        6416.31, 6457.28, 6531.34, 6554.16, 6577.21, 6583.91, 6588.54,
        6591.48, 6593.94, 6604.85, 6643.70, 6662.27, 6677.28, 6684.29,
        6727.46, 6752.83, 6756.45, 6766.61, 6780.41, 6871.29, 6911.23,
        6937.66, 6943.61, 6965.43, 6989.66,
    ]),
    poss_wls=np.array([
        6182.62, 6583.91, 6588.54, 6591.48, 6752.83, 6871.29, 6965.43,
    ]),
    lsf_wls=np.array([6583.91, 6752.83, 6965.43]),
)

_LINES = {"blue": _BLUE, "red": _RED}


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
        The known, possible and LSF wavelengths.

    Raises
    ------
    ValueError
        If *channel* is not a known channel.

    Examples
    --------
    >>> lines = thar_lines("blue")
    >>> lines.poss_wls.shape
    (7,)
    """
    key = channel.strip().lower()
    if key not in _LINES:
        raise ValueError(
            f"unknown channel {channel!r}; expected one of {list(CHANNELS)}"
        )
    return _LINES[key]
