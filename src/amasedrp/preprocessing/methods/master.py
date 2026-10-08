#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         master.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Combine several frames into one master calibration frame.
"""

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.image import Image

__all__ = ["combine_master"]


def combine_master(images: Sequence[Image]) -> Image:
    """Combine frames into a master frame by taking the per-pixel median.

    The median rejects a cosmic ray or a warm pixel in a single input frame,
    which a mean would carry into the master.

    Parameters
    ----------
    images
        Frames to combine.  All must hold data of the same shape.

    Returns
    -------
    Image
        The master frame.  It carries a copy of the header of the first input,
        and one ``HISTORY`` entry per input frame.

    Raises
    ------
    ValueError
        If *images* is empty, if a frame holds no data, or if the shapes
        differ.

    Notes
    -----
    The whole stack is held in memory.  Combining the 11 bias frames of the
    2025-07 sweep (9600 x 6422, ``float32``) needs about 2.7 GB.
    """
    images = list(images)
    if not images:
        raise ValueError("combine_master needs at least one frame.")

    for image in images:
        if image.data is None:
            filename = Path(image.filename).name if image.filename else "a frame"
            raise ValueError(f"{filename} holds no data.")

    shapes = {image.data.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"frame shapes differ: {sorted(shapes)}")

    stack = np.stack([image.data for image in images], axis=0)
    master = Image(
        data=np.nanmedian(stack, axis=0),
        header=images[0].header.copy(),
        filename=None,
        unit="electron",
    )

    master.header.add_history(
        f"Master frame combined from {len(images)} frames: "
        f"{datetime.now().isoformat(timespec='seconds')}"
    )
    for image in images:
        if image.filename:
            master.header.add_history(f"  input: {Path(image.filename).name}")

    return master
