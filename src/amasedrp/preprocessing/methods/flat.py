#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         flat.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Pixel flat-field correction.
"""

from datetime import datetime

import numpy as np

from ..core.image import Image


def apply_pixel_flat(
    image: Image, master_flat: Image, master_bias: Image, master_dark: Image
) -> Image:
    """Apply pixel flat-field correction."""
    flat_exptime: float | None = master_flat.exptime
    assert flat_exptime is not None and flat_exptime > 0.0

    dark_exptime: float | None = master_dark.exptime
    assert dark_exptime is not None and dark_exptime > 0.0

    darkcurr = (
        master_dark.data.astype(np.float32) - master_bias.data.astype(np.float32)
    ) / dark_exptime

    pixflat = (
        master_flat.data.astype(np.float32) - master_bias.data.astype(np.float32)
    ) - (darkcurr * flat_exptime)

    median = float(np.nanmedian(pixflat))
    if median == 0.0 or np.isnan(median):
        raise ValueError(
            f"Median of pixel flat field is {median}; "
            "cannot compute pixel flat-field correction (division by zero)."
        )

    pixresp = pixflat / median  # pixel response relative to the median

    result: Image = image.copy()
    result.data /= pixresp
    result.header.add_history(
        f"Pixel flat-field corrected: {datetime.now().isoformat(timespec='seconds')}"
    )
    return result
