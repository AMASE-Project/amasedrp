#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         dark.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Dark current subtraction.
"""

from datetime import datetime

import numpy as np

from ..core.image import Image


def subtract_dark(image: Image, master_dark: Image, master_bias: Image) -> Image:
    """Subtract dark current, scaled by exposure time."""
    dark_exptime: float | None = master_dark.exptime
    assert dark_exptime is not None and dark_exptime > 0.0

    darkcurr = (
        master_dark.data.astype(np.float32) - master_bias.data.astype(np.float32)
    ) / dark_exptime

    result: Image = image.copy()
    input_exptime: float | None = image.exptime
    assert input_exptime is not None and input_exptime > 0.0
    result.data -= darkcurr * input_exptime
    result.header.add_history(
        f"Dark subtracted: {datetime.now().isoformat(timespec='seconds')}"
    )
    return result
