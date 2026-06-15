#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         bias.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Image bias subtraction.
"""

from datetime import datetime

import numpy as np

from ..core.image import Image


def subtract_bias(image: Image, master_bias: Image) -> Image:
    """Subtract bias."""
    result: Image = image.copy()
    result.data = result.data.astype(np.float32) - master_bias.data.astype(np.float32)
    result.header.add_history(
        f"Bias subtracted: {datetime.now().isoformat(timespec='seconds')}"
    )
    return result
