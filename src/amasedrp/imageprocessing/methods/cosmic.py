#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         cosmic.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Cosmic ray detection and removal.
"""

from datetime import datetime
from typing import Any

import numpy as np
from astroscrappy import detect_cosmics  # type: ignore[import]

from ..core.image import Image


def detect_cosmic_rays(image: Image, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect cosmic rays in the image using the L.A.Cosmic algorithm,
    based on Laplacian edge detection (van Dokkum 2001).

    This function is designed to be efficient and fast, utilizing
    the C/Cython implementation of `astroscrappy.detect_cosmics()`.

    References:
    - van Dokkum (2001):
        https://iopscience.iop.org/article/10.1086/323894
    - astroscrappy GitHub:
        https://github.com/astropy/astroscrappy
    - astroscrappy Docs:
        https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html
    """
    crmask, cleanarr = detect_cosmics(image.data, **kwargs)
    return crmask, cleanarr


def remove_cosmic_rays(image: Image, **kwargs: Any) -> Image:
    """Detect and remove cosmic rays, return cleaned image."""
    crmask, cleanarr = detect_cosmic_rays(image, **kwargs)
    result: Image = image.copy()
    result.data = cleanarr.astype(np.float32)
    result.header.add_history(
        f"Cosmic rays removed: {datetime.now().isoformat(timespec='seconds')}"
    )
    return result


# NOTE: the 2D & 1D mask system is TODO.
