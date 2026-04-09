#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         plotting.py
@Time:         2026/04/09 22:28:23
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Plotting functions for visualizing e.g.,
               fits images, spectra, etc.
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_fits_image(
        ax, data, cmap='gray', vmin=None, vmax=None,
        draw_colorbar=False,
        cbar_label=None, cbar_orientation='vertical', cbar_extend='neither'):
    """Plot a 2D fits image."""
    # pre-process
    if vmin is None:
        vmin = np.percentile(data, 5)
    if vmax is None:
        vmax = np.percentile(data, 95)
    # plot
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    # colorbar
    if draw_colorbar:
        cbar_width, cbar_height = 0.05, 1.0
        ax_ins = ax.inset_axes(
            [1.03, 0., cbar_width, cbar_height],
            transform=ax.transAxes)
        cbar = plt.colorbar(
            im, cax=ax_ins,
            orientation=cbar_orientation, extend=cbar_extend)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    return ax
