#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         image_preprocessing.py
@Time:         2026/04/09 16:59:59
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Main functions for image preprocessing,
               including e.g., bias subtraction, dark subtraction,
               pixel flat-field correction, cosmic ray removal, etc.
'''

import numpy as np


def image_calibration(
        input_image, master_bias_image, master_dark_image,
        master_pixflat_image):
    """Image calibration by applying bias subtraction, dark subtraction,
    and pixel flat-field correction."""
    # For each image, the detector pixel value in principal is as follows:
    # bias image       = bias level + readout noise
    # dark image       = dark current * exposure time
    #                    + bias level + readout noise
    # pixel flat image = uniform illumination * pixel response
    #                    + dark current * exposure time
    #                    + bias level + readout noise
    # science image    = science signal * pixel response
    #                    + dark current * exposure time
    #                    + bias level + readout noise
    #
    # Thus, the calibration steps can be expressed as:
    # (1) calclute the needed calibration data
    # darkcurr' = (dark - bias) / dark exposure time
    # pixflat'  = (pixel flat - bias) - darkcurr' * pixel flat exposure time
    # pixresp'  = pixflat' / median(pixflat')
    # (2) apply the calibration to the science image:
    # bias subtraction:    S1 = science - bias
    # dark subtraction:    S2 = S1 - darkcurr' * science exposure time
    # pixel flat fielding: S3 = S2 / pixresp'

    # calculate the needed calibration data
    darkcurr = ((master_dark_image.data - master_bias_image.data)
                / master_dark_image.exptime)
    pixflat = ((master_pixflat_image.data - master_bias_image.data)
               - darkcurr * master_pixflat_image.exptime)
    pixresp = pixflat / np.nanmedian(pixflat)

    # create a copy of the input image for preprocessing
    output_image = input_image.copy()
    output_image.data = output_image.data.astype(np.float32)

    # bias subtraction
    print("Applying bias subtraction...")
    output_image.data -= master_bias_image.data
    print("Bias subtraction applied.")
    print('\n')

    # dark subtraction
    print("Applying dark subtraction...")
    output_image.data -= darkcurr * input_image.exptime
    print("Dark subtraction applied.")
    print('\n')

    # pixel flat-fielding
    print("Applying pixel flat-field correction...")
    output_image.data /= pixresp
    print("Pixel flat-field correction applied.")
    print('\n')

    return output_image


def image_preprocessing():
    # NOTE: e.g., given path for bias, dark, flat, and science images
    # read the images, then conduct the pre-processing operations,
    # and finally save the processed images.
    pass
