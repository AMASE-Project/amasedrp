#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         imagePreprocessing.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Functions used for image preprocessing (such as
               bias subtraction, dark subtraction, pixel flat fielding, etc.).
'''

import os
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from ..core.image import Image


def load_image(filename: str) -> Image:
    """Load an image from a FITS file.
    Parameters
    ----------
    filename : str
        Path to the FITS file.
    Returns -------
    Image
        An instance of the Image class.
    """
    image = Image()
    image.readFitsData(filename)
    return image


def preprocess_image(
        input_image: str,
        output_image: str,
        master_bias_frame: Optional[str] = None,
        master_dark_frame: Optional[str] = None,
        master_pixflat_frame: Optional[str] = None,
        reject_cosmic_rays: bool = True,
        replace_with_nan: bool = True,
        display_plots: bool = False,
):
    """Preprocess a raw 2D image by applying bias subtraction,
    dark subtraction, pixel flat fielding, cosmic rays rejection, etc.

    Parameters
    ----------
    input_image : str
        path to the input raw image
    output_image : str
        path to the output image
    master_bias_frame : Optional[str], optional
        path to the master bias frame, by default None
    master_dark_frame : Optional[str], optional
        path to the master dark frame, by default None
    master_pixflat_frame : Optional[str], optional
        path to the master pixel flat frame, by default None
    reject_cosmic_rays : bool, optional
        whether to reject cosmic rays from the output preprocessed image,
        by default True
    replace_with_nan : bool, optional
        whether to replace masked pixels with NaNs in the preprocessed image,
        by default Trues
    display_plots : bool, optional
        whether to show plots of the images, by default False

    Returns
    -------
    # TODO: specify the return
    _type_
        _description_
    """
    in_img = load_image(input_image)
    # ensure the input image is in float format
    if not np.issubdtype(in_img.data.dtype, np.floating):
        in_img.data = in_img.data.astype(np.float32)
    print(f"Input image loaded from {input_image}")
    print('\n')
    # TODO: add a logger to log the preprocessing steps

    # read the master bias frame
    if master_bias_frame and os.path.isfile(master_bias_frame):
        mbias_img = load_image(master_bias_frame)
        # ensure the bias frame is in float format
        if not np.issubdtype(mbias_img.data.dtype, np.floating):
            mbias_img.data = mbias_img.data.astype(np.float32)
        print(f"Master bias frame loaded from {master_bias_frame}")
    else:
        mbias_img = Image(data=np.zeros_like(in_img.data))
        print("No master bias frame provided or file not found.")
        print("Using dummy array as master bias frame.")
    print('\n')

    # read the master dark frame
    if master_dark_frame and os.path.isfile(master_dark_frame):
        mdark_img = load_image(master_dark_frame)
        # ensure the dark frame is in float format
        if not np.issubdtype(mdark_img.data.dtype, np.floating):
            mdark_img.data = mdark_img.data.astype(np.float32)
        print(f"Master dark frame loaded from {master_dark_frame}")
    else:
        mdark_img = Image(data=np.zeros_like(in_img.data))
        print("No master dark frame provided or file not found.")
        print("Using dummy array as master dark frame.")
    print('\n')

    # read the master pixel flat frame
    if master_pixflat_frame and os.path.isfile(master_pixflat_frame):
        mpflat_img = load_image(master_pixflat_frame)
        # ensure the pixel flat frame is in float format
        if not np.issubdtype(mpflat_img.data.dtype, np.floating):
            mpflat_img.data = mpflat_img.data.astype(np.float32)
        print(f"Master pixel flat frame loaded from {master_pixflat_frame}")
    else:
        mpflat_img = Image(data=np.ones_like(in_img.data), header={})
        mpflat_img.data *= np.max([np.nanmax(mbias_img.data)*100., 10000.])
        mpflat_img.header['EXPTIME'] = in_img.header['EXPTIME']
        print("No master pixel flat frame provided or file not found.")
        print("Using dummy array as master pixel flat frame.")
    print('\n')

    # NOTE:
    # bias frame       = bias level + readout noise
    # dark frame       = dark current * exposure time
    #                    + bias level + readout noise
    # pixel flat frame = uniform illumination * pixel response
    #                    + dark current * exposure time
    #                    + bias level + readout noise
    # science frame    = signal * pixel response
    #                    + dark current * exposure time
    #                    + bias level + readout noise
    #
    # darkcurr' = (dark - bias) / dark exposure time
    # flat'     = (flat - bias) - darkcurr' * flat exposure time
    # pixresp'  = flat' / median(flat')
    #
    # bias subtraction:    S1 = science - bias
    # dark subtraction:    S2 = S1 - darkcurr' * science exposure time
    # pixel flat fielding: S3 = S2 / pixresp'

    # preparation for preprocessing
    darkcurr_arr = ((mdark_img.data - mbias_img.data)
                    / mdark_img.header['EXPTIME'])
    flat_arr = ((mpflat_img.data - mbias_img.data)
                - darkcurr_arr * mpflat_img.header['EXPTIME'])
    pixresp_arr = flat_arr / np.nanmedian(flat_arr)

    # create a copy of the input image for preprocessing
    out_img = in_img.copy()
    # ensure the output image is in float format
    if not np.issubdtype(out_img.data.dtype, np.floating):
        out_img.data = out_img.data.astype(np.float32)

    # bias subtraction
    print("Applying master bias subtraction...")
    out_img.data -= mbias_img.data
    print("Master bias subtraction applied.")
    print('\n')

    # dark subtraction
    print("Applying master dark subtraction...")
    out_img.data -= darkcurr_arr * in_img.header['EXPTIME']
    print("Master dark subtraction applied.")
    print('\n')

    # pixel flat fielding
    print("Applying master pixel flat fielding...")
    out_img.data / pixresp_arr
    print("Master pixel flat fielding applied.")
    print('\n')

    # apply cosmic rays rejection
    if reject_cosmic_rays:
        print("Rejecting cosmic rays...")
        # out_img.rejectCosmicRays()  # TODO: finish this function
        print("Cosmic rays rejected.")
        print('\n')

    # propagate the pixel mask
    out_img_nan_pixels = np.isnan(out_img.data)
    out_img_inf_pixels = np.isinf(out_img.data)
    out_img.mask = np.logical_or(out_img.mask, out_img_nan_pixels)
    out_img.mask = np.logical_or(out_img.mask, out_img_inf_pixels)

    # replace masked pixels with NaNs if requested
    # # TODO: Check how to handle with the "Bitmasks"
    if replace_with_nan:
        # out_img.data[out_img.mask] = np.nan
        pass
        # print("Masked pixels replaced with NaNs")
        # print('\n')

    # propagate the header
    # # TODO: Check the following keywords
    # out_img.header = in_img.header.copy()
    # out_img.header['IMGTYPE'] = 'PREPROCESSED'
    # out_img.header['PREPROCESSED'] = True
    # out_img.header['BIASFILE'] = master_bias_frame
    # out_img.header['DARKFILE'] = master_dark_frame
    # out_img.header['FLATFILE'] = master_pixflat_frame

    # save the preprocessed image
    print("Saving the preprocessed image...")
    out_img.writeFitsData(output_image)
    print(f"Preprocessed image saved to {output_image}")
    print('\n')

    # save (and show if requested) the figures of images
    # TODO: Finish this part
    print("Plotting the images...")
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("test")
    # # TODO: plot the images
    # fig.savefig(output_image.replace('.fits', '.pdf'))
    if display_plots:
        plt.show()
    else:
        plt.close(fig)
    # print(f"Image plots saved to {output_image.replace('.fits', '.pdf')}")

    return in_img, mbias_img, mdark_img, mpflat_img, out_img


# Example usage
if __name__ == "__main__":
    input_image = "path/to/input_image.fits"
    output_image = "path/to/output_image.fits"
    master_bias_frame = "path/to/master_bias.fits"
    master_dark_frame = "path/to/master_dark.fits"
    master_pixflat_frame = "path/to/master_pixflat.fits"
    in_img, mbias_img, mdark_img, mpflat_img, out_img = preprocess_image(
        input_image=input_image,
        output_image=output_image,
        master_bias_frame=master_bias_frame,
        master_dark_frame=master_dark_frame,
        master_pixflat_frame=master_pixflat_frame,
        replace_with_nan=False,
        reject_cosmic_rays=True,
        display_plots=False,
    )
