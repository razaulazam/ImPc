# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image filters based on convolutions

import cv2
import numpy as np

from typing import Union, Optional, Tuple, List
from collections import namedtuple
from commons.exceptions import FilteringError, WrongArgumentsType, WrongArgumentsValue
from common.warning import DefaultSetting, ImageDataTypeConversion
from image.common.interfaces.loader import BaseImage
from image.common.decorators import check_image_exist_external
from image.common.helpers import image_array_check_conversion, check_user_provided_ndarray
from image.common.datastructs import SKIMAGE_SAMPLING_REGISTRY, AllowedDataType
from image.common.datastructs import CV_BORDER_INTERPOLATION
from image.filter.common_methods import is_not_namedtuple
from scipy.ndimage import convolve as scipy_convolve
from scipy.ndimage import correlate as scipy_correlate

# -------------------------------------------------------------------------

@check_image_exist_external
def corr2d(
    image: BaseImage,
    kernel: Union[namedtuple, np.ndarray],
    delta: Optional[Union[float, int]] = 0,
    border: Optional[str] = "default",
) -> BaseImage:
    """Outputs the image with the same depth. Places the computed/filtered value in the center of the area covered by the kernel.
       Note: Kernel windows are not variable sized here. They have a constant size over each pixel neighborhood."""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    if not isinstance(delta, (float, int)):
        raise WrongArgumentsType("Provided value for delta is not either an integer or a float")

    if not isinstance(border, str):
        raise WrongArgumentsType("Provided border type is not a string")

    check_image = image_array_check_conversion(image)

    border = border.lower()
    border_actual = CV_BORDER_INTERPOLATION.get(border, None)
    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    try:
        check_image._set_image(
            cv2.filter2D(
                check_image.image, -1, kernel, delta=float(delta), borderType=border_actual
            )
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def average_blur(
    image: BaseImage,
    kernel_size: Union[List[int], Tuple[int, int]],
    normalize: Optional[bool] = True,
    border: Optional[str] = "default"
) -> BaseImage:
    """Wrap border is not supported here. Normalize = False can be used to extract useful image chracteristics e.g. covariance matrix of the image gradients
    can help with extracting images demonstrating optical flow for object tracking."""

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType("Kernel size can only be defined in form of a tuple or list")

    if len(kernel_size) != 2:
        raise WrongArgumentsValue("Kernal size can only span in the height and the width direction")

    if not all(i > 0 for i in kernel_size):
        raise WrongArgumentsValue("Kernel size cannot be negative")

    if not isinstance(normalize, bool):
        raise WrongArgumentsType("Normalize argument can only be specified as a boolean")

    if not isinstance(border, str):
        raise WrongArgumentsType("Border argument can only be specified as a string")

    check_image = image_array_check_conversion(image)

    border = border.lower()
    if border == "wrap":
        DefaultSetting(
            "Provided border option is not supported for this operation. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]
    else:
        border_actual = CV_BORDER_INTERPOLATION.get(border, None)

    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    kernel_size = [int(i) for i in kernel_size]
    try:
        check_image._set_image(
            cv2.boxFilter(
                check_image.image, -1, kernel_size, normalize=normalize, borderType=border_actual
            )
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def gaussian_blur(
    image: BaseImage,
    kernel_size: Union[List[int], Tuple[int, int]],
    sigma_x: float,
    sigma_y: Optional[float] = 0.0,
    border: Optional[str] = "default"
) -> BaseImage:
    """Warp border is not supported here. It is better to supply both the kernel size and the sigma_x. If the kernel size is zero
    then it is computed from the sigma's provided by the user. If sigma_y is zero, it is computed from sigma_x. If both sigma_x and sigma_y is 
    zero then it is computed from the kernel_size."""

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType("Kernel size can only be defined in form of a tuple or list")

    if len(kernel_size) != 2:
        raise WrongArgumentsValue("Kernal size can only span in the height and the width direction")

    if not all(i > 0 for i in kernel_size):
        raise WrongArgumentsValue("Kernel size cannot be negative")

    if all(i == 0 for i in kernel_size):
        DefaultSetting(
            "Kernel size would be computed from the standard deviations both in the x and y direction"
        )

    if not isinstance(sigma_x, (int, float)):
        raise WrongArgumentsType(
            "Provided value of sigma in the x direction does not have the accurate type"
        )

    if not isinstance(sigma_y, (float, int)):
        raise WrongArgumentsType(
            "Provided value of sigma in the y direction does not have the accurate type"
        )

    if not isinstance(border, str):
        raise WrongArgumentsType("Border argument can only be specified as a string")

    check_image = image_array_check_conversion(image)

    border = border.lower()
    if border == "wrap":
        DefaultSetting(
            "Provided border option is not supported for this operation. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]
    else:
        border_actual = CV_BORDER_INTERPOLATION.get(border, None)

    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    kernel_size = [int(i) for i in kernel_size]
    try:
        check_image._set_image(
            cv2.GaussianBlur(
                check_image.image,
                kernel_size,
                float(sigma_x),
                float(sigma_y),
                borderType=border_actual
            )
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def median_blur(image: BaseImage, kernel_size: Union[List[int], Tuple[int, int]]) -> BaseImage:
    """Kernel size has to be odd. Different data types for different kernel size"""

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType("Kernel size can only be defined as either tuple or list")

    if len(kernel_size) != 2:
        raise WrongArgumentsValue("Kernal size can only span in the height and the width direction")

    if kernel_size[0] != kernel_size[1]:
        raise WrongArgumentsValue("Kernel should have the same width and the height dimension")

    if not all(i > 1 for i in kernel_size):
        raise WrongArgumentsValue("Kernel size should be positive and greater than 1")

    if kernel_size[0] % 2 == 0:
        raise WrongArgumentsValue("Kernel width/height should be an odd integer")

    check_image = image_array_check_conversion(image)

    if kernel_size[0] > 5 and image.dtype != AllowedDataType.Uint8:
        ImageDataTypeConversion(
            "Converting the image type to uint8 since for kernel sizes > 5 only this type is supported"
        )
        check_image._image_conversion_helper(AllowedDataType.Uint8)

    try:
        check_image._set_image(cv2.medianBlur(check_image.image, int(kernel_size[0])))
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def bilateral_filter(
    image: BaseImage,
    kernel_diameter: int,
    color_sigma: float,
    spatial_sigma: float,
    border: Optional[str] = "default"
) -> BaseImage:
    """Only 8-bit and 32-bit floating point images are supported"""
    """Does not work with RGBA images"""

    if image.channels > 3:
        raise FilteringError(
            "This filter cannot operate on images that have color channels more than 3"
        )

    if not isinstance(kernel_diameter, (float, int)):
        raise WrongArgumentsType("Diameter value can only be either an integer or float value")

    if kernel_diameter < 0:
        DefaultSetting("Diameter value will be computed from the spatial sigma value")

    if not isinstance(color_sigma, (float, int)):
        raise WrongArgumentsType("Color sigma value can either be an integer or a float value")

    if not isinstance(spatial_sigma, (float, int)):
        raise WrongArgumentsType("Spatial sigma value can either be an integer or a float value")

    if not isinstance(border, str):
        raise WrongArgumentsType("Border argument can only be specified as a string")

    check_image = image_array_check_conversion(image)

    if check_image.dtype is AllowedDataType.Uint16:
        ImageDataTypeConversion(
            "Converting the image from 16 bits to 8 bits per channel since this is what is only supported for this filter"
        )
        check_image._image_conversion_helper(AllowedDataType.Uint8)

    border = border.lower()
    border_actual = CV_BORDER_INTERPOLATION.get(border, None)
    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    try:
        check_image._set_image(
            cv2.bilateralFilter(
                check_image.image, int(kernel_diameter), float(color_sigma), float(spatial_sigma),
                border_actual
            )
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def convolve(image: BaseImage, weights: np.ndarray, mode: Optional[str] = "reflect") -> BaseImage:
    """Convolves the provided image with the weights (provided as np.ndarray). Result is returned as the 
    image with the same datatype as the provided image."""

    if not isinstance(weights, np.ndarray):
        raise WrongArgumentsType("Weights must be provided as a np.ndarray")

    if image.is_rgb() or image.is_hsv() or image.is_lab() or image.is_rgba() or image.is_ycbcr():
        if len(weights.shape) != 3:
            raise WrongArgumentsValue("Weights must have same number of dimensions as the image")
        width, height, channels = weights.shape
        is_correct_width = width > 0 and width <= image.width
        is_correct_height = height > 0 and height <= image.height
        is_correct_channels = channels == image.channels
        if not is_correct_channels or not is_correct_height or not is_correct_width:
            raise WrongArgumentsValue("Provided weights do not have the right dimensions")
    elif image.is_gray():
        if len(weights.shape) != 2:
            raise WrongArgumentsValue("Weights must have same number of dimensions as the image")
        width, height = weights.shape
        is_correct_width = width > 0 and width <= image.width
        is_correct_height = height > 0 and height <= image.height
        if not is_correct_height or not is_correct_width:
            raise WrongArgumentsValue("Provided weights do not have the right dimensions")
    else:
        raise WrongArgumentsValue("Provided image has a mode which is not supported by this method")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode must be provided as str")

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode, None)
    if mode_arg is None:
        DefaultSetting("Using default mode since the provided mode is not supported by the library")
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]

    check_image = image_array_check_conversion(image)
    try:
        check_image._set_image(scipy_convolve(check_image.image, weights, mode=mode_arg))
    except Exception as e:
        raise FilteringError("Failed to convolve the image with the provided weights") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def correlate(image: BaseImage, weights: np.ndarray, mode: Optional[str] = "reflect") -> BaseImage:
    """Correlates the provided image with the weights (provided as np.ndarray). Result is returned as the 
    image with the same datatype as the provided image."""

    if not isinstance(weights, np.ndarray):
        raise WrongArgumentsType("Weights must be provided as a np.ndarray")

    if image.is_rgb() or image.is_hsv() or image.is_lab() or image.is_rgba() or image.is_ycbcr():
        if len(weights.shape) != 3:
            raise WrongArgumentsValue("Weights must have same number of dimensions as the image")
        width, height, channels = weights.shape
        is_correct_width = width > 0 and width <= image.width
        is_correct_height = height > 0 and height <= image.height
        is_correct_channels = channels == image.channels
        if not is_correct_channels or not is_correct_height or not is_correct_width:
            raise WrongArgumentsValue("Provided weights do not have the right dimensions")
    elif image.is_gray():
        if len(weights.shape) != 2:
            raise WrongArgumentsValue("Weights must have same number of dimensions as the image")
        width, height = weights.shape
        is_correct_width = width > 0 and width <= image.width
        is_correct_height = height > 0 and height <= image.height
        if not is_correct_height or not is_correct_width:
            raise WrongArgumentsValue("Provided weights do not have the right dimensions")
    else:
        raise WrongArgumentsValue("Provided image has a mode which is not supported by this method")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode must be provided as str")

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode, None)
    if mode_arg is None:
        DefaultSetting("Using default mode since the provided mode is not supported by the library")
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]

    check_image = image_array_check_conversion(image)
    try:
        check_image._set_image(scipy_correlate(check_image.image, weights, mode=mode_arg))
    except Exception as e:
        raise FilteringError("Failed to correlate the image with the provided weights") from e

    return check_image

# -------------------------------------------------------------------------