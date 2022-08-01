# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
# \brief Image filters based on convolutions

from re import L
import cv2
import numpy as np

from typing import Union, Optional, Tuple, List
from collections import namedtuple
from commons.exceptions import FilteringError, WrongArgumentsType, WrongArgumentsValue
from commons.warning import DefaultSetting, ImageDataTypeConversion
from image.load._interface import PyFaroImage
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion, check_user_provided_ndarray

# -------------------------------------------------------------------------

BORDER_INTERPOLATION = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "transparent": cv2.BORDER_TRANSPARENT,
    "default": cv2.BORDER_DEFAULT
}

# -------------------------------------------------------------------------

@check_image_exist_external
def corr2d(
    image: PyFaroImage,
    kernel: Union[namedtuple, np.ndarray],
    delta: Optional[Union[float, int]] = 0,
    border: Optional[str] = "default",
) -> PyFaroImage:
    """Outputs the image with the same depth. Places the computed/filtered value in the center of the area covered by the kernel.
       Note: Kernel windows are not variable sized here. They have a constant size over each pixel neighborhood."""

    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, (np.ndarray, namedtuple)):
        raise WrongArgumentsType(
            "Provided kernel has a type that is not supported by the library. Try using get_kernel() for getting the kernel instead"
        )

    if isinstance(kernel, namedtuple):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    if not isinstance(delta, (float, int)):
        raise WrongArgumentsType("Provided value for delta is not either an integer or a float")

    if not isinstance(border, str):
        raise WrongArgumentsType("Provided border type is not a string")

    border_actual = BORDER_INTERPOLATION.get(border, None)
    if not border_actual:
        DefaultSetting(
            "Provided border option is not supported currently. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]

    try:
        new_im = image.copy()
        new_im.image = cv2.filter2D(
            new_im.image, -1, kernel, delta=float(delta), borderType=border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def average_blur(
    image: PyFaroImage,
    kernel_size: Union[List[int], Tuple[int, int]],
    normalize: Optional[bool] = True,
    border: Optional[str] = "default"
) -> PyFaroImage:
    """Wrap border is not supported here. Normalize = False can be used to extract useful image chracteristics e.g. covariance matrix of the image gradients
    can help with extracting images demonstrating optical flow for object tracking."""

    image_array_check_conversion(image, "openCV")

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

    if border == "wrap":
        DefaultSetting(
            "Provided border option is not supported for this operation. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]
    else:
        border_actual = BORDER_INTERPOLATION.get(border, "None")

    if not border_actual:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]

    kernel_size = [int(i) for i in kernel_size]
    try:
        new_im = image.copy()
        new_im.image = cv2.boxFilter(
            new_im.image, -1, kernel_size, normalize=normalize, borderType=border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def gaussian_blur(
    image: PyFaroImage,
    kernel_size: Union[List[int], Tuple[int, int]],
    sigma_x: float,
    sigma_y: Optional[float] = 0.0,
    border: Optional[str] = "default"
) -> PyFaroImage:
    """Warp border is not supported here. It is better to supply both the kernel size and the sigma_x. If the kernel size is zero
    then it is computed from the sigma's provided by the user. If sigma_y is zero, it is computed from sigma_x. If both sigma_x and sigma_y is 
    zero then it is computed from the kernel_size."""

    image_array_check_conversion(image, "openCV")

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

    if border == "wrap":
        DefaultSetting(
            "Provided border option is not supported for this operation. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]
    else:
        border_actual = BORDER_INTERPOLATION.get(border, "None")

    if not border_actual:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]

    kernel_size = [int(i) for i in kernel_size]
    try:
        new_im = image.copy()
        new_im.image = cv2.GaussianBlur(
            new_im.image, kernel_size, float(sigma_x), float(sigma_y), borderType=border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def median_blur(image: PyFaroImage, kernel_size: Union[List[int], Tuple[int, int]]) -> PyFaroImage:
    """Kernel size has to be odd. Different data types for different kernel size"""

    image_array_check_conversion(image, "openCV")

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

    if kernel_size[0] > 5 and image.dtype != np.uint8:
        ImageDataTypeConversion(
            "Converting the image type to uint8 since for kernel sizes > 5 only this type is supported"
        )
        image._image_conversion_helper(np.uint8)
        image.update_file_stream()
        image.set_loader_properties()

    try:
        new_im = image.copy()
        new_im.image = cv2.medianBlur(new_im.image, int(kernel_size[0]))
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def bilateral_filter(
    image: PyFaroImage,
    kernel_diameter: int,
    color_sigma: float,
    spatial_sigma: float,
    border: Optional[str] = "default"
) -> PyFaroImage:
    """Only 8-bit and 32-bit floating point images are supported"""
    """Does not work with RGBA images"""

    if image.channels > 3:
        raise FilteringError(
            "This filter cannot operate on images that have color channels more than 3"
        )

    image_array_check_conversion(image, "openCV")

    if image.dtype == "uint16":
        ImageDataTypeConversion(
            "Converting the image from 16 bits to 8 bits per channel since this is what is only supported for this filter"
        )
        image._image_conversion_helper(np.uint8)
        image.update_file_stream()
        image.set_loader_properties()

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

    border_actual = BORDER_INTERPOLATION.get(border, "None")
    if not border_actual:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = BORDER_INTERPOLATION["default"]

    try:
        new_im = image.copy()
        new_im.image = cv2.bilateralFilter(
            new_im.image, int(kernel_diameter), float(color_sigma), float(spatial_sigma),
            border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------