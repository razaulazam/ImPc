# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image thresholding methods

import cv2

from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion
from image._common_datastructs import AllowedDataType
from skimage.filters.thresholding import threshold_niblack as sk_niblack
from skimage.filters.thresholding import threshold_sauvola as sk_sauvola
from typing import Union, Optional

# -------------------------------------------------------------------------

CV_THRESHOLD_STRATEGY = {
    "binary": cv2.THRESH_BINARY,
    "binary_inverse": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozeroinv": cv2.THRESH_TOZERO_INV,
    "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}

# -------------------------------------------------------------------------

@check_image_exist_external
def simple_threshold(
    image: BaseImage, threshold: Union[float, int], max_val: Union[float, int], method: str
) -> BaseImage:
    """Applied fixed level threshold to each image pixel. Maximum value is ignored for trunc, tozero and tozeroinv methods.
    Threshold values can be calculated from different strategies e.g. Isodata etc."""

    if not isinstance(threshold, (float, int)):
        raise WrongArgumentsType("Threshold can only be provided as float or integer")

    if not isinstance(max_val, (float, int)):
        raise WrongArgumentsType("Maximum value can only be provided as float or integer")

    if not isinstance(method, str):
        raise WrongArgumentsType("Method can only be provided as a string")

    method = method.lower()
    method_arg = CV_THRESHOLD_STRATEGY.get(method, None)
    if method_arg is None:
        raise WrongArgumentsValue("Provided thresholding method is wrong")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            cv2.threshold(check_image.image, float(threshold), float(max_val),
                          method_arg)[1].astype(check_image.dtype.value, copy=False)
        )
    except Exception as e:
        raise FilteringError(
            "Failed to filter the image with the specified thresholding strategy"
        ) from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def adaptive_threshold(
    image: BaseImage, max_val: Union[float, int], adaptive_method: str, threshold_method: str,
    block_size: int, subtract_val: Union[float, int]
) -> BaseImage:
    """Works on grayscale 8 bit images only"""

    if not image.is_gray():
        raise WrongArgumentsValue("This method only works with grayscale images")

    if not isinstance(max_val, (float, int)):
        raise WrongArgumentsType("Maximum value can only be provided as integer or float")

    if not isinstance(adaptive_method, str):
        raise WrongArgumentsType("Adaptive method argument can only be provided as a string")

    if not isinstance(threshold_method, str):
        raise WrongArgumentsType("Threshold method argument can only be provided as a string")

    if not isinstance(block_size, int):
        raise WrongArgumentsType("Block size argument can only be provided as a integer")

    if block_size <= 1 or block_size % 2 == 1:
        raise WrongArgumentsValue(
            "Block size can not be less than or equal to 1. Further it should be odd e.g. 3, 5, 7, 9 ..."
        )

    if not isinstance(subtract_val, (float, int)):
        raise WrongArgumentsType("Block size argument can only be provided as a integer")

    threshold_method = threshold_method.lower()
    threshold_method_arg = CV_THRESHOLD_STRATEGY.get(threshold_method, None)
    if threshold_method_arg is None:
        raise WrongArgumentsValue("Provided thresholding method is wrong")

    adaptive_method_arg = CV_THRESHOLD_STRATEGY.get(adaptive_method, None)
    if adaptive_method_arg is None:
        raise WrongArgumentsValue("Provided adaptive thresholding method is wrong")

    check_image = image_array_check_conversion(image)
    if check_image.dtype is not AllowedDataType.Uint8.value:
        check_image._image_conversion_helper(AllowedDataType.Uint8)

    try:
        check_image._set_image(
            cv2.adaptiveThreshold(
                check_image.image, float(max_val), adaptive_method_arg, threshold_method_arg,
                int(block_size), float(subtract_val)
            )
        )
    except Exception as e:
        raise FilteringError("Failed to apply the adaptive threshold method to the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def niblack_threshold(
    image: BaseImage,
    kernel_size: Optional[Union[float, int]] = 15,
    factor: Optional[float] = 0.2
) -> BaseImage:
    """Computes the local Niblack threshold"""

    if not isinstance(kernel_size, (float, int)):
        raise WrongArgumentsType("Kernel size must be either an integer or a float value")

    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise WrongArgumentsValue("Kernel size can not be <= 0 and must always be odd")

    if not isinstance(factor, float):
        raise WrongArgumentsType("Factor must only be defined as a float value")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            sk_niblack(check_image.image, kernel_size,
                       factor).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FilteringError("Failed to apply the Niblack threshold to the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def sauvola_threshold(
    image: BaseImage,
    kernel_size: Optional[Union[float, int]] = 15,
    factor: Optional[float] = 0.2
) -> BaseImage:
    """Applies local Sauvola threshold to the image"""

    if not isinstance(kernel_size, (float, int)):
        raise WrongArgumentsType("Kernel size must be either an integer or a float value")

    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise WrongArgumentsValue("Kernel size can not be <= 0 and must always be odd")

    if not isinstance(factor, float):
        raise WrongArgumentsType("Factor must only be defined as a float value")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            sk_sauvola(check_image.image, kernel_size,
                       factor).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FilteringError("Failed to apply the Niblack threshold to the image") from e

    return check_image

# -------------------------------------------------------------------------