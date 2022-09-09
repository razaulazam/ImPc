# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image thresholding methods

import cv2

from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion
from image._common_datastructs import AllowedDataType
from typing import Union

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

    threshold_method_arg = CV_THRESHOLD_STRATEGY.get(threshold_method, None)
    if threshold_method_arg is None:
        raise WrongArgumentsValue("Provided thresholding method is wrong")

    adaptive_method_arg = CV_THRESHOLD_STRATEGY.get(adaptive_method, None)
    if adaptive_method_arg is None:
        raise WrongArgumentsValue("Provided adaptive thresholding method is wrong")

    check_image = image_array_check_conversion(image)
    if check_image.dtype is not AllowedDataType.Uint8.value:
        check_image._image_conversion_helper(AllowedDataType.Uint8)
        check_image._update_dtype()

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

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    import numpy as np
    from image.load.loader import open_image
    from image.transform.color_conversion import convert
    from skimage.filters.thresholding import threshold_li
    image_path = Path(__file__).parent.parent.parent / "sample.jpg"

    image = open_image(str(image_path))
    #image = convert(image, "rgb2gray")
    #image = image.image.astype(np.uint8)

    im1 = threshold_li(image.image)

    print("hallo")