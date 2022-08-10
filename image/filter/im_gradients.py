# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image gradient methods

import cv2

from image.load._interface import BaseImage
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from typing import Union, List, Tuple, Optional
from conv_filters import BORDER_INTERPOLATION

# -------------------------------------------------------------------------

@check_image_exist_external
def laplacian(
    image: BaseImage,
    kernel_size: Union[List[int], Tuple[int, int]],
    scale: Optional[float] = 0.0,
    delta: Optional[float] = 0.0,
    border: Optional[str] = "default"
) -> BaseImage:
    """Scale factor, delta and border are optional
    Kernal size must be odd and positive"""

    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType("Kernel size must either be provided as tuple or list")

    if len(kernel_size) != 2:
        raise WrongArgumentsValue(
            "Kernel size must have a length of 2 representing its width and height"
        )

    if kernel_size[0] != kernel_size[1]:
        raise WrongArgumentsValue("Kernel size must be equal in both the dimensions")

    if not all(i > 0 for i in kernel_size):
        raise WrongArgumentsValue("Kernel size must be a positive value")

    if kernel_size[0] % 2 == 0:
        raise WrongArgumentsValue(
            "Kernel size can only have odd values for this filter to work properly"
        )

    if not isinstance(scale, (int, float)):
        raise WrongArgumentsType("Provided value of scale does not have the accurate type")

    if not isinstance(delta, (float, int)):
        raise WrongArgumentsType("Provided value of delta does not have the accurate type")

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

    try:
        new_im = image.copy()
        new_im.image = cv2.Laplacian(
            new_im.image, int(kernel_size[0]), float(scale), float(delta), border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def sobel(
    image: BaseImage,
    xorder: int,
    yorder: int,
    scale: float,
    delta: float,
    border: Optional[str] = "default"
) -> BaseImage:
    """First, second, and mixed image derivatives can be calculated from this function."""

    image_array_check_conversion(image, "openCV")

    if not isinstance(xorder, int):
        raise WrongArgumentsType("Provided argument (xorder) should have a integer type")

    if not isinstance(yorder, int):
        raise WrongArgumentsType("Provided argument (yorder) should have a integer type")

    if not isinstance(scale, (int, float)):
        raise WrongArgumentsType("Provided value of scale does not have the accurate type")

    if not isinstance(delta, (float, int)):
        raise WrongArgumentsType("Provided value of delta does not have the accurate type")

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

    try:
        new_im = image.copy()
        new_im.image = cv2.Sobel(
            new_im.image, -1, xorder, yorder, cv2.FILTER_SCHARR, float(scale), float(delta),
            border_actual
        )
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

# -------------------------------------------------------------------------
