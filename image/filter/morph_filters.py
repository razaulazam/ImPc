# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Morphological Image Filters

import cv2

import numpy as np
from typing import Optional, Union
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from image.load._interface import BaseImage
from image.common.helpers import image_array_check_conversion, check_user_provided_ndarray
from image.common.decorators import check_image_exist_external
from collections import namedtuple
from image.filter._common_methods import is_not_namedtuple

# -------------------------------------------------------------------------

@check_image_exist_external
def erode(
    image: BaseImage,
    kernel: Union[namedtuple, np.ndarray],
    iterations: Optional[Union[int, float]] = 1
) -> BaseImage:
    """Erode Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    if not isinstance(iterations, (int, float)):
        raise WrongArgumentsType(
            "Please check the type of the iterations argument. Only integers are allowed"
        )
    if int(iterations) < 0:
        raise WrongArgumentsValue("Value of the iterations cannot be less than 0")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.erode(check_image.image, kernel, int(iterations)))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def dilate(
    image: BaseImage,
    kernel: Union[namedtuple, np.ndarray],
    iterations: Optional[int] = 1
) -> BaseImage:
    """Dilate Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    if not isinstance(iterations, (int, float)):
        raise WrongArgumentsType(
            "Please check the type of the iterations argument. Only integers are allowed"
        )
    if int(iterations) < 0:
        raise WrongArgumentsValue("Value of the iterations cannot be less than 0")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.dilate(check_image.image, kernel, int(iterations)))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def closing(image: BaseImage, kernel: Union[namedtuple, np.ndarray]) -> BaseImage:
    """Closing Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.morphologyEx(check_image.image, cv2.MORPH_CLOSE, kernel))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def morph_gradient(image: BaseImage, kernel: Union[namedtuple, np.ndarray]) -> BaseImage:
    """Morph Gradient Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.morphologyEx(check_image.image, cv2.MORPH_GRADIENT, kernel))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def top_hat(image: BaseImage, kernel: Union[namedtuple, np.ndarray]) -> BaseImage:
    """Tophar Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.morphologyEx(check_image.image, cv2.MORPH_TOPHAT, kernel))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def black_hat(image: BaseImage, kernel: Union[namedtuple, np.ndarray]) -> BaseImage:
    """Blackhat Morphological Filter"""

    if not isinstance(kernel, np.ndarray) and is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel)

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(cv2.morphologyEx(check_image.image, cv2.MORPH_BLACKHAT, kernel))
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return check_image

# -------------------------------------------------------------------------