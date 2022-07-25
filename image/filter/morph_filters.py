# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
# \brief Morphological Image Filters

import cv2

import numpy as np
from typing import Optional, Union
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from image.load._interface import PyFaroImage
from image._helpers import image_array_check_conversion, check_user_provided_ndarray
from image._decorators import check_image_exist_external
from collections import namedtuple

# -------------------------------------------------------------------------

@check_image_exist_external
def erode(
    image: PyFaroImage,
    kernel: Union[namedtuple, np.ndarray],
    iterations: Optional[Union[int, float]] = 1
) -> PyFaroImage:

    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    if not isinstance(iterations, (int, float)):
        raise WrongArgumentsType(
            "Please check the type of the iterations argument. Only integers are allowed"
        )
    if int(iterations) < 0:
        raise WrongArgumentsValue("Value of the iterations cannot be less than 0")

    try:
        new_im = image.copy()
        new_im.image = cv2.erode(new_im.image, kernel, int(iterations))
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def dilate(
    image: PyFaroImage,
    kernel: Union[namedtuple, np.ndarray],
    iterations: Optional[int] = 1
) -> PyFaroImage:
    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    if not isinstance(iterations, (int, float)):
        raise WrongArgumentsType(
            "Please check the type of the iterations argument. Only integers are allowed"
        )
    if int(iterations) < 0:
        raise WrongArgumentsValue("Value of the iterations cannot be less than 0")

    try:
        new_im = image.copy()
        new_im.image = cv2.dilate(new_im.image, kernel, int(iterations))
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def closing(image: PyFaroImage, kernel: Union[namedtuple, np.ndarray]) -> PyFaroImage:
    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    try:
        new_im = image.copy()
        new_im.image = cv2.morphologyEx(new_im.image, cv2.MORPH_CLOSE, kernel)
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def morph_gradient(image: PyFaroImage, kernel: Union[namedtuple, np.ndarray]) -> PyFaroImage:
    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    try:
        new_im = image.copy()
        new_im.image = cv2.morphologyEx(new_im.image, cv2.MORPH_GRADIENT, kernel)
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def top_hat(image: PyFaroImage, kernel: Union[namedtuple, np.ndarray]) -> PyFaroImage:
    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    try:
        new_im = image.copy()
        new_im.image = cv2.morphologyEx(new_im.image, cv2.MORPH_TOPHAT, kernel)
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def black_hat(image: PyFaroImage, kernel: Union[namedtuple, np.ndarray]) -> PyFaroImage:
    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, np.ndarray) and _is_not_namedtuple(kernel):
        raise WrongArgumentsType(
            "Please check the type of the provided kernel. Use get_kernel() method of the library instead"
        )

    if not _is_not_namedtuple(kernel):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    try:
        new_im = image.copy()
        new_im.image = cv2.morphologyEx(new_im.image, cv2.MORPH_BLACKHAT, kernel)
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to dilate the image") from e

    return new_im

# -------------------------------------------------------------------------

def _is_not_namedtuple(source: namedtuple) -> bool:
    return not (
        isinstance(source, tuple) and hasattr(source, "_asdict") and hasattr(source, "_fields")
    )

# -------------------------------------------------------------------------

if __name__ == "__main__":
    a = "C:\\dev\\pyfaro\\sample.jpg"
    from image.load.loader import open_image
    b = open_image(a)
    c = top_hat(b, np.ones((5, 5), dtype=np.uint8))

    print(c)