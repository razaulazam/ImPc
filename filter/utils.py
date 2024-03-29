# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Utility methods that support filtering operations

import cv2
import numpy as np

from typing import Union, List, Tuple, Optional
from common.interfaces.loader import BaseImage
from common.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from collections import namedtuple
from skimage.filters.thresholding import threshold_isodata as sk_thresh_isodata
from skimage.filters.thresholding import threshold_li as sk_thresh_li
from skimage.filters.thresholding import threshold_mean as sk_thresh_mean
from skimage.filters.thresholding import threshold_minimum as sk_thresh_minimum
from skimage.filters.thresholding import threshold_otsu as sk_thresh_otsu
from skimage.filters.thresholding import threshold_triangle as sk_thresh_triangle
from skimage.filters.thresholding import threshold_yen as sk_thresh_yen
from skimage.filters.thresholding import threshold_multiotsu as sk_thresh_multiotsu
from common.helpers import image_array_check_conversion
from common.decorators import check_image_exist_external

# -------------------------------------------------------------------------

ALLOWED_KERNELS = {
    "rectangle": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS
}

# -------------------------------------------------------------------------

def get_kernel(kernel_shape: str, kernel_size: Union[List[int], Tuple[int, int]]) -> namedtuple:
    """Utility for getting the kernel of a particular shape"""

    if not isinstance(kernel_shape, str):
        raise WrongArgumentsType(
            "Please check the type of the first argument. Only strings are allowed"
        )

    kernel_shape = kernel_shape.lower()
    shape_strategy = ALLOWED_KERNELS.get(kernel_shape, None)
    if shape_strategy is None:
        raise WrongArgumentsValue(
            "Provided shape of the kernel is currently not supported. Please provide it yourself or stick to the provided ones by the library"
        )

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType(
            "Please check the type of the size argument. Only tuples and lists are allowed"
        )

    if len(kernel_size) != 2:
        raise WrongArgumentsValue("Expected a tuple/list of length two for the kernel size")

    if not all(i > 0 for i in kernel_size):
        raise WrongArgumentsValue(
            "Provided width or height of the kernel is negative which is not allowed"
        )

    kernel_size = [int(i) for i in kernel_size]

    structuring_element = cv2.getStructuringElement(shape_strategy, kernel_size)
    _impc_array = namedtuple("ImPcArray", "array_")

    return _impc_array(structuring_element)

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_isodata(image: BaseImage,
                              bins: Optional[Union[float, int]] = 256) -> Union[float, int]:
    """Computes the threshold based on isodata strategy"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_isodata(check_image.image, nbins=int(bins))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on isodata strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_li(
    image: BaseImage, start_guess: Optional[Union[float, int]] = None
) -> float:
    """Computes the threshold based on Li's iterative minimum cross entropy method"""

    if start_guess and not isinstance(start_guess, (float, int)):
        raise WrongArgumentsType(
            "Starting guess value can only be provided as either integer or float"
        )

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_li(check_image.image, initial_guess=float(start_guess))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on li's strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_mean(image: BaseImage) -> float:
    """Computes the threshold based on the mean of pixel values in the image provided"""

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_mean(check_image.image)
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on mean strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_minimum(image: BaseImage, bins: Optional[Union[float, int]] = 256) -> float:
    """Computes the threshold based on the minimum method"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_minimum(check_image.image, nbins=int(bins))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on minimum strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_otsu(image: BaseImage, bins: Optional[Union[float, int]] = 256) -> float:
    """Computes the threshold based on otsu's method"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_otsu(check_image.image, nbins=int(bins))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on otsu's strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_triangle(image: BaseImage, bins: Optional[Union[float, int]] = 256) -> float:
    """Computes the threshold based on the triangle method"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_triangle(check_image.image, nbins=int(bins))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on triangle strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_yen(image: BaseImage, bins: Optional[Union[float, int]] = 256) -> float:
    """Computes the threshold based on the yen's method"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_yen(check_image.image, nbins=int(bins))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on yen's strategy") from e

    return threshold

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_threshold_multiotsu(
    image: BaseImage,
    classes: Optional[Union[float, int]] = 3,
    bins: Optional[Union[float, int]] = 256
) -> np.ndarray:
    """Computes the threshold based on the multiotsu method"""

    if not isinstance(bins, (float, int)):
        raise WrongArgumentsType("Bins can only be provided as either integer or float")

    if not isinstance(classes, (float, int)):
        raise WrongArgumentsType("Classes can only be provided as either integer or float")

    check_image = image_array_check_conversion(image)

    try:
        threshold = sk_thresh_multiotsu(check_image.image, nbins=int(bins), classes=int(classes))
    except Exception as e:
        raise FilteringError("Failed to compute the threshold based on multiotsu strategy") from e

    return threshold

# -------------------------------------------------------------------------