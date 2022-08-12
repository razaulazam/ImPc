# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image processing methods

import cv2
import numpy as np

from typing import Union, Tuple, List
from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue
from image._helpers import image_array_check_conversion
from commons.exceptions import ProcessingError, ImageAlreadyClosed

# -------------------------------------------------------------------------

def blend(image_one: BaseImage, image_two: BaseImage, alpha: float) -> BaseImage:
    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_one argument")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_two argument")

    if image_one.closed:
        raise ImageAlreadyClosed("Provided first image is already closed")

    if image_two.closed:
        raise ImageAlreadyClosed("Provided second image is already closed")

    if not isinstance(alpha, (float, int)):
        raise WrongArgumentsType("Please check the type of the alpha argument")

    if image_one.dims != image_two.dims:
        raise WrongArgumentsValue("Dimensions of the images are not the same")

    if image_one.channels != image_two.channels:
        raise WrongArgumentsValue("Number of channels should be same for both the images")

    if image_one.mode != image_two.mode:
        raise WrongArgumentsValue("Mode should be similar for both the images")

    if image_one.dtype != image_two.dtype:
        raise WrongArgumentsValue("Provided images should have the same data type")

    checked_image_one = image_array_check_conversion(image_one, "openCV")
    checked_image_two = image_array_check_conversion(image_two, "openCV")

    try:
        new_im = checked_image_one.copy()
        new_im._set_image(
            cv2.addWeighted(
                checked_image_one.image, alpha, checked_image_two.image, float(1.0 - alpha)
            )
        )
        new_im._update_dtype()
    except Exception as e:
        raise ProcessingError("Failed to alpha blend the image") from e

    return new_im

# -------------------------------------------------------------------------

def composite(image_one: BaseImage, image_two: BaseImage, mask: np.ndarray) -> BaseImage:

    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_one argument")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_two argument")

    if image_one.closed:
        raise ImageAlreadyClosed("Provided first image is already closed")

    if image_two.closed:
        raise ImageAlreadyClosed("Provided second image is already closed")

    if image_one.channels != image_two.channels:
        raise WrongArgumentsValue("Number of channels should be same for both the images")

    if image_one.mode != image_two.mode:
        raise WrongArgumentsValue("Mode should be similar for both the images")

    if image_one.dtype != image_two.dtype:
        raise WrongArgumentsValue("Provided images should have the same data type")

    try:
        mask_dims = _compute_mask_dims(mask)
    except Exception as e:
        raise RuntimeError("Provide mask image does not have the accurate dimensions") from e

    if image_one.dims == image_two.dims == mask_dims:
        raise WrongArgumentsValue("Dimensions of the provided images do not match")

    mask = _adjust_mask_dtype(mask, np.uint8)
    mask = _normalize_mask(mask)

    checked_image_one = image_array_check_conversion(image_one, "openCV")
    checked_image_two = image_array_check_conversion(image_two, "openCV")

    new_im = checked_image_one.copy()
    raw_image_one = checked_image_one.image
    raw_image_two = checked_image_two.image
    raw_image_one[mask == 1] = raw_image_two

    new_im._set_image(raw_image_one)
    new_im._update_dtype()

    return new_im

# -------------------------------------------------------------------------

def gaussian_pyramid(image: BaseImage, level: int) -> List[BaseImage]:
    """Computes the gaussian pyramid"""

    if not isinstance(image, BaseImage):
        raise WrongArgumentsType(
            "Provided image should be opened by the open_image() function first"
        )

    if not isinstance(level, (int, float)):
        raise WrongArgumentsType("Provided level value does not have the accurate type")

    if level <= 0:
        raise WrongArgumentsValue("Level cannot be zero or less than zero")

    check_image = image_array_check_conversion(image, "openCV")
    pyramid = []

    for _ in range(level):
        pyr_level = check_image.copy()
        pyr_level._set_image(cv2.pyrDown(pyr_level.image))
        pyr_level._update_dtype()
        pyramid.append(pyr_level)
        check_image = pyr_level

    return pyramid

# -------------------------------------------------------------------------

def _adjust_mask_dtype(mask: np.ndarray, desired_type: np.dtype):
    return mask.astype(desired_type, copy=False)

# -------------------------------------------------------------------------

def _compute_mask_dims(mask: np.ndarray) -> Union[Tuple[int, int, int], Tuple[int, int]]:
    """Computes the shape of the mask"""

    mask_shape = mask.shape
    length_mask_shape = len(mask_shape)

    if length_mask_shape < 2:
        raise ValueError("Mask must be at least two dimensional")

    if length_mask_shape == 2 or (length_mask_shape == 3 and mask_shape[-1] == 3):
        return mask_shape

    if length_mask_shape == 3 and mask_shape[-1] == 1:
        return mask_shape[:-1]

# -------------------------------------------------------------------------

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalizes the mask"""

    mask_data_type = str(mask.dtype)
    max_val = np.max(mask)

    if max_val != float(1) if "float" in mask_data_type else 1:
        mask = mask / max_val
    return mask.astype(mask_data_type)

# -------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    from image.load.loader import open_image
    path_image = "C:\\dev\\ImProcMagic\\sample.jpg"
    image = open_image(path_image)

    b = gaussian_pyramid(image, 2)
    print("hallo")