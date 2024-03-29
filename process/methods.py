# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image processing methods

import cv2
import numpy as np

from typing import Union, Tuple, List
from common.interfaces.loader import BaseImage
from common.exceptions import WrongArgumentsType, WrongArgumentsValue
from common.helpers import image_array_check_conversion
from common.exceptions import ProcessingError, ImageAlreadyClosed
from transform import resize
from common.decorators import check_image_exist_external
from common.datastructs import AllowedDataType

# -------------------------------------------------------------------------

def blend(image_one: BaseImage, image_two: BaseImage, alpha: float) -> BaseImage:
    """Functionality for alpha blending two images"""

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

    checked_image_one = image_array_check_conversion(image_one)
    checked_image_two = image_array_check_conversion(image_two)

    try:
        checked_image_one._set_image(
            cv2.addWeighted(
                checked_image_one.image,
                alpha,
                checked_image_two.image,
                float(1.0 - alpha),
                gamma=0
            ).astype(checked_image_one.dtype.value, copy=False)
        )
    except Exception as e:
        raise ProcessingError("Failed to blend the image") from e

    return checked_image_one

# -------------------------------------------------------------------------

def composite(image_one: BaseImage, image_two: BaseImage, mask: np.ndarray) -> BaseImage:
    """Composes two images based on a mask"""

    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_one argument")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_two argument")

    if image_one.closed:
        raise ImageAlreadyClosed("Provided first image is already closed")

    if image_two.closed:
        raise ImageAlreadyClosed("Provided second image is already closed")

    if image_one.dims != image_two.dims:
        raise WrongArgumentsValue("Dimensions of the images are not the same")

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

    if image_one.dims != mask_dims:
        raise WrongArgumentsValue("Dimensions of the provided images do not match")

    mask = _adjust_mask_dtype(mask, AllowedDataType.Uint8)
    mask = _normalize_mask(mask)

    checked_image_one = image_array_check_conversion(image_one)
    checked_image_two = image_array_check_conversion(image_two)

    raw_image_one = checked_image_one.image
    raw_image_two = checked_image_two.image

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] == 1:
                raw_image_one[row, col] = raw_image_two[row, col]

    checked_image_one._set_image(raw_image_one)

    return checked_image_one

# -------------------------------------------------------------------------

@check_image_exist_external
def gaussian_pyramid(image: BaseImage, level: Union[int, float]) -> List[BaseImage]:
    """Computes the gaussian pyramid where the first image is always the original image itself"""

    if not isinstance(level, (int, float)):
        raise WrongArgumentsType("Provided level value does not have the accurate type")

    if level <= 0:
        raise WrongArgumentsValue("Level cannot be zero or less than zero")

    check_image = image_array_check_conversion(image)
    pyramid = []
    pyramid.append(check_image)

    for _ in range(int(level)):
        pyr_level = check_image
        pyr_level._set_image(cv2.pyrDown(pyr_level.image).astype(pyr_level.dtype.value, copy=False))
        pyramid.append(pyr_level)
        check_image = pyr_level.copy()

    assert len(pyramid) == int(level) + 1, ProcessingError(
        "Failed to compute the gaussian pyramid with accurate number of levels"
    )

    return pyramid

# -------------------------------------------------------------------------

def laplacian_pyramid(image: BaseImage, level: Union[int, float]) -> List[BaseImage]:
    """Computes the laplacian pyramid from the gaussian pyramid"""

    gauss_pyramid = gaussian_pyramid(image, level)
    laplacian_pyramid = []

    for i in range(len(gauss_pyramid) - 1, 0, -1):
        pyr_level = gauss_pyramid[i]
        pyr_level._set_image(cv2.pyrUp(pyr_level.image).astype(pyr_level.dtype.value, copy=False))
        pyr_level_down = gauss_pyramid[i - 1]
        if pyr_level.dims != pyr_level_down.dims:
            pyr_level_down = resize(pyr_level_down, pyr_level.dims)
        pyr_level._set_image(
            cv2.subtract(pyr_level_down.image,
                         pyr_level.image).astype(pyr_level.dtype.value, copy=False)
        )
        laplacian_pyramid.append(pyr_level)

    assert len(laplacian_pyramid) == int(level), ProcessingError(
        "Failed to compute the laplacian pyramid with accurate number of levels"
    )

    return laplacian_pyramid

# -------------------------------------------------------------------------

def pyramid_blend(
    image_one: BaseImage, image_two: BaseImage, level: Union[int, float]
) -> BaseImage:
    """Blends two images after computing their laplacian pyramids"""

    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Provided image is not a ImageLoader instance")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Provided image is not a ImageLoader instance")

    if image_one.closed:
        raise ImageAlreadyClosed("Processing cannot be performed on closed images")

    if image_two.closed:
        raise ImageAlreadyClosed("Processing cannot be performed on closed images")

    if image_one.dims != image_two.dims:
        raise WrongArgumentsValue("Dimensions of the images are not the same")

    if image_one.channels != image_two.channels:
        raise WrongArgumentsValue("Number of channels should be same for both the images")

    if image_one.mode != image_two.mode:
        raise WrongArgumentsValue("Mode should be similar for both the images")

    if image_one.dtype != image_two.dtype:
        raise WrongArgumentsValue("Provided images should have the same data type")

    lap_pyr_one = laplacian_pyramid(image_one, level)
    lap_pyr_two = laplacian_pyramid(image_two, level)

    combined_pyr = []

    # Blend the halves
    for i in range(int(level)):
        lap_image_one = lap_pyr_one[i].image
        lap_image_two = lap_pyr_two[i].image

        image_dims = lap_pyr_one[i].dims
        combined_image = np.hstack(
            (
                lap_image_one[:, :image_dims[1] // 2, ...], lap_image_two[:, image_dims[1] // 2:,
                                                                          ...]
            )
        )

        lap_pyr_one[i]._set_image(combined_image)
        combined_pyr.append(lap_pyr_one[i])

    assert len(combined_pyr) == int(level), ProcessingError("Failed to perform pyramid blending")

    # Reconstruction
    start_level = combined_pyr[0]
    for i in range(1, len(combined_pyr)):
        level = combined_pyr[i]
        start_level._set_image(
            cv2.pyrUp(start_level.image).astype(start_level.dtype.value, copy=False)
        )
        dim_level, dim_start_level = level.dims, start_level.dims
        if dim_level > dim_start_level:
            level = _adjust_dims(level, dim_start_level)
        else:
            start_level = _adjust_dims(start_level, dim_level)
        combined_image = cv2.add(level.image, start_level.image).astype(
            start_level.dtype.value, copy=False
        )
        start_level._set_image(combined_image)

    return start_level

# -------------------------------------------------------------------------

def _adjust_mask_dtype(mask: np.ndarray, desired_type: AllowedDataType):
    return mask.astype(desired_type.value, copy=False)

# -------------------------------------------------------------------------

def _compute_mask_dims(mask: np.ndarray) -> Union[Tuple[int, int, int], Tuple[int, int]]:
    """Computes the shape of the mask"""

    mask_shape = mask.shape
    length_mask_shape = len(mask_shape)

    if length_mask_shape < 2 or length_mask_shape > 3:
        raise ValueError(
            "Mask must be at least two dimensional and must not exceed three dimensions"
        )

    if length_mask_shape == 2:
        return mask_shape

    if length_mask_shape == 3:
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

def _adjust_dims(image_one: BaseImage, dims: Tuple[int, int]) -> BaseImage:
    """Adjust the dimensions of the image"""

    return resize(image_one, dims)

# -------------------------------------------------------------------------