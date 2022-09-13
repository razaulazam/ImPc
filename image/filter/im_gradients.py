# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image gradient methods

import cv2
import numpy as np

from image.load._interface import BaseImage
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion, check_user_provided_ndarray
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from typing import Union, List, Tuple, Optional
from skimage.filters.edges import scharr as sk_scharr
from skimage.filters._unsharp_mask import unsharp_mask as sk_unsharp_mask
from image._common_datastructs import SKIMAGE_SAMPLING_REGISTRY, AllowedDataType
from image._common_datastructs import CV_BORDER_INTERPOLATION

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

    check_image = image_array_check_conversion(image)

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

    try:
        check_image._set_image(
            cv2.Laplacian(
                check_image.image, int(kernel_size[0]), float(scale), float(delta), border_actual
            ).astype(check_image.dtype.value, copy=False)
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

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

    check_image = image_array_check_conversion(image)

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

    try:
        check_image._set_image(
            cv2.Sobel(
                check_image.image, -1, xorder, yorder, cv2.FILTER_SCHARR, float(scale),
                float(delta), border_actual
            )
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def scharr(
    image: BaseImage,
    mask: Optional[np.ndarray] = None,
    mode: Optional[str] = "reflect"
) -> BaseImage:
    """Edge magnitude using Scharr transform. Result is returned as float32 image"""

    if mask and not isinstance(mask, np.ndarray):
        raise WrongArgumentsType(
            "Mask should be provided as a numpy array with the same size as the image"
        )

    if mask and (len(mask.shape) > 3 or len(mask.shape) < 2):
        raise WrongArgumentsValue("Incorrect dimensions for the mask provided")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode should be provided as a string")

    if image.is_gray() and image.dims != mask.shape:
        raise WrongArgumentsValue("Dimensions of the image and the mask does not match")

    if image.is_rgb():
        if len(mask.shape) == 2 and image.dims != mask.shape:
            raise WrongArgumentsValue("Dimensions of the image and the mask does not match")
        elif len(mask.shape) == 3 and (image.dims + (image.channels) != mask.shape):
            raise WrongArgumentsValue(
                "Dimensions of the image and the mask does not match. Please look at the dimensions"
            )

    check_mask = check_user_provided_ndarray(mask) if mask else None
    check_image = image_array_check_conversion(image)

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode.lower(), None)
    if mode_arg is None:
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["reflect"]
        DefaultSetting("Using default mode (reflect) since the provided mode type is not supported")

    try:
        check_image._set_image(
            sk_scharr(check_image.image, mask=check_mask,
                      mode=mode_arg).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to filter the image using Prewitt transform") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def unsharp_mask_filter(
    image: BaseImage,
    radius: Optional[Union[float, int]] = 1.0,
    scale_factor: Optional[Union[float, int]] = 1.0
) -> BaseImage:
    """Identifies the sharp details in the image and adds them back to the original image"""

    if not isinstance(radius, (float, int)):
        raise WrongArgumentsType("Radius can only be specified as int or float")

    if not isinstance(scale_factor, (float, int)):
        raise WrongArgumentsType("Scale factor can only be specified as int or float")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_unsharp_mask(
                check_image.image,
                radius=float(radius),
                amount=float(scale_factor),
                preserve_range=True,
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value),
            copy=False
        )
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to filter the image with the Unsharp mask filter") from e

    return check_image

# -------------------------------------------------------------------------
