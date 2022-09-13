# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Image restoration methods

import cv2

from typing import Optional, Union
from commons.exceptions import WrongArgumentsType, RestorationError
from commons.warning import DefaultSetting, ImageDataTypeConversion
from image.load._interface import BaseImage
from image._helpers import image_array_check_conversion
from image._common_datastructs import AllowedDataType, CV_BORDER_INTERPOLATION
from image._decorators import check_image_exist_external
from skimage.restoration import denoise_tv_bregman as sk_denoise_bregman
from skimage.restoration import denoise_tv_chambolle as sk_denoise_tv_chambolle

# -------------------------------------------------------------------------

@check_image_exist_external
def bilateral_filter(
    image: BaseImage,
    kernel_diameter: int,
    color_sigma: float,
    spatial_sigma: float,
    border: Optional[str] = "default"
) -> BaseImage:
    """Only 8-bit and 32-bit floating point images are supported"""
    """Does not work with RGBA images"""

    if image.is_rgba():
        raise RestorationError(
            "This filter cannot operate on images that have color channels more than 3"
        )

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

    check_image = image_array_check_conversion(image)

    if check_image.dtype is AllowedDataType.Uint16:
        ImageDataTypeConversion(
            "Converting the image from 16 bits to 8 bits per channel since this is what is only supported for this filter"
        )
        check_image._image_conversion_helper(AllowedDataType.Uint8)
        check_image._update_dtype()

    border_actual = CV_BORDER_INTERPOLATION.get(border, None)
    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    try:
        check_image._set_image(
            cv2.bilateralFilter(
                check_image.image, int(kernel_diameter), float(color_sigma), float(spatial_sigma),
                border_actual
            )
        )
    except Exception as e:
        raise RestorationError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def bregman_denoising(
    image: BaseImage,
    weight: Optional[Union[float, int]] = 5.0,
    num_iter: Optional[Union[float, int]] = 100,
    stop_tolerance: Optional[float] = 0.001,
    isotropic: Optional[bool] = True
) -> BaseImage:
    """Total variation denoising using split-Bregman optimization. Result is returned as float32"""

    if not isinstance(weight, (float, int)):
        raise WrongArgumentsType("Weight can only be provided as either float or integer")

    if not isinstance(num_iter, (float, int)):
        raise WrongArgumentsType("Number of iterations can only be specified as integer or float")

    if not isinstance(stop_tolerance, float):
        raise WrongArgumentsType("Stop tolerance can only be provided as float")

    if not isinstance(isotropic, bool):
        raise WrongArgumentsType(
            "Option between isotropic and anisotropic mode can only be provided as bool"
        )

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_denoise_bregman(
                check_image.image,
                weight=float(weight),
                max_num_iter=int(num_iter),
                eps=float(stop_tolerance),
                isotropic=isotropic,
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise RestorationError(
            "Failed to denoise the image using total variations filter with split-bregman optimization"
        ) from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def tv_chambolle_denoising(
    image: BaseImage,
    weight: Optional[Union[float, int]] = 0.1,
    stop_tolerance: Optional[float] = 0.0002,
    num_iter: Optional[Union[float, int]] = 200
) -> BaseImage:
    """Total variations denoising"""

    if not isinstance(weight, (float, int)):
        raise WrongArgumentsType("Weight can only be provided as either float or integer")

    if not isinstance(num_iter, (float, int)):
        raise WrongArgumentsType("Number of iterations can only be specified as integer or float")

    if not isinstance(stop_tolerance, float):
        raise WrongArgumentsType("Stop tolerance can only be provided as float")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_denoise_tv_chambolle(
                check_image.image,
                weight=float(weight),
                max_num_iter=int(num_iter),
                eps=float(stop_tolerance),
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise RestorationError("Failed to denoise the image using total variations filter") from e

    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from image.load.loader import open_image
    from skimage.restoration import denoise_tv_bregman

    image_path = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(image_path))

    output = denoise_tv_bregman(image.image)

    print("dsad")