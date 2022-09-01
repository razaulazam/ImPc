# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image filtering methods

from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from image._decorators import check_image_exist_external
from image._common_datastructs import AllowedDataType, SKIMAGE_SAMPLING_REGISTRY
from image._helpers import image_array_check_conversion
from skimage.filters._fft_based import butterworth as sk_butterworth
from skimage.filters._gaussian import difference_of_gaussians as sk_difference_gaussians
from typing import Optional, Union

# -------------------------------------------------------------------------

@check_image_exist_external
def butterworth(
    image: BaseImage,
    f_cutoff_ratio: Optional[float] = 0.005,
    high_pass: Optional[bool] = True,
    filter_order: Optional[Union[int, float]] = 2.0,
    padding: Optional[int] = 0
) -> BaseImage:
    """Applies the butterworth filter. The result is returned as float32"""

    if not isinstance(f_cutoff_ratio, float):
        raise WrongArgumentsType("Cutoff frequency ratio must have float type")

    if f_cutoff_ratio < 0 or f_cutoff_ratio > 0.5:
        raise WrongArgumentsValue("Cutoff frequency ratio must be between 0 and 0.5 (inclusive)")

    if not isinstance(high_pass, bool):
        raise WrongArgumentsType("High pass filtering argument must be provided as boolean")

    if not isinstance(filter_order, (int, float)):
        raise WrongArgumentsType("Filter order argument must be provided as integer or float")

    if not isinstance(padding, int):
        raise WrongArgumentsType("Padding argument must be provided as integer")

    if padding < 0:
        raise WrongArgumentsValue("Padding value must be >= 0")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_butterworth(
                check_image.image,
                cutoff_frequency_ratio=f_cutoff_ratio,
                high_pass=high_pass,
                channel_axis=channel_axis,
                order=filter_order,
                n_pad=padding
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to apply butterworth filter to the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def difference_gaussians(
    image: BaseImage,
    sigma_low: float,
    sigma_high: Optional[float] = None,
    mode: Optional[str] = "nearest"
) -> BaseImage:
    """Bandpass filter applied based on computing two gaussians. Result is return as float32"""

    if not isinstance(sigma_low, float):
        raise WrongArgumentsType("Cutoff frequency ratio must have float type")

    if sigma_low < 0:
        raise WrongArgumentsValue("Low sigma value cannot be less than zero")

    if sigma_high and sigma_high < sigma_low:
        raise WrongArgumentsValue(
            "High sigma value should be greater than or equal to low sigma value"
        )

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode should have string argument type")

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode.lower(), None)
    if mode_arg is None:
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["nearest"]
        DefaultSetting("Using default mode (nearest) since the provided mode type is not supported")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_difference_gaussians(
                check_image.image,
                low_sigma=sigma_low,
                high_sigma=sigma_high,
                mode=mode_arg,
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FilteringError("Failed to filter the image with difference of gaussians strategy") from 

    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from image.load.loader import open_image
    import cv2
    from image.transform.color_conversion import convert
    import numpy as np
    from pathlib import Path
    from skimage.filters._gaussian import difference_of_gaussians
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image_ = open_image(str(path_image))
    #image_ = convert(image_, "rgb2hsv")
    image_ = image_.image

    im1 = difference_of_gaussians(image_, 2.0, 1.0)
    im1 = im1.astype(np.float32)
    max_im1 = np.max(im1)
    im1 = (im1/max_im1) * 255
    im1 = np.clip(im1, 0, 255)

    print("hallo")