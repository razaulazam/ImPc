# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image filtering methods

import numpy as np

from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from image._decorators import check_image_exist_external
from image._common_datastructs import AllowedDataType, SKIMAGE_SAMPLING_REGISTRY
from image._helpers import image_array_check_conversion, check_user_provided_ndarray
from skimage.filters._fft_based import butterworth as sk_butterworth
from skimage.filters._gaussian import difference_of_gaussians as sk_difference_gaussians
from skimage.filters._gabor import gabor as sk_gabor
from skimage.filters.edges import farid as sk_farid
from skimage.filters.edges import prewitt as sk_prewitt
from typing import Optional, Union, Tuple

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
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError(
            "Failed to filter the image with difference of gaussians strategy"
        ) from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def farid(
    image: BaseImage,
    mask: Optional[np.ndarray] = None,
    mode: Optional[str] = "reflect"
) -> BaseImage:
    """Computes the Farid transform which finds the edge magnitude. Result is return as float32"""

    if not image.is_gray():
        raise WrongArgumentsValue("This method only work with gray images")

    if mask and not isinstance(mask, np.ndarray):
        raise WrongArgumentsType(
            "Mask should be provided as a numpy array with the same size as the image"
        )

    if mask and (len(mask.shape) > 2 or len(mask.shape) < 2):
        raise WrongArgumentsValue("Mask can only be provided as a 2D array")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode should be provided as a string")

    if image.dims != mask.shape:
        raise WrongArgumentsValue("Dimensions of the image and the mask does not match")

    check_mask = check_user_provided_ndarray(mask) if mask else None
    check_image = image_array_check_conversion(image)

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode.lower(), None)
    if mode_arg is None:
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["reflect"]
        DefaultSetting("Using default mode (reflect) since the provided mode type is not supported")

    try:
        check_image._set_image(
            sk_farid(check_image.image, mask=check_mask,
                     mode=mode_arg).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to filter the image using Farid transform") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def gabor(
    image: BaseImage,
    frequency: float,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    mode: Optional[str] = "constant"
) -> Tuple[BaseImage, BaseImage]:
    """Return imaginary and real response to the gabor filter. Responses are returned as a tuple of float32 image loader instances"""

    if not image.is_gray():
        raise WrongArgumentsValue("This method only work with gray images")

    if not isinstance(frequency, float):
        raise WrongArgumentsType("Frequency argument should be specified as a float value")

    if sigma_x and not isinstance(sigma_x, float):
        raise WrongArgumentsType("Sigma (X) argument should be specified as a float value")

    if sigma_y and not isinstance(sigma_y, float):
        raise WrongArgumentsType("Sigma (X) argument should be specified as a float value")

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode.lower(), None)
    if mode_arg is None:
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["reflect"]
        DefaultSetting("Using default mode (reflect) since the provided mode type is not supported")

    check_image = image_array_check_conversion(image)
    check_image_imaginary = check_image.copy()

    try:
        real_resp, im_resp = sk_gabor(
            check_image.image, frequency=frequency, sigma_x=sigma_x, sigma_y=sigma_y, mode=mode_arg
        )
        check_image._set_image(real_resp.astype(AllowedDataType.Float32.value, copy=False))
        check_image._update_dtype()
        check_image_imaginary._set_image(im_resp.astype(AllowedDataType.Float32.value, copy=False))
        check_image_imaginary._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to filter the image using Gabor transform") from e

    return (check_image, check_image_imaginary)

# -------------------------------------------------------------------------

@check_image_exist_external
def prewitt(image: BaseImage, mask: Optional[np.ndarray] = None, mode: Optional[str] = "reflect") -> BaseImage:
    """Edge magnitude using prewitt transform"""
    
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
            raise WrongArgumentsValue("Dimensions of the image and the mask does not match. Please look at the dimensions")

    check_mask = check_user_provided_ndarray(mask) if mask else None
    check_image = image_array_check_conversion(image)

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode.lower(), None)
    if mode_arg is None:
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["reflect"]
        DefaultSetting("Using default mode (reflect) since the provided mode type is not supported")
    
    try:
        check_image._set_image(sk_prewitt(check_image.image, mask=check_mask, mode=mode_arg).astype(AllowedDataType.Float32.value, copy=False))
        check_image._update_dtype()
    except Exception as e:
        raise FilteringError("Failed to filter the image using Prewitt transform") from e
    
    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from image.load.loader import open_image
    import cv2
    from image.transform.color_conversion import convert
    import numpy as np
    from pathlib import Path
    from skimage.filters.edges import prewitt
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    #mask = np.ones((400, 750), dtype=np.uint8)
    image_ = open_image(str(path_image))
    #image_ = convert(image_, "rgb2gray")
    image_ = image_.image

    im1 = prewitt(image_)
    im1 = im1.astype(np.float32)
    max_im1 = np.max(im1)
    im1 = (im1/max_im1) * 255
    im1 = np.clip(im1, 0, 255)

    print("hallo")