# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image feature detection methods

import numpy as np

from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FeatureError
from commons.warning import ImageModeConversion, DefaultSetting
from image._decorators import check_image_exist_external
from image._common_datastructs import SKIMAGE_SAMPLING_REGISTRY
from image.load._interface import BaseImage
from image._helpers import AllowedDataType
from image.transform.color_conversion import convert
from image._helpers import image_array_check_conversion
from typing import List, Optional, Tuple, Union
from skimage.feature import canny as sk_canny
from skimage.feature import blob_dog as sk_blob_dog
from skimage.feature import blob_doh as sk_blob_doh
from skimage.feature import blob_log as sk_blob_log
from skimage.feature import corner_fast as sk_corner_fast
from skimage.feature import corner_foerstner as sk_corner_foerstner
from skimage.feature import corner_harris as sk_corner_harris
from skimage.feature import corner_kitchen_rosenfeld as sk_corner_kr
from skimage.feature import corner_moravec as sk_corner_moravec

# -------------------------------------------------------------------------

@check_image_exist_external
def canny(
    image: BaseImage,
    sigma: Optional[float] = 1.0,
    thresh_low: Optional[Union[float, int]] = None,
    thresh_high: Optional[Union[float, int]] = None
) -> BaseImage:
    """Edge detection using Canny algorithm. Result is returned as float32 image."""

    if not image.is_gray():
        ImageModeConversion(
            "Canny algorithm only works with grayscale images. Performing the conversion automatically ..."
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma must have the float type")

    if not isinstance(thresh_low, (float, int)):
        raise WrongArgumentsType("Low threshold value can either be float or integer")

    if not isinstance(thresh_high, (float, int)):
        raise WrongArgumentsType("High threshold value can either be integer or float")

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_canny(
                check_image.image,
                sigma=sigma,
                low_threshold=thresh_low,
                high_threshold=thresh_high
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute the edges with the Canny algorithm") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_diff_gaussian(
    image: BaseImage,
    sigma_min: Optional[float] = 1.0,
    sigma_max: Optional[float] = 50.0,
    sigma_ratio: Optional[float] = 1.6,
    threshold: Optional[float] = 0.5,
    overlap: Optional[float] = 0.5
) -> np.ndarray:
    """Compute blobs in a grayscale image using difference of gaussian method"""

    if not image.is_gray():
        raise WrongArgumentsType("Input image should be grayscale for this method to work")

    if not isinstance(sigma_min, float):
        raise WrongArgumentsType("Minimum sigma should be provided as float")

    if not isinstance(sigma_max, float):
        raise WrongArgumentsType("Maximum sigma should be provided as float")

    if not isinstance(sigma_ratio, float):
        raise WrongArgumentsType("Sigma ratio should be provided as float")

    if not isinstance(threshold, float):
        raise WrongArgumentsType("Threshold must be provided as float")

    if not isinstance(overlap, float):
        raise WrongArgumentsType("Overlap must be provided as float")

    if overlap <= 0 or overlap >= 1:
        raise WrongArgumentsValue("Overlap value should be between 0 and 1")

    check_image = image_array_check_conversion(image)

    try:
        found_blobs = sk_blob_dog(
            check_image.image, sigma_min, sigma_max, sigma_ratio, threshold, overlap
        )
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_determinant_hessian(
    image: BaseImage,
    sigma_min: Optional[float] = 1.0,
    sigma_max: Optional[float] = 50.0,
    sigma_ratio: Optional[float] = 1.6,
    sigma_num: Optional[Union[int, float]] = 10,
    overlap: Optional[float] = 0.5
) -> np.ndarray:
    """Compute blobs in a grayscale image using determinant of hessian method"""

    if not image.is_gray():
        raise WrongArgumentsType("Input image should be grayscale for this method to work")

    if not isinstance(sigma_min, float):
        raise WrongArgumentsType("Minimum sigma should be provided as float")

    if not isinstance(sigma_max, float):
        raise WrongArgumentsType("Maximum sigma should be provided as float")

    if not isinstance(sigma_ratio, float):
        raise WrongArgumentsType("Sigma ratio should be provided as float")

    if not isinstance(sigma_num, (int, float)):
        raise WrongArgumentsType("Number of sigmas must be provided as float or int")

    if not isinstance(overlap, float):
        raise WrongArgumentsType("Overlap must be provided as float")

    if overlap <= 0 or overlap >= 1:
        raise WrongArgumentsValue("Overlap value should be between 0 and 1")

    check_image = image_array_check_conversion(image)

    try:
        found_blobs = sk_blob_doh(
            check_image.image, sigma_min, sigma_max, sigma_ratio, int(sigma_num), overlap
        )
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_laplacian_gaussian(
    image: BaseImage,
    sigma_min: Optional[float] = 1.0,
    sigma_max: Optional[float] = 50.0,
    sigma_num: Optional[Union[int, float]] = 10,
    threshold: Optional[float] = 0.2,
    overlap: Optional[float] = 0.5
) -> np.ndarray:
    """Compute blobs in a grayscale image using laplacian of gaussian method"""

    if not image.is_gray():
        raise WrongArgumentsType("Input image should be grayscale for this method to work")

    if not isinstance(sigma_min, float):
        raise WrongArgumentsType("Minimum sigma should be provided as float")

    if not isinstance(sigma_max, float):
        raise WrongArgumentsType("Maximum sigma should be provided as float")

    if not isinstance(threshold, float):
        raise WrongArgumentsType("Threshold should be provided as float")

    if not isinstance(sigma_num, (int, float)):
        raise WrongArgumentsType("Number of sigmas must be provided as float or int")

    if not isinstance(overlap, float):
        raise WrongArgumentsType("Overlap must be provided as float")

    if overlap <= 0 or overlap >= 1:
        raise WrongArgumentsValue("Overlap value should be between 0 and 1")

    check_image = image_array_check_conversion(image)

    try:
        found_blobs = sk_blob_log(
            check_image.image, sigma_min, sigma_max, int(sigma_num), threshold, overlap
        )
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_fast_corners(
    image: BaseImage,
    num_pixels: Optional[Union[int, float]] = 12,
    threshold: Optional[float] = 0.15
) -> BaseImage:
    """Compute fast corners for a given image. Result is returned as float32 image"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale image since this method works only with 2D arrays"
        )
        converted_image = convert(image)

    if not isinstance(num_pixels, (float, int)):
        raise WrongArgumentsType("Num pixels can only be provided as either integer or float")

    if not isinstance(threshold, float):
        raise WrongArgumentsType("Threshold can only be provided as float value")

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_corner_fast(check_image.image, int(num_pixels),
                           threshold).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute the FAST corners of the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_foerstner_corners(image: BaseImage, sigma: Optional[float] = 1.0) -> Tuple[BaseImage]:
    """Compute foerstner corners of an image. Result is returned as a tuple of float32 images"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method only supports 2D images"
        )
        converted_image = convert(image)

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma must be provided as a float value")

    check_image = image_array_check_conversion(converted_image)
    check_image_one = check_image.copy()

    try:
        error_ellipse, roundness_ellipse = sk_corner_foerstner(check_image.image, sigma)
        check_image._set_image(error_ellipse.astype(AllowedDataType.Float32.value, copy=False))
        check_image_one._set_image(
            roundness_ellipse.astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
        check_image_one._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute the foerstner corner of the image") from e

    return (check_image, check_image_one)

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_harris_corners(
    image: BaseImage,
    method: Optional[str] = "k",
    sens_factor: Optional[float] = 0.05,
    sigma: Optional[float] = 1.0
) -> BaseImage:
    """Compute harris corner measure response image. Result is returned as a float32 image"""

    methods = {"k": "k", "eps": "eps"}

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied on 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(method, str):
        raise WrongArgumentsType("Method should be supplied as a string")

    if not isinstance(sens_factor, float):
        raise WrongArgumentsType("Sensitivity factor can only be supplied as float")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma should be supplied as float")

    method = method.lower()
    method_arg = methods.get(method, None)
    if method_arg is None:
        DefaultSetting(
            "Provided method is not supported by the library. Using the default method -> k"
        )
        method_arg = methods["k"]

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_corner_harris(check_image.image, method=method_arg, k=sens_factor,
                             sigma=sigma).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute harris corners of the provided image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_kitchen_rosenfeld_corners(
    image: BaseImage, mode: Optional[str] = "constant"
) -> BaseImage:
    """Computes kitchen and rosenfeld corners in a grayscale image. Result is returned as a float32 image"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode can only be supplied as a string")
    
    mode = mode.lower()
    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode, None)
    if mode_arg is None:
        DefaultSetting("Choosing constant as sampling strategy for handling values outside the image borders since the provided mode is not currently supported")
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]
    
    check_image = image_array_check_conversion(converted_image)
    
    try:
        check_image._set_image(sk_corner_kr(check_image.image, mode=mode_arg).astype(AllowedDataType.Float32.value, copy=False))
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute kitchen rosenfeld corners") from e
    
    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_moravec_corners(image: BaseImage, kernel_size: Optional[Union[int, float]] = 1) -> BaseImage:
    """Compute moravec corners in the provided image. Result is returned as float32 image"""
    
    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")
    
    if not isinstance(kernel_size, (float, int)):
        raise WrongArgumentsType("Kernel size must be provided as either integer or float")
    
    check_image = image_array_check_conversion(converted_image)
    
    try:
        

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from image.load.loader import open_image
    from image.transform.color_conversion import convert
    from skimage.feature import corner_harris
    import cv2
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(path_image))
    image = convert(image, "rgb2gray")
    image_input = image.image.astype(np.uint16)

    #out = cv2.Canny(image_input, 100, 200)
    out1 = corner_harris(image.image, k=0.3)
    out2 = out1.astype(np.uint8)

    print("hallo")