# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image feature detection methods

import numpy as np

from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FeatureError
from commons.warning import ImageModeConversion
from image._decorators import check_image_exist_external
from image.load._interface import BaseImage
from image.transform.color_conversion import convert
from image._helpers import image_array_check_conversion
from typing import Optional, Union
from skimage.feature import canny as sk_canny
from skimage.feature import blob_dog as sk_blob_dog
from skimage.feature import blob_doh as sk_blob_doh
from skimage.feature import blob_log as sk_blob_log

# -------------------------------------------------------------------------

@check_image_exist_external
def canny(image: BaseImage, sigma: Optional[float] = 1.0, thresh_low: Optional[Union[float, int]] = None, thresh_high: Optional[Union[float, int]] = None) -> BaseImage:
    """Edge detection using Canny algorithm"""

    if not image.is_gray():
        ImageModeConversion("Canny algorithm only works with grayscale images. Performing the conversion automatically ...")
        converted_image = convert(image, "rgb2gray")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma must have the float type")

    if not isinstance(thresh_low, (float, int)):
        raise WrongArgumentsType("Low threshold value can either be float or integer")

    if not isinstance(thresh_high, (float, int)):
        raise WrongArgumentsType("High threshold value can either be integer or float")

    check_image = image_array_check_conversion(converted_image)
    
    try:
        check_image._set_image(sk_canny(check_image.image, sigma=sigma, low_threshold=thresh_low, high_threshold=thresh_high))
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute the edges with the Canny algorithm") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_diff_gaussian(image: BaseImage, sigma_min: Optional[float] = 1.0, sigma_max: Optional[float] = 50.0, sigma_ratio: Optional[float] = 1.6, threshold: Optional[float] = 0.5, overlap: Optional[float] = 0.5) -> np.ndarray:
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
        found_blobs = sk_blob_dog(check_image.image, sigma_min, sigma_max, sigma_ratio, threshold, overlap)
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_determinant_hessian(image: BaseImage, sigma_min: Optional[float] = 1.0, sigma_max: Optional[float] = 50.0, sigma_ratio: Optional[float] = 1.6, sigma_num: Optional[Union[int, float]] = 10, overlap: Optional[float] = 0.5) -> np.ndarray:
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
        found_blobs = sk_blob_doh(check_image.image, sigma_min, sigma_max, sigma_ratio, int(sigma_num), overlap)
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------

@check_image_exist_external
def blob_laplacian_gaussian(image: BaseImage, sigma_min: Optional[float] = 1.0, sigma_max: Optional[float] = 50.0, sigma_num: Optional[Union[int, float]] = 10, threshold: Optional[float] = 0.2, overlap: Optional[float] = 0.5) -> np.ndarray:
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
        found_blobs = sk_blob_log(check_image.image, sigma_min, sigma_max, int(sigma_num), threshold, overlap)
    except Exception as e:
        raise FeatureError("Failed to find the blobs in the image") from e

    return found_blobs

# -------------------------------------------------------------------------


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from image.load.loader import open_image
    from image.transform.color_conversion import convert
    from skimage.feature import blob_dog
    import cv2
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(path_image))
    image_input = image.image.astype(np.uint16)


    #out = cv2.Canny(image_input, 100, 200)
    out1 = blob_dog(image.image, float(2.0))

    print("hallo")
