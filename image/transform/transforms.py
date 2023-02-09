# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Image transforms

import cv2
import numpy as np

from skimage.transform import rotate as sk_rotate
from typing import Tuple, Optional, Union, List
from image._decorators import check_image_exist_external
from image.load._interface import BaseImage
from commons.warning import DefaultSetting
from commons.exceptions import WrongArgumentsType, TransformError, WrongArgumentsValue
from image._helpers import image_array_check_conversion
from image._common_datastructs import AllowedDataType, SKIMAGE_SAMPLING_REGISTRY
from image.transform.color_conversion import convert

# -------------------------------------------------------------------------

CV2_SAMPLING_REGISTRY = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "bilinear_exact": cv2.INTER_LINEAR_EXACT,
    "max": cv2.INTER_MAX,
    "fill_outliers": cv2.WARP_FILL_OUTLIERS,
    "inverse_map": cv2.WARP_INVERSE_MAP
}

# -------------------------------------------------------------------------

@check_image_exist_external
def resize(
    image: BaseImage,
    size: Union[Tuple[int, int], List[int]],
    resample: Optional[str] = "bilinear",
) -> BaseImage:
    """Resizes the image"""

    from cv2 import resize as cv_resize

    if not isinstance(size, (tuple, list)):
        raise WrongArgumentsType("Please check the type of the size argument")

    if len(size) != 2:
        raise WrongArgumentsValue("Insufficient arguments in the size tuple")

    if not all(i > 0 for i in size):
        raise WrongArgumentsValue("Arguments in the size tuple cannot be negative")

    if not isinstance(resample, str):
        raise WrongArgumentsType("Please check the type of the resample argument")

    sample_arg = CV2_SAMPLING_REGISTRY.get(resample.lower(), None)
    if sample_arg is None:
        DefaultSetting(
            "Using default sampling strategy (nearest) since the provided filter type is not supported"
        )

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            cv_resize(check_image.image, size[::-1],
                      sample_arg).astype(check_image.dtype.value, copy=False)
        )
    except Exception as e:
        raise TransformError("Failed to resize the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def rotate(
    image: BaseImage,
    angle: Union[int, float],
    resize: Optional[bool] = False,
    center: Optional[Union[Tuple[int, int], List[int]]] = None,
    resample: Optional[str] = "constant",
) -> BaseImage:
    """Rotates the image. For the constant mode, the image is padded with zeros"""

    if not isinstance(angle, (float, int)):
        raise WrongArgumentsType(
            "Please check the type of the angle argument. It should be either float or int"
        )

    if not isinstance(resample, str):
        raise WrongArgumentsType("Please check the type of the resample argument")

    if not isinstance(resize, bool):
        raise WrongArgumentsType("Please check the type of the resize argument")

    if center: # provided like (height, width)
        if not isinstance(center, (tuple, list)):
            raise WrongArgumentsType("Please check the type of the center argument")
        if len(center) != int(2):
            raise WrongArgumentsValue("Invalid number of arguments for the center")

    sample_arg = SKIMAGE_SAMPLING_REGISTRY.get(resample.lower(), None)
    if sample_arg is None:
        sample_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]
        DefaultSetting(
            "Using default sampling strategy (constant) since the provided filter type is not supported"
        )

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            sk_rotate(
                check_image.image,
                float(angle),
                resize,
                center[::-1],
                mode=resample,
                preserve_range=True
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise TransformError("Failed to rotate the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def transpose(image: BaseImage,) -> BaseImage:

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            cv2.transpose(check_image.image).astype(check_image.dtype.value, copy=False)
        )
    except Exception as e:
        raise TransformError("Failed to transpose the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def kmeans_quantize(
    image: BaseImage,
    clusters: Optional[Union[int, float]] = 8,
) -> BaseImage:
    """Reduces the color space of the image. This method only applies to images with either of these modes: RGB or LAB"""

    from sklearn.cluster import MiniBatchKMeans

    allowed_modes = ["RGB", "LAB"]
    if image.mode not in allowed_modes:
        raise WrongArgumentsValue(
            "This method only works with images with either of these modes: RGB or LAB"
        )

    if not isinstance(clusters, (int, float)):
        raise WrongArgumentsType("Please check the type of the colors argument")

    if clusters <= 0:
        raise WrongArgumentsValue("Value of the clusters arguments should be > 0")

    check_image = image_array_check_conversion(image)
    quantize_image = None

    if check_image.mode == "LAB":
        quantize_image = check_image
    elif check_image.mode == "RGB":
        quantize_image = convert(check_image, "rgb2lab")

    try:
        quantize_image_flatten = quantize_image.image.reshape(
            (np.prod(quantize_image.dims), quantize_image.channels)
        )
        kmeans_manager = MiniBatchKMeans(n_clusters=int(clusters))
        compute_clusters = kmeans_manager.fit_predict(quantize_image_flatten)
        final_result = kmeans_manager.cluster_centers_.astype(
            quantize_image.dtype.value, copy=False
        )[compute_clusters]
        final_result = final_result.reshape(
            *quantize_image.dims,
            quantize_image.channels,
        )
        quantize_image._set_image(final_result)
        if quantize_image.mode == "LAB":
            quantize_image = convert(quantize_image, "lab2rgb")
    except Exception as e:
        raise TransformError("Failed to transform the image") from e

    return quantize_image

# -------------------------------------------------------------------------