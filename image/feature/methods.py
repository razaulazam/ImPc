# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image feature detection methods

from signal import default_int_handler
from threading import local
import numpy as np

from commons.exceptions import ImageAlreadyClosed, WrongArgumentsType, WrongArgumentsValue, FeatureError
from commons.warning import ImageModeConversion, DefaultSetting
from image._decorators import check_image_exist_external
from image._common_datastructs import SKIMAGE_SAMPLING_REGISTRY
from image.load._interface import BaseImage
from image._helpers import AllowedDataType
from image.transform.color_conversion import convert
from image._helpers import image_array_check_conversion, check_user_provided_ndarray
from typing import Optional, Tuple, Union, List
from skimage.feature import canny as sk_canny
from skimage.feature import blob_dog as sk_blob_dog
from skimage.feature import blob_doh as sk_blob_doh
from skimage.feature import blob_log as sk_blob_log
from skimage.feature import corner_fast as sk_corner_fast
from skimage.feature import corner_foerstner as sk_corner_foerstner
from skimage.feature import corner_harris as sk_corner_harris
from skimage.feature import corner_kitchen_rosenfeld as sk_corner_kr
from skimage.feature import corner_moravec as sk_corner_moravec
from skimage.feature import corner_shi_tomasi as sk_corner_shi_tomasi
from skimage.feature import daisy as sk_daisy
from skimage.feature import haar_like_feature as sk_haar_like_feature
from skimage.feature import hessian_matrix as sk_hessian_matrix
from skimage.feature import hessian_matrix_eigvals as sk_hessian_matrix_eigvals
from skimage.feature import hog as sk_hog_descriptors
from skimage.feature import local_binary_pattern as sk_local_binary_pattern
from skimage.feature import match_descriptors as sk_match_descriptors

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
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

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

    check_image = image_array_check_conversion(converted_image)

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
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

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

    check_image = image_array_check_conversion(converted_image)

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
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

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

    check_image = image_array_check_conversion(converted_image)

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
        converted_image = convert(image, "rgb2gray")

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
        converted_image = convert(image, "rgb2gray")

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
        DefaultSetting(
            "Choosing constant as sampling strategy for handling values outside the image borders since the provided mode is not currently supported"
        )
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_corner_kr(check_image.image,
                         mode=mode_arg).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FeatureError("Failed to compute kitchen rosenfeld corners") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_moravec_corners(
    image: BaseImage, kernel_size: Optional[Union[int, float]] = 1
) -> BaseImage:
    """Compute moravec corners in the provided image. Result is returned as float32 image"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(kernel_size, (float, int)):
        raise WrongArgumentsType("Kernel size must be provided as either integer or float")

    if kernel_size <= 0:
        raise WrongArgumentsValue("Kernel size must be > 0")

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_corner_moravec(check_image.image, window_size=int(kernel_size)
                              ).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FeatureError("Failed to compute moravec corners in the provided image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_shi_tomasi_corners(image: BaseImage, sigma: Optional[float] = 1.0) -> BaseImage:
    """Compute shi tomasi corners in the provided image. Result is returned as a float32 image"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma should be provided as a float value")

    check_image = image_array_check_conversion(converted_image)

    try:
        check_image._set_image(
            sk_corner_shi_tomasi(check_image.image,
                                 float(sigma)).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FeatureError("Failed to compute shi tomasi corners in the provided image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_daisy_features(
    image: BaseImage,
    sample_step: Optional[int] = 4,
    radius_outer: Optional[int] = 15,
    num_rings: Optional[int] = 3,
    num_histograms: Optional[int] = 8,
    normalization: Optional[str] = "l1",
    visualize: Optional[bool] = False
) -> Union[np.ndarray, tuple[np.ndarray, BaseImage]]:
    """Computes daisy features of the provided image. If visualize = True, we return an array of 
    descriptors and a image instance for visualizing those features otherwise we just 
    return array of descriptors."""

    norm_methods = {"l1": "l1", "l2": "l2", "daisy": "daisy", "off": "off"}

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(sample_step, int):
        raise WrongArgumentsType("Sampling step argument can only be supplied as integer")

    if not isinstance(radius_outer, int):
        raise WrongArgumentsType(
            "Radius of the outermost ring (radius_outer) can only be supplied as integer"
        )

    if radius_outer <= 0:
        raise WrongArgumentsValue("Outer radius can not be less than or equal to zero")

    if not isinstance(num_rings, int):
        raise WrongArgumentsType("Number of rings can only be supplied as integer")

    if num_rings <= 0:
        raise WrongArgumentsValue("Number of rings can not be less than or equal to zero")

    if not isinstance(num_histograms, int):
        raise WrongArgumentsType("Number of histograms can only be supplied as integer")

    if not isinstance(normalization, str):
        raise WrongArgumentsType("Normalization method must be provided as string")

    if not isinstance(visualize, bool):
        raise WrongArgumentsType("Visualize argument must be provided as boolean")

    check_image = image_array_check_conversion(converted_image)

    norm_method_arg = norm_methods.get("normalization", None)
    if norm_method_arg is None:
        DefaultSetting(
            "Using the default normalization method (L1) since the provided normalization method is currently not supported by the library"
        )
        norm_method_arg = norm_methods["l1"]

    try:
        descriptors = sk_daisy(
            check_image.image,
            step=sample_step,
            radius=radius_outer,
            rings=num_rings,
            histograms=num_histograms,
            normalization=norm_method_arg,
            visualize=visualize
        )
        if visualize:
            check_image._set_image(descriptors[1].astype(AllowedDataType.Float32.value, copy=False))
    except Exception as e:
        raise FeatureError("Failed to compute daisy features of the provided image") from e

    if visualize:
        return (descriptors[0], check_image)
    else:
        return descriptors

# -------------------------------------------------------------------------

def compute_haar_like_features(
    image: BaseImage, row: int, col: int, width: int, height: int
) -> np.ndarray:
    """Computes haar like features. Result is returned as a numpy array of float32 feature values"""

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this operation can only be applied to 2D images"
        )
        converted_image = convert(image, "rgb2gray")

    if not isinstance(row, int):
        raise WrongArgumentsType("Row can only be supplied as an integer")

    if not isinstance(col, int):
        raise WrongArgumentsType("Col can only be supplied as an integer")

    if not isinstance(width, int):
        raise WrongArgumentsType("Width can only be supplied as an integer")

    if not isinstance(height, int):
        raise WrongArgumentsType("Height can only be supplied as an integer")

    if row < 0:
        raise WrongArgumentsValue(
            "Row should be supplied as an integer greater than or equal to zero"
        )

    if col < 0:
        raise WrongArgumentsValue(
            "Col should be supplied as an integer greater than or equal to zero"
        )

    if width <= 0:
        raise WrongArgumentsValue("Width should be supplied as an integer greater than zero")

    if height <= 0:
        raise WrongArgumentsValue("Height should be supplied as an integer greater than zero")

    check_image = check_image_exist_external(converted_image)
    try:
        features = sk_haar_like_feature(
            check_image.image, r=row, c=col, width=width, height=height
        ).astype(
            AllowedDataType.Float32.value, copy=False
        )
    except Exception as e:
        raise FeatureError("Failed to compute haar like features for the provided image") from e

    return features

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_hessian_matrix(
    image: BaseImage,
    sigma: Optional[float] = 1.0,
    mode: Optional[str] = "constant"
) -> List[np.ndarray]:
    """Computes the hessian matrix of the provided image. Result is returned as a list of float32 gradient arrays."""

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma must be provided as a float value")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode must be provided as string")

    mode_arg = SKIMAGE_SAMPLING_REGISTRY.get(mode, None)
    if mode_arg is None:
        DefaultSetting(
            "Using constant as the mode for filling in the boundary pixels since the provided mode is not supported by the library yet"
        )
        mode_arg = SKIMAGE_SAMPLING_REGISTRY["constant"]

    check_image = image_array_check_conversion(image)
    try:
        gradients = sk_hessian_matrix(check_image.image, sigma, mode_arg)
        gradients = [
            gradient.astype(AllowedDataType.Float32.value, copy=False) for gradient in gradients
        ]
    except Exception as e:
        raise FeatureError("Failed to compute the hessian matrix") from e

    return gradients

# -------------------------------------------------------------------------

def compute_hessian_matrix_eigvals(hessian_matrix: List[np.ndarray]) -> np.ndarray:
    """Computes the eigen values of the hessian matrix. Result is returned as float """

    for i in range(len(hessian_matrix)):
        hessian_matrix[i] = check_user_provided_ndarray(hessian_matrix[i])

    try:
        eigen_values = sk_hessian_matrix_eigvals(hessian_matrix).astype(
            AllowedDataType.Float32.value, copy=False
        )
    except Exception as e:
        raise FeatureError("Failed to compute the eigen values of the provided hessian matrix")

    return eigen_values

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_hog_descriptors(
    image: BaseImage,
    orientations: Optional[int] = 9,
    pixels_in_cell: Optional[Tuple[int, int]] = (8, 8),
    cells_per_block: Optional[Tuple[int, int]] = (3, 3)
) -> Tuple[np.ndarray, BaseImage]:
    """Computes the hog descriptor features of the provided image. Result is returned as a tuple of float32 array and image
    where the first element represents the flattened hog features and the second element is a library image
    for visualization."""

    if not isinstance(orientations, int):
        raise WrongArgumentsType("Orientations must be provided as an integer value")

    for pixel in pixels_in_cell:
        if not isinstance(pixel, int):
            raise WrongArgumentsType(
                "Pixels in cell must be provided as a tuple with only integer values"
            )

    if len(pixels_in_cell) != 2:
        raise WrongArgumentsValue(
            "Pixels in cell must be provided as a tuple with values in just row and col direction"
        )

    for cell in cells_per_block:
        if not isinstance(cell, int):
            raise WrongArgumentsType(
                "Cells per block must be provided as a tuple with only integer values"
            )

    if len(cells_per_block) != 2:
        raise WrongArgumentsValue(
            "Cells per block must be provided as a tuple with values in just row and col direction"
        )

    check_image = image_array_check_conversion(image)
    channel_axis = -1 if check_image.is_rgb() else None
    try:
        descriptors, image_hog = sk_hog_descriptors(
            check_image.image,
            pixels_per_cell=pixels_in_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            channel_axis=channel_axis
        )
        check_image._set_image(image_hog.astype(AllowedDataType.Float32.value, copy=False))
        descriptors.astype(AllowedDataType.Float32.value, copy=False)
    except Exception as e:
        raise FeatureError("Failed to compute the hog descriptors of the image") from e

    return (descriptors, check_image)

# -------------------------------------------------------------------------

@check_image_exist_external
def compute_local_binary_pattern(
    image: BaseImage,
    neigbour_points: int,
    radius: int,
    method: Optional[str] = "default"
) -> BaseImage:
    """Computes the local binary pattern of a gray-scale image. Result is returned as a float32 image."""

    possible_methods = {"default": "default", "ror": "ror", "uniform": "uniform", "var": "var"}

    if not image.is_gray():
        ImageModeConversion(
            "Converting the image to grayscale since this method is only supported for grayscale images"
        )
        image = convert(image, "rgb2gray")

    if not isinstance(neigbour_points, int):
        raise WrongArgumentsType("Neighbour points must be provided as an integer")

    if not isinstance(radius, int):
        raise WrongArgumentsType("Radius must be provided as an integer")

    if neigbour_points < 0:
        raise WrongArgumentsValue("Neigbour points can not be a negative number")

    if radius < 0:
        raise WrongArgumentsValue("Radius can not be a negative number")

    method_arg = possible_methods.get(method, None)
    if method_arg is None:
        DefaultSetting(
            "Using default method since the provided method is not supported by the library yet"
        )
        method_arg = possible_methods["default"]

    check_image = image_array_check_conversion(image)
    try:
        check_image._set_image(
            sk_local_binary_pattern(check_image.image, neigbour_points, radius,
                                    method_arg).astype(AllowedDataType.Float32.value, copy=False)
        )
    except Exception as e:
        raise FeatureError("Failed to compute the local binary pattern") from e

    return check_image

# -------------------------------------------------------------------------

def match_image_descriptors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Matches the provided first and second descriptors array with a brute-force approach.
    It returns a (Q, 2) dimensional array, where the first column denotes the matched indices in the first and 
    the second column denotes the matched indices in the second array of descriptors. Result is returned as a float32 array."""

    dim_first, dim_second = first.shape, second.shape

    if len(dim_first) != 2 or len(dim_second) != 2:
        raise WrongArgumentsValue(
            "Either of the provided descriptors array does not have the right dimensions"
        )

    if dim_first[1] != dim_second[1]:
        raise WrongArgumentsValue("Dimensions of the provided descriptors array does not match")

    check_first = check_user_provided_ndarray(first)
    check_second = check_user_provided_ndarray(second)

    try:
        matches = sk_match_descriptors(check_first, check_second).astype(
            AllowedDataType.Float32.value, copy=False
        )
    except Exception as e:
        raise FeatureError("Failed to match the provided descriptors") from e

    return matches

# -------------------------------------------------------------------------
# Check this function and the return types of this function
@check_image_exist_external
def match_image_template(image: BaseImage, template: Union[BaseImage, np.ndarray]) -> np.ndarray:
    check_template = None
    if isinstance(template, BaseImage):
        if template.closed:
            raise ImageAlreadyClosed("The provided template is already closed")
        else:
            check_template = image_array_check_conversion(template)
    elif isinstance(template, np.ndarray):
        check_tempalate = check_user_provided_ndarray(template)
    else:
        raise WrongArgumentsType(
            "Provide template has a type which is not supported by this method"
        )
    if isinstance(template, BaseImage):
        if template.width <= 0 or template.height or template.channels != image.channels:
            raise WrongArgumentsValue(
                "Image cannot have a width or height of zero or less than zero"
            )
        else:
            width, height, channels = template.shape[:-1]
            if width <= 0 or height <= 0 or channels != image.channels:
                raise WrongArgumentsValue("Image cannot have a width or height of less than zero")

    check_image = image_array_check_conversion(image)
    try:
        check_image._set_image(
            check_image.image,
            check_template if isinstance(check_template, np.ndarray) else check_image.image
        )
    except Exception as e:
        raise FeatureError("Failed to peform the template matching") from e
    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from skimage import data
    import numpy as np
    import napari
    from image.load.loader import open_image
    from image.transform.color_conversion import convert
    from skimage.feature import daisy, local_binary_pattern, match_template
    import cv2
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(path_image))
    #viewer = napari.view_image(image.image)
    image = convert(image, "rgb2gray")
    #image_input = image.image.astype(np.uint16)

    #out = cv2.Canny(image_input, 100, 200)
    out1 = local_binary_pattern(image.image, 1, 1)
    out2 = out1.astype(np.uint8)

    print("hallo")