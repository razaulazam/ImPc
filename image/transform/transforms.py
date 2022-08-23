# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Image transforms

import cv2


from PIL import Image
from typing import Any, Tuple, Optional, Union, List
from image._decorators import check_image_exist_external
from image.load._interface import BaseImage
from commons.warning import DefaultSetting
from commons.exceptions import WrongArgumentsType, TransformError, WrongArgumentsValue, ImageAlreadyClosed
from image._helpers import image_array_check_conversion, ConversionMode

# -------------------------------------------------------------------------

SAMPLING_REGISTRY = {
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

TRANSPOSE_REGISTRY = {
    "flip_left_right": Image.Transpose.FLIP_LEFT_RIGHT,
    "rotate_90": Image.Transpose.ROTATE_90,
    "flip_top_bottom": Image.Transpose.FLIP_TOP_BOTTOM,
    "rotate_180": Image.Transpose.ROTATE_180,
    "rotate_270": Image.Transpose.ROTATE_270,
    "transpose": Image.Transpose.TRANSPOSE,
    "transverse": Image.Transpose.TRANSVERSE,
}

TRANSFORM_REGISTRY = {
    "extent": Image.Transform.EXTENT,
    "affine": Image.Transform.AFFINE,
    "perspective": Image.Transform.PERSPECTIVE,
    "quad": Image.Transform.QUAD,
    "mesh": Image.Transform.MESH,
}

QUANTIZE_REGISTRY = {
    "mediancut": Image.Quantize.MEDIANCUT,
    "maxcoverage": Image.Quantize.MAXCOVERAGE,
    "fastoctree": Image.Quantize.FASTOCTREE,
}

# -------------------------------------------------------------------------

@check_image_exist_external
def resize(
    image: BaseImage,
    size: Union[Tuple[int, int], List[int]],
    resample: Optional[str] = "bilinear",
) -> BaseImage:

    if not isinstance(image, BaseImage):
        raise WrongArgumentsType("Provided image is not a ImageLoader instance")

    if not isinstance(size, (tuple, list)):
        raise WrongArgumentsType("Please check the type of the size argument")

    if len(size) != 2:
        raise WrongArgumentsValue("Insufficient arguments in the size tuple")

    if not all(i > 0 for i in size):
        raise WrongArgumentsValue("Arguments in the size tuple cannot be negative")

    if not isinstance(resample, str):
        raise WrongArgumentsType("Please check the type of the resample argument")

    sample_arg = SAMPLING_REGISTRY.get(resample.lower(), None)
    if not sample_arg:
        DefaultSetting(
            "Using default sampling strategy (nearest) since the provided filter type is not supported"
        )
        
    check_image = image_array_check_conversion(image, ConversionMode.OpenCV)

    try:
        check_image._set_image(cv2.resize(check_image.image, size[::-1], sample_arg).astype(check_image.dtype.value, copy=False))
    except Exception as e:
        raise TransformError("Failed to resize the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def rotate(
    image: BaseImage,
    angle: float,
    resample: Optional[str] = "nearest",
    expand: Optional[int] = 0,
    center: Optional[Union[Tuple[int, int], List[int]]] = None,
    translate: Optional[Union[Tuple[int, int], List[int]]] = None,
    fill_color: Optional[Any] = None,
) -> BaseImage:

    image = image_array_check_conversion(image, "PIL")

    if resample and not isinstance(resample, str):
        raise WrongArgumentsType("Please check the type of the resample argument")

    if expand and not isinstance(expand, int):
        raise WrongArgumentsType("Please check the type of the expand argument")

    if center:
        if not isinstance(center, (tuple, list)):
            raise WrongArgumentsType("Please check the type of the center argument")
        if len(center) != int(2):
            raise WrongArgumentsValue("Invalid number of arguments for the center")

    if translate:
        if not isinstance(translate, (tuple, list)):
            raise WrongArgumentsType("Please check the type of the translate argument")
        if len(translate) != 2:
            raise WrongArgumentsValue("Invalid number of arguments for the translate")

    sample_arg = SAMPLING_REGISTRY.get(resample.lower(), None)
    if not sample_arg:
        sample_arg = Image.Resampling.NEAREST
        DefaultSetting(
            "Using default sampling strategy (nearest) since the provided filter type is not supported"
        )

    try:
        new_im = image.copy()
        new_im.file_stream = new_im.file_stream.rotate(
            angle, sample_arg, expand, center, translate, fill_color
        )
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise TransformError("Failed to rotate the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def transpose(
    image: BaseImage,
    method: str,
) -> BaseImage:

    image = image_array_check_conversion(image, "PIL")

    if not isinstance(method, str):
        raise WrongArgumentsType("Please check the type of the method argument")

    method_arg = TRANSPOSE_REGISTRY.get(method.lower(), None)
    if not method_arg:
        raise WrongArgumentsValue("Defined method is not supported currently")

    try:
        new_im = image.copy()
        new_im.file_stream = new_im.file_stream.transpose(method_arg)
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise TransformError("Failed to transpose the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def reduce(
    image: BaseImage,
    factor: Union[int, Union[Tuple[int, int], List[int]]],
    box: Optional[Union[Tuple[int, int, int, int], List[int]]] = None,
) -> BaseImage:

    image = image_array_check_conversion(image, "PIL")

    if not (isinstance(factor, int) or isinstance(factor, tuple)):
        raise WrongArgumentsType("Please check the type of the factor argument")

    if isinstance(factor, int) and factor < 0:
        raise WrongArgumentsValue("Value of the factor can not be less than 0")

    if isinstance(factor, (tuple, list)):
        if len(factor) != int(2):
            raise WrongArgumentsValue("Factor tuple can not have more than four values")
        if not all(i > 0 for i in factor):
            raise WrongArgumentsValue("The arguments of the bouding box tuple cannot be negative")

    if box:
        if not isinstance(box, (tuple, list)):
            raise WrongArgumentsType("Please check the type of the box argument")
        if len(box) != int(4):
            raise WrongArgumentsValue("Insufficient arguments in the bounding box tuple")
        if not all(i > 0 for i in box):
            raise WrongArgumentsValue("The arguments of the bounding box tuple cannot be negative")
        if box[2] > image.width:
            raise WrongArgumentsValue("Bounding box width cannot be greater than the image width")
        if box[3] > image.height:
            raise WrongArgumentsValue("Bounding box height cannot be greater than the image height")

    try:
        new_im = image.copy()
        new_im.file_stream = new_im.file_stream.reduce(factor, box)
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise TransformError("Failed to reduce the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def transform(
    image: BaseImage,
    size: Union[Tuple[int, int], List[int]],
    method: str,
    data: Optional[Union[Tuple[int, int, int, int], List[int]]] = None,
    resample: Optional[str] = "nearest",
    fill_color: Optional[Any] = None
) -> BaseImage:

    image = image_array_check_conversion(image, "PIL")

    if not isinstance(size, (tuple, list)):
        raise WrongArgumentsType("Please check the type of the size argument")

    if len(size) != 2:
        raise WrongArgumentsValue("Insufficient arguments in the size tuple")

    if not all(i > 0 for i in size):
        raise WrongArgumentsValue("Arguments in the size tuple cannot be negative")

    if method and not isinstance(method, str):
        raise WrongArgumentsType("Please check the type of the method argument")

    if data:
        if not isinstance(data, (tuple, list)):
            raise WrongArgumentsType("Please check the type of the data argument")
        if len(data) != int(4):
            raise WrongArgumentsValue("Insufficient arguments in the data tuple")
        if not all(i > 0 for i in data):
            raise WrongArgumentsValue("The arguments of the data tuple cannot be negative")
        if data[2] > image.width:
            raise WrongArgumentsValue("Data width cannot be greater than the image width")
        if data[3] > image.height:
            raise WrongArgumentsValue("Data height cannot be greater than the image height")

    method_arg = TRANSFORM_REGISTRY.get(method.lower(), None)
    if not method_arg:
        raise WrongArgumentsValue("Defined method is not supported currently")

    resample_arg = SAMPLING_REGISTRY.get(resample.lower(), None)
    if not resample_arg:
        resample_arg = Image.Resampling.NEAREST
        DefaultSetting(
            "Using default sampling strategy (nearest) since the provided filter type is not supported"
        )

    try:
        new_im = image.copy()
        new_im.file_stream = new_im.file_stream.transform(
            size, method=method_arg, data=data, resample=resample_arg, fillcolor=fill_color
        )
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise TransformError("Failed to transform the image") from e

    return new_im

# -------------------------------------------------------------------------

@check_image_exist_external
def quantize(
    image: BaseImage,
    colors: int = 256,
    method: str = None,
    kmeans: int = 0,
    dither: str = None
) -> BaseImage:

    image = image_array_check_conversion(image, "PIL")

    if not isinstance(colors, int):
        raise WrongArgumentsType("Please check the type of the colors argument")
    if colors > 256:
        raise WrongArgumentsValue("Value of the colors arguments should be <= 256")

    if method and not isinstance(method, str):
        raise WrongArgumentsType("Please check the type of the method argument")

    if not isinstance(kmeans, int):
        raise WrongArgumentsType("Please check the type of the kmeans argument")

    if kmeans < int(0):
        raise WrongArgumentsValue("Value of kmeans cannot be less than 0")

    if dither and not isinstance(dither, str):
        raise WrongArgumentsType("Please check the type of the dither argument")

    if image.mode == "RGBA" or image.mode == "PA":
        method_arg = Image.Quantize.FASTOCTREE # Silently uses this as default for RGBA
    else:
        method_arg = QUANTIZE_REGISTRY.get(method.lower(), None)
        if not method_arg:
            DefaultSetting(
                "Using default method (fastoctree) since the provided method is not supported"
            )
            method_arg = Image.Quantize.MEDIANCUT

    dither = dither.lower()
    if dither == "floydsteinburg":
        dither_arg = Image.Dither.FLOYDSTEINBURG
    else:
        DefaultSetting(
            "Using default dithering strategy (None) since the provided strategy is not supported"
        )
        dither_arg = Image.Dither.NONE

    try:
        new_im = image.copy()
        new_im.file_stream = new_im.file_stream.quantize(
            colors, method=method_arg, kmeans=kmeans, dither=dither_arg
        )
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise TransformError("Failed to transform the image") from e

    return new_im

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    import cv2
    import time
    im = cv2.imread(str(path_image))
    from scipy.ndimage import rotate as rotate1
    from skimage.transform import rotate as rotate2

    start_time = time.time()
    rotate1(im, 45, (1, 0))
    print(time.time() - start_time)

    start_time = time.time()
    rotate2(im, 45)
    print(time.time() - start_time)