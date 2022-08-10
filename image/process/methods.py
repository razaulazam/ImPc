# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image processing methods

import cv2

from PIL import Image
from image.load._interface import BaseImage
from image.load.loader import ImageLoader
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue
from image._helpers import image_array_check_conversion
from commons.exceptions import ProcessingError, ImageAlreadyClosed

# -------------------------------------------------------------------------

def blend(image_one: BaseImage, image_two: BaseImage, alpha: float) -> BaseImage:
    
    image_array_check_conversion(image_one, "openCV")
    image_array_check_conversion(image_two, "openCV")
    
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

    try:
        new_im = image_one.copy()
        new_im._set_image(cv2.addWeighted(image_one.image, alpha, image_two.image, float(1.0 - alpha)))
        new_im._update_dtype()
    except Exception as e:
        raise ProcessingError("Failed to alpha blend the image") from e

    return new_im

# -------------------------------------------------------------------------

def composite(image_one: BaseImage, image_two: BaseImage, mask: BaseImage) -> BaseImage:

    image_one = image_array_check_conversion(image_one, "PIL")
    image_two = image_array_check_conversion(image_two, "PIL")
    mask = image_array_check_conversion(mask, "PIL")

    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_one argument")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_two argument")

    if not isinstance(mask, BaseImage):
        raise WrongArgumentsType("Please check the type of the mask argument")

    if image_one.closed:
        raise ImageAlreadyClosed("Provided first image is already closed")

    if image_two.closed:
        raise ImageAlreadyClosed("Provided second image is already closed")

    if mask.closed:
        raise ImageAlreadyClosed("Provided mask image is already closed")

    assert image_one.dims == image_two.dims == mask.dims, RuntimeError(
        "Dimensions of the images do not match"
    )

    try:
        new_im = ImageLoader.create_loader()
        new_im.file_stream = Image.composite(
            image_one.file_stream, image_two.file_stream, mask.file_stream
        )
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise ProcessingError("Failed to perform the composite operation") from e

    return new_im

# -------------------------------------------------------------------------
