# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
# \brief Image processing methods

from PIL import Image
from image.load._interface import BaseImage
from image.load.loader import ImageLoader
from commons.exceptions import WrongArgumentsType
from image._helpers import image_array_check_conversion
from commons.exceptions import ProcessingError, ImageAlreadyClosed

# -------------------------------------------------------------------------

def alpha_composite(image_one: BaseImage, image_two: BaseImage) -> BaseImage:

    image_one = image_array_check_conversion(image_one, "PIL")
    image_two = image_array_check_conversion(image_two, "PIL")

    if not isinstance(image_one, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_one argument")

    if not isinstance(image_two, BaseImage):
        raise WrongArgumentsType("Please check the type of the image_two argument")

    if image_one.closed:
        raise ImageAlreadyClosed("Provided first image is already closed")

    if image_two.closed:
        raise ImageAlreadyClosed("Provided second image is already closed")

    assert image_one.dims == image_two.dims, RuntimeError(
        "Dimensions of the images are not the same"
    )

    try:
        new_im = ImageLoader.create_loader()
        new_im.file_stream = Image.alpha_composite(image_one.file_stream, image_two.file_stream)
        new_im.update_image()
        new_im.set_loader_properties()
    except Exception as e:
        raise ProcessingError("Failed to alpha compose the image") from e

    return new_im

# -------------------------------------------------------------------------

def blend(image_one: BaseImage, image_two: BaseImage, alpha: float) -> BaseImage:

    image_one = image_array_check_conversion(image_one, "PIL")
    image_two = image_array_check_conversion(image_two, "PIL")

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

    assert image_one.dims == image_two.dims, RuntimeError(
        "Dimensions of the images are not the same"
    )

    try:
        new_im = ImageLoader.create_loader()
        new_im.file_stream = Image.blend(image_one.file_stream, image_two.file_stream, alpha)
        new_im.update_image()
        new_im.set_loader_properties()
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
