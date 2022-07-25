import cv2
import numpy as np

from typing import Union, Optional
from collections import namedtuple
from commons.exceptions import FilteringError, WrongArgumentsType
from image.load._interface import PyFaroImage
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion, check_user_provided_ndarray

BORDER_INTERPOLATION = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "transparent": cv2.BORDER_TRANSPARENT,
    "default": cv2.BORDER_DEFAULT
}

@check_image_exist_external
def corr2d(
    image: PyFaroImage,
    kernel: Union[namedtuple, np.ndarray],
    delta: Optional[Union[float, int]] = 0,
    border: Optional[str] = "default",
) -> PyFaroImage:

    image_array_check_conversion(image, "openCV")

    if not isinstance(kernel, (np.ndarray, namedtuple)):
        raise WrongArgumentsType(
            "Provided kernel has a type that is not supported by the library. Try using get_kernel() for getting the kernel instead"
        )

    if isinstance(kernel, namedtuple):
        kernel = kernel.array_
    else:
        kernel = check_user_provided_ndarray(kernel, "openCV")

    if not isinstance(delta, (float, int)):
        raise WrongArgumentsType("Provided value for delta is not either an integer or a float")

    if not isinstance(border, str):
        raise WrongArgumentsType("Provided border type is not a string")

    border_actual = BORDER_INTERPOLATION.get(border, None)
    if not border_actual:
        border_actual = cv2.BORDER_DEFAULT

    try:
        new_im = image.copy()
        new_im.image = cv2.filter2D(new_im.image, -1, kernel, delta=delta, borderType=border_actual)
        new_im.update_file_stream()
        new_im.set_loader_properties()
    except Exception as e:
        raise FilteringError("Failed to filter the image") from e

    return new_im

if __name__ == "__main__":
    path_image = "C:\\dev\\pyfaro\\sample.jpg"
    a = cv2.imread(path_image)
    print("hallo")