# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image thresholding methods

import cv2

from image.load._interface import BaseImage
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FilteringError
from commons.warning import DefaultSetting
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion
from typing import Union

# -------------------------------------------------------------------------

CV_THRESHOLD_STRATEGY = {
    "binary": cv2.THRESH_BINARY,
    "binary_inverse": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozeroinv": cv2.THRESH_TOZERO_INV,
}

# -------------------------------------------------------------------------

@check_image_exist_external
def simple_threshold(
    image: BaseImage, threshold: Union[float, int], max_val: Union[float, int], method: str
) -> BaseImage:
    """Applied fixed level threshold to each image pixel"""

    if not isinstance(threshold, (float, int)):
        raise WrongArgumentsType("Threshold can only be provided as float or integer")

    if not isinstance(max_val, (float, int)):
        raise WrongArgumentsType("Maximum value can only be provided as float or integer")

    if not isinstance(method, str):
        raise WrongArgumentsType("Method can only be provided as a string")

    method_arg = CV_THRESHOLD_STRATEGY.get(method, None)
    if method_arg is None:
        raise WrongArgumentsValue("Provided thresholding method is wrong")

    check_image = image_array_check_conversion(image)

    try:
        check_image._set_image(
            cv2.threshold(check_image.image, float(threshold), float(max_val),
                          method_arg).astype(check_image.dtype.value, copy=False)
        )
    except Exception as e:
        raise FilteringError(
            "Failed to filter the image with the specified thresholding strategy"
        ) from e

    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    import numpy as np
    from image.load.loader import open_image
    image_path = Path(__file__).parent.parent.parent / "sample.jpg"

    image = open_image(str(image_path))
    image = image.image.astype(np.float32)

    im1 = cv2.threshold(image, -1, 1000, cv2.THRESH_BINARY_INV)

    print("hallo")