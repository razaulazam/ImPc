# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image thresholding methods

import cv2

from image.load._interface import BaseImage
from image._decorators import check_image_exist_external
from image._helpers import image_array_check_conversion

def simple_threshold(image: BaseImage, threshold: float, max_val: float, method: str) -> BaseImage:
    """Works on 8-bit and 32-bit floating point image"""

    image_array_check_conversion(image, "openCV")
