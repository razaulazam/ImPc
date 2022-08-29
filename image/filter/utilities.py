# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Utilites required for filtering methods

import cv2

from typing import Union, List, Tuple
from commons.exceptions import WrongArgumentsType, WrongArgumentsValue
from collections import namedtuple

# -------------------------------------------------------------------------

ALLOWED_KERNELS = {
    "rectangle": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS
}

# -------------------------------------------------------------------------

def get_kernel(kernel_shape: str, kernel_size: Union[List[int], Tuple[int, int]]) -> namedtuple:
    """Utility for getting the kernel of a particular shape"""

    if not isinstance(kernel_shape, str):
        raise WrongArgumentsType(
            "Please check the type of the first argument. Only strings are allowed"
        )

    shape_strategy = ALLOWED_KERNELS.get(kernel_shape, None)
    if shape_strategy is None:
        raise WrongArgumentsValue(
            "Provided shape of the kernel is currently not supported. Please provide it yourself or stick to the provided ones by the library"
        )

    if not isinstance(kernel_size, (tuple, list)):
        raise WrongArgumentsType(
            "Please check the type of the size argument. Only tuples and lists are allowed"
        )

    if len(kernel_size) != 2:
        raise WrongArgumentsValue("Expected a tuple/list of length two for the kernel size")

    if not all(i > 0 for i in kernel_size):
        raise WrongArgumentsValue(
            "Provided width or height of the kernel is negative which is not allowed"
        )

    kernel_size = [int(i) for i in kernel_size]

    structuring_element = cv2.getStructuringElement(shape_strategy, kernel_size)
    _impc_array = namedtuple("ImPcArray", "array_")

    return _impc_array(structuring_element)

# -------------------------------------------------------------------------