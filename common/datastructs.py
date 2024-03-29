# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal common data structs

import cv2
import numpy as np

from enum import Enum, unique

# -------------------------------------------------------------------------

@unique
class DataType(Enum):
    ...

# -------------------------------------------------------------------------

class AllowedDataType(DataType):
    Uint8 = np.uint8
    Uint16 = np.uint16
    Float32 = np.float32

# -------------------------------------------------------------------------

class ConversionDataType(DataType):
    Uint32 = np.float32
    Float16 = np.float16
    Float64 = np.float64

# -------------------------------------------------------------------------

ALLOWED_DATA_TYPES = {
    "uint8": AllowedDataType.Uint8,
    "uint16": AllowedDataType.Uint16,
    "float32": AllowedDataType.Float32,
}

# -------------------------------------------------------------------------

CONVERSION_DATA_TYPES = {
    "uint32": ConversionDataType.Uint32,
    "float16": ConversionDataType.Float16,
    "float64": ConversionDataType.Float64,
}

# -------------------------------------------------------------------------

SKIMAGE_SAMPLING_REGISTRY = {
    "constant": "constant",
    "edge": "edge",
    "nearest": "nearest",
    "mirror": "mirror",
    "symmetric": "symmetric",
    "reflect": "reflect",
    "wrap": "wrap"
}

# -------------------------------------------------------------------------

CV_BORDER_INTERPOLATION = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "transparent": cv2.BORDER_TRANSPARENT,
    "default": cv2.BORDER_DEFAULT
}

# -------------------------------------------------------------------------
