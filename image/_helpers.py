# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal library helpers

import numpy as np

from image.load._interface import BaseImage
from commons.exceptions import NotSupportedDataType
from commons.warning import ImageDataTypeConversion

# -------------------------------------------------------------------------

def check_user_provided_ndarray(array_: np.ndarray, strategy: str):
    data_type = array_.dtype
    if not _correct_type(data_type, strategy):
        if not _conversion_type(data_type, strategy):
            raise NotSupportedDataType("Image has a data-type which is currently not supported")
        array_ = _convert_array_dtype(array_)
    return array_

# -------------------------------------------------------------------------

def image_array_check_conversion(image: BaseImage, strategy: str) -> BaseImage:
    data_type = image.dtype
    if not _correct_type(data_type, strategy):
        if not _conversion_type(data_type, strategy):
            raise NotSupportedDataType("Image has a data-type which is currently not supported")
        _convert_image_dtype(image)

# -------------------------------------------------------------------------

def _correct_type(data_type: np.dtype, strategy: str):
    if strategy == "PIL":
        flag = data_type in ["uint8", "uint16", "uint32", "float32"]
    elif strategy == "openCV":
        flag = data_type in ["uint8", "uint16", "float32"]
    return flag

# -------------------------------------------------------------------------

def _conversion_type(data_type: np.dtype, strategy: str):
    if strategy == "PIL":
        flag = data_type == "float16"
    elif strategy == "openCV":
        flag = data_type in ["uint32", "float16"]
    return flag

# -------------------------------------------------------------------------

def _convert_image_dtype(image: BaseImage):
    data_type = image.dtype
    stored_image = image.image
    if data_type == "float16":
        ImageDataTypeConversion(
            "Converting the data type from float16 to float32 which is supported by the library"
        )
        image.image = stored_image.astype(np.float32)

    elif data_type == "float64":
        ImageDataTypeConversion(
            "Converting the data type from float64 to float32 which this method supports. This can possibly result in loss of precision/data"
        )
        image.image = stored_image.astype(np.float32)

    elif data_type == "uint32":
        ImageDataTypeConversion(
            "Converting the data type from uint32 to uint16 which this method supports. This can possibly result in loss of precision/data"
        )
        image.image = stored_image.astype(np.uint16)

    else:
        raise NotSupportedDataType("Image has a data-type which is currently not supported")

    image.update_file_stream()
    image.set_loader_properties()

# -------------------------------------------------------------------------

def _convert_array_dtype(array_: np.ndarray) -> np.ndarray:
    data_type = array_.dtype
    if data_type == "float16":
        ImageDataTypeConversion(
            "Converting the data type from float16 to float32 which is supported by the library"
        )
        array_ = array_.astype(np.float32)

    elif data_type == "float64":
        ImageDataTypeConversion(
            "Converting the data type from float64 to float32 which this method supports. This can possibly result in loss of precision/data"
        )
        array_ = array_.astype(np.float32)

    elif data_type == "uint32":
        ImageDataTypeConversion(
            "Converting the data type from uint32 to uint16 which this method supports. This can possibly result in loss of precision/data"
        )
        array_ = array_.astype(np.uint16)

    else:
        raise NotSupportedDataType("Image has a data-type which is currently not supported")

    return array_

# -------------------------------------------------------------------------
