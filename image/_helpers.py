# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal library helpers

import numpy as np

from enum import Enum, unique
from image.load._interface import BaseImage
from commons.exceptions import NotSupportedDataType
from commons.warning import ImageDataTypeConversion
from image._common_datastructs import DataType, ConversionDataType, AllowedDataType

# -------------------------------------------------------------------------

@unique
class ConversionMode(Enum):
    OpenCV = 1

# -------------------------------------------------------------------------

def check_user_provided_ndarray(array_: np.ndarray, strategy: str):
    data_type = array_.dtype
    if not _correct_type(data_type, strategy):
        if not _conversion_type(data_type, strategy):
            raise NotSupportedDataType("Image has a data-type which is currently not supported")
        array_ = _convert_array_dtype(array_)
    return array_ # need to check on this

# -------------------------------------------------------------------------

def image_array_check_conversion(image: BaseImage, strategy: ConversionMode) -> BaseImage:
    data_type = image.dtype
    if not _correct_type(data_type, strategy):
        if not _conversion_type(data_type, strategy):
            raise NotSupportedDataType("Image has a data-type which is currently not supported")
        image_new = image.copy()
        _convert_image_dtype(image_new)
        image_new._update_dtype()
        return image_new
    return image

# -------------------------------------------------------------------------

def _correct_type(data_type: DataType, strategy: ConversionMode):
    if strategy is ConversionMode.OpenCV:
        allowed_type = [val.value for val in AllowedDataType.__members__.values()]
        flag = data_type.value in allowed_type
    return flag

# -------------------------------------------------------------------------

def _conversion_type(data_type: DataType, strategy: ConversionMode):
    if strategy is ConversionMode.OpenCV:
        conversion_type = [val.value for val in ConversionDataType.__members__.values()]
        flag = data_type.value in conversion_type
    return flag

# -------------------------------------------------------------------------

def _convert_image_dtype(image_new: BaseImage):
    data_type = image_new.dtype
    stored_image = image_new.image
    if data_type is ConversionDataType.Float16:
        ImageDataTypeConversion(
            "Converting the data type from float16 to float32 which is supported by the library"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Float32, copy=False))

    elif data_type is ConversionDataType.Float64:
        ImageDataTypeConversion(
            "Converting the data type from float64 to float32 which this method supports. This can possibly result in loss of precision/data"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Float32, copy=False))

    elif data_type is ConversionDataType.Uint32:
        ImageDataTypeConversion(
            "Converting the data type from uint32 to uint16 which this method supports. This can possibly result in loss of precision/data"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Uint16, copy=False))

    else:
        raise NotSupportedDataType("Image has a data-type which is currently not supported")

# -------------------------------------------------------------------------

# needs revision
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

    return array_ # need to check on this

# -------------------------------------------------------------------------
