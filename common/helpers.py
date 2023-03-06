# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal library helpers

import re
import numpy as np

from common.interfaces.loader import BaseImage
from common.exceptions import NotSupportedDataType
from common.warning import ImageDataTypeConversion
from common.decorators import check_image_exist_external
from common.datastructs import DataType, ConversionDataType, AllowedDataType, ALLOWED_DATA_TYPES, CONVERSION_DATA_TYPES

# -------------------------------------------------------------------------

def check_user_provided_ndarray(array_: np.ndarray):
    data_type = array_.dtype
    internal_data_type = ALLOWED_DATA_TYPES.get(str(data_type), None)
    if internal_data_type is None:
        conversion_data_type = CONVERSION_DATA_TYPES.get(str(data_type), None)
        if conversion_data_type is None:
            raise NotSupportedDataType(
                "Provided numpy array does not have the supported internal data type"
            )
        array_ = _convert_array_dtype(array_, conversion_data_type)
    return array_

# -------------------------------------------------------------------------

def image_array_check_conversion(image: BaseImage) -> BaseImage:
    data_type = image.dtype
    if not _correct_type(data_type):
        if not _conversion_type(data_type):
            raise NotSupportedDataType("Image has a data-type which is currently not supported")
        image_new = image.copy()
        _convert_image_dtype(image_new)
        return image_new
    return image.copy()

# -------------------------------------------------------------------------

def _correct_type(data_type: DataType):
    allowed_type = [val.value for val in AllowedDataType.__members__.values()]
    return data_type.value in allowed_type

# -------------------------------------------------------------------------

def _conversion_type(data_type: DataType):
    conversion_type = [val.value for val in ConversionDataType.__members__.values()]
    return data_type.value in conversion_type

# -------------------------------------------------------------------------

def _convert_image_dtype(image_new: BaseImage):
    data_type = image_new.dtype
    stored_image = image_new.image
    if data_type is ConversionDataType.Float16:
        ImageDataTypeConversion(
            "Converting the data type from float16 to float32 which is supported by the library"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Float32.value, copy=False))

    elif data_type is ConversionDataType.Float64:
        ImageDataTypeConversion(
            "Converting the data type from float64 to float32 which this method supports. This can possibly result in loss of precision/data"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Float32.value, copy=False))

    elif data_type is ConversionDataType.Uint32:
        ImageDataTypeConversion(
            "Converting the data type from uint32 to uint16 which this method supports. This can possibly result in loss of precision/data"
        )
        image_new._set_image(stored_image.astype(AllowedDataType.Uint16.value, copy=False))

# -------------------------------------------------------------------------

def _convert_array_dtype(array: np.ndarray, internal_type: DataType) -> np.ndarray:
    """Converts the provided numpy array to the internal valid data type"""

    if internal_type is ConversionDataType.Float16:
        ImageDataTypeConversion(
            "Converting the data type from float16 to float32 which is supported by the library"
        )
        return array.astype(AllowedDataType.Float32.value, copy=False)
    elif internal_type is ConversionDataType.Float64:
        ImageDataTypeConversion(
            "Converting the data type from float64 to float32 which is supported by the library"
        )
        return array.astype(AllowedDataType.Float32.value, copy=False)
    elif internal_type is ConversionDataType.Uint32:
        ImageDataTypeConversion(
            "Converting the data type from Uint32 to Uint16 which is supported by the library"
        )
        return array.astype(AllowedDataType.Uint16.value, copy=False)

# -------------------------------------------------------------------------

@check_image_exist_external
def safe_cast(image: BaseImage, desired_type: str):
    """Casts the image to the desired type"""

    internal_data_type = ALLOWED_DATA_TYPES.get(desired_type, None)
    if internal_data_type is None:
        raise RuntimeError("Failed to cast the image to the type provided")

    data_type = internal_data_type.value
    bit_depth = re.search("(?<=uint)\d+|(?<=float)\d+", str(data_type))

    assert bit_depth is not None, RuntimeError("Failed to retrieve the bit depth")
    num_bits = int(bit_depth.group(0))
    norm_factor = (2**num_bits) - 1

    check_image = image_array_check_conversion(image)
    internal_image = check_image.image

    max_pixel_val = np.max(internal_image)
    internal_image = (internal_image/max_pixel_val) * norm_factor
    internal_image = np.clip(internal_image, np.min(internal_image), norm_factor).astype(
        internal_data_type.value, copy=False
    )

    check_image._set_image(internal_image)
    return check_image

# -------------------------------------------------------------------------