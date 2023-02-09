# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image loader for 2D images

import os
import copy
import re
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from functools import singledispatch
from typing import Tuple, List, Optional, Any, Union
from image.common.decorators import check_image_exist_internal
from commons.exceptions import WrongArgumentsValue, NotSupportedDataType, NotSupportedMode
from commons.exceptions import PathDoesNotExist, WrongArgumentsType, LoaderError
from image.load._interface import BaseImage
from image.common.datastructs import AllowedDataType, ALLOWED_DATA_TYPES, DataType

# -------------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = 15000000000

# -------------------------------------------------------------------------

IMAGE_LOADER_MODES = {
    "L": ("Gray", "8-bit pixels, black and white (grayscale)"),
    "RGB": ("RGB", "3x8-bit pixels, true color"),
    "RGBA": ("RGBA", "4x8-bit pixels, true color with transparency mask"),
    "CMYK": ("CMYK", "4x8-bit pixels, color separation"),
    "YCbCr": ("YCbCr", "3x8-bit pixels, color video format (refers to the JPEG)"),
    "LAB": ("LAB", "3x8-bit pixels, the L*a*b color space"),
    "HSV": ("HSV", "3x8-bit pixels, Hue, Saturation, Value color space"),
    "F": ("Float32", "32-bit floating point pixels"),
    "I;16": ("Uint16", "16 bit unsigned integer pixels"),
    "I;16L": ("Uint16LE", "16 bit little endian unsigned integer pixels"),
    "I;16B": ("Uint16BE", "16 bit big endian unsigned integer pixels"),
}

# -------------------------------------------------------------------------

@BaseImage.register
class ImageLoader:

    def __init__(self):
        self._image: np.ndarray = None
        self._file_extension: str = ""
        self._mode: str = ""
        self._mode_description: str = ""
        self._data_type: AllowedDataType = None
        self.__original_mode: str = ""
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._image is not None:
            del self._image
            self.closed = True

    @classmethod
    def create_loader(cls):
        return cls()

    def __valid_image_mode(self, mode: str) -> bool:
        if mode not in IMAGE_LOADER_MODES:
            return False
        return True

    def _load_image(
        self, path: Union[io.BytesIO, str], formats: Optional[Union[List[str], Tuple[str]]] = None
    ):
        try:
            file_stream = Image.open(path, formats=formats)
            self._image = np.ascontiguousarray(file_stream)
        except Exception as e:
            raise LoaderError("Failed to load the image file") from e

        if not self.__valid_image_mode(file_stream.mode):
            raise NotSupportedMode("The provide image can not be loaded by the library")
        self._set_initial_loader_properties(file_stream)

        return self

    def _set_initial_loader_properties(self, file_stream: Image.Image):
        self._file_extension = file_stream.format
        self.__original_mode = file_stream.mode
        self._mode, self._mode_description = IMAGE_LOADER_MODES[file_stream.mode]
        self._data_type = ALLOWED_DATA_TYPES.get(str(self._image.dtype), None)
        if self._data_type is None:
            raise NotSupportedDataType(
                "The data type of this image is currently not supported by the library"
            )

    @check_image_exist_internal
    def _get_original_image_mode(self) -> str:
        return self.__original_mode

    def _set_mode_description(self, mode_desc: str):
        assert isinstance(
            mode_desc, str
        ), WrongArgumentsValue("Provide mode description does not have the valid type")
        self._mode_description = mode_desc

    def _set_mode(self, mode: str):
        assert isinstance(mode, str
                          ), WrongArgumentsValue("Provided mode does not have the accurate type")
        self._mode = mode

    def _image_conversion_helper(self, desired_type: DataType):
        self._image = self._image.astype(desired_type.value, copy=False)
        self.__set_dtype()

    def _set_image(self, image: np.ndarray):
        assert isinstance(image, np.ndarray
                          ), WrongArgumentsValue("Trying to set the wrong image instance type")
        self._image = image
        self.__set_dtype()

    def __set_dtype(self):
        self._data_type = ALLOWED_DATA_TYPES.get(str(self._image.dtype), None)
        if self._data_type is None:
            raise NotSupportedDataType(
                "The data type of the image supplied to be set is currently not supported by the library"
            )

    @property
    @check_image_exist_internal
    def image(self) -> np.ndarray:
        return self._image

    @property
    @check_image_exist_internal
    def height(self) -> int:
        return self._image.shape[0]

    @property
    @check_image_exist_internal
    def width(self) -> int:
        return self._image.shape[1]

    @property
    @check_image_exist_internal
    def dims(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    @check_image_exist_internal
    def channels(self) -> int:
        channels = 0
        image_dims = self.image.shape
        if len(image_dims) == 3:
            channels = image_dims[-1]
        return channels

    @property
    @check_image_exist_internal
    def extension(self) -> str:
        return self._file_extension

    @property
    @check_image_exist_internal
    def dtype(self) -> AllowedDataType:
        return self._data_type

    @property
    @check_image_exist_internal
    def mode(self) -> str:
        return self._mode

    @property
    @check_image_exist_internal
    def mode_description(self) -> str:
        return self._mode_description

    @check_image_exist_internal
    def is_rgb(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 3 and image_dims[-1] == 3 and self.mode == "RGB":
            return True
        return False

    @check_image_exist_internal
    def is_gray(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 2 and self.mode == "Gray":
            return True
        return False

    @check_image_exist_internal
    def is_rgba(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 3 and image_dims[-1] == 4 and self.mode == "RGBA":
            return True
        return False

    @check_image_exist_internal
    def is_lab(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 3 and image_dims[-1] == 3 and self.mode == "LAB":
            return True
        return False

    @check_image_exist_internal
    def is_hsv(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 3 and image_dims[-1] == 3 and self.mode == "HSV":
            return True
        return False

    @check_image_exist_internal
    def is_ycbcr(self) -> bool:
        image_dims = self.image.shape
        if len(image_dims) == 3 and image_dims[-1] == 3 and self.mode == "YCbCr":
            return True
        return False

    @check_image_exist_internal
    def normalize(self):
        """Normalizes the image. Supports only 8-bit, 16-bit and 32-bit encoding"""

        self._image = ImageLoader.normalize_image(self._image)
        self._update_dtype()

        return self

    @staticmethod
    def normalize_image(image: np.ndarray):
        """Internal helper for normalizing the image"""

        max_pixel_value = np.max(image)
        image = (image / max_pixel_value).astype(AllowedDataType.Float32.value, copy=False)

        return image

    @check_image_exist_internal
    def copy(self) -> Any:
        new_im = copy.deepcopy(self)
        return new_im

    def show(self, normalize: Optional[bool] = False):
        show_image: np.ndarray = None
        if normalize:
            show_image = ImageLoader.normalize_image(self._image, self.dtype)
        else:
            show_image = self._image
        plt.imshow(show_image)
        plt.show()

    @check_image_exist_internal
    def save(self, path_: Union[str, io.BytesIO], extension: Optional[str] = "png"):
        if not isinstance(path_, (str, io.BytesIO)):
            raise WrongArgumentsType(
                "Please check the type of the first argument. It should either be a string or a bytesIO object"
            )
        if isinstance(path_, str) and not os.path.exists(os.path.dirname(path_)):
            raise PathDoesNotExist("Path does not exist. Please check the path again")

        try:
            if isinstance(path_, str):
                cv2.imwrite(path_, self._image)
            elif isinstance(path_, io.BytesIO):
                file_stream = Image.fromarray(self._image)
                file_stream.save(path_, extension)
        except Exception as e:
            raise LoaderError("Failed to save the image") from e

    @check_image_exist_internal
    def putpixel(self, xy: Union[List[int], Tuple[int, int]], value: Union[Tuple[int], List[int]]):
        if not isinstance(xy, (list, tuple)):
            raise WrongArgumentsType("Please provide the index as a tuple or list")

        if len(xy) != 2:
            raise WrongArgumentsValue("Insufficient arguments in the xy tuple")

        row, col = xy
        height, width = self.dims

        if row < 0 and row >= width:
            raise WrongArgumentsValue(f"Row index = {row} exceeds image width dimension = {width}")
        if col < 0 and col >= height:
            raise WrongArgumentsValue(
                f"Col index = {col} exceeds image height dimension = {height}"
            )

        try:
            self._image[row, col, ...] = value
        except Exception as e:
            raise LoaderError("Failed to put the pixel at the defined location") from e

    @check_image_exist_internal
    def getpixel(
        self, index: Union[Tuple[int, int], List[int]]
    ) -> Union[Tuple[int, int], Tuple[int, int, int]]:

        if not isinstance(index, (tuple, list)):
            raise WrongArgumentsType("Please provide the index as a tuple or list")

        row, col = index
        width, height = self.dims

        if row < 0 and row >= width:
            raise WrongArgumentsValue(f"Row index = {row} exceeds image width dimension = {width}")
        if col < 0 and col >= height:
            raise WrongArgumentsValue(
                f"Col index = {col} exceeds image height dimension = {height}"
            )

        try:
            pixel = self._image[row, col]
        except Exception as e:
            raise LoaderError(f"Failed to get the pixel value at the index: {repr(index)}") from e

        return pixel

    @check_image_exist_internal
    def tobytes(self) -> bytes:
        return self._image.tobytes()

    @check_image_exist_internal
    def getbbox(self):
        bbox = (0, 0, self.width, self.height)
        return bbox

    @check_image_exist_internal
    def close(self):
        del self._image
        self.closed = True

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and self.mode == other.mode
            and self.dims == other.dims and self.tobytes() == other.tobytes()
        )

    def __repr__(self):
        return 'ImageLoader'

# -------------------------------------------------------------------------

@singledispatch
def open_image() -> BaseImage:
    ...

# -------------------------------------------------------------------------

@open_image.register(str)
def _(path, formats: Optional[Union[List[str], Tuple[str]]] = None) -> BaseImage:
    if formats and not isinstance(formats, (list, tuple)):
        raise WrongArgumentsType("Please check the type of the formats argument")

    path = Path(path)
    if not path.exists():
        res = _find_path(path)
        if not res[0]:
            raise FileNotFoundError(f"Unable to find the image file at the path: {str(path)}")
        else:
            raise LoaderError(
                f"File with the same name exists at another location: {str(res[1])}. Please look at the path again!"
            )
    im = ImageLoader().create_loader()._load_image(path=str(path), formats=formats)
    return im

# -------------------------------------------------------------------------

@open_image.register(io.BytesIO)
def _(path, formats: Optional[Union[List[str], Tuple[str]]] = None) -> BaseImage:
    if formats and not isinstance(formats, (list, tuple)):
        raise WrongArgumentsType("Please check the type of the formats argument")

    im = ImageLoader().create_loader()._load_image(path=path, formats=formats)
    return im

# -------------------------------------------------------------------------

def _find_path(path: Path) -> Tuple[bool, Union[Path, None]]:
    stop_signal = path.anchor
    while path != stop_signal:
        if path.exists():
            return (True, path)
        path = path.parent
    return (False, None)

# -------------------------------------------------------------------------