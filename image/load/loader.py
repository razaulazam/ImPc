# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image loader for 2D images

import os
import copy
import re
import io
import numpy as np

from PIL import Image
from pathlib import Path
from functools import singledispatch
from typing import Mapping, Tuple, List, Optional, Any, BinaryIO, Union
from image._decorators import check_image_exist_internal
from commons.exceptions import WrongArgumentsValue, NotSupportedDataType, NotSupportedMode
from commons.exceptions import PathDoesNotExist, WrongArgumentsType, LoaderError
from image.load._interface import BaseImage

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
} # allows uint8, uint16 and float32

# Error handling needs to be more explicit. Very short errors might not prove to be that descriptive
# mode would need to change as well as the mode description
# data type might need to change as well

# -------------------------------------------------------------------------

@BaseImage.register
class ImageLoader:

    def __init__(self):
        self._image: np.ndarray = None
        self._file_extension: str = ""
        self._mode: str = ""
        self._mode_description: str = ""
        self._data_type: np.dtype = None
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
        self, path: Union[BinaryIO, str], formats: Optional[Union[List[str], Tuple[str]]] = None
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
        self._data_type = self._image.dtype

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

    def _image_conversion_helper(self, desired_type: np.dtype):
        self._image = self._image.astype(desired_type, casting="same_kind", copy=False)

    @check_image_exist_internal
    def _update_dtype(self):
        self._data_type = self._image.dtype

    def _set_image(self, image: np.ndarray):
        assert isinstance(image, np.ndarray
                          ), WrongArgumentsValue("Trying to set the wrong image instance type")
        self._image = image

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
    # What happens for the grayscale and rgb images as well as for float?
    def channels(self) -> int:
        return self._image.shape[-1]

    @property
    @check_image_exist_internal
    def extension(self) -> str:
        return self._file_extension

    @property
    @check_image_exist_internal
    def dtype(self) -> np.dtype:
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
    def normalize(self):
        """Normalizes the image. Supports only 8-bit, 16-bit and 32-bit encoding"""

        data_type = self.dtype
        bit_depth = re.search("(?<=uint)\d+|(?<=float)\d+", str(data_type))
        if not bit_depth:
            raise NotSupportedDataType("Image has a data-type which is currently not supported")

        num_bits = int(bit_depth.group(0))
        if num_bits > 32:
            raise NotSupportedDataType(
                "Number of bits used to encode pixel information is higher than 32. Only images with 8, 16 and 32-bit encoding are supported"
            )

        norm_factor = (2**num_bits) - 1
        self._image = (self._image / norm_factor).astype(data_type, copy=False)

        return self

    @check_image_exist_internal
    def copy(self) -> Any:
        new_im = copy.deepcopy(self)
        return new_im

    # This needs to go away since we would get rid of the file stream
    @check_image_exist_internal
    def show(self):
        self.__file_stream.show()

    @check_image_exist_internal # needs to change
    def save(self, path_: Union[str, io.BytesIO], format: Optional[str] = None):
        if not isinstance(path_, (str, io.BytesIO)):
            raise WrongArgumentsType(
                "Please check the type of the first argument. It should either be a string or a bytesIO object"
            )
        if isinstance(path_, str) and not os.path.exists(path_):
            raise PathDoesNotExist("Path does not exist. Please check the path again")
        if format and not isinstance(format, str):
            raise WrongArgumentsType("Please check the type of the format argument")
        try:
            self.__file_stream.save(path_, format=format)
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

@open_image.register(BinaryIO)
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

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    image_path = Path(__file__).parent.parent.parent / "sample.jpg"
    import numpy as np
    a = open_image(str(image_path))
    b = a
    a.close()
    print("hallo")
