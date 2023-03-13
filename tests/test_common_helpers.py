# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Common helper tests

import pytest
import numpy as np

from pathlib import Path
from common.helpers import check_user_provided_ndarray, image_array_check_conversion, safe_cast
from common.exceptions import NotSupportedDataType
from load import open_image

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_check_user_provided_ndarray(sample_data_path):
    # Open the image and check the type of the array.
    im = open_image(sample_data_path)
    im_array = im.image
    _ = check_user_provided_ndarray(im_array)

    # Set the data type to something which is not supported
    dummy_image = np.ones((28, 28), dtype=np.uint64)
    with pytest.raises(NotSupportedDataType):
        _ = im._set_image(dummy_image)

# -------------------------------------------------------------------------

def test_image_array_check_conversion(sample_data_path):
    # Open the image and check the type of the array.
    im = open_image(sample_data_path)
    _ = image_array_check_conversion(im)

# -------------------------------------------------------------------------

def test_safe_cast(sample_data_path):
    # Open the image and check the type of the array.
    im = open_image(sample_data_path)

    # Cast the data type of the image.
    _ = safe_cast(im, "float32")

    # Cast the data type to something wrong.
    with pytest.raises(RuntimeError):
        _ = safe_cast(im, "float128")

# -------------------------------------------------------------------------