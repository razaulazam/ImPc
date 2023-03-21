# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Transform methods tests

import pytest

from pathlib import Path
from load import open_image
from common.exceptions import WrongArgumentsType, WrongArgumentsValue
from transform import resize, rotate, transpose, kmeans_quantize, histogram_equalization, convert_color

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_bilateral_filter(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = resize(im, (40, 50))

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = resize(im, (-30, 20))
    with pytest.raises(WrongArgumentsType):
        _ = resize(im, "(30, 30)")

# -------------------------------------------------------------------------

def test_bilateral_filter(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = rotate(im, 10)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = rotate(im, angle="40")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    a = str(Path(__file__).parent / "data" / "sample.jpg")
    b = open_image(a)
    c = rotate(b, 10)