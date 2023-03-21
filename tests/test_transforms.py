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

def test_resize(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = resize(im, (40, 50))

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = resize(im, (-30, 20))
    with pytest.raises(WrongArgumentsType):
        _ = resize(im, "(30, 30)")

# -------------------------------------------------------------------------

def test_rotate(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = rotate(im, 10)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = rotate(im, angle="40")

# -------------------------------------------------------------------------

def test_transpose(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = transpose(im)

# -------------------------------------------------------------------------

def test_kmeans_quantize(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = kmeans_quantize(im)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = kmeans_quantize(im, clusters="40")

# -------------------------------------------------------------------------

def test_histogram_equalization(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = histogram_equalization(im)

# -------------------------------------------------------------------------

def test_convert_color(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = convert_color(im, "rgb2gray")

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = convert_color(im, "rg2gr")

# -------------------------------------------------------------------------