# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Processing methods tests

import pytest
import numpy as np

from pathlib import Path
from common.exceptions import ImageAlreadyClosed, WrongArgumentsValue
from load import open_image
from process import blend, composite, gaussian_pyramid, laplacian_pyramid, pyramid_blend

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_blend(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = blend(im, im, 0.1)

    # Trigger errors and warnings.
    im.close()
    with pytest.raises(ImageAlreadyClosed):
        _ = blend(im, im, 0.1)

# -------------------------------------------------------------------------

def test_composite(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    mask = np.zeros((145, 335))
    _ = composite(im, im, mask)

    # Trigger errors and warnings.
    im.close()
    with pytest.raises(ImageAlreadyClosed):
        _ = composite(im, im, mask)

# -------------------------------------------------------------------------

def test_gaussian_pyramid(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = gaussian_pyramid(im, 5)

    # Trigger errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = gaussian_pyramid(im, -1)

# -------------------------------------------------------------------------

def test_laplacian_pyramid(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = laplacian_pyramid(im, 5)

    # Trigger errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = laplacian_pyramid(im, -1)

# -------------------------------------------------------------------------

def test_pyramid_blend(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = pyramid_blend(im, im, 4)

    # Trigger errors and warnings.
    im.close()
    with pytest.raises(ImageAlreadyClosed):
        _ = pyramid_blend(im, im, 4)

# -------------------------------------------------------------------------