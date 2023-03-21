# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Restoration methods tests

import pytest
import numpy as np

from pathlib import Path
from load import open_image
from common.exceptions import WrongArgumentsType, WrongArgumentsValue
from restoration import bilateral_filter, bregman_denoising, tv_chambolle_denoising, wavelet_denoising, biharmonic_inpainting
from restoration import deconv_richardson_lucy, rolling_ball, unwrap_phase

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_bilateral_filter(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = bilateral_filter(im, 3, 1.0, 1.0)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = bilateral_filter(im, 3, 1.0, "1.0")
    with pytest.warns(UserWarning):
        _ = bilateral_filter(im, -1.0, 1.0, 1.0)

# -------------------------------------------------------------------------

def test_bregman_denoising(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = bregman_denoising(im)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = bregman_denoising(im, "1.0")

# -------------------------------------------------------------------------

def test_tv_chambolle_denoising(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = tv_chambolle_denoising(im)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = tv_chambolle_denoising(im, "1.0")

# -------------------------------------------------------------------------

def test_wavelet_denoising(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = wavelet_denoising(im)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = wavelet_denoising(im, sigma="1.0")

# -------------------------------------------------------------------------

def test_biharmonic_inpainting(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    mask = np.ones((im.height, im.width))
    mask[:30, :30] = 0
    _ = biharmonic_inpainting(im, mask)

    # Trigger the errors and warnings.
    wrong_mask = np.ones((30, 30))
    with pytest.raises(WrongArgumentsValue):
        _ = biharmonic_inpainting(im, wrong_mask)

# -------------------------------------------------------------------------

def test_deconv_richardson_lucy(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = np.ones((3, 3, 1))
    _ = deconv_richardson_lucy(im, kernel)

    # Trigger the errors and warnings.
    wrong_kernel = np.ones((3, 3, 3, 3))
    with pytest.raises(WrongArgumentsValue):
        _ = deconv_richardson_lucy(im, wrong_kernel)

# -------------------------------------------------------------------------

def test_rolling_ball(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsValue):
        _ = rolling_ball(im, radius=-1.0)

# -------------------------------------------------------------------------

def test_unwrap_phase(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = unwrap_phase(im)

    # Trigger the errors and warnings.
    with pytest.raises(WrongArgumentsType):
        _ = unwrap_phase(im, wrap="true")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    a = str(Path(__file__).parent / "data" / "sample.jpg")
    b = open_image(a)
    kernel = np.ones((3, 3))
    c = rolling_ball(b, ball_kernel=kernel)
