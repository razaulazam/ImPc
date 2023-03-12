# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Filter methods tests

import pytest
import numpy as np

from pathlib import Path
from common.exceptions import WrongArgumentsType, WrongArgumentsValue
from load import open_image
from filter.conv import corr2d, average_blur, gaussian_blur, median_blur
from filter.conv import bilateral_filter, convolve, correlate
from filter.general import butterworth, difference_gaussians, farid, gabor, prewitt, rank_order, roberts, roberts_neg_diag, roberts_pos_diag

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_corr_2d(sample_data_path):
    # Open the image.
    kernel = np.ones((3, 3), dtype=np.uint8)
    im = open_image(sample_data_path)
    _ = corr2d(im, kernel)

    # Try with wrong kernel.
    wrong_kernel = [1, 2, 3, 4]
    with pytest.raises(WrongArgumentsType):
        _ = corr2d(im, wrong_kernel)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = corr2d(im, kernel, border="blah")

# -------------------------------------------------------------------------

def test_average_blur(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = average_blur(im, (3, 3))

    # Try with wrong kernel size.
    with pytest.raises(WrongArgumentsValue):
        _ = average_blur(im, (3, 3, 3))

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = average_blur(im, (3, 3), border="blah")

# -------------------------------------------------------------------------

def test_gaussian_blur(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = gaussian_blur(im, (3, 3), 1.0)

    # Try with wrong kernel size.
    with pytest.raises(WrongArgumentsValue):
        _ = gaussian_blur(im, (3, 3, 3), 1.0)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = gaussian_blur(im, (3, 3), 1.0, border="blah")

# -------------------------------------------------------------------------

def test_median_blur(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = median_blur(im, (3, 3))

    # Try with wrong kernel size.
    with pytest.raises(WrongArgumentsValue):
        _ = median_blur(im, (3, 3, 3))
    with pytest.raises(WrongArgumentsValue):
        _ = median_blur(im, (2, 2))

# -------------------------------------------------------------------------

def test_bilateral_filter(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = bilateral_filter(im, 3, 1.0, 1.0)

    # Try with wrong kernel size.
    with pytest.warns(UserWarning):
        _ = bilateral_filter(im, -1, 1.0, 1.0)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = bilateral_filter(im, 3, 1.0, 1.0, border="blah")

# -------------------------------------------------------------------------

def test_convolve(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    weights = np.ones((3, 3, 3))
    _ = convolve(im, weights)

    # Try with wrong kernel.
    wrong_weights = np.ones((1000, 1000, 3))
    with pytest.raises(WrongArgumentsValue):
        _ = convolve(im, wrong_weights)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = convolve(im, weights, mode="blah")

# -------------------------------------------------------------------------

def test_correlate(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    weights = np.ones((3, 3, 3))
    _ = correlate(im, weights)

    # Try with wrong kernel.
    wrong_weights = np.ones((1000, 1000, 3))
    with pytest.raises(WrongArgumentsValue):
        _ = correlate(im, wrong_weights)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = correlate(im, weights, mode="blah")

# -------------------------------------------------------------------------

def test_butterworth(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = butterworth(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsValue):
        _ = butterworth(im, f_cutoff_ratio=20.0)

# -------------------------------------------------------------------------

def test_diff_gaussians(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = difference_gaussians(im, 1.0)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsValue):
        _ = difference_gaussians(im, sigma_low=-2.0)

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = difference_gaussians(im, sigma_low=1.0, mode="blah")

# -------------------------------------------------------------------------

def test_farid(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = farid(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = farid(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

def test_gabor(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = gabor(im, frequency=5.0)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = gabor(im, frequency="5.0")

# -------------------------------------------------------------------------

def test_prewitt(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = prewitt(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = prewitt(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

def test_rank_order(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = rank_order(im)

# -------------------------------------------------------------------------

def test_roberts(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = roberts(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = roberts(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

def test_roberts_neg_diag(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = roberts_neg_diag(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = roberts_neg_diag(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

def test_roberts_pos_diag(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = roberts_pos_diag(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = roberts_pos_diag(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

if __name__ == "__main__":
    a = str(Path(__file__).parent / "data" / "sample.jpg")
    b = open_image(a)
    c = rank_order(b)