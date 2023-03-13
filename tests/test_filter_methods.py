# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Filter methods tests

import pytest
import numpy as np

from pathlib import Path
from common.exceptions import WrongArgumentsType, WrongArgumentsValue
from load import open_image
from filter.utils import get_kernel
from filter.conv import corr2d, average_blur, gaussian_blur, median_blur
from filter.conv import bilateral_filter, convolve, correlate
from filter.general import butterworth, difference_gaussians, farid, gabor, prewitt, rank_order, roberts, roberts_neg_diag, roberts_pos_diag
from filter.gradients import laplacian, sobel, scharr, unsharp_mask_filter
from filter.morph import erode, dilate, closing, morph_gradient, top_hat, black_hat
from filter.thresholding import simple_threshold, adaptive_threshold, niblack_threshold, sauvola_threshold

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

def test_laplacian(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = laplacian(im, (3, 3))

    # Try with wrong kernel size.
    with pytest.raises(WrongArgumentsValue):
        _ = laplacian(im, (3, 3, 3))

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = laplacian(im, (3, 3), border="blah")

# -------------------------------------------------------------------------

def test_sobel(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = sobel(im, 1, 0, 1.0, 1.0)

    # Try with wrong kernel size.
    with pytest.raises(WrongArgumentsValue):
        _ = sobel(im, 3, 3, 1.0, "1")

    # Trigger the warnings.
    with pytest.warns(UserWarning):
        _ = sobel(im, 0, 1, 1.0, 1.0, border="blah")

# -------------------------------------------------------------------------

def test_scharr(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = scharr(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = scharr(im, mask=[1, 2, 3])

# -------------------------------------------------------------------------

def test_unsharp_mask_filter(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = unsharp_mask_filter(im)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = unsharp_mask_filter(im, radius="1")

# -------------------------------------------------------------------------

def test_erode(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("rectangle", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = erode(im, kernel)
    _ = erode(im, kernel_array)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = erode(im, kernel, iterations="1")

# -------------------------------------------------------------------------

def test_dilate(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("ellipse", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = dilate(im, kernel)
    _ = dilate(im, kernel_array)

    # Try with wrong value.
    with pytest.raises(WrongArgumentsType):
        _ = dilate(im, kernel, iterations="1")

# -------------------------------------------------------------------------

def test_closing(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("rectangle", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = closing(im, kernel)
    _ = closing(im, kernel_array)

# -------------------------------------------------------------------------

def test_morph_gradient(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("rectangle", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = morph_gradient(im, kernel)
    _ = morph_gradient(im, kernel_array)

# -------------------------------------------------------------------------

def test_top_hat(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("rectangle", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = top_hat(im, kernel)
    _ = top_hat(im, kernel_array)

# -------------------------------------------------------------------------

def test_black_hat(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    kernel = get_kernel("rectangle", (3, 3))
    kernel_array = np.ones((3, 3))
    _ = black_hat(im, kernel)
    _ = black_hat(im, kernel_array)

# -------------------------------------------------------------------------
if __name__ == "__main__":
    a = str(Path(__file__).parent / "data" / "sample.jpg")
    b = open_image(a)
    c = unsharp_mask_filter(b)