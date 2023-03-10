# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Feature method tests

import pytest

from pathlib import Path
from load import open_image
from common.exceptions import WrongArgumentsValue, WrongArgumentsType
from feature import canny, blob_diff_gaussian, blob_determinant_hessian
from feature import blob_laplacian_gaussian, compute_fast_corners, compute_foerstner_corners, compute_harris_corners
from feature import compute_kitchen_rosenfeld_corners, compute_moravec_corners
from transform import convert_color

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------

def test_canny(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = canny(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = canny(im)

# -------------------------------------------------------------------------

def test_blob_diff_gaussian(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = blob_diff_gaussian(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = blob_diff_gaussian(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = blob_diff_gaussian(im, overlap=2.0)
    with pytest.raises(WrongArgumentsType):
        _ = blob_diff_gaussian(im, sigma_max=int(2))

# -------------------------------------------------------------------------

def test_blob_determinant_hessian(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = blob_determinant_hessian(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = blob_determinant_hessian(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = blob_determinant_hessian(im, overlap=2.0)
    with pytest.raises(WrongArgumentsType):
        _ = blob_determinant_hessian(im, sigma_max=int(2))

# -------------------------------------------------------------------------

def test_blob_laplacian_gaussian(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = blob_laplacian_gaussian(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = blob_laplacian_gaussian(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = blob_laplacian_gaussian(im, overlap=2.0)
    with pytest.raises(WrongArgumentsType):
        _ = blob_laplacian_gaussian(im, sigma_max=int(2))

# -------------------------------------------------------------------------

def test_compute_fast_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_fast_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_fast_corners(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsType):
        _ = compute_fast_corners(im, threshold=int(2))
    with pytest.raises(WrongArgumentsType):
        _ = compute_fast_corners(im, num_pixels=str(2))

# -------------------------------------------------------------------------

def test_compute_foerstner_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_foerstner_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_foerstner_corners(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsType):
        _ = compute_foerstner_corners(im, sigma=str(2))

# -------------------------------------------------------------------------

def test_compute_harris_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_harris_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_harris_corners(im)

    # Call the function with wrong arguments.
    with pytest.warns(UserWarning):
        _ = compute_harris_corners(im, method="blah")
    with pytest.raises(WrongArgumentsType):
        _ = compute_harris_corners(im, sigma=str(2))

# -------------------------------------------------------------------------

def test_compute_kitchen_rosenfeld_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_kitchen_rosenfeld_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_kitchen_rosenfeld_corners(im)

    # Call the function with wrong arguments.
    with pytest.warns(UserWarning):
        _ = compute_kitchen_rosenfeld_corners(im, mode="blah")

# -------------------------------------------------------------------------

def test_compute_moravec_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_moravec_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_moravec_corners(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = compute_moravec_corners(im, kernel_size=-1)

if __name__ == "__main__":
    a = str(Path(__file__).parent / "data" / "sample.jpg")
    b = open_image(a)
    c = blob_determinant_hessian(b)