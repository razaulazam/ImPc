# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Feature method tests

import pytest

from pathlib import Path
from load import open_image
from common.exceptions import WrongArgumentsValue, WrongArgumentsType, ImageAlreadyClosed
from feature import canny, blob_diff_gaussian, blob_determinant_hessian
from feature import blob_laplacian_gaussian, compute_fast_corners, compute_foerstner_corners, compute_harris_corners
from feature import compute_kitchen_rosenfeld_corners, compute_moravec_corners, compute_shi_tomasi_corners
from feature import compute_daisy_features, compute_haar_like_features, compute_hessian_matrix, compute_hessian_matrix_eigvals
from feature import compute_hog_descriptors, compute_local_binary_pattern
from feature import match_image_descriptors, match_image_template, compute_structure_tensor
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

# -------------------------------------------------------------------------

def test_compute_shi_tomasi_corners(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_shi_tomasi_corners(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_shi_tomasi_corners(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsType):
        _ = compute_shi_tomasi_corners(im, sigma="1")

# -------------------------------------------------------------------------

def test_compute_daisy_features(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_daisy_features(im)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_daisy_features(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = compute_daisy_features(im, radius_outer=-1)
    with pytest.raises(WrongArgumentsValue):
        _ = compute_daisy_features(im, num_rings=-1)
    with pytest.warns(UserWarning):
        _ = compute_daisy_features(im, normalization="blah")
    with pytest.raises(WrongArgumentsValue):
        _ = compute_moravec_corners(im, kernel_size=-1)

# -------------------------------------------------------------------------

def test_compute_haar_like_features(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_haar_like_features(im, row=20, col=20, width=4, height=4)

    # Convert the image to gray.
    im = convert_color(im, "rgb2gray")
    _ = compute_haar_like_features(im, row=20, col=20, width=4, height=4)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = compute_haar_like_features(im, row=-1, col=20, width=4, height=4)
    with pytest.raises(WrongArgumentsValue):
        _ = compute_haar_like_features(im, row=20, col=-1, width=4, height=4)

# -------------------------------------------------------------------------

def test_compute_hessian_matrix(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = compute_hessian_matrix(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsType):
        _ = compute_hessian_matrix(im, mode=1)
    with pytest.warns(UserWarning):
        _ = compute_hessian_matrix(im, mode="blah")

# -------------------------------------------------------------------------

def test_compute_hessian_matrix_eig_vals(sample_data_path):
    # Open the image and compute the vals
    im = open_image(sample_data_path)
    hessian = compute_hessian_matrix(im)
    _ = compute_hessian_matrix_eigvals(hessian)

# -------------------------------------------------------------------------

def test_compute_hog_descriptors(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = compute_hog_descriptors(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = compute_hog_descriptors(im, pixels_in_cell=(3, 3, 3))
    with pytest.raises(WrongArgumentsValue):
        _ = compute_hog_descriptors(im, cells_per_block=(2, 2, 2))

# -------------------------------------------------------------------------

def test_compute_local_binary_pattern(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.warns(UserWarning):
        _ = compute_local_binary_pattern(im, radius=2, neigbour_points=2)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsValue):
        _ = compute_local_binary_pattern(im, radius=-1, neigbour_points=2)
    with pytest.raises(WrongArgumentsValue):
        _ = compute_local_binary_pattern(im, radius=2, neigbour_points=-1)

# -------------------------------------------------------------------------

def test_match_image_descriptors(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    with pytest.raises(WrongArgumentsValue):
        _ = match_image_descriptors(im.image, im.image)

    im = convert_color(im, "rgb2gray")
    _ = match_image_descriptors(im.image, im.image)

    # Call the function with wrong arguments.
    import numpy as np
    dummy_image = np.ones((300, 300))
    with pytest.raises(WrongArgumentsValue):
        _ = match_image_descriptors(im.image, dummy_image)

# -------------------------------------------------------------------------

def test_match_image_template(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    template = open_image(sample_data_path)
    template.close()
    with pytest.raises(ImageAlreadyClosed):
        _ = match_image_template(im, template)

    template_new = open_image(sample_data_path)
    _ = match_image_template(im, template_new)

# -------------------------------------------------------------------------

def test_compute_structure_tensor(sample_data_path):
    # Open the image.
    im = open_image(sample_data_path)
    _ = compute_structure_tensor(im)

    # Call the function with wrong arguments.
    with pytest.raises(WrongArgumentsType):
        _ = compute_structure_tensor(im, sigma="1")
    with pytest.warns(UserWarning):
        _ = compute_structure_tensor(im, mode="blah")

# -------------------------------------------------------------------------