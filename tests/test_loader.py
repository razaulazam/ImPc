import pytest

from pathlib import Path
from load import open_image
from common.exceptions import LoaderError, ImageAlreadyClosed

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

def test_open_image(sample_data_path):
    # Open the image and check its properties
    im = open_image(sample_data_path)
    assert im.height > 0 and im.width > 0
    assert im.channels == 3
    assert not im.closed
    assert im.dims == (im.height, im.width)
    assert im.extension == "JPEG"
    assert im.is_rgb()

    # Close the image
    im.close()
    with pytest.raises(ImageAlreadyClosed):
        _ = im.height

    # Check if wrong path yields the right exception
    wrong_path = str(Path(__file__).parent / "data" / "wrong.jpg")
    with pytest.raises(LoaderError):
        _ = open_image(wrong_path)
