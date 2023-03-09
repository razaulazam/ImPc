# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Feature method tests

import pytest

from pathlib import Path
from load import open_image
from common.exceptions import LoaderError, ImageAlreadyClosed

# -------------------------------------------------------------------------

@pytest.fixture
def sample_data_path():
    return str(Path(__file__).parent / "data" / "sample.jpg")

# -------------------------------------------------------------------------