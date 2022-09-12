# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal common methods for filtering

import cv2

from collections import namedtuple

# -------------------------------------------------------------------------

ALLOWED_KERNELS = {
    "rectangle": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS
}

# -------------------------------------------------------------------------

def is_not_namedtuple(source: namedtuple) -> bool:
    return not (
        isinstance(source, tuple) and hasattr(source, "_asdict") and hasattr(source, "_fields")
    )

# ------------------------------------------------------------------------
