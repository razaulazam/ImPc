# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal common structs for filtering methods

import cv2

# -------------------------------------------------------------------------

BORDER_INTERPOLATION = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "transparent": cv2.BORDER_TRANSPARENT,
    "default": cv2.BORDER_DEFAULT
}

# -------------------------------------------------------------------------
