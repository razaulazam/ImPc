# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal library warnings

import warnings

# -------------------------------------------------------------------------

class BaseWarning:

    def __init__(self, message: str):
        warnings.warn(message, stacklevel=3)

# -------------------------------------------------------------------------

class ImageAlreadyOpen(BaseWarning):

    def __init__(self, message: str):
        super().__init__(message)

# -------------------------------------------------------------------------

class DefaultSetting(BaseWarning):

    def __init__(self, message: str):
        super().__init__(message)

# -------------------------------------------------------------------------

class ImageDataTypeConversion(BaseWarning):

    def __init__(self, message: str):
        super().__init__(message)

# -------------------------------------------------------------------------
