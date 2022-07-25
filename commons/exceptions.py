# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
# \brief Internal library exceptions

class _BaseException(Exception):
    pass

# -------------------------------------------------------------------------

class PathDoesNotExist(_BaseException):
    pass

# -------------------------------------------------------------------------

class WrongArgumentsType(_BaseException):
    pass

# -------------------------------------------------------------------------

class TransformError(_BaseException):
    pass

# -------------------------------------------------------------------------

class ProcessingError(_BaseException):
    pass

# -------------------------------------------------------------------------

class FilteringError(_BaseException):
    pass

# -------------------------------------------------------------------------

class LoaderError(_BaseException):
    pass

# -------------------------------------------------------------------------

class WrongArgumentsValue(_BaseException):
    pass

# -------------------------------------------------------------------------

class ImageAlreadyClosed(_BaseException):
    pass

# -------------------------------------------------------------------------

class NotSupportedDataType(_BaseException):
    pass

# -------------------------------------------------------------------------