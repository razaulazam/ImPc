# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Internal library decorators

from functools import wraps
from commons.exceptions import ImageAlreadyClosed, WrongArgumentsType
from image.common.interfaces.loader import BaseImage

# -------------------------------------------------------------------------

def check_image_exist_internal(fn):

    @wraps(fn)
    def inner_fn(instance_ref, *args, **kwargs):
        if instance_ref.closed:
            raise ImageAlreadyClosed("This method cannot be called before loading the image")
        ans = fn(instance_ref, *args, **kwargs)
        if ans is not None:
            return ans

    return inner_fn

# -------------------------------------------------------------------------

def check_image_exist_external(fn):

    @wraps(fn)
    def inner_fn(*args, **kwargs):
        image = args[0]
        if not isinstance(image, BaseImage):
            raise WrongArgumentsType(
                "Provide image is not a library compatible instance. Use open_image() to get the right image instance"
            )
        if image.closed:
            raise ImageAlreadyClosed(
                "Provided image loader does not encapsulate an accurately loaded image"
            )
        ans = fn(*args, **kwargs)
        if ans is not None:
            return ans

    return inner_fn

# -------------------------------------------------------------------------
