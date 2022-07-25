# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
# \brief Internal library decorators

from functools import wraps
from commons.exceptions import ImageAlreadyClosed

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
        if image.closed:
            raise ImageAlreadyClosed("Provided image loader does not encapsulate an accurately loaded image")
        ans = fn(*args, **kwargs)
        if ans is not None:
            return ans

    return inner_fn

# -------------------------------------------------------------------------
