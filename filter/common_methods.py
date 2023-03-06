# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Internal common methods for filtering

from collections import namedtuple

# -------------------------------------------------------------------------

def is_not_namedtuple(source: namedtuple) -> bool:
    return not (
        isinstance(source, tuple) and hasattr(source, "_asdict") and hasattr(source, "_fields")
    )

# ------------------------------------------------------------------------
