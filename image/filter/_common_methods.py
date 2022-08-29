# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Image gradient methods

from collections import namedtuple

# -------------------------------------------------------------------------

def _is_not_namedtuple(source: namedtuple) -> bool:
    return not (
        isinstance(source, tuple) and hasattr(source, "_asdict") and hasattr(source, "_fields")
    )

# ------------------------------------------------------------------------