import numpy as np

from enum import Enum, unique

# -------------------------------------------------------------------------

@unique
class DataType(Enum):
    ...

# -------------------------------------------------------------------------

class AllowedDataType(DataType):
    Uint8 = np.uint8
    Uint16 = np.uint16
    Float32 = np.float32

# -------------------------------------------------------------------------

class ConversionDataType(DataType):
    Uint32 = np.float32
    Float16 = np.float16
    Float64 = np.float64

# -------------------------------------------------------------------------
