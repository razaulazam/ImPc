# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Utilities for image restoration module

import numpy as np

from image.common.interfaces.loader import BaseImage
from typing import Optional, Union
from commons.exceptions import WrongArgumentsType, RestorationError
from image.common.helpers import image_array_check_conversion
from image.common.decorators import check_image_exist_external
from skimage.restoration import estimate_sigma as sk_sigma_estimator
from skimage.restoration import ellipsoid_kernel as sk_ellipsoid_kernel

# -------------------------------------------------------------------------

@check_image_exist_external
def calculate_sigma(image: BaseImage,
                    sigma_every_channel: Optional[bool] = False) -> Union[float, list]:
    """Wavelet based estimator of the standard deviation of the gaussian noise"""

    if not isinstance(sigma_every_channel, bool):
        raise WrongArgumentsType(
            "Argument (sigma_every_channel) which controls the averaging operation of sigmas over the channels should be provided as bool"
        )

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        calculated_sigma = sk_sigma_estimator(
            check_image.image, average_sigmas=sigma_every_channel, channel_axis=channel_axis
        )
    except Exception as e:
        raise RestorationError("Failed to calculate the sigma for this image") from e

    return calculated_sigma

# -------------------------------------------------------------------------

def get_ellipsoid_kernel(shape: Union[tuple, list], intensity: Union[int, float]) -> np.ndarray:
    """Creates an ellipsoid kernel for use in rolling ball algorithms"""

    if not isinstance(shape, (list, tuple)):
        raise WrongArgumentsType("Shape argument can only be provided as a tuple or a list")

    if not isinstance(intensity, (float, int)):
        raise WrongArgumentsType("Intensity should be provided as either integer or float")

    try:
        kernel = sk_ellipsoid_kernel(shape, int(intensity))
    except Exception as e:
        raise RestorationError(
            "Failed to compute the ellipsoid kernel for the specified shape and intensity"
        ) from e

    return kernel

# -------------------------------------------------------------------------