from commons.exceptions import WrongArgumentsType, WrongArgumentsValue, FeatureError
from commons.warning import ImageModeConversion
from image._decorators import check_image_exist_external
from image.load._interface import BaseImage
from image.transform.color_conversion import convert
from image._helpers import image_array_check_conversion
from typing import Optional, Union
from skimage.feature import canny as sk_canny

@check_image_exist_external
def canny(image: BaseImage, sigma: Optional[float] = 1.0, thresh_low: Optional[Union[float, int]] = None, thresh_high: Optional[Union[float, int]] = None) -> BaseImage:
    """Edge detection using Canny algorithm"""

    if not image.is_gray():
        ImageModeConversion("Canny algorithm only works with grayscale images. Performing the conversion automatically ...")
        converted_image = convert(image, "rgb2gray")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma must have the float type")

    if not isinstance(thresh_low, (float, int)):
        raise WrongArgumentsType("Low threshold value can either be float or integer")

    if not isinstance(thresh_high, (float, int)):
        raise WrongArgumentsType("High threshold value can either be integer or float")

    check_image = image_array_check_conversion(converted_image)
    
    try:
        check_image._set_image(sk_canny(check_image.image, sigma=sigma, low_threshold=thresh_low, high_threshold=thresh_high))
        check_image._update_dtype()
    except Exception as e:
        raise FeatureError("Failed to compute the edges with the Canny algorithm") from e

    return check_image


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from image.load.loader import open_image
    from image.transform.color_conversion import convert
    from skimage.feature import canny
    import cv2
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(path_image))
    image_input = image.image.astype(np.uint16)


    #out = cv2.Canny(image_input, 100, 200)
    out1 = canny(convert(image, "rgb2gray").image)

    print("hallo")
