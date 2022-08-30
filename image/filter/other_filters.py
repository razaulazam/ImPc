from image.load._interface import BaseImage
from image._decorators import check_image_exist_external
from skimage.filters._fft_based import butterworth as sk_butterworth

from skimage.transform import rotate

@check_image_exist_external
def butterworth(image: BaseImage,):
    ...

if __name__ == "__main__":
    from image.load.loader import open_image
    import cv2
    from image.transform.color_conversion import convert
    import numpy as np
    from pathlib import Path
    path_image = Path(__file__).parent.parent.parent / "sample.jpg"
    image_ = open_image(str(path_image))
    image_ = convert(image_, "rgb2hsv")
    image_ = image_.image

    im1 = sk_butterworth(image_)
    max_im1 = np.max(im1)
    im1 = (im1/max_im1) * 255
    im1 = np.clip(im1, 0, 255)

    print("hallo")