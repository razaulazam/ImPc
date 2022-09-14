# Copyright (C) Raza Ul Azam., All Rights Reserved.
# \brief Image restoration methods

from tabnanny import check
import cv2

from typing import Optional, Union
from commons.exceptions import WrongArgumentsType, RestorationError
from commons.warning import DefaultSetting, ImageDataTypeConversion
from image.load._interface import BaseImage
from image._helpers import image_array_check_conversion
from image._common_datastructs import AllowedDataType, CV_BORDER_INTERPOLATION
from image._decorators import check_image_exist_external
from skimage.restoration import denoise_tv_bregman as sk_denoise_bregman
from skimage.restoration import denoise_tv_chambolle as sk_denoise_tv_chambolle
from skimage.restoration import denoise_wavelet as sk_denoise_wavelet

# -------------------------------------------------------------------------

@check_image_exist_external
def bilateral_filter(
    image: BaseImage,
    kernel_diameter: int,
    color_sigma: float,
    spatial_sigma: float,
    border: Optional[str] = "default"
) -> BaseImage:
    """Only 8-bit and 32-bit floating point images are supported"""
    """Does not work with RGBA images"""

    if image.is_rgba():
        raise RestorationError(
            "This filter cannot operate on images that have color channels more than 3"
        )

    if not isinstance(kernel_diameter, (float, int)):
        raise WrongArgumentsType("Diameter value can only be either an integer or float value")

    if kernel_diameter < 0:
        DefaultSetting("Diameter value will be computed from the spatial sigma value")

    if not isinstance(color_sigma, (float, int)):
        raise WrongArgumentsType("Color sigma value can either be an integer or a float value")

    if not isinstance(spatial_sigma, (float, int)):
        raise WrongArgumentsType("Spatial sigma value can either be an integer or a float value")

    if not isinstance(border, str):
        raise WrongArgumentsType("Border argument can only be specified as a string")

    check_image = image_array_check_conversion(image)

    if check_image.dtype is AllowedDataType.Uint16:
        ImageDataTypeConversion(
            "Converting the image from 16 bits to 8 bits per channel since this is what is only supported for this filter"
        )
        check_image._image_conversion_helper(AllowedDataType.Uint8)
        check_image._update_dtype()

    border_actual = CV_BORDER_INTERPOLATION.get(border, None)
    if border_actual is None:
        DefaultSetting(
            "Provided border option is not supported by the library currently. Using the default strategy (reflect)"
        )
        border_actual = CV_BORDER_INTERPOLATION["default"]

    try:
        check_image._set_image(
            cv2.bilateralFilter(
                check_image.image, int(kernel_diameter), float(color_sigma), float(spatial_sigma),
                border_actual
            )
        )
    except Exception as e:
        raise RestorationError("Failed to filter the image") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def bregman_denoising(
    image: BaseImage,
    weight: Optional[Union[float, int]] = 5.0,
    num_iter: Optional[Union[float, int]] = 100,
    stop_tolerance: Optional[float] = 0.001,
    isotropic: Optional[bool] = True
) -> BaseImage:
    """Total variation denoising using split-Bregman optimization. Result is returned as float32"""

    if not isinstance(weight, (float, int)):
        raise WrongArgumentsType("Weight can only be provided as either float or integer")

    if not isinstance(num_iter, (float, int)):
        raise WrongArgumentsType("Number of iterations can only be specified as integer or float")

    if not isinstance(stop_tolerance, float):
        raise WrongArgumentsType("Stop tolerance can only be provided as float")

    if not isinstance(isotropic, bool):
        raise WrongArgumentsType(
            "Option between isotropic and anisotropic mode can only be provided as bool"
        )

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_denoise_bregman(
                check_image.image,
                weight=float(weight),
                max_num_iter=int(num_iter),
                eps=float(stop_tolerance),
                isotropic=isotropic,
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise RestorationError(
            "Failed to denoise the image using total variations filter with split-bregman optimization"
        ) from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def tv_chambolle_denoising(
    image: BaseImage,
    weight: Optional[Union[float, int]] = 0.1,
    stop_tolerance: Optional[float] = 0.0002,
    num_iter: Optional[Union[float, int]] = 200
) -> BaseImage:
    """Total variations denoising"""

    if not isinstance(weight, (float, int)):
        raise WrongArgumentsType("Weight can only be provided as either float or integer")

    if not isinstance(num_iter, (float, int)):
        raise WrongArgumentsType("Number of iterations can only be specified as integer or float")

    if not isinstance(stop_tolerance, float):
        raise WrongArgumentsType("Stop tolerance can only be provided as float")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_denoise_tv_chambolle(
                check_image.image,
                weight=float(weight),
                max_num_iter=int(num_iter),
                eps=float(stop_tolerance),
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise RestorationError("Failed to denoise the image using total variations filter") from e

    return check_image

# -------------------------------------------------------------------------

@check_image_exist_external
def wavelet_denoising(
    image: BaseImage,
    sigma: Optional[float] = None,
    wavelet: Optional[str] = "db1",
    mode: Optional[str] = "soft",
    levels: Optional[Union[int, float]] = None,
    conversion_ycbcr: Optional[bool] = False,
    method: Optional[str] = "BayesShrink"
) -> BaseImage:
    """Wavelet denoising on the image. Result is returned as float32"""
    
    mode_list = {"soft": "soft", "hard": "hard"}
    method_list = {"bayes": "BayesShrink", "visu": "VisuShrink"}
    wavelist = {
        "db1": "db1",
        "db2": "db2",
        "db3": "db3",
        "db4": "db4",
        "db5": "db5",
        "db6": "db6",
        "db7": "db7",
        "db8": "db8",
        "db9": "db9",
        "db10": "db10",
        "db11": "db11",
        "db12": "db12",
        "db13": "db13",
        "db14": "db14",
        "db15": "db15",
        "db16": "db16",
        "db17": "db17",
        "db18": "db18",
        "db19": "db19",
        "db20": "db20",
        "db21": "db21",
        "db22": "db22",
        "db23": "db23",
        "db24": "db24",
        "db25": "db25",
        "db26": "db26",
        "db27": "db27",
        "db28": "db28",
        "db29": "db29",
        "db30": "db30",
        "db31": "db31",
        "db32": "db32",
        "db33": "db33",
        "db34": "db34",
        "db35": "db35",
        "db36": "db36",
        "db37": "db37",
        "db38": "db38",
        "haar": "haar",
        "sym2": "sym2",
        "sym3": "sym3",
        "sym4": "sym4",
        "sym5": "sym5",
        "sym6": "sym6",
        "sym7": "sym7",
        "sym8": "sym8",
        "sym9": "sym9",
        "sym10": "sym10",
        "sym11": "sym11",
        "sym12": "sym12",
        "sym13": "sym13",
        "sym14": "sym14",
        "sym15": "sym15",
        "sym16": "sym16",
        "sym17": "sym17",
        "sym18": "sym18",
        "sym19": "sym19",
        "sym20": "sym20",
        "coif1": "coif1",
        "coif2": "coif2",
        "coif3": "coif3",
        "coif4": "coif4",
        "coif5": "coif5",
        "coif6": "coif6",
        "coif7": "coif7",
        "coif8": "coif8",
        "coif9": "coif9",
        "coif10": "coif10",
        "coif11": "coif11",
        "coif12": "coif12",
        "coif13": "coif13",
        "coif14": "coif14",
        "coif15": "coif15",
        "coif16": "coif16",
        "coif17": "coif17",
        "bior1.1": 'bior1.1',
        "bior1.3": 'bior1.3',
        "bior1.5": 'bior1.5',
        "bior2.2": 'bior2.2',
        "bior2.4": 'bior2.4',
        "bior2.6": 'bior2.6',
        "bior2.8": 'bior2.8',
        "bior3.1": 'bior3.1',
        "bior3.3": 'bior3.3',
        "bior3.5": 'bior3.5',
        "bior3.7": 'bior3.7',
        "bior3.9": 'bior3.9',
        "bior4.4": 'bior4.4',
        "bior5.5": 'bior5.5',
        "bior6.8": 'bior6.8',
        "rbio1.1": 'rbio1.1',
        "rbio1.3": 'rbio1.3',
        "rbio1.5": 'rbio1.5',
        "rbio2.2": 'rbio2.2',
        "rbio2.4": 'rbio2.4',
        "rbio2.6": 'rbio2.6',
        "rbio2.8": 'rbio2.8',
        "rbio3.1": 'rbio3.1',
        "rbio3.3": 'rbio3.3',
        "rbio3.5": 'rbio3.5',
        "rbio3.7": 'rbio3.7',
        "rbio3.9": 'rbio3.9',
        "rbio4.4": 'rbio4.4',
        "rbio5.5": 'rbio5.5',
        "rbio6.8": 'rbio6.8',
        "dmey": "dmey",
        "gaus1": "gaus1",
        "gaus2": "gaus2",
        "gaus3": "gaus3",
        "gaus4": "gaus4",
        "gaus5": "gaus5",
        "gaus6": "gaus6",
        "gaus7": "gaus7",
        "gaus8": "gaus8",
        "mexh": "mexh",
        "morl": "morl",
        "cgau1": "cgau1",
        "cgau2": "cgau2",
        "cgau3": "cgau3",
        "cgau4": "cgau4",
        "cgau5": "cgau5",
        "cgau6": "cgau6",
        "cgau7": "cgau7",
        "cgau8": "cgau8",
        "shan": "shan",
        "fbsp": "fbsp",
        "cmor": "cmor",
    }

    if image.is_gray() and conversion_ycbcr:
        raise RestorationError("Conversion to YCbCr is only possible for multichannel images")

    if not isinstance(sigma, float):
        raise WrongArgumentsType("Sigma value should be provided as float")

    if not isinstance(wavelet, str):
        raise WrongArgumentsType("Wavelet value should be provided as string")

    if not isinstance(mode, str):
        raise WrongArgumentsType("Mode value should be provided as string")

    if not isinstance(levels, (int, float)):
        raise WrongArgumentsType("Levels value should be provided as integer")

    if not isinstance(conversion_ycbcr, bool):
        raise WrongArgumentsType("Conversion YCbCr value should be provided as bool")

    if not isinstance(method, str):
        raise WrongArgumentsType("Method value should be provided as string")

    wavelet_arg = wavelist.get(wavelet, None)
    if wavelet_arg is None:
        DefaultSetting(
            "Using the default wavelet (db1) since the provided wavelet is not supported by the library"
        )
        wavelet_arg = wavelist.get("db1")
    
    mode_arg = mode_list.get(mode, None)
    if mode_arg is None:
        DefaultSetting(
            "Using the default mode (soft) since the provided mode is not supported by the library"
        )
        mode_arg = mode_list.get("soft")
        
    method_arg = method_list.get(method, None)
    if method_arg is None:
        DefaultSetting(
            "Using the default mode (bayes) since the provided method is not supported by the library"
        )
        method_arg = method_list.get("bayes")

    check_image = image_array_check_conversion(image)
    channel_axis = None if check_image.channels == 0 else 2

    try:
        check_image._set_image(
            sk_denoise_wavelet(
                check_image.image,
                sigma=float(sigma),
                wavelet=wavelet_arg,
                mode=mode_arg,
                wavelet_levels=int(levels),
                convert2ycbcr=conversion_ycbcr,
                method=method_arg,
                rescale_sigma=True,
                channel_axis=channel_axis
            ).astype(AllowedDataType.Float32.value, copy=False)
        )
        check_image._update_dtype()
    except Exception as e:
        raise RestorationError("Failed to denoise the image with wavelet denosing") from e

    return check_image

# -------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from image.load.loader import open_image
    from skimage.restoration import denoise_wavelet

    image_path = Path(__file__).parent.parent.parent / "sample.jpg"
    image = open_image(str(image_path))

    output = denoise_wavelet(image.image, convert2ycbcr=True, channel_axis=2)

    print("dsad")