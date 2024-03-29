# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Color conversion methods

import cv2

from common.exceptions import TransformError, WrongArgumentsValue, WrongArgumentsType
from common.decorators import check_image_exist_external
from common.interfaces.loader import BaseImage
from common.helpers import image_array_check_conversion

# -------------------------------------------------------------------------

COLOR_REGISTRY = {
    "bgr2bgra": cv2.COLOR_BGR2BGRA,
    "rgb2rgba": cv2.COLOR_RGB2RGBA,
    "bgra2bgr": cv2.COLOR_BGRA2BGR,
    "rgba2rgb": cv2.COLOR_RGBA2RGB,
    "bgr2rgba": cv2.COLOR_BGR2RGBA,
    "rgb2bgra": cv2.COLOR_RGB2BGRA,
    "rgba2bgr": cv2.COLOR_RGBA2BGR,
    "bgra2rgb": cv2.COLOR_BGRA2RGB,
    "bgr2rgb": cv2.COLOR_BGR2RGB,
    "rgb2bgr": cv2.COLOR_RGB2BGR,
    "bgra2rgba": cv2.COLOR_BGRA2RGBA,
    "rgba2bgra": cv2.COLOR_RGBA2BGRA,
    "bgr2gray": cv2.COLOR_BGR2GRAY,
    "rgb2gray": cv2.COLOR_RGB2GRAY,
    "gray2bgr": cv2.COLOR_GRAY2BGR,
    "gray2rgb": cv2.COLOR_GRAY2RGB,
    "gray2bgra": cv2.COLOR_GRAY2BGRA,
    "gray2rgba": cv2.COLOR_GRAY2RGBA,
    "bgra2gray": cv2.COLOR_BGRA2GRAY,
    "rgba2gray": cv2.COLOR_RGBA2GRAY,
    "bgr2bgr565": cv2.COLOR_BGR2BGR565,
    "rgb2bgr565": cv2.COLOR_RGB2BGR565,
    "bgr5652bgr": cv2.COLOR_BGR5652BGR,
    "bgr5652rgb": cv2.COLOR_BGR5652RGB,
    "bgra2bgr565": cv2.COLOR_BGRA2BGR565,
    "rgba2bgr565": cv2.COLOR_RGBA2BGR565,
    "bgr5652bgra": cv2.COLOR_BGR5652BGRA,
    "bgr5652rbga": cv2.COLOR_BGR5652RGBA,
    "gray2bgr565": cv2.COLOR_GRAY2BGR565,
    "bgr5652gray": cv2.COLOR_BGR5652GRAY,
    "bgr2bgr555": cv2.COLOR_BGR2BGR555,
    "rgb2bgr555": cv2.COLOR_RGB2BGR555,
    "bgr5552bgr": cv2.COLOR_BGR5552BGR,
    "bgr5552rgb": cv2.COLOR_BGR5552RGB,
    "bgra2bgr555": cv2.COLOR_BGRA2BGR555,
    "rgba2bgr555": cv2.COLOR_RGBA2BGR555,
    "bgr5552bgra": cv2.COLOR_BGR5552BGRA,
    "bgr5552rgba": cv2.COLOR_BGR5552RGBA,
    "gray2bgr555": cv2.COLOR_GRAY2BGR555,
    "bgr5552gray": cv2.COLOR_BGR5552GRAY,
    "bgr2xyz": cv2.COLOR_BGR2XYZ,
    "rgb2xyz": cv2.COLOR_RGB2XYZ,
    "xyz2bgr": cv2.COLOR_XYZ2BGR,
    "xyz2rgb": cv2.COLOR_XYZ2RGB,
    "bgr2YCrCb": cv2.COLOR_BGR2YCrCb,
    "rgb2YCrCb": cv2.COLOR_RGB2YCrCb,
    "YCrCb2bgr": cv2.COLOR_YCrCb2BGR,
    "YCrCb2rgb": cv2.COLOR_YCrCb2RGB,
    "bgr2hsv": cv2.COLOR_BGR2HSV,
    "rgb2hsv": cv2.COLOR_RGB2HSV,
    "bgr2lab": cv2.COLOR_BGR2LAB,
    "rgb2lab": cv2.COLOR_RGB2LAB,
    "bgr2luv": cv2.COLOR_BGR2LUV,
    "rgb2luv": cv2.COLOR_RGB2LUV,
    "bgr2hls": cv2.COLOR_BGR2HLS,
    "rgb2hls": cv2.COLOR_RGB2HLS,
    "hsv2bgr": cv2.COLOR_HSV2BGR,
    "hsv2rgb": cv2.COLOR_HSV2RGB,
    "lab2bgr": cv2.COLOR_LAB2BGR,
    "lab2rgb": cv2.COLOR_LAB2RGB,
    "luv2bgr": cv2.COLOR_LUV2BGR,
    "luv2rgb": cv2.COLOR_LUV2RGB,
    "hls2bgr": cv2.COLOR_HLS2BGR,
    "hls2rgb": cv2.COLOR_HLS2RGB,
    "bgr2hsvfull": cv2.COLOR_BGR2HSV_FULL,
    "rgb2hsvfull": cv2.COLOR_RGB2HSV_FULL,
    "bgr2hlsfull": cv2.COLOR_BGR2HLS_FULL,
    "rgb2hlsfull": cv2.COLOR_RGB2HLS_FULL,
    "hsv2bgrfull": cv2.COLOR_HSV2BGR_FULL,
    "hsv2rgbfull": cv2.COLOR_HSV2RGB_FULL,
    "hls2bgrfull": cv2.COLOR_HLS2BGR_FULL,
    "hls2rgbfull": cv2.COLOR_HLS2RGB_FULL,
    "lbgr2lab": cv2.COLOR_LBGR2LAB,
    "lrgb2lab": cv2.COLOR_LRGB2LAB,
    "lbgr2luv": cv2.COLOR_LBGR2LUV,
    "lrgb2luv": cv2.COLOR_LRGB2Luv,
    "lab2lbgr": cv2.COLOR_LAB2LBGR,
    "lab2lrgb": cv2.COLOR_LAB2LRGB,
    "luv2lbgr": cv2.COLOR_LUV2LBGR,
    "luv2lrgb": cv2.COLOR_LUV2LRGB,
    "bgr2yuv": cv2.COLOR_BGR2YUV,
    "rgb2yuv": cv2.COLOR_RGB2YUV,
    "yuv2bgr": cv2.COLOR_YUV2BGR,
    "yuv2rgb": cv2.COLOR_YUV2RGB,
    "yuv2rgb_nv12": cv2.COLOR_YUV2RGB_NV12,
    "yuv2bgr_nv12": cv2.COLOR_YUV2BGR_NV12,
    "yuv2rgb_nv21": cv2.COLOR_YUV2RGB_NV21,
    "yuv2bgr_nv21": cv2.COLOR_YUV2BGR_NV21,
    "yuv420sp2rgb": cv2.COLOR_YUV420sp2RGB,
    "yuv420sp2bgr": cv2.COLOR_YUV420sp2BGR,
    "yuv2rgba_nv12": cv2.COLOR_YUV2RGBA_NV12,
    "yuv2bgra_nv12": cv2.COLOR_YUV2BGRA_NV12,
    "yuv2rgba_nv21": cv2.COLOR_YUV2RGBA_NV21,
    "yuv2bgranv21": cv2.COLOR_YUV2BGRA_NV21,
    "yuv420sp2rgba": cv2.COLOR_YUV420sp2RGBA,
    "yuv420sp2bgra": cv2.COLOR_YUV420sp2BGRA,
    "yuv2rgb_yv12": cv2.COLOR_YUV2RGB_YV12,
    "yuv2bgr_yv12": cv2.COLOR_YUV2BGR_YV12,
    "yuv2rgb_iyuv": cv2.COLOR_YUV2RGB_IYUV,
    "yuv2bgr_iyuv": cv2.COLOR_YUV2BGR_IYUV,
    "yuv2rgb_i420": cv2.COLOR_YUV2RGB_I420,
    "yuv2bgr_i420": cv2.COLOR_YUV2BGR_I420,
    "yuv420p2rgb": cv2.COLOR_YUV420p2RGB,
    "yuv420p2bgr": cv2.COLOR_YUV420p2BGR,
    "yuv2rgba_yv12": cv2.COLOR_YUV2RGBA_YV12,
    "yuv2bgra_yv12": cv2.COLOR_YUV2BGRA_YV12,
    "yuv2rgba_iyuv": cv2.COLOR_YUV2RGBA_IYUV,
    "yuv2bgra_iyuv": cv2.COLOR_YUV2BGRA_IYUV,
    "yuv2rgba_i420": cv2.COLOR_YUV2RGBA_I420,
    "yuv2bgra_i420": cv2.COLOR_YUV2BGRA_I420,
    "yuv420p2rgba": cv2.COLOR_YUV420p2RGBA,
    "yuv420p2bgra": cv2.COLOR_YUV420p2RGBA,
    "yuv2gray_420": cv2.COLOR_YUV2GRAY_420,
    "yuv2gray_nv21": cv2.COLOR_YUV2GRAY_NV21,
    "yuv2gray_nv12": cv2.COLOR_YUV2GRAY_NV12,
    "yuv2gray_yv12": cv2.COLOR_YUV2GRAY_YV12,
    "yuv2gray_iyuv": cv2.COLOR_YUV2GRAY_IYUV,
    "yuv2gray_i420": cv2.COLOR_YUV2GRAY_I420,
    "yuv420sp2gray": cv2.COLOR_YUV420SP2GRAY,
    "yuv420p2gray": cv2.COLOR_YUV420p2GRAY,
    "yuv2rgb_uyvy": cv2.COLOR_YUV2RGB_UYVY,
    "yuv2bgr_uyvy": cv2.COLOR_YUV2BGR_UYVY,
    "yuv2rgb_y422 ": cv2.COLOR_YUV2RGB_Y422,
    "yuv2bgr_y422 ": cv2.COLOR_YUV2BGR_Y422,
    "yuv2rgb_uynv": cv2.COLOR_YUV2RGB_UYNV,
    "yuv2bgr_uynv": cv2.COLOR_YUV2BGR_UYNV,
    "yuv2rgba_uyvy": cv2.COLOR_YUV2RGBA_UYVY,
    "yuv2bgra_uyvy": cv2.COLOR_YUV2BGRA_UYVY,
    "yuv2rgba_y422": cv2.COLOR_YUV2RGBA_Y422,
    "yuv2bgra_y422": cv2.COLOR_YUV2BGRA_Y422,
    "yuv2rgba_uynv": cv2.COLOR_YUV2RGBA_UYNV,
    "yuv2bgra_uynv": cv2.COLOR_YUV2BGRA_UYNV,
    "yuv2rgb_yuy2": cv2.COLOR_YUV2RGB_YUY2,
    "yuv2bgr_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuv2rgb_yvyu": cv2.COLOR_YUV2RGB_YVYU,
    "yuv2bgr_yvyu": cv2.COLOR_YUV2BGR_YVYU,
    "yuv2rgb_yuyv": cv2.COLOR_YUV2RGB_YUYV,
    "yuv2bgr_yuyv": cv2.COLOR_YUV2BGR_YUYV,
    "yuv2rgb_yunv": cv2.COLOR_YUV2RGB_YUNV,
    "yuv2bgr_yunv": cv2.COLOR_YUV2BGR_YUNV,
    "yuv2rgba_yuy2": cv2.COLOR_YUV2RGBA_YUY2,
    "yuvbgra_yuy2": cv2.COLOR_YUV2BGRA_YUY2,
    "yuv2rgba_yvyu": cv2.COLOR_YUV2RGBA_YVYU,
    "yuv2bgra_yvyu": cv2.COLOR_YUV2BGRA_YVYU,
    "yuv2rgba_yuyv": cv2.COLOR_YUV2RGBA_YUYV,
    "yuv2bgra_yuyv": cv2.COLOR_YUV2BGRA_YUYV,
    "yuv2rgba_yunv": cv2.COLOR_YUV2RGBA_YUNV,
    "yuv2bgra_yunv": cv2.COLOR_YUV2BGRA_YUNV,
    "yuv2gray_uyvy": cv2.COLOR_YUV2GRAY_UYVY,
    "yuv2gray_y422": cv2.COLOR_YUV2GRAY_Y422,
    "yuv2gray_uynv": cv2.COLOR_YUV2GRAY_UYNV,
    "yuv2gray_yvyu": cv2.COLOR_YUV2GRAY_YVYU,
    "yuv2gray_yuyv": cv2.COLOR_YUV2GRAY_YUYV,
    "yuv2gray_yunv": cv2.COLOR_YUV2GRAY_YUNV,
    "rgba2mrgba": cv2.COLOR_RGBA2M_RGBA,
    "mrgba2rgba": cv2.COLOR_mRGBA2RGBA,
    "rgb2yuv_i420": cv2.COLOR_RGB2YUV_I420,
    "bgr2yuv_i420": cv2.COLOR_BGR2YUV_I420,
    "rgb2yuv_iyuv": cv2.COLOR_RGB2YUV_IYUV,
    "bgr2yuv_iyuv": cv2.COLOR_BGR2YUV_IYUV,
    "rgba2yuv_i420": cv2.COLOR_RGBA2YUV_I420,
    "bgra2yuv_i420": cv2.COLOR_BGRA2YUV_I420,
    "rgba2yuv_iyuv": cv2.COLOR_RGBA2YUV_IYUV,
    "bgra2yuv_iyuv": cv2.COLOR_BGRA2YUV_IYUV,
    "rgb2yuv_yv12": cv2.COLOR_RGB2YUV_YV12,
    "bgr2yuv_yv12": cv2.COLOR_BGR2YUV_YV12,
    "rgba2yuv_yv12": cv2.COLOR_RGBA2YUV_YV12,
    "bgra2yuv_yv12": cv2.COLOR_BGRA2YUV_YV12,
    "rgba2yuv_iyuv": cv2.COLOR_RGBA2YUV_IYUV,
    "bgra2yuv_iyuv": cv2.COLOR_BGRA2YUV_IYUV,
    "rgb2yuv_yv12": cv2.COLOR_RGB2YUV_YV12,
    "bgr2yuv_yv12": cv2.COLOR_BGR2YUV_YV12,
    "rgba2yuv_yv12": cv2.COLOR_RGBA2YUV_YV12,
    "BGRA2yuv_yv12": cv2.COLOR_BGRA2YUV_YV12,
    "bayerbg2bgr": cv2.COLOR_BayerBG2BGR,
    "bayergb2bgr": cv2.COLOR_BayerGB2BGR,
    "bayerrg2bgr": cv2.COLOR_BayerRG2BGR,
    "bayergr2bgr": cv2.COLOR_BayerGR2BGR,
}

# -------------------------------------------------------------------------

INTERNAL_CONVERSION_MODES = {
    "bgr2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "rgb2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "bgra2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "rgba2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "rgb2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "rgba2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bgra2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "rgb2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bgra2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "rgba2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "bgr2gray": ("Gray", "Gray with the native data type as of the original image"),
    "rgb2gray": ("Gray", "Gray with the native data type as of the original image"),
    "gray2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "gray2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "gray2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "gray2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "bgra2gray": ("Gray", "Gray with the native data type as of the original image"),
    "rgba2gray": ("Gray", "Gray with the native data type as of the original image"),
    "bgr2bgr565": ("BGR565", "BGR565 with the native data type as of the original image"),
    "rgb2bgr565": ("BGR565", "BGR565 with the native data type as of the original image"),
    "bgr5652bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bgr5652rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgra2bgr565": ("BGR565", "BGR565 with the native data type as of the original image"),
    "rgba2bgr565": ("BGR565", "BGR565 with the native data type as of the original image"),
    "bgr5652bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "bgr5652rbga": ("RGBA", "RGBA with the native data type as of the original image"),
    "gray2bgr565": ("BGR565", "BGR565 with the native data type as of the original image"),
    "bgr5652gray": ("Gray", "Gray with the native data type as of the original image"),
    "bgr2bgr555": ("BGR555", "BGR555 with the native data type as of the original image"),
    "rgb2bgr555": ("BGR555", "BGR555 with the native data type as of the original image"),
    "bgr5552bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bgr5552rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgra2bgr555": ("BGR555", "BGR555 with the native data type as of the original image"),
    "rgba2bgr555": ("BGR555", "BGR555 with the native data type as of the original image"),
    "bgr5552bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "bgr5552rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "gray2bgr555": ("BGR555", "BGR555 with the native data type as of the original image"),
    "bgr5552gray": ("Gray", "Gray with the native data type as of the original image"),
    "bgr2xyz": ("XYZ", "XYZ with the native data type as of the original image"),
    "rgb2xyz": ("XYZ", "XYZ with the native data type as of the original image"),
    "xyz2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "xyz2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2YCrCb": ("YCbCr", "YCbCr with the native data type as of the original image"),
    "rgb2YCrCb": ("YCbCr", "YCbCr with the native data type as of the original image"),
    "YCrCb2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "YCrCb2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2hsv": ("HSV", "HSV with the native data type as of the original image"),
    "rgb2hsv": ("BGRA", "BGRA with the native data type as of the original image"),
    "bgr2lab": ("LAB", "LAB with the native data type as of the original image"),
    "rgb2lab": ("LAB", "LAB with the native data type as of the original image"),
    "bgr2luv": ("LUV", "LUV with the native data type as of the original image"),
    "rgb2luv": ("LUV", "LUV with the native data type as of the original image"),
    "bgr2hls": ("HLS", "HLS with the native data type as of the original image"),
    "rgb2hls": ("HLS", "HLS with the native data type as of the original image"),
    "hsv2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "hsv2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "lab2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "lab2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "luv2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "luv2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "hls2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "hls2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2hsvfull": ("HSVFULL", "HSVFULL with the native data type as of the original image"),
    "rgb2hsvfull": ("HSVFULL", "HSVFULL with the native data type as of the original image"),
    "bgr2hlsfull": ("HLSFULL", "HLSFULL with the native data type as of the original image"),
    "rgb2hlsfull": ("HLSFULL", "HLSFULL with the native data type as of the original image"),
    "hsv2bgrfull": ("BGRFULL", "BGRFULL with the native data type as of the original image"),
    "hsv2rgbfull": ("RGBFULL", "RGBFULL with the native data type as of the original image"),
    "hls2bgrfull": ("BGRFULL", "BGRFULL with the native data type as of the original image"),
    "hls2rgbfull": ("RGBFULL", "RGBFULL with the native data type as of the original image"),
    "lbgr2lab": ("LAB", "LAB with the native data type as of the original image"),
    "lrgb2lab": ("LAB", "LAB with the native data type as of the original image"),
    "lbgr2luv": ("LUV", "LUV with the native data type as of the original image"),
    "lrgb2luv": ("LUV", "LUV with the native data type as of the original image"),
    "lab2lbgr": ("BGR", "BGR with the native data type as of the original image"),
    "lab2lrgb": ("RGB", "RGB with the native data type as of the original image"),
    "luv2lbgr": ("BGR", "BGR with the native data type as of the original image"),
    "luv2lrgb": ("RGB", "RGB with the native data type as of the original image"),
    "bgr2yuv": ("YUV", "YUV with the native data type as of the original image"),
    "rgb2yuv": ("YUV", "YUV with the native data type as of the original image"),
    "yuv2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "yuv2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "yuv2rgb_nv12": ("RGBNV12", "RGBNV12 with the native data type as of the original image"),
    "yuv2bgr_nv12": ("BGRNV12", "BGRNV12 with the native data type as of the original image"),
    "yuv2rgb_nv21": ("RGBNV21", "RGBNV21 with the native data type as of the original image"),
    "yuv2bgr_nv21": ("RGBNV21", "RGBNV21 with the native data type as of the original image"),
    "yuv420sp2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "yuv420sp2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "yuv2rgba_nv12": ("RGBANV12", "RGBANV12 with the native data type as of the original image"),
    "yuv2bgra_nv12": ("BGRANV12", "BGRANV12 with the native data type as of the original image"),
    "yuv2rgba_nv21": ("RGBANV21", "RGBANV21 with the native data type as of the original image"),
    "yuv2bgranv21": ("BGRANV21", "BGRANV21 with the native data type as of the original image"),
    "yuv420sp2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "yuv420sp2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "yuv2rgb_yv12": ("RGBYV12", "RGBYV12 with the native data type as of the original image"),
    "yuv2bgr_yv12": ("BGRYV12", "BGRYV12 with the native data type as of the original image"),
    "yuv2rgb_iyuv": ("RGBIYUV", "RGBIYUV with the native data type as of the original image"),
    "yuv2bgr_iyuv": ("BGRIYUV", "BGRIYUV with the native data type as of the original image"),
    "yuv2rgb_i420": ("RGBI420", "RGBI420 with the native data type as of the original image"),
    "yuv2bgr_i420": ("BGRI420", "BGRI420 with the native data type as of the original image"),
    "yuv420p2rgb": ("RGB", "RGB with the native data type as of the original image"),
    "yuv420p2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "yuv2rgba_yv12": ("RGBAYV12", "RGBAYV12 with the native data type as of the original image"),
    "yuv2bgra_yv12": ("BGRAYV12", "BGRAYV12 with the native data type as of the original image"),
    "yuv2rgba_iyuv": ("RGBAIYUV", "RGBAIYUV with the native data type as of the original image"),
    "yuv2bgra_iyuv": ("BGRAIYUV", "RGBAIYUV with the native data type as of the original image"),
    "yuv2rgba_i420": ("RGBAI420", "RGBAI420 with the native data type as of the original image"),
    "yuv2bgra_i420": ("BGRAI420", "BGRAI420 with the native data type as of the original image"),
    "yuv420p2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "yuv420p2bgra": ("BGRA", "BGRA with the native data type as of the original image"),
    "yuv2gray_420": ("Gray420", "Gray420 with the native data type as of the original image"),
    "yuv2gray_nv21": ("GrayNV21", "GrayNV21 with the native data type as of the original image"),
    "yuv2gray_nv12": ("GrayNV12", "GrayNV12 with the native data type as of the original image"),
    "yuv2gray_yv12": ("GrayYV12", "GrayYV12 with the native data type as of the original image"),
    "yuv2gray_iyuv": ("GrayIYUV", "GrayIYUV with the native data type as of the original image"),
    "yuv2gray_i420": ("GrayI420", "GrayI420 with the native data type as of the original image"),
    "yuv420sp2gray": ("Gray", "Gray with the native data type as of the original image"),
    "yuv420p2gray": ("Gray", "Gray with the native data type as of the original image"),
    "yuv2rgb_uyvy": ("RGBUYVY", "RGBUYVY with the native data type as of the original image"),
    "yuv2bgr_uyvy": ("BGRUYVY", "BGRUYVY with the native data type as of the original image"),
    "yuv2rgb_y422 ": ("RGBY422", "RGBY422 with the native data type as of the original image"),
    "yuv2bgr_y422 ": ("BGRY422", "BGRY422 with the native data type as of the original image"),
    "yuv2rgb_uynv": ("RGBUYNV", "RGBUYNV with the native data type as of the original image"),
    "yuv2bgr_uynv": ("BGRUYNV", "BGRUYNV with the native data type as of the original image"),
    "yuv2rgba_uyvy": ("RGBAUYVY", "RGBAUYVY with the native data type as of the original image"),
    "yuv2bgra_uyvy": ("BGRAUYVY", "BGRAUYVY with the native data type as of the original image"),
    "yuv2rgba_y422": ("RGBAY422", "RGBAY422 with the native data type as of the original image"),
    "yuv2bgra_y422": ("BGRAY422", "BGRAY422 with the native data type as of the original image"),
    "yuv2rgba_uynv": ("RGBAUYNV", "RGBAUYNV with the native data type as of the original image"),
    "yuv2bgra_uynv": ("BGRAUYNV", "BGRAUYNV with the native data type as of the original image"),
    "yuv2rgb_yuy2": ("RGBYUY2", "RGBYUY2 with the native data type as of the original image"),
    "yuv2bgr_yuy2": ("BGRYUY2", "BGRYUY2 with the native data type as of the original image"),
    "yuv2rgb_yvyu": ("RGBYVYU", "RGBYVYU with the native data type as of the original image"),
    "yuv2bgr_yvyu": ("BGRYVYU", "BGRYVYU with the native data type as of the original image"),
    "yuv2rgb_yuyv": ("RGBYUYV", "RGBYUYV with the native data type as of the original image"),
    "yuv2bgr_yuyv": ("BGRYUYV", "BGRYUYV with the native data type as of the original image"),
    "yuv2rgb_yunv": ("RGBYUNV", "RGBYUNV with the native data type as of the original image"),
    "yuv2bgr_yunv": ("BGRYUNV", "BGRYUNV with the native data type as of the original image"),
    "yuv2rgba_yuy2": ("BGRAYUY2", "BGRAYUY2 with the native data type as of the original image"),
    "yuvbgra_yuy2": ("BGRAYUY2", "BGRAYUY2 with the native data type as of the original image"),
    "yuv2rgba_yvyu": ("RGBAYVYU", "RGBAYVYU with the native data type as of the original image"),
    "yuv2bgra_yvyu": ("BGRAYVYU", "BGRAYVYU with the native data type as of the original image"),
    "yuv2rgba_yuyv": ("RGBAYUYV", "RGBAYUYV with the native data type as of the original image"),
    "yuv2bgra_yuyv": ("BGRAYUYV", "BGRAYUYV with the native data type as of the original image"),
    "yuv2rgba_yunv": ("RGBAYUNV", "RGBAYUNV with the native data type as of the original image"),
    "yuv2bgra_yunv": ("BGRAYUNV", "BGRAYUNV with the native data type as of the original image"),
    "yuv2gray_uyvy": ("GrayUYVY", "GrayUYVY with the native data type as of the original image"),
    "yuv2gray_y422": ("GrayY422", "GrayY422 with the native data type as of the original image"),
    "yuv2gray_uynv": ("GrayUYNV", "GrayUYNV with the native data type as of the original image"),
    "yuv2gray_yvyu": ("GrayYVYU", "GrayYVYU with the native data type as of the original image"),
    "yuv2gray_yuyv": ("GrayYUYV", "GrayYUYV with the native data type as of the original image"),
    "yuv2gray_yunv": ("GrayYUNV", "GrayYUNV with the native data type as of the original image"),
    "rgba2mrgba": ("MRGBA", "MRGBA with the native data type as of the original image"),
    "mrgba2rgba": ("RGBA", "RGBA with the native data type as of the original image"),
    "rgb2yuv_i420": ("YUVI420", "YUVI420 with the native data type as of the original image"),
    "bgr2yuv_i420": ("YUVI420", "YUVI420 with the native data type as of the original image"),
    "rgb2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "bgr2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "rgba2yuv_i420": ("YUVI420", "YUVI420 with the native data type as of the original image"),
    "bgra2yuv_i420": ("YUVI420", "YUVI420 with the native data type as of the original image"),
    "rgba2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "bgra2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "rgb2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "bgr2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "rgba2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "bgra2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "rgba2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "bgra2yuv_iyuv": ("YUVIYUV", "YUVIYUV with the native data type as of the original image"),
    "rgb2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "bgr2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "rgba2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "BGRA2yuv_yv12": ("YUVYV12", "YUVYV12 with the native data type as of the original image"),
    "bayerbg2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bayergb2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bayerrg2bgr": ("BGR", "BGR with the native data type as of the original image"),
    "bayergr2bgr": ("BGR", "BGR with the native data type as of the original image"),
}

# -------------------------------------------------------------------------

@check_image_exist_external
def convert_color(image: BaseImage, code: str) -> BaseImage:
    """Converts the image to different color spaces"""

    if not isinstance(code, str):
        raise WrongArgumentsType(
            "Please check the type of the provided code. Only strings are accepted"
        )

    check_image = image_array_check_conversion(image)

    opencv_code = COLOR_REGISTRY.get(code.lower(), None)
    if opencv_code is None:
        raise WrongArgumentsValue("Provided conversion code is currently not supported")

    mode, mode_description = INTERNAL_CONVERSION_MODES.get(code.lower())

    try:
        check_image._set_image(
            cv2.cvtColor(check_image.image, opencv_code).astype(check_image.dtype.value)
        )
        check_image._set_mode(mode)
        check_image._set_mode_description(mode_description)
    except Exception as e:
        raise TransformError(f"Conversion to {code} is not possible") from e

    return check_image

# -------------------------------------------------------------------------
