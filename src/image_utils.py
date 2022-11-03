import cv2 as cv
import numpy as np
from nd2 import ND2File


def openImage(file_path: str, rgb_required: bool = True) -> np.ndarray:
    if 'nd2' in file_path:
        return npNDArrayFromND2File(file_path, rgb_required)
    return cv.imread(file_path)


def npNDArrayFromND2File(nd2_file_path: str, rgb_required: bool = True) -> np.ndarray:
    """ Extract and return an image from an ND2 file """
    nd2_file_path = nd2_file_path
    nd2_file = ND2File(nd2_file_path)

    if nd2_file.ndim > 2:  # just return the first frame
        nd2_array = nd2_file.to_xarray(delayed=False)[0].to_numpy()
    else:
        nd2_array = nd2_file.to_xarray(delayed=False).to_numpy()
    if rgb_required and not nd2_file.is_rgb:
        nd2_array = grayToRGB(nd2_array)
    return nd2_array


def rgbConverter(type: str = 'eye') -> np.ndarray:
    if type == 'eye':
        return np.asarray([0.2989, 0.5870, 0.1140])
    if type == 'sensor':
        return np.asarray([0.25, 0.5, 0.25])
    else:
        return np.asarray([1.0, 1.0, 1.0])/3.0


def rescaledToUint8(intensity, intensity_min, intensity_range) -> np.uint8:
    return np.uint8(255.0*(intensity - intensity_min)/intensity_range)


def asUINT8(array_to_convert: np.ndarray) -> np.ndarray:
    frame_min = np.min(array_to_convert)
    frame_max = np.max(array_to_convert)
    frame_range = frame_max - frame_min
    return rescaledToUint8(array_to_convert, frame_min, frame_range)


def rgbToGray(array_to_convert: np.ndarray, rgb_conversion: np.ndarray) -> np.ndarray:
    return np.dot(array_to_convert[..., :3], rgb_conversion)


def grayToRGB(array_to_convert: np.ndarray) -> np.ndarray:
    frame_as_uint8 = asUINT8(array_to_convert)
    return np.stack((frame_as_uint8, frame_as_uint8, frame_as_uint8), axis=-1)
