 
from typing import Dict, List, Tuple
from fractions import Fraction
import numpy as np
import nd2
import av

# TODO: add a var to the VideoReader constructor called duplicate
#       that takes a VideoReader object and copies all it's settings 
#       to make the new VideoReader object.

# TODO: add a new a class VideoWriter (only using PyAv) because theres no api to write nd2 files anyway.


class VideoReader:
    ''' Unified interface for reading videos of various formats ND2, mp4/avi ... '''
    def __init__(self, video_path: str, reader_api: str=None):
        self.video_path = video_path
        self.reader = self._set_reader(reader_api)

    def _apiFromPath(self) -> str:
        self.format_apis = [
            {'format': 'avi', 'api': 'av'}, 
            {'format': 'mp4', 'api': 'av'}, 
            {'format': 'nd2', 'api': 'nd2'}
        ]
        for video_format_api in self.format_apis:
            if '.' + video_format_api['format'] in self.video_path:
                return video_format_api['api']
        return None

    def _set_reader(self, reader_api: str):
        if reader_api is None:
            reader_api = self._apiFromPath()
        if reader_api == 'nd2':
            return ND2VideoReader(self.video_path)
        return PYAVReader(self.video_path)

    def isOpened(self) -> bool:
        return self.reader.isOpened()

    def setFramePosition(self, frame_position: int) -> bool:
        return self.reader.setFramePosition(frame_position)

    def frame(self) -> np.ndarray:
        return self.reader.frame()

    def frameGray(self) -> np.ndarray:
        return self.reader.frameGray()

    def frameRGB(self) -> np.ndarray:
        return self.reader.frameRGB()

    def frameVideoRGB(self) -> np.ndarray:
        return self.reader.frameVideoRGB()

    def next(self) -> bool:
        return self.reader.next()

    def frameWidth(self) -> int:
        return self.reader.frameWidth()

    def frameHeight(self) -> int:
        return self.reader.frameHeight()

    def frameVideoWidth(self) -> int:
        return self.reader.frameVideoWidth()

    def frameVideoHeight(self) -> int:
        return self.reader.frameVideoHeight()

    def framesPerSecond(self) -> float:
        return self.reader.framesPerSecond()

    def avgFPS(self):
        return self.reader.avgFPS()

    def numFrames(self) -> int:
        return self.reader.numFrames()

    def framePosition(self) -> int:
        return self.reader.framePosition()

    def timeStamp(self) -> float:
        return self.reader.timeStamp()

    def duration(self) -> float:
        return self.reader.duration()

    def framePTS(self) -> float:
        return self.reader.framePTS()

    def frameVideoPTS(self) -> float:
        return self.reader.frameVideoPTS()

    def timeBase(self) -> Fraction:
        return self.reader.timeBase()

    def codecName(self):
        return self.reader.codecName()

    def pixelFormat(self):
        return self.reader.pixelFormat()

    def bitRate(self):
        return self.reader.bitRate()

    def initialiseStream(self):
        return self.reader.initialiseStream()

    def close(self):
        return self.reader.close()

    def videoFormat(self):
        return self.reader.videoFormat()


class PYAVReader():
    ''' Video reader interface using pyav for mp4, avi, mov etc (anything other than ND2)'''
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.container = av.open(self.video_path)
        self.video_stream = self.container.streams.video[0]
        self.num_frames = self.video_stream.frames
        self.video_frames = self.container.decode(video=0)
        self.current_frame = None
        self.frame_num = -1
        self.initialiseStream()

    def next(self) -> bool:
        return_this_at_end_of_frames = None
        self.current_frame = next(self.video_frames, return_this_at_end_of_frames)
        if self.current_frame is None:
            return False
        self.frame_num += 1
        return True

    def initialiseStream(self):
        self.container.seek(0)
        self.frame_num = -1
        self.next()

    def frame(self) -> np.ndarray:
        return self.current_frame.to_ndarray()

    def frameRGB(self) -> np.ndarray:
        return self.current_frame.to_ndarray(format='rgb24')

    def frameVideoRGB(self) -> np.ndarray:
        # unlike nd2 files, video files won't ever need to have
        # thier width or height adjusted to be a multple of 2,
        # so we just use the same frameRGB()) method
        return self.frameRGB()

    def frameGray(self) -> np.ndarray:
        return self.current_frame.to_ndarray(format='gray')

    def timeStamp(self) -> float:
        return self.current_frame.time

    def framePTS(self) -> int:
        return self.current_frame.pts

    # TODO: record the first PTS so we can subtract it. 
    # TODO: adjust duration etc so everything is relative to 0
    #       because video doesn't like the non zero starts

    def frameVideoPTS(self) -> int:
        return self.current_frame.pts

    def timeBase(self) -> Fraction:
        return self.video_stream.time_base

    def duration(self) -> float:
        self.duration = float(self.video_stream.duration * self.timeBase())

    def codecName(self):
        return self.video_stream.codec_context.name

    def pixelFormat(self):
        return self.video_stream.pix_fmt

    def bitRate(self):
        '''bit rate in kpbs'''
        return self.video_stream.bit_rate

    def isOpened(self) -> bool:
        return self.video_stream is not None and self.container is not None

    def close(self):
        if self.video_frames is not None:
            self.video_frames.close()
            self.video_frames = None
        if self.container is not None:
            self.container.close()
            self.container = None

    def videoFormat(self):
        return self.current_frame.format.name 

    def frameWidth(self) -> int:
        return int(self.video_stream.codec_context.width)
    
    def frameHeight(self) -> int:
        return int(self.video_stream.codec_context.height)

    def frameVideoWidth(self) -> int:
        return self.frameWidth()

    def frameVideoHeight(self) -> int:
        return self.frameHeight()

    def framesPerSecond(self) -> float:
        return float(self.video_stream.guessed_rate)

    def avgFPS(self):
        return float(self.video_stream.average_rate)

    def numFrames(self) -> int:
        return self.num_frames

    def framePosition(self) -> int:
        return self.frame_num

    def setFramePosition(self, frame_position: int) -> bool:
        if frame_position < 0 or frame_position >= self.num_frames:
            return False
        if frame_position < self.frame_num:
            self.initialiseStream()
        while self.frame_num < frame_position:
            self.next()
        return True


class ND2VideoReader():
    ''' Video reader interface using nd2 for ND2 files: https://github.com/tlambert03/nd2 '''
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.nd2_file = nd2.ND2File(self.video_path)
        self.is_rgb = self.nd2_file.is_rgb
        self.rgb_conversion = rgbConverter(type='eye')

        self.is_legacy = self.nd2_file.is_legacy

        if 'C' in self.nd2_file.sizes:
            num_data_channels = int(self.nd2_file.sizes['C'])
        else:
            num_data_channels = 0
        if num_data_channels > 1:
            error_message = 'ERROR. Data Channels is not 1. Unsupported format.'
            raise TypeError(error_message)

        self.width = int(self.nd2_file.sizes['X'])
        self.height = int(self.nd2_file.sizes['Y'])
        self.num_frames = int(self.nd2_file.sizes['T'])
        self.nd2_frames = self.nd2_file.to_xarray(delayed=True)

        self.time_base = Fraction(1,1000)  # is always milliseconds for nd2 files
        self.time_steps = self.timeStamps()
        self.duration = float(self.time_steps[-1])
        self.avg_fps = float(self.num_frames)/self.duration

        self.frame_num = -1

    def next(self) -> bool:
        if self.frame_num < self.num_frames - 1:
            self.frame_num += 1
            return True
        else:
            self.frame_num = self.num_frames
            return False

    def initialiseStream(self):
        self.frame_num = 0

    def frame(self) -> np.ndarray:
        return self.nd2_frames[self.frame_num].to_numpy()

    def frameRGB(self) -> np.ndarray:
        if self.is_rgb:
            return self.nd2_frames[self.frame_num].to_numpy()
        else:
            return grayToRGB(self.nd2_frames[self.frame_num].to_numpy())

    def frameVideoRGB(self) -> np.ndarray:
        new_height = self.height
        if self.height % 2 != 0:
            new_height += 1
        new_width = self.width
        if self.width % 2 != 0:
             new_width += 1
        if self.is_rgb:
            old_frame = self.nd2_frames[self.frame_num].to_numpy()
        else:
            old_frame = grayToRGB(self.nd2_frames[self.frame_num].to_numpy())
        frame_to_return = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        frame_to_return[0:old_frame.shape[0], 0:old_frame.shape[1],:] = old_frame
        return frame_to_return

    def frameGray(self) -> np.ndarray:
        if self.is_rgb:
            return rgbToGray(self.nd2_frames[self.frame_num].to_numpy(), self.rgb_conversion)
        else:
            return self.nd2_frames[self.frame_num].to_numpy()

    def framePTS(self, frame_num: int=None) -> int:
        if frame_num is None:
            frame_num = self.frame_num
        _chs = self.nd2_file._rdr._frame_metadata(frame_num).get('channels')
        time_stamp = _chs[0]['time']['relativeTimeMs']
        return time_stamp

    def frameVideoPTS(self, frame_num: int=None) -> int:
        if frame_num is None:
            frame_num = self.frame_num        
        return self.framePTS(frame_num) - self.framePTS(0)

    def timeStamp(self, frame_num: int=None) -> float:
        if frame_num is None:
            frame_num = self.frame_num
        return self.frameVideoPTS(frame_num)*self.time_base

    def timeStamps(self):
        time_stamps = []
        for frame_num in range(self.num_frames):
            time_stamps.append(self.timeStamp(frame_num))
        return time_stamps

    def timeBase(self) -> Fraction:
        return self.time_base

    def duration(self) -> float:
        return self.duration

    def codecName(self):
        # we just return a fixed value for cases where
        # the user wants to copy the input stream metadata
        return 'ffv1'  #'h264'  # 'rawvideo'

    def pixelFormat(self):
        # we just return a fixed value for cases where 
        # the user wants to copy the input stream metadata
        return 'yuv420p'  #  'rgb24'

    def bitRate(self, for_rgb: bool=True) -> float:
        ''' estimated kpbs '''
        bits_per_pixel = self.nd2_file.attributes.bitsPerComponentInMemory
        pixels_per_frame = self.width * self.height
        bits_per_frame = float(bits_per_pixel * pixels_per_frame)
        kilobits_per_frame = bits_per_frame / 1024.0
        # max_kbps = float(2**16)
        max_kbps = float(2**32 - 1)
        kbps = kilobits_per_frame * self.avg_fps
        return min(kbps, max_kbps)

    def isOpened(self) -> bool:
        return not self.nd2_file.closed

    def close(self):
        self.nd2_file.close()

    def videoFormat(self):
        return 'mkv'

    def frameWidth(self) -> int:
        return self.width

    def frameHeight(self) -> int:
        return self.height

    def frameVideoWidth(self) -> int:
        ''' adjust to be multiple of 2 because we'll do this for frameVideoRGB'''
        return self.width + (self.width % 2)

    def frameVideoHeight(self) -> int:
        ''' adjust to be multiple of 2 because we'll do this for frameVideoRGB'''        
        return self.height + (self.height % 2)

    def framesPerSecond(self) -> float:  # TODO:
        return self.avgFPS()

    def avgFPS(self):
        return self.avg_fps

    def numFrames(self) -> int:
        return self.num_frames

    def framePosition(self) -> int:
        return self.frame_num

    def setFramePosition(self, frame_position: int) -> bool:
        if frame_position < 0 or frame_position >= self.num_frames:
            return False
        self.frame_num = frame_position
        return True


def rgbConverter(type: str='eye') -> np.ndarray:
    if type =='eye':
        return np.asarray([0.2989, 0.5870, 0.1140])
    if type =='sensor':
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
    return np.dot(array_to_convert[...,:3], rgb_conversion)


def grayToRGB(array_to_convert: np.ndarray) -> np.ndarray:
    frame_as_uint8 = asUINT8(array_to_convert)
    return np.stack((frame_as_uint8, frame_as_uint8, frame_as_uint8), axis=-1)    
