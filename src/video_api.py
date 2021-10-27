 
from typing import Dict, List, Tuple
from fractions import Fraction
import numpy as np
import av
from nd2reader import ND2Reader


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

    def next(self) -> bool:
        return self.reader.next()

    def frameWidth(self) -> int:
        return self.reader.frameWidth()

    def frameHeight(self) -> int:
        return self.reader.frameHeight()

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

    def pts(self) -> float:
        return self.reader.pts()

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
        if self.frame_num < self.num_frames - 1:
            self.current_frame = next(self.video_frames)
            self.frame_num += 1
            return True
        else:
            self.current_frame = None
            self.frame_num = self.num_frames
            return False

    def initialiseStream(self):
        self.container.seek(0)
        self.frame_num = -1
        self.next()

    def frame(self) -> np.ndarray:
        return self.current_frame.to_ndarray()

    def frameRGB(self) -> np.ndarray:
        return self.current_frame.to_ndarray(format='rgb24')

    def frameGray(self) -> np.ndarray:
        return self.current_frame.to_ndarray(format='gray')

    def timeStamp(self) -> float:
        return self.current_frame.time

    def pts(self) -> int:
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
        return self.video_stream.bit_rate

    def isOpened(self) -> bool:
        return self.video_stream is not None and self.container is not None

    # def release(self):
    #     self.video_stream.release()
    #     self.video_frames = None

    def close(self):
        # self.release()
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


# TODO: figure out if the format of these ND2 files is 
#       [z, y, x, T, b[, g, r]] or [T, z, y, x, b[, g, r]] etc
class ND2VideoReader():
    ''' Video reader interface using nd2 for ND2 files: https://github.com/tlambert03/nd2 '''
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.nd2 = ND2Reader(self.video_path)

        if 'C' in self.nd2.sizes:
            num_data_channels = int(self.nd2.sizes['c'])
        else:
            num_data_channels = 0
        if num_data_channels != num_data_channels:
            error_message = 'ERROR. Data Channels is not 1. Unsupported format.'
            raise TypeError(error_message)

        if 'S' in self.nd2.sizes:
            num_colours = int(self.nd2.sizes['s'])
            if num_colours == 3:
                self.is_rgb = True
            else:
                error_message = 'ERROR. Colour channels is not 0 or 3. Unsupported format.'
                raise TypeError(error_message)
        else:
            self.is_rgb = False
        self.rgb_conversion = self._rgb_converter()

        self.num_frames = int(self.nd2.sizes['t'])      
        self.height = int(self.nd2.sizes['y'])
        self.width = int(self.nd2.sizes['x'])

        self.time_base = Fraction(1,1000)  # appears to always be 1/1000 for nd2 files?
        self.time_steps = self.nd2.timesteps*float(self.time_base)  
        self.duration = float(self.time_steps[-1] - self.time_steps[0])
        self.avg_fps = float(self.num_frames)/self.duration

        self.frame_num = -1

    def _rgb_converter(self, type: str='eye') -> np.ndarray:
            if type =='eye':
                return np.asarray([0.2989, 0.5870, 0.1140])
            if type =='sensor':
                np.asarray([0.25, 0.5, 0.25])
            else:
                np.asarray([1.0, 1.0, 1.0])/3.0

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
        return self.nd2[self.frame_num]

    def frameRGB(self) -> np.ndarray:
        if self.is_rgb:
            return self.nd2[self.frame_num]
        else:
            frame_min = np.min(self.nd2[self.frame_num])
            frame_max = np.max(self.nd2[self.frame_num])
            frame_range = frame_max - frame_min
            frame_as_uint8 = rescaledToUint8(self.nd2[self.frame_num], frame_min, frame_range)
            return np.stack((frame_as_uint8, frame_as_uint8, frame_as_uint8), axis=-1)

    def frameGray(self) -> np.ndarray:
        if self.is_rgb:
            return np.dot(self.nd2[self.frame_num][...,:3], self.rgb_conversion)        
        else:
            return self.nd2[self.frame_num]

    def timeStamp(self) -> float:
        return self.time_steps[self.frame_num]

    def pts(self) -> int:
        return self.time_steps[self.frame_num]/float(self.time_base)

    def timeBase(self) -> Fraction:
        return self.time_base

    def duration(self) -> float:
        return self.duration

    def codecName(self):
        # we just return a fixed value for cases where 
        # the user wants to copy the input stream metadata        
        # return 'h264'
        return 'rawvideo'

    def pixelFormat(self):
        # we just return a fixed value for cases where 
        # the user wants to copy the input stream metadata
        return 'rgb24'
        # return 'yuv420p'

    def bitRate(self) -> float:
        # TODO: Figure out if this actually works since it's a hack
        #       i have no idea if itemsize of an nd2 frame
        #       returns the actual num bytes per array element
        return self.nd2[0].itemsize * 8 * self.width * self.height * self.avg_fps

    def isOpened(self) -> bool:
        return self.nd2 is not None

    def close(self):
        self.nd2.close()

    def videoFormat(self):
        return 'avi'

    def frameWidth(self) -> int:
        return self.width

    def frameHeight(self) -> int:
        return self.height

    def framesPerSecond(self) -> float:
        return self.nd2.frame_rate()

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

def rescaledToUint8(intensity, intensity_min, intensity_range) -> np.uint8:
    return np.uint8(255.0*(intensity - intensity_min)/intensity_range)
