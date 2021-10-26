 
from fractions import Fraction
import av
import nd2
import numpy as np
from typing import Dict, List, Tuple


# TODO: add a readAsGrey() method that performs the conversion to gray scale
# TODO: change the name of read and readAsGray to nextFrame() and nextFrameGray()
# TODO: might want a nicer way to iterate too, so perhaps we can have nextFrame() which can return None
#       if there's no more frames, or nextFrame() only fetches the frame into a 
#       var in the object and returns True if there was a next frame and false otherwise,
#       and then the user needs to call frame() or frameGrey() to get the current frame  
#       
# TODO: add methods to get all the metadata we use, bitRate, codec etc
#       and get the pts, dts, time etc so we can 

# TODO: add a var to the VideoReader constructor called duplicate
#       that takes a VideoReader object and copies all it's settings 
#       to make the new VideoReader object.

# TODO: add a new a class VideoWriter (only using PyAv) because theres no api to write nd2 files anyway.


# TODO: remove this VideoAPI class since we'll only have av and nd2Reader libs
#       and therefore the only choice is, 
#       if .nd2 is in the filename, use nd2Reader api, otherewise, use PyAv api


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
            return ND2Reader(self.video_path)
        return AVReader(self.video_path)

    def isOpened(self) -> bool:
        return self.reader.isOpened()

    def setFramePosition(self, frame_position: int) -> bool:
        return self.reader.setFramePosition(frame_position)

    def frame(self) -> np.ndarray:
        return self.reader.frame()

    def frameGray(self) -> np.ndarray:
        return self.reader.frameGray()

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

    def pixFMT(self):
        return self.reader.pixFMT()

    def bitRate(self):
        return self.reader.bitRate()

    def reset(self):
        return self.reader.reset()

    def close(self):
        return self.reader.close()

    def videoFormat(self):
        return self.reader.videoFormat()


class AVReader():
    ''' Video reader interface using pyav for mp4, avi, mov etc (anything other than ND2)'''
    def __init__(self, video_path: str, return_grayscale: bool=True):
        self.format = 'gray' if return_grayscale else None
        self.video_path = video_path
        self.container = av.open(self.video_path)

        self.video_stream = self.container.streams.video[0]
        self.video_stream.thread_type = 'AUTO'  # go faster according to docs
        self.num_frames = self.video_stream.frames
                
        self.video_frames = self.container.decode()
        self.current_frame = None
        self.frame_num = -1
        self.next()  # get the first frame

    def next(self) -> bool:
        if self.frame_num < self.num_frames - 2:
            self.current_frame = next(self.video_frames)
            self.frame_num += 1
            return True
        else:
            self.current_frame = None
            self.frame_num = self.num_frames
            return False

    def reset(self):
        self.container.seek(0)
        self.frame_num = -1
        self.next()

    def frame(self) -> np.ndarray:
        return self.current_frame.to_ndarray()

    def frameGray(self) -> np.ndarray:
        return self.current_frame.to_ndarray(format=self.format)

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

    def pixFMT(self):
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
            self.reset()
        while self.frame_num < frame_position:
            self.next()
        return True



# TODO: figure out if the format of these ND2 files is [z, y, x, [b, g, r]] or
#       [z, T, y, x, [b,g,r]] or perhaps [z, T, [b,g,r], y, x]
# perhaps the T isn't there for some videos? or perhaps it's always there
# if it is there then we need to account for this when accessing the array obviously
# same with the RGB,  

class ND2Reader():
    ''' Video reader interface using nd2 for ND2 files: https://github.com/tlambert03/nd2 '''
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.nd2 = nd2.ND2File(self.video_path)
        self.num_frames = int(self.nd2.sizes['T'])
        self.num_channels = int(self.nd2.sizes['C'])
        self.height = int(self.nd2.sizes['Y'])
        self.width = int(self.nd2.sizes['X'])
        self.is_rgb = self.nd2.is_rgb
        if self.is_rgb:
            self.num_colours = int(self.nd2.sizes['S'])
        else:
            self.num_colours = 0
        self.avg_fps = 1
        self.codec_name = 'rawvideo'
        self.pix_fmt = 'yuv420p'
        self.bitrate = 1  # bits per frame * frames_per_second ?
        
        self.frame_position = 0
        self.video_stream = self.nd2.to_xarray(delayed=True)

    def avgFPS(self):
        return self.avg_fps

    def codecName(self):
        return self.codec_name

    def pixFMT(self):
        return self.pix_fmt

    def bitRate(self):
        return self.bitrate

    def isOpened(self) -> bool:
        return not self.video_stream.closed()

    def release(self):
        self.video_stream.close()

    def setFramePosition(self, frame_position: int) -> bool:
        if frame_position >= 0 and frame_position < self.num_frames:
            self.frame_position = frame_position
            return True
        else:
            return False

    def read(self, next=True) -> Tuple[bool, np.ndarray]:
        if self.frame_position < self.num_frames:
            current_frame = self.video_stream[self.frame_position, :]
            self.frame_position += 1
            return (True, current_frame)
        else:            
            return (False, None)

    def frameWidth(self) -> int:
        return self.width

    def frameHeight(self) -> int:
        return self.height

    def framesPerSecond(self) -> float:
        return 1  # TODO

    def numFrames(self) -> int:
        return self.num_frames

    def framePosition(self) -> int:
        return self.frame_position

    def timeStamp(self) -> int:
        return 0  # TODO



class OpenCVReader():
    ''' Video reader interface using opencv for mp4, avi (currently anything other than ND2)'''
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_stream = cv.VideoCapture(self.video_path)
        self.num_frames = int(self.video_stream.get(cv.CAP_PROP_FRAME_COUNT))
        # meta data
        input_video_conainer = av.open(self.video_path)
        self.duration = input_video_conainer.duration
        self.time_base = input_video_conainer.streams.video[0].time_base
        self.avg_fps_fraction = input_video_conainer.streams.video[0].average_rate
        self.codec_name = input_video_conainer.streams.video[0].codec_context.name
        self.pix_fmt = input_video_conainer.streams.video[0].pix_fmt
        self.bitrate = input_video_conainer.streams.video[0].bit_rate
        input_video_conainer.close()

    def avgFPS(self):
        return float(self.avg_fps_fraction)

    def codecName(self):
        return self.codec_name

    def pixFMT(self):
        return self.pix_fmt

    def bitRate(self):
        return self.bitrate

    def isOpened(self) -> bool:
        return self.video_stream.isOpened()

    def release(self):
        self.video_stream.release()

    def setFramePosition(self, frame_position: int) -> bool:
        if frame_position >= 0 and frame_position < self.num_frames:
            self.video_stream.set(cv.CAP_PROP_POS_FRAMES, frame_position)
            return True
        else:
            return False

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.video_stream.read()

    def frameWidth(self) -> int:
        return int(self.video_stream.get(cv.CAP_PROP_FRAME_WIDTH))

    def frameHeight(self) -> int:
        return int(self.video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))

    def framesPerSecond(self) -> float:
        return self.video_stream.get(cv.CAP_PROP_FPS)

    def numFrames(self) -> int:
        return self.num_frames

    def framePosition(self) -> int:
        return self.video_stream.get(cv.CAP_PROP_POS_FRAMES)

    def timeStamp(self) -> int:
        return self.video_stream.get(cv.CAP_PROP_POS_MSEC)
