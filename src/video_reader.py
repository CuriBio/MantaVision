 
import av
import nd2
import numpy as np
from cv2 import cv2 as cv
from typing import Dict, List, Tuple


class VideoReader:
    ''' Unified interface for reading videos of various formats ND2, mp4/avi ... '''
    def __init__(self, video_path: str, reader_api=None):
        self.video_format_apis = [
            {'format': 'avi', 'api': 'cv'}, 
            {'format': 'mp4', 'api': 'cv'}, 
            {'format': 'nd2', 'api': 'nd2'}
        ]
        self.video_path = video_path
        if reader_api is None:
            for video_format_api in self.video_format_apis:
                if '.' + video_format_api['format'] in self.video_path:
                    reader_api = video_format_api['api']
            if reader_api is None:
                reader_api = 'cv'  # default to opencv video reader
        self.reader = self._select_reader(reader_api)

    def _select_reader(self, reader_api: str):
        if reader_api == 'nd2':
            return ND2Reader(self.video_path)
        else:
            return OpenCVReader(self.video_path)

    def isOpened(self) -> bool:
        return self.reader.isOpened()

    def release(self):
        self.reader.release()

    def setFramePosition(self, frame_position: int) -> bool:
        return self.reader.setFramePosition(frame_position)

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.reader.read()

    def frameWidth(self) -> int:
        return self.reader.frameWidth()

    def frameHeight(self) -> int:
        return self.reader.frameHeight()

    def framesPerSecond(self) -> float:
        return self.reader.framesPerSecond()

    def numFrames(self) -> int:
        return self.reader.numFrames()

    def framePosition(self) -> int:
        return self.reader.framePosition()

    def timeStamp(self) -> int:
        return self.reader.timeStamp()

    def avgFPS(self):
        return self.reader.avgFPS()

    def codecName(self):
        return self.reader.codecName()

    def pixFMT(self):
        return self.reader.pixFMT()

    def bitRate(self):
        return self.reader.bitRate()


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
        return self.avg_fps_fraction.numerator / self.avg_fps_fraction.denominator

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
