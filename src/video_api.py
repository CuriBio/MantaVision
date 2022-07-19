 
import numpy as np
from typing import Dict
from fractions import Fraction
from av import open as OpenVideo
from av import VideoFrame
from nd2 import ND2File


supported_file_extensions = ['.mp4', '.avi', '.mov', '.nd2']


# TODO: figure out why stream initialisation doesn't work

class VideoWriter:
    """ Unified interface for writing videos """
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        time_base: Fraction,        
        fps: float,
        bitrate: float,
        codec: str = None,
        pixel_format: str = None
    ):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.codec = codec
        self.pixel_format = pixel_format
        self.time_base = time_base

        if self.width % 2 != 0:
            error_message = 'ERROR in constructing VideoWriter object.'
            error_message += '\nDimensions of a video are required to be a multiple of 2.'
            error_message += '\nThe requested width of ' + str(self.width) + ' is not a multiple of 2'
            raise ValueError(error_message)
        if self.height % 2 != 0:
            error_message = 'ERROR in constructing VideoWriter object.'
            error_message += '\nDimensions of a video are required to be a multiple of 2.'
            error_message += '\nThe requested height of ' + str(self.height) + ' is not a multiple of 2'
            raise ValueError(error_message)

        if self.pixel_format is None:
            self.pixel_format = 'yuv420p'  # 'yuv444p' #

        if self.codec is None:
            if '.mkv' in self.path:
                self.codec = 'ffv1'
            else:
                self.codec = 'libx264'

        if self.time_base is None:
            self.time_base = Fraction(1/1000)

        self.container = OpenVideo(self.path, mode='w')
        self.video_stream = self.container.add_stream(
            self.codec,
            rate=str(round(self.fps, 2))  # NOTE: this must be a str (not a float) and can't be too long 
        )
        # set some options for libx264 videos see https://trac.ffmpeg.org/wiki/Encode/H.264#crf for details
        if self.codec == 'libx264':
            self.video_stream.options['crf'] = '1'  # higher is more lossy compression, 0 is psuedo lossless             
            self.video_stream.options['preset'] = 'ultrafast'
            self.video_stream.options['tune'] = 'film'

        self.video_stream.codec_context.time_base = self.time_base
        self.video_stream.bit_rate = self.bitrate # can be small i.e. 2**20 & very still very viewable
        self.video_stream.pix_fmt = self.pixel_format
        self.video_stream.height = self.height
        self.video_stream.width = self.width

    def writeFrame(self, frame_to_write: np.ndarray, frame_pts: float):
        video_frame = VideoFrame.from_ndarray(frame_to_write, format='rgb24')
        video_frame.pts = frame_pts
        video_packet = self.video_stream.encode(video_frame)
        if video_packet:
            self.container.mux(video_packet)

    def close(self):
        # flush any remaining data in the output stream
        for video_packet in self.video_stream.encode():
            if video_packet.dts is None:
                continue
            try:
                self.container.mux(video_packet)
            except:
                print('Warning. an error occurred while closing the output stream. the output may be corrupt.')   
        self.container.close()


class VideoReader:
    """ Unified interface for reading videos of various formats ND2, mp4/avi ... """
    def __init__(self, video_path: str, reader_api: str = None, direction_sense: Dict = None):
        self.video_path = video_path
        self.reader = self._set_reader(reader_api, direction_sense)

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

    def _set_reader(self, reader_api: str, direction_sense: Dict = None):
        if reader_api is None:
            reader_api = self._apiFromPath()
        if reader_api == 'nd2':
            return ND2VideoReader(self.video_path, direction_sense)
        return PYAVReader(self.video_path, direction_sense)

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


class PYAVReader():
    """ Video reader interface using pyav for mp4, avi, mov etc (anything other than ND2) """
    def __init__(self, video_path: str, direction_sense: Dict = None):
        self.direction_sense = direction_sense
        self.video_path = video_path
        self.container = OpenVideo(self.video_path)
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
        return orientedFrame(
            self.current_frame.to_ndarray(),
            self.direction_sense
        )

    def frameRGB(self) -> np.ndarray:
        return orientedFrame(
            self.current_frame.to_ndarray(format='rgb24'),
            self.direction_sense
        )

    def frameVideoRGB(self) -> np.ndarray:
        # unlike nd2 files, video files won't ever need to have
        # their width or height adjusted to be a multiple of 2,
        # so we just use the same frameRGB()) method
        return self.frameRGB()

    def frameGray(self) -> np.ndarray:
        return orientedFrame(
            self.current_frame.to_ndarray(format='gray'),
            self.direction_sense
        )

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
        return float(self.video_stream.duration * self.timeBase())

    def codecName(self):
        return self.video_stream.codec_context.name

    def pixelFormat(self):
        return self.video_stream.pix_fmt

    def bitRate(self):
        """ bit rate in kpbs """
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
    """ Video reader interface using nd2 for ND2 files: https://github.com/tlambert03/nd2 """
    def __init__(self, video_path: str, direction_sense: Dict = None):
        self.video_path = video_path
        self.nd2_file = ND2File(self.video_path)
        self.is_rgb = self.nd2_file.is_rgb
        self.rgb_conversion = rgbConverter(type='eye')
        self.direction_sense = direction_sense
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
        self.nd2_frames = self.nd2_file.to_xarray(delayed=False)
        # if delayed is set to true, the xarray will be backed by a memmapped dask array
        # which might be necessary BUT,
        # it then prints something that seems like the frame or z slice i.e. (1,) (2,)
        # and I have no idea why or how to stop it from doing that

        self.time_base = Fraction(1, 1000)  # is always milliseconds for nd2 files
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
            current_frame = self.nd2_frames[self.frame_num].to_numpy()
        else:
            current_frame = grayToRGB(self.nd2_frames[self.frame_num].to_numpy())
        return orientedFrame(
            current_frame,
            self.direction_sense
        )

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
        frame_to_return[0:old_frame.shape[0], 0:old_frame.shape[1], :] = old_frame
        return orientedFrame(frame_to_return, self.direction_sense)

    def frameGray(self) -> np.ndarray:
        current_frame = self.nd2_frames[self.frame_num].to_numpy()
        if self.is_rgb:
            current_frame = rgbToGray(current_frame, self.rgb_conversion)
        return orientedFrame(current_frame, self.direction_sense)

    def framePTS(self, frame_num: int = None) -> int:
        if frame_num is None:
            frame_num = self.frame_num
        _chs = self.nd2_file._rdr._frame_metadata(frame_num).get('channels')
        time_stamp = _chs[0]['time']['relativeTimeMs']
        return time_stamp

    def frameVideoPTS(self, frame_num: int = None) -> int:
        if frame_num is None:
            frame_num = self.frame_num        
        return self.framePTS(frame_num) - self.framePTS(0)

    def timeStamp(self, frame_num: int = None) -> float:
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
        return 'libx264'  # 'ffv1'  # 'rawvideo'

    def pixelFormat(self):
        # we just return a fixed value for cases where 
        # the user wants to copy the input stream metadata
        return 'yuv420p'  # 'yuv444p'  # 'rgb24'

    def bitRate(self) -> float:
        """ estimated kpbs """
        bits_per_pixel = self.nd2_file.attributes.bitsPerComponentInMemory
        pixels_per_frame = self.width * self.height
        bits_per_frame = float(bits_per_pixel * pixels_per_frame)
        kilobits_per_frame = bits_per_frame / 1024.0
        # max_kbps = float(2**16 - 1)
        max_kbps = float(2**32 - 1)
        kbps = kilobits_per_frame * self.avg_fps
        return min(kbps, max_kbps)

    def isOpened(self) -> bool:
        return not self.nd2_file.closed

    def close(self):
        self.nd2_file.close()

    def frameWidth(self) -> int:
        return self.width

    def frameHeight(self) -> int:
        return self.height

    def frameVideoWidth(self) -> int:
        """ report a width that is increased if necessary to be multiple of 2 because
            frameVideoRGB increases the width if necessary to be a multiple of 2 """
        return self.width + (self.width % 2)

    def frameVideoHeight(self) -> int:
        """ report a height that is increased if necessary to be multiple of 2 because
            frameVideoRGB increases the height if necessary to be a multiple of 2 """
        return self.height + (self.height % 2)

    def framesPerSecond(self) -> float:
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


def orientedFrame(current_frame, direction_sense: Dict) -> np.ndarray:
    if direction_sense is not None:
        flip_directions = []
        if direction_sense['y'] < 0:
            flip_directions.append(0)
        if direction_sense['x'] < 0:
            flip_directions.append(1)
        # NOTE: the copy() that follows is necessary because
        # openCV can't use the "view" returned by np.flip()
        current_frame = np.flip(current_frame, flip_directions).copy()
    return current_frame


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
