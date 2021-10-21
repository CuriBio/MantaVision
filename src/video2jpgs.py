import argparse
import os
import sys
from cv2 import cv2 as cv # pip install --user opencv-python
from track_template import intensityAdjusted, rescaled


def video_to_jpgs(
  input_video_path: str=None,
  output_dir_path: str=None,
  enhance_contrast: bool=False,
  frame_number_to_write: int=None,
  image_extension: str='tif'
) -> int:
  '''
  Converts an input video to a sequence of jpg images.

  Args:
    input_video_path:  path of the input video to be converted.
    output_dir_path:   path to a directory where the output images are to be saved.
  Returns:
    A simple status code, 0 for success, anything else for an error.
  '''
  # check that all the input parameters have been provided and are valid
  error_code = 1
  conversion_success = 0
  if input_video_path is None:
    print("ERROR. No path provided to an input video. Nothing has been converted.")
    return error_code
  if output_dir_path is None:
    print("ERROR. No output directory provided to write output images. Nothing has been converted.")
    return error_code
  if not os.path.isdir(output_dir_path):
    print("ERROR. Output directory provided does not exist. Nothing has been converted.")
    return error_code
  if image_extension != "jpg" and image_extension != 'tif' and image_extension != 'png':
    print(f'ERROR. extension {image_extension} for output images not valid. Nothing has been converted.')
    return error_code    

  # open the video capture stream
  video_stream = cv.VideoCapture(input_video_path)
  if not video_stream.isOpened():
    print("Error. Can't open videos stream for capture. Nothing has been converted.")
    return error_code

  # set the video basename for each frame
  frame_base_name = os.path.basename(input_video_path)

  # write out the video frames as images
  number_of_frames = int(video_stream.get(cv.CAP_PROP_FRAME_COUNT))
  zero_padding_length = len(str(number_of_frames))  
  if frame_number_to_write is not None: # only write the single frame
    video_stream.set(cv.CAP_PROP_POS_FRAMES, frame_number_to_write)
    frame_returned, frame = video_stream.read()
    if not frame_returned:
      print("ERROR. Unexpected problem during video frame capture. Exiting.")
      return error_code
    frame_file_name = frame_base_name + "_frame_" + str(frame_number_to_write).zfill(zero_padding_length) + "." + image_extension
    frame_path = os.path.join(output_dir_path, frame_file_name)
    if enhance_contrast:
      frame = intensityAdjusted(frame) 
    cv.imwrite(frame_path, frame)
  else: # write out all the frames
    frame_number = 0
    frame_returned, frame = video_stream.read()
    while frame_returned:
      frame_file_name = frame_base_name + "_frame_" + str(frame_number).zfill(zero_padding_length) + "." + image_extension
      frame_path = os.path.join(output_dir_path, frame_file_name)
      # auto_contrast_msg = ''
      if enhance_contrast:
        frame = intensityAdjusted(frame) 
      cv.imwrite(frame_path, frame)
      frame_returned, frame = video_stream.read()
      frame_number += 1
    video_stream.release()
    if frame_number != number_of_frames:
      warning_msg = '\nWARNING.\n'
      warning_msg += ' Number of expected frames ' + str(number_of_frames)
      warning_msg += ' does not match actual number of frames ' + str(frame_number) + '.\n'

  return conversion_success


if __name__ == '__main__':

    # parse the input args
    parser = argparse.ArgumentParser(
        description='convert a video to a sequence of images',
    )
    parser.add_argument(
        'input_video_path',
        default=None,
        help='path of the input video to be converted.',
    )
    parser.add_argument(
        'output_dir_path',
        default=None,
        help='path to a directory where the output images are to be saved.',
    )
    parser.add_argument(
        '-enhance_contrast',
        action='store_true',
        help='automatically adjust intensity to enhance contrast.',
    )    
    args = parser.parse_args()

    # convert the video to a sequence of jpegs
    conversion_status = video_to_jpgs(args.input_video_path, args.output_dir_path, args.enhance_contrast)
    sys.exit(conversion_status)
