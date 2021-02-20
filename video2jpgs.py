import argparse
import os
import sys
import cv2 as cv # pip install --user opencv-python
from skimage import filters as skimage_filters # pip install --user scikit-image
from skimage import exposure as skimage_exposure 


def contrast_enhanced(image_to_adjust):
  '''
  Performs an automatic adjustment of the input intensity range to enhance contrast.
  
  Args:
    image_to_adjust: the image to adjust the contrast of. 
  '''
  optimal_threshold = skimage_filters.threshold_yen(image_to_adjust)
  rgb_range = (0, 255)
  return skimage_exposure.rescale_intensity(image_to_adjust, in_range=(0, optimal_threshold), out_range=rgb_range)


def video_to_jpgs(input_video_path: str = None, output_dir_path: str = None, enhance_contrast: bool = False) -> int:
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

  # open the video capture stream
  video_stream = cv.VideoCapture(input_video_path)
  if not video_stream.isOpened():
    # try to open it once in case there was an initialization error
    video_stream.open()
  if not video_stream.isOpened():
    print("Error. Can't open videos stream for capture. Nothing has been converted.")
    return error_code

  # set the video basename for each frame
  frame_base_name = os.path.basename(input_video_path)

  # write out all the video frames as images
  number_of_frames = int(video_stream.get(cv.CAP_PROP_FRAME_COUNT))
  zero_padding_length = len(str(number_of_frames))
  for frame_number in range(number_of_frames):
    frame_returned, frame = video_stream.read()
    if not frame_returned:
      print("Error. Unexpected problem during video frame capture. Exiting.")
      return error_code
    frame_file_name = frame_base_name + "_frame_" + str(frame_number).zfill(zero_padding_length) + ".jpg"
    frame_path = os.path.join(output_dir_path, frame_file_name)
    auto_contrast_msg = ''
    if enhance_contrast:
      frame = contrast_enhanced(frame) 
      auto_contrast_msg = '(auto contrast enhanced)'
    cv.imwrite(frame_path, frame)
    print(f'Saved frame {frame_number} to {frame_path} {auto_contrast_msg}')

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
