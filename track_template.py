import argparse
import os
import sys
import json
import numpy as np
import pathlib
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
  uint8_range = (0, 255)
  return skimage_exposure.rescale_intensity(image_to_adjust, in_range=(0, optimal_threshold), out_range=uint8_range)


def track_template(input_video_path: str, template_image_path: str, output_video_path: str = None) -> {}:
  '''
  Tracks a template image through each frame of a video.

  Args:
    input_video_path:     path of the input video to track the template.
    template_image_path:  path to an image that will be used as a template to match.
  Returns:
    tuple with .
  '''
  if input_video_path is None:
    error_msg = "ERROR. No path provided to an input video. Nothing has been tracked."
    print(error_msg)
    return {'STATUS': 'FAILURE', 'STATUS_DETAIL': error_msg}    

  # open a video reader stream
  input_video_stream = cv.VideoCapture(input_video_path)
  if not input_video_stream.isOpened():
    # try to open it once in case there was an initialization error
    input_video_stream.open()
    if not input_video_stream.isOpened():
      error_msg = "Error. Can't open videos stream for capture. Nothing has been tracked."
      print(error_msg)
      return {'STATUS': 'FAILURE', 'STATUS_DETAIL': error_msg}

  # open the template as a gray scale image
  template = cv.cvtColor(cv.imread(template_image_path), cv.COLOR_BGR2GRAY)
  template_width = template.shape[::-1][0]
  template_height = template.shape[::-1][1]

  frames_per_second = input_video_stream.get(cv.CAP_PROP_FPS)

  # open a video writer stream if required
  if output_video_path is not None:
    format_extension = pathlib.Path(output_video_path).suffix
    if format_extension == '.avi':
      output_video_codec = cv.VideoWriter_fourcc('M','J','P','G')
    elif format_extension == '.mp4':
      output_video_codec = cv.VideoWriter_fourcc(*'mp4v') 
    else:
      error_msg = "Error. File format extension " + format_extension + " is not supported. "
      error_msg += "Only .mp4 and .avi are allowed. Nothing has been tracked."
      print(error_msg)
      return {'STATUS': 'FAILURE', 'STATUS_DETAIL': error_msg}

    output_video_codec = cv.VideoWriter_fourcc(*'mp4v') #cv.VideoWriter_fourcc(*'DIVX') #     
    frame_width  = int(input_video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    output_video_stream = cv.VideoWriter(
      output_video_path,
      output_video_codec,
      frames_per_second,
      frame_size,
      False  # isColor = True by default
    )
  else:
    output_video_stream = None

  # track the template in the video stream
  tracker_results = [] # will be a list of dicts
  number_of_frames = int(input_video_stream.get(cv.CAP_PROP_FRAME_COUNT))
  for frame_number in range(number_of_frames):
    frame_returned, frame = input_video_stream.read()
    if not frame_returned:
      error_msg = "Error. Unexpected problem during video frame capture. Exiting."
      print(error_msg)
      return {'STATUS': 'FAILURE', 'STATUS_DETAIL': error_msg}
    
    frame = contrast_enhanced(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).astype(np.uint8)

    # track the template
    match_results = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)    
    template_origin_x = match_coordinates[0]
    template_origin_y = match_coordinates[1]
    tracker_results.append(
      {
        'FRAME_NUMBER': frame_number,
        'ELAPSED_TIME': frame_number/frames_per_second,
        'MATCH_MEASURE': match_measure,
        'TEMPLATE_MATCH_ORIGIN_X': template_origin_x,
        'TEMPLATE_MATCH_ORIGIN_Y': template_origin_y,
        'TEMPLATE_WIDTH': template_width,
        'TEMPLATE_HEIGHT': template_height   
      }
    )

    if output_video_stream is not None:
      # draw a rectangle around the area we matched with the template
      template_bottom_right_x = template_origin_x + template_width    
      template_bottom_right_y = template_origin_y + template_height
      cv.rectangle(
        frame,
        (template_origin_x, template_origin_y),
        (template_bottom_right_x, template_bottom_right_y),
        color=125,
        thickness=1
      )
      output_video_stream.write(frame)

  input_video_stream.release()
  if output_video_stream is not None:
    output_video_stream.release()

  # TODO: go through all frames of the results, find the "highest" y point
  #       and add in a new entry for each dict that is the displacement in y
  #       which is computed as highest_y_point - match_origin_y
  return tracker_results


if __name__ == '__main__':

    # parse the input args
    parser = argparse.ArgumentParser(
        description='Tracks a template image through each frame of a video.',
    )
    parser.add_argument(
        'input_video_path',
        default=None,
        help='Path of the input video to track the templated.',
    )
    parser.add_argument(
        'template_image_path',
        default=None,
        help='Path to an image that will be used as a template to match.',
    )    
    parser.add_argument(
        'output_json_path',
        default=None,
        help='Path to write the output results json.',
    )
    parser.add_argument(
        'output_video_path',
        default=None,
        help='Path to write a video with the tracking results visualized.',
    )    
    args = parser.parse_args()

    output_json_path = args.output_json_path
    if output_json_path is None:
      print("ERROR. No output path provided to write results to. Nothing tracked.")
      sys.exit(1)

    output_video_path = args.output_video_path
    if output_video_path is None:
      print("ERROR. No output path provided to write video results to. Nothing tracked.")
      sys.exit(1)

    tracker_results = track_template(args.input_video_path, args.template_image_path, output_video_path)

    with open(output_json_path, 'w') as outfile:
        json.dump(tracker_results, outfile, indent=4)
