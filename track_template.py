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


def best_template_match(video_to_search, template_to_find) -> np.ndarray:
  new_template = None
  initial_frame_pos = video_to_search.get(cv.CAP_PROP_POS_FRAMES)
  if initial_frame_pos != 0:
    video_to_search.set(cv.CAP_PROP_POS_FRAMES, 0)

  best_match_measure = 0.0
  best_match_coordinates = ''
  best_match_frame = None
  number_of_frames = int(video_to_search.get(cv.CAP_PROP_FRAME_COUNT))
  for _ in range(number_of_frames):
    frame_returned, frame = video_to_search.read()
    if not frame_returned:
      print("Error. Unexpected problem during video frame capture. Exiting.")

    # track the template
    frame = contrast_enhanced(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).astype(np.uint8)
    match_results = cv.matchTemplate(frame, template_to_find, cv.TM_CCOEFF_NORMED)
    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)    
    if match_measure > best_match_measure:
      best_match_measure = match_measure
      best_match_coordinates = match_coordinates
      best_match_frame = frame

  # reset the video to where it was initially
  video_to_search.set(cv.CAP_PROP_POS_FRAMES, initial_frame_pos)

  # cut out the best match template from the best match frame
  template_width = template_to_find.shape[::-1][0]
  template_height = template_to_find.shape[::-1][1]  
  new_template_start_x = best_match_coordinates[0]
  new_template_end_x = new_template_start_x + template_width
  new_template_start_y = best_match_coordinates[1]
  new_template_end_y = new_template_start_y + template_height
  new_template = best_match_frame[new_template_start_y:new_template_end_y, new_template_start_x:new_template_end_x]
  
  return new_template


def track_template(
  input_video_path: str,
  template_image_path: str,
  output_video_path: str = None,
  template_as_guide: bool = False
) -> {}:
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
  if template_as_guide:
    template = best_template_match(input_video_stream, template)
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
      True
      # False  # isColor = True by default
    )
  else:
    output_video_stream = None

  # track the template in the video stream
  min_template_origin_x = frame_width
  max_template_origin_x = 0
  min_template_origin_y = frame_height
  max_template_origin_y = 0
  tracking_results = [] # will be a list of dicts
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

    tracking_results.append(
      {
        'FRAME_NUMBER': frame_number,
        'ELAPSED_TIME': frame_number/frames_per_second,
        'MATCH_MEASURE': match_measure,
        'Y_DISPLACEMENT_FROM_LOWEST_POINT': 0,
        'X_DISPLACEMENT_FROM_LEFTMOST_POINT': 0,
        'TEMPLATE_MATCH_ORIGIN_X': template_origin_x,
        'TEMPLATE_MATCH_ORIGIN_Y': template_origin_y,
        # 'TEMPLATE_WIDTH': template_width,
        # 'TEMPLATE_HEIGHT': template_height
      }
    )
    if template_origin_x < min_template_origin_x:
      min_template_origin_x = template_origin_x
    if template_origin_x > max_template_origin_x:
      max_template_origin_x = template_origin_x
    if template_origin_y < min_template_origin_y: 
      min_template_origin_y = template_origin_y
    if template_origin_y > max_template_origin_y:
      max_template_origin_y = template_origin_y

    if output_video_stream is not None:
      # TODO: put in bounds checks for drawing the grid
      grid_colour_bgr = (255, 128, 0)      
      grid_square_diamter = 10
      frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
      # mark the template ROI on each frame
      # first draw a rectangle around the template border
      grid_width = int(template_width/grid_square_diamter)*grid_square_diamter
      grid_height = int(template_height/grid_square_diamter)*grid_square_diamter
      template_bottom_right_x = template_origin_x + grid_width
      template_bottom_right_y = template_origin_y + grid_height
      cv.rectangle(
        frame,
        (template_origin_x, template_origin_y),
        (template_bottom_right_x, template_bottom_right_y),
        color=grid_colour_bgr,
        thickness=1
      )

      # then add horizontal grid lines within the rectangle
      line_start_x = template_origin_x + 1
      line_end_x = template_bottom_right_x - 1
      for line_pos_y in range(template_origin_y + grid_square_diamter, template_bottom_right_y, grid_square_diamter):
        cv.line(
          frame,
          (line_start_x, line_pos_y),
          (line_end_x, line_pos_y),
          color=grid_colour_bgr,
          thickness=1
        )
      
      # then add vertical grid lines within the rectangle 
      line_start_y = template_origin_y + 1
      line_end_y = template_bottom_right_y - 1
      for line_pos_x in range(template_origin_x + grid_square_diamter, template_bottom_right_x, grid_square_diamter):
        cv.line(
          frame,
          (line_pos_x, line_start_y),
          (line_pos_x, line_end_y),
          color=grid_colour_bgr,
          thickness=1,
          lineType=cv.LINE_AA
        )

      output_video_stream.write(frame)

  input_video_stream.release()
  if output_video_stream is not None:
    output_video_stream.release()

  adjusted_tracking_results = []
  for frame_info in tracking_results:
    frame_info['Y_DISPLACEMENT_FROM_LOWEST_POINT'] = max_template_origin_y - frame_info['TEMPLATE_MATCH_ORIGIN_Y']
    frame_info['X_DISPLACEMENT_FROM_LEFTMOST_POINT'] = frame_info['TEMPLATE_MATCH_ORIGIN_X'] - min_template_origin_x       
    adjusted_tracking_results.append(frame_info)
      
  return adjusted_tracking_results

# TODO: need to find a better brightness and contrast adjustment method.

# TODO: We need to be able to match a rotated and scaled magnet with our template.
#       We could use SIFT/SURF/ORB etc to find the template in the first video frame
#       and then warp the template image and use that with the current method
#       Of course we could just use SIFT/SURF/ORB etc for the whole thing if it's computationall feasible
#       and robust and doesn't require loads of twiddle factors.

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
    parser.add_argument(
        '-template_as_guide',
        action='store_true',
        help='Use the template as a guide to find the a template within the video.',
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

    tracking_results = track_template(
      args.input_video_path,
      args.template_image_path,
      output_video_path,
      args.template_as_guide
    )

    with open(output_json_path, 'w') as outfile:
        json.dump(tracking_results, outfile, indent=4)
