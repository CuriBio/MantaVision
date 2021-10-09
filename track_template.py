#! /usr/bin/env python

import numpy as np
from scipy.ndimage import shift
import math
import pathlib
from cv2 import cv2 as cv # pip install --user opencv-python
from typing import Tuple, List, Dict
import sys

import av
from fractions import Fraction

# TODO: parallelise the computation of matching for each frame. i.e. if we have 10 processors, split up the search space into
#       10 disjoint regions and have each thread process those regions independently then combine results
#       to find the min among them.

# TODO: for the contrast adjustment, maybe instead of taking the actual min after gamma adjusting
#       we should be taking the intensity value at the 2/5/?% in the increasing ordered image array
#       i.e. the 2nd percentile, or 5th percentile. The reason being that just a single very low value
#       in the gamma adjusted image can make the image extremely bright.
#       so then the question is, what do we do with values below that min, do we just set them to the
#       xth percentile or do we perform some sort of binning on the whole image...?

# TODO: the template we use in the algo should be called the roi_template and
#       the template we get from the user should be called guide_template.
#       so we get the guide_template and use it to find an roi_template in the video we're searching.

# TODO: in trackTemplate(), there needs to be a proper, named, return struct,
#       that is initialized on entry, it's contents updated as state changes. 
#       like when an error is returned, the named struct can be returned 
#       with sensible default values and just the updated error val.

# TODO: we could try adding in rotation of the template +/- some small range of angles for each position per frame.

def trackTemplate(
  input_video_path: str,
  template_guide_image_path: str,
  output_video_path: str = None,
  guide_match_search_seconds: float = None,
  microns_per_pixel: float = None,
  output_conversion_factor: float = None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  user_roi_selection: bool = True, 
  max_movement_per_frame = None
) -> Tuple[str, List[Dict], float, np.ndarray, int]:
  '''
  Tracks a template image through each frame of a video.

  Args:
    input_video_path:           path of the input video to track the template.
    template_guide_image_path:  path to an image that will be used as a template to match.
    output_video_path:          path to write a video with the tracking results visualized. 
    guide_match_search_seconds: approximate number of seconds for template to complete a full period of movement.',
  Returns:
    error string (or None if no errors occurred), 
    list of per frame tracking results, 
    frame rate,
    template actually used for the tracking
  '''

  if microns_per_pixel is None:
    microns_per_pixel = 1.0
  if output_conversion_factor is None:
    output_conversion_factor = 1.0

  error_msg = None
  warning_msg = None
  frames_per_second = float(0.0)  
  if input_video_path is None:
    error_msg = "ERROR. No path provided to an input video. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second, None, -1)

  # open a video reader stream
  input_video_stream = cv.VideoCapture(input_video_path)
  if not input_video_stream.isOpened():
    error_msg = "Error. Can't open videos stream for capture. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second, None, -1)
  frame_width  = int(input_video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
  frame_height = int(input_video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
  frames_per_second = input_video_stream.get(cv.CAP_PROP_FPS)
    
  # open the template image
  if user_roi_selection:
    template = None
  else:
    template = cv.imread(template_guide_image_path)
    if template is None:
      error_msg = "ERROR. The path provided for template does not point to an image file. Nothing has been tracked."
      return (error_msg, [{}], frames_per_second, None, -1)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
  
  if guide_match_search_seconds is None:
    max_frames_to_check = None
  else:
    max_frames_to_check = int(math.ceil(frames_per_second*float(guide_match_search_seconds)))

  template = templateFromInputROI(
    input_video_stream,
    template,
    max_frames_to_check,
    user_roi_selection=user_roi_selection
  )
  template_height = template.shape[0]
  template_width = template.shape[1]

  input_video_conainer = av.open(input_video_path)
  # input_video_duration = input_video_conainer.duration
  # input_video_time_base = input_video_conainer.streams.video[0].time_base
  input_video_avg_fps_fraction = input_video_conainer.streams.video[0].average_rate
  input_video_codec_name = input_video_conainer.streams.video[0].codec_context.name
  input_video_pix_fmt = input_video_conainer.streams.video[0].pix_fmt
  input_video_bitrate = input_video_conainer.streams.video[0].bit_rate
  input_video_conainer.close()

  # open a writable video stream if required
  if output_video_path is not None:
    output_video_bitrate = input_video_bitrate
    output_video_fps = input_video_avg_fps_fraction.numerator/input_video_avg_fps_fraction.denominator
    if input_video_codec_name == 'rawvideo':
      # mp4 doesn't support rawvideo and neither do a lot of players
      output_video_codec = 'h264'
      output_video_pix_fmt = 'yuv420p'
    else:
      output_video_codec = input_video_codec_name  # just use 'h264' if this ever fails
      output_video_pix_fmt = input_video_pix_fmt
    output_video_container = av.open(output_video_path, mode='w')
    output_video_stream = output_video_container.add_stream(output_video_codec, rate=str(round(output_video_fps, 2)))
    output_video_stream.codec_context.time_base = Fraction(1, 1000)  # milliseconds
    output_video_stream.bit_rate = output_video_bitrate # can be small i.e. 2**20 & very still very viewable
    output_video_stream.pix_fmt = output_video_pix_fmt  # use 'yuv420p' if this ever fails 
    output_video_stream.height = frame_height
    output_video_stream.width = frame_width
  else:
    output_video_stream = None

  # track the template in the video stream
  min_x_origin = (frame_width, frame_height)
  max_x_origin = (0, 0)
  min_x_frame = 0
  min_y_origin = (frame_width, frame_height)  
  max_y_origin = (0, 0)
  min_y_frame = 0

  tracking_results = [] # will be a list of dicts
  number_of_frames = int(input_video_stream.get(cv.CAP_PROP_FRAME_COUNT))
  best_match_origin_x = None
  best_match_origin_y = None
  match_points = []

  frame_number = 0
  frame_returned, raw_frame = input_video_stream.read()
  while frame_returned:
    # crop out a smaller sub region to search if required
    if max_movement_per_frame is None:
      sub_region_padding = None
    else:
      sub_region_padding = (
        math.ceil(max_movement_per_frame[0]/microns_per_pixel), math.ceil(max_movement_per_frame[1]/microns_per_pixel)
      )
    input_image_sub_region_to_search, input_image_sub_region_origin = inputImageSubRegion(
      input_image=raw_frame,
      sub_region_base_shape=(template_width, template_height),
      sub_region_origin=(best_match_origin_x, best_match_origin_y),
      sub_region_padding=sub_region_padding
    )
    match_measure, match_coordinates = matchResults(
      input_image_to_search=input_image_sub_region_to_search,
      template_to_match=template,
      sub_pixel_search_increment=sub_pixel_search_increment,
      sub_pixel_refinement_radius=sub_pixel_refinement_radius
    )

    best_match_origin_x = match_coordinates[0] + input_image_sub_region_origin[0]
    best_match_origin_y = match_coordinates[1] + input_image_sub_region_origin[1]

    original_time_stamp = input_video_stream.get(cv.CAP_PROP_POS_MSEC)
    milliseconds_per_second = 1000.0
    time_stamp_in_seconds = original_time_stamp/milliseconds_per_second
    tracking_results.append({
      'FRAME_NUMBER': frame_number,
      'FRAME_TIME': frame_number/frames_per_second,
      'TIME_STAMP': time_stamp_in_seconds,
      'MATCH_MEASURE': match_measure,
      'Y_DISPLACEMENT': 0,
      'X_DISPLACEMENT': 0,
      'XY_DISPLACEMENT': 0,
      'TEMPLATE_MATCH_ORIGIN_X': best_match_origin_x,
      'TEMPLATE_MATCH_ORIGIN_Y': best_match_origin_y,
    })
    match_points.append((best_match_origin_x, best_match_origin_y))

    # update the min and max positions of the template origin for ALL frames
    # using the position in the y dimension only as the reference measure
    if best_match_origin_y < min_y_origin[1]:
      min_y_origin = (best_match_origin_x, best_match_origin_y)
      min_y_frame = frame_number
    if best_match_origin_y > max_y_origin[1]:
      max_y_origin = (best_match_origin_x, best_match_origin_y)
    if best_match_origin_x < min_x_origin[0]:
      min_x_origin = (best_match_origin_x, best_match_origin_y)
      min_x_frame = frame_number
    if best_match_origin_x > max_x_origin[0]:
      max_x_origin = (best_match_origin_x, best_match_origin_y)

    # draw a grid over the region in the frame where the template matched
    if output_video_stream is not None:
      frame = raw_frame

      # opencv drawing functions require integer coordinates
      # so we convert them to the nearest pixel
      grid_origin_y = int(math.floor(best_match_origin_y + 0.5))
      grid_origin_x = int(math.floor(best_match_origin_x + 0.5))

      # mark the template ROI on each frame
      # first draw a rectangle around the template border
      grid_square_diameter = 20  # make the roi we draw a multiple of a set size
      grid_width = int(template_width/grid_square_diameter)*grid_square_diameter
      grid_height = int(template_height/grid_square_diameter)*grid_square_diameter
      template_bottom_right_y = min(frame_height, grid_origin_y + grid_height)
      template_bottom_right_x = min(frame_width, grid_origin_x + grid_width)
      grid_colour_bgr = (0, 255, 0)
      cv.rectangle(
        img=frame,
        pt1=(grid_origin_x, grid_origin_y),
        pt2=(template_bottom_right_x, template_bottom_right_y),
        color=grid_colour_bgr,
        thickness=1,
        lineType=cv.LINE_AA
      )

      output_video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
      output_video_frame.pts = original_time_stamp
      for output_video_packet in output_video_stream.encode(output_video_frame):
        output_video_container.mux(output_video_packet)

    frame_returned, raw_frame = input_video_stream.read()
    frame_number += 1

  if frame_number != number_of_frames:
    warning_msg = '\nWARNING.\n'
    warning_msg += ' Number of expected frames ' + str(number_of_frames)
    warning_msg += ' does not match actual number of frames ' + str(frame_number) + '.\n'

  if output_video_stream is not None:
    # flush any remaining data in the output stream
    for output_video_packet in output_video_stream.encode():
      output_video_container.mux(output_video_packet)   
    output_video_container.close()
  input_video_stream.release()

  extreme_points = [min_x_origin, min_y_origin, max_x_origin, max_y_origin]
  min_frame_numbers = (min_x_frame, min_y_frame)
  adjusted_tracking_results, min_frame_number = adjustTrackingResults(
    tracking_results=tracking_results,
    microns_per_pixel=microns_per_pixel,
    output_conversion_factor=output_conversion_factor,
    extreme_points=extreme_points,
    min_frame_numbers=min_frame_numbers
  )
 
  return ((warning_msg, error_msg), adjusted_tracking_results, frames_per_second, template, min_frame_number)


def adjustTrackingResults(
  tracking_results: List[Dict],
  microns_per_pixel: float,
  output_conversion_factor: float,
  extreme_points: List[Tuple[float, float]],
  min_frame_numbers: Tuple[int, int],
  contraction_vector: Tuple[int, int]=None
) -> List[Dict]:
  
  # TODO: currently, positive movement is defined as down or right
  #       we can use the contraction_vector parameter to define
  #       which direction of contration is positive. i.e.
  #       (0,  -1) means positive contractions are right to left,
  #       (-1, -1) means positive contractions are bottom right to top left.
  #       then readjust match origin coordinates accordingly so that
  #       XY displacement is reported as 0 to some positive value that direction
  min_x_origin, min_y_origin, max_x_origin, max_y_origin = extreme_points
  range_of_x_movement = max_x_origin[0] - min_x_origin[0]
  range_of_y_movement = max_y_origin[1] - min_y_origin[1]
  if range_of_x_movement > range_of_y_movement:
    main_movement_axis = 'x'
    min_template_origin_x = min_x_origin[0]
    min_template_origin_y = min_x_origin[1]
    min_frame_number = min_frame_numbers[0]
  else:
    main_movement_axis = 'y'
    min_template_origin_x = min_y_origin[0]
    min_template_origin_y = min_y_origin[1]
    min_frame_number = min_frame_numbers[1]
  print(f'main axis of movement detected along {main_movement_axis}')

  adjusted_tracking_results = []
  for frame_info in tracking_results:
    # re-adjust x and y positions so the min of the main axis of movement is the reference
    x_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_X'] - min_template_origin_x)*float(microns_per_pixel)
    x_displacement *= float(output_conversion_factor)
    frame_info['X_DISPLACEMENT'] = x_displacement
    y_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_Y'] - min_template_origin_y)*float(microns_per_pixel)
    y_displacement *= float(output_conversion_factor)
    frame_info['Y_DISPLACEMENT'] = y_displacement
    frame_info['XY_DISPLACEMENT'] = math.sqrt(x_displacement*x_displacement + y_displacement*y_displacement)
    adjusted_tracking_results.append(frame_info)
  return adjusted_tracking_results, min_frame_number


def inputImageSubRegion(
  input_image: np.ndarray,
  sub_region_base_shape: Tuple[int, int],
  sub_region_origin: Tuple[int, int],
  sub_region_padding: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
  ''' '''

  sub_region_origin_x, sub_region_origin_y = sub_region_origin
  if sub_region_origin_x is None or sub_region_origin_y is None:
    return input_image, (0,0)

  if sub_region_padding is None:
    return input_image, (0, 0)    
  sub_region_padding_x, sub_region_padding_y = sub_region_padding

  input_shape_x, input_shape_y = (input_image.shape[1], input_image.shape[0])
  sub_region_base_width, sub_region_base_height = sub_region_base_shape

  # define the sub region end points
  sub_region_start_x = math.floor(max(0, sub_region_origin_x - sub_region_padding_x))
  sub_region_end_x = math.ceil(min(
    input_shape_x, 
    sub_region_origin_x + sub_region_base_width + sub_region_padding_x
  ))
  sub_region_start_y = math.floor(max(0, sub_region_origin_y - sub_region_padding_y))
  sub_region_end_y = math.ceil(min(
    input_shape_y, 
    sub_region_origin_y + sub_region_base_height + sub_region_padding_y
  ))
  # crop out the new sub region
  sub_region = input_image[sub_region_start_y:sub_region_end_y, sub_region_start_x:sub_region_end_x]
  # return the new sub region and it's origin relateve to the input_image
  return sub_region, (sub_region_start_x, sub_region_start_y)


def templateFromInputROI(
  video_to_search,
  template_to_find,
  max_frames_to_check: int,
  user_roi_selection: bool=False,
) -> np.ndarray:
  '''
  '''
  initial_frame_pos = video_to_search.get(cv.CAP_PROP_POS_FRAMES)
  if initial_frame_pos != 0:
    video_to_search.set(cv.CAP_PROP_POS_FRAMES, 0)

  best_match_measure = 0.0
  best_match_coordinates = None
  best_match_frame = None
  number_of_frames = int(video_to_search.get(cv.CAP_PROP_FRAME_COUNT))
  if max_frames_to_check is None:
    number_of_frames_to_check = number_of_frames
  else:
    number_of_frames_to_check = min(number_of_frames, max_frames_to_check)
  for _ in range(number_of_frames_to_check):
    frame_returned, frame = video_to_search.read()
    if not frame_returned:
      error_msg = "Error. No Frame returned during video capture in templateFromInputROI function. Exiting."
      raise RuntimeError(error_msg)

    if user_roi_selection:
      print("Wating on user to manually select ROI...")
      roi = userDrawnROI(frame)
      if roi is None:
        print("...No ROI selected")
        continue
      else:
        print("...ROI selection complete")
        # reset the video to where it was initially before we return
        video_to_search.set(cv.CAP_PROP_POS_FRAMES, initial_frame_pos)      
        # return the selected roi
        return roi

    # track the template
    frame_adjusted = contrastAdjusted(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).astype(np.uint8)
    match_results = cv.matchTemplate(frame_adjusted, template_to_find, cv.TM_CCOEFF)
    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)
    if match_measure > best_match_measure:
      best_match_measure = match_measure
      best_match_coordinates = match_coordinates
      best_match_frame = frame

  # reset the video to where it was initially
  video_to_search.set(cv.CAP_PROP_POS_FRAMES, initial_frame_pos)
  if best_match_frame is None:
    error_msg = "ERROR. No ROI drawn by user for template. Cannot perform matching without a template. Exiting."
    raise RuntimeError(error_msg)

  # cut out a new best match template from the best match frame
  template_height = template_to_find.shape[0]
  template_width = template_to_find.shape[1]
  new_template_start_x = best_match_coordinates[0]
  new_template_end_x = new_template_start_x + template_width
  new_template_start_y = best_match_coordinates[1]
  new_template_end_y = new_template_start_y + template_height
  new_template = best_match_frame[new_template_start_y:new_template_end_y, new_template_start_x:new_template_end_x]
  return new_template


def userDrawnROI(input_image: np.ndarray) -> np.ndarray:
  '''
  Show the user a window with an image they can draw a ROI on.
  Args:
    input_image: the image to show the user.
  Returns:
    ROI selected by the user from the input image.
  '''
  # create a window that can be resized
  roi_selector_window_name = "DRAW RECTANGULAR ROI"
  roi_gui_flags = cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL  # can resize the window
  cv.namedWindow(roi_selector_window_name, flags=roi_gui_flags)

  # open a roi selector in the resizeable window we just created
  roi_selection = cv.selectROI(roi_selector_window_name, input_image, showCrosshair=False)
  cv.destroyAllWindows()

  x_start = roi_selection[0]
  x_end = x_start + roi_selection[2]
  if x_end - x_start <= 0:
    return None
  y_start = roi_selection[1]
  y_end = y_start + roi_selection[3]
  if y_end - y_start <= 0:
    return None

  return input_image[
    y_start:y_end,
    x_start:x_end,    
  ]


def contrastAdjusted(image_to_adjust: np.ndarray):
  '''
  Performs an automatic adjustment of the input intensity range to enhance contrast.
  
  Args:
    image_to_adjust: the image to adjust the contrast of. 
  '''
  image_stddev = np.std(image_to_adjust)
  gamma_value = 1.0/np.sqrt(np.log2(image_stddev))
  gamma_adjusted_image = gammaAdjusted(
    intensity=image_to_adjust,
    gamma=gamma_value
  )

  gamma_adjusted_image_min: float = np.min(gamma_adjusted_image)
  gamma_adjusted_image_max: float = np.max(gamma_adjusted_image)
  gamma_adjusted_image_range: float = gamma_adjusted_image_max - gamma_adjusted_image_min
  final_adjusted_image = rescaled(
    intensity=gamma_adjusted_image,
    intensity_min=gamma_adjusted_image_min,
    intensity_range=gamma_adjusted_image_range,
    new_scale=255.0
  ).astype(np.uint8)

  return final_adjusted_image


def gammaAdjusted(intensity: float, gamma: float, intensity_range: float=255.0) -> float:
  return intensity**gamma


def rescaled(intensity: float, intensity_min: float, intensity_range: float, new_scale: float = 1.0) -> float:
  return new_scale*(intensity - intensity_min)/intensity_range


def matchResults(
  input_image_to_search: np.ndarray,
  template_to_match: np.ndarray,
  sub_pixel_search_increment: float=None,
  sub_pixel_refinement_radius: int=None
) -> Tuple[float, List[float]]:
  '''
    Computes the coordinates of the best match between the input image and template.
    Accuracy is +/-1 pixel if sub_pixel_search_increment is None or >= 1.0.
    Accuracy is +/-sub_pixel_search_increment if |sub_pixel_search_increment| < 1.0 and not None.
  '''

  if sub_pixel_search_increment is not None and not abs(sub_pixel_search_increment) < 1.0:
    print('WARNING. Passing sub_pixel_search_increment >= 1.0 to bestMatch() is pointless. Ignoring.')
    sub_pixel_search_increment = None
  
  input_image = contrastAdjusted(cv.cvtColor(input_image_to_search, cv.COLOR_BGR2GRAY)).astype(np.uint8)
  template = contrastAdjusted(cv.cvtColor(template_to_match, cv.COLOR_BGR2GRAY)).astype(np.uint8)

  # find the best match for template in input_image_to_search
  match_results = cv.matchTemplate(input_image, template, cv.TM_CCOEFF)
  _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)  # opencv returns in x, y order

  if sub_pixel_search_increment is None:
    return (match_measure, match_coordinates)
  # else
  #   refine the results with a sub pixel search

  # convert images to appropriate types for sub pixel match step
  input_image = cv.cvtColor(input_image_to_search, cv.COLOR_BGR2GRAY).astype(np.float32)
  template = cv.cvtColor(template_to_match, cv.COLOR_BGR2GRAY).astype(np.float32)

  if sub_pixel_refinement_radius is None:
    sub_pixel_search_radius = 1
  else:
    sub_pixel_search_radius = sub_pixel_refinement_radius
  match_coordinates_origin_x = match_coordinates[0]
  match_coordinates_origin_y = match_coordinates[1]
  sub_region_y_start = max(
    match_coordinates_origin_y - sub_pixel_search_radius,
    0
  )
  sub_region_y_end = min(
    match_coordinates_origin_y + template.shape[0] + sub_pixel_search_radius, 
    input_image.shape[0]
  )
  sub_region_x_start = max(
    match_coordinates_origin_x - sub_pixel_search_radius,
    0
  )
  sub_region_x_end = min(
    match_coordinates_origin_x + template.shape[1] + sub_pixel_search_radius,
    input_image.shape[1]
  )
  match_measure, match_sub_coordinates = bestSubPixelMatch(
    input_image_to_search=input_image[
      sub_region_y_start:sub_region_y_end, 
      sub_region_x_start:sub_region_x_end
    ],
    template_to_match=template,
    search_increment=sub_pixel_search_increment
  )

  match_coordinates = [
    sub_region_x_start + match_sub_coordinates[0],
    sub_region_y_start + match_sub_coordinates[1]
  ]
  return (match_measure, match_coordinates)


def bestSubPixelMatch(
  input_image_to_search: np.ndarray,
  template_to_match: np.ndarray,  # must be .astype(np.float64)
  search_increment: float
) -> Tuple[float, List[float]]:
    '''
      Computes the coordinates of the best sub pixel match between the input image and template.
    '''
    input_dim_y, input_dim_x = input_image_to_search.shape
    template_dim_y, template_dim_x = template_to_match.shape
    search_length_y = input_dim_y - template_dim_y
    search_length_x = input_dim_x - template_dim_x
    max_match_measure = 0.0
    max_coordinates = [0.0, 0.0]
    shifted_input = np.ndarray(shape=[template_dim_y + 1, template_dim_x + 1], dtype=np.float32)
    for y_origin in np.arange(search_increment, search_length_y, search_increment):
      for x_origin in np.arange(search_increment, search_length_x, search_increment):
        sub_region_start_y = math.floor(y_origin)
        sub_region_end_y = sub_region_start_y + template_dim_y + 1
        sub_region_start_x = math.floor(x_origin)
        sub_region_end_x = sub_region_start_x + template_dim_x + 1        
        sub_image_to_shift = input_image_to_search[
          sub_region_start_y:sub_region_end_y,
          sub_region_start_x:sub_region_end_x,
        ]
        shift(input=sub_image_to_shift, shift=[-y_origin, -x_origin], output=shifted_input)
        input_to_match = shifted_input[:template_dim_y, :template_dim_x]
        match_results = cv.matchTemplate(input_to_match, template_to_match, cv.TM_CCOEFF)
        _, match_measure, _, _ = cv.minMaxLoc(match_results)
        if match_measure > max_match_measure:
          max_match_measure = match_measure
          max_coordinates = [x_origin, y_origin]

    return (max_match_measure, max_coordinates)


def fourccNum(c1, c2, c3, c4) -> int:
  return (
    (ord(c1) & 255) + ((ord(c2) & 255) << 8) + ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)
  )


def fourccChars(codec_num: int) -> List[str]:
  chars = []
  chars.append(chr(  codec_num & 255) )
  chars.append(chr( (codec_num >> 8) & 255 ))
  chars.append(chr( (codec_num >> 16) & 255 ))
  chars.append(chr( (codec_num >> 24) & 255 ))
  return chars
  # avc1 is the codec returned by videos from curi's microscope
