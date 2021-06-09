#! /usr/bin/env python

import numpy as np
from scipy.ndimage import shift
import math
import pathlib
import cv2 as cv # pip install --user opencv-python


# TODO: add a parameter for the direction of contraction and allow down, right, up, down, left, right
#       from this we just rotate the image 0, 90, 180, -90 or 270 degrees

# TODO: try implementing our own match measure. i.e.
#       normalize each image,
#       compute the diffs of the two overlapping image sections,
#       compute the variance i.e. average of the diffs squared
#       could also introduce additional measures like gradient orientation, MI. etc etc.
#
# OR
#
# TODO: experiment with using a CC method that isn't computeECC() to make it faster

# TODO: we could introduce a max movement parameter that limited how far from the last results
#       we look for a new match. i.e. if after the first frame we find a best match at (x, y)
#       and max_movement = 50 pixels, then we'd look within the region: x - 50 and x + 50, and y - 50 and y + 50

# TODO: parallelise the check of each frame. i.e. if we have 10 processors, split up the search space into
#       10 disjoint regions and have each thread process those regions independently then combine results
#       to find the min among them.

# TODO: we could try adding in rotation of the template +/- some small range of angles for each position per frame.

# TODO: the template we use in the algo should be called the roi_template and
#       the template we get from the user should be called guide_template.
#       so we get the guide_template and use it to find an roi_template in the video we're searching.

# TODO: in trackTemplate(), there needs to be a proper, named, return struct,
#       that is initialized on entry, it's contents updated as state changes. 
#       like when an error is returned, the named struct can be returned 
#       with sensible default values and just the updated error val.


def trackTemplate(
  input_video_path: str,
  template_guide_image_path: str,
  output_video_path: str = None,
  guide_match_search_seconds: float = None,
  microns_per_pixel: float = None,
  output_conversion_factor: float = None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  user_roi_selection: bool = True
) -> (str, [{}], float, np.ndarray):
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
  error_msg = None
  frames_per_second = float(0.0)

  if input_video_path is None:
    error_msg = "ERROR. No path provided to an input video. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second, None)

  # open a video reader stream
  input_video_stream = cv.VideoCapture(input_video_path)
  if not input_video_stream.isOpened():
    error_msg = "Error. Can't open videos stream for capture. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second, None)
  frame_width  = int(input_video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
  frame_height = int(input_video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
  frame_size = (frame_width, frame_height)
  frames_per_second = input_video_stream.get(cv.CAP_PROP_FPS)

  # open the template image
  if user_roi_selection:
    template = None
  else:
    template = cv.imread(template_guide_image_path)
    if template is None:
      error_msg = "ERROR. The path provided for template does not point to an image file. Nothing has been tracked."
      return (error_msg, [{}], frames_per_second, None)
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
      return (error_msg, [{}], frames_per_second, None)

    output_video_codec = cv.VideoWriter_fourcc(*'mp4v') #cv.VideoWriter_fourcc(*'DIVX') #     
    output_video_stream = cv.VideoWriter(
      output_video_path,
      output_video_codec,
      frames_per_second,
      frame_size,
      True # False  # isColor = True by default
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

    frame_returned, raw_frame = input_video_stream.read()
    if not frame_returned:
      error_msg = "Error. Unexpected problem occurred during video frame capture. Exiting."
      return (error_msg, [{}], frames_per_second, None)
    
    match_measure, match_coordinates = bestMatch(
      input_image_to_search=raw_frame,
      template_to_match=template,
      sub_pixel_search_increment=sub_pixel_search_increment,
      sub_pixel_refinement_radius=sub_pixel_refinement_radius
    )

    best_match_origin_x = match_coordinates[0]  
    best_match_origin_y = match_coordinates[1]
    tracking_results.append(
      {
        'FRAME_NUMBER': frame_number,
        'ELAPSED_TIME': frame_number/frames_per_second,
        'MATCH_MEASURE': match_measure,
        'Y_DISPLACEMENT': 0,
        'X_DISPLACEMENT': 0,
        'XY_DISPLACEMENT': 0,
        'TEMPLATE_MATCH_ORIGIN_X': best_match_origin_x,
        'TEMPLATE_MATCH_ORIGIN_Y': best_match_origin_y,
      }
    )

    # update the min and max positions of the template origin for ALL frames
    # using the position in the y dimension only as the reference measure 
    if best_match_origin_y < min_template_origin_y: 
      min_template_origin_y = best_match_origin_y
      min_template_origin_x = best_match_origin_x
    if best_match_origin_y > max_template_origin_y:
      max_template_origin_y = best_match_origin_y
      max_template_origin_x = best_match_origin_x

    # draw a grid over the region in the frame where the template matched
    if output_video_stream is not None:
      # opencv drawing functions require integer coordinates
      # so we convert them to the nearest pixel
      grid_origin_y = int(math.floor(best_match_origin_y + 0.5))
      grid_origin_x = int(math.floor(best_match_origin_x + 0.5))

      grid_colour_bgr = (255, 128, 0)      
      grid_square_diameter = 20
      frame = raw_frame

      # mark the template ROI on each frame
      # first draw a rectangle around the template border
      grid_width = int(template_width/grid_square_diameter)*grid_square_diameter
      grid_height = int(template_height/grid_square_diameter)*grid_square_diameter
      template_bottom_right_x = min(frame_width, grid_origin_x + grid_width)
      template_bottom_right_y = min(frame_height, grid_origin_y + grid_height)
      cv.rectangle(
        img=frame,
        pt1=(grid_origin_x, grid_origin_y),
        pt2=(template_bottom_right_x, template_bottom_right_y),
        color=grid_colour_bgr,
        thickness=1,
        lineType=cv.LINE_AA
      )

      output_video_stream.write(frame)

  input_video_stream.release()
  if output_video_stream is not None:
    output_video_stream.release()

  # adjust the displacement values relative to a fixed orientation
  # the top most position is always zero for movement in the y dimension, 
  # zero for movement in the x dimension depends on whether the tracking movement
  # is top left to bottom right, or bottom left to top right
  # and of course compute the actual displacement in x-y i.e euclidean distance
  if min_template_origin_x > max_template_origin_x:
    positive_x_movement_slope = True
  else:
    positive_x_movement_slope = False
  adjusted_tracking_results = []
  if microns_per_pixel is None:
    microns_per_pixel = 1.0
  if output_conversion_factor is None:
    output_conversion_factor = 1.0
  output_conversion_factor
  for frame_info in tracking_results:
    y_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_Y'] - min_template_origin_y)*float(microns_per_pixel)
    y_displacement *= float(output_conversion_factor)
    frame_info['Y_DISPLACEMENT'] = y_displacement
    if positive_x_movement_slope:
      x_displacement = (max_template_origin_x - frame_info['TEMPLATE_MATCH_ORIGIN_X'])*float(microns_per_pixel)
    else:
      x_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_X'] - min_template_origin_x)*float(microns_per_pixel)
    x_displacement *= float(output_conversion_factor)
    frame_info['X_DISPLACEMENT'] = x_displacement
    frame_info['XY_DISPLACEMENT'] = math.sqrt(x_displacement*x_displacement + y_displacement*y_displacement)
    adjusted_tracking_results.append(frame_info)

  return (error_msg, adjusted_tracking_results, frames_per_second, template)



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
    error_msg = "ERROR. No ROI drawn by user for template. Cannot perform matcning without a template. Exiting."
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
  '''
  # create a window that can be resized
  roi_selector_window_name = "DRAW RECTANGULAR ROI"
  # roi_gui_flags = cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL  # can resize the window
  # roi_gui_flags = cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO | cv.WINDOW_AUTOSIZE  # can not resize the window
  roi_gui_flags = cv.WINDOW_GUI_NORMAL  # cv.WINDOW_GUI_EXPANDED
  cv.namedWindow(roi_selector_window_name, flags=roi_gui_flags)
  cv.setWindowProperty(roi_selector_window_name, cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
  # cv.setWindowProperty(roi_selector_window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_NORMAL)
  cv.setWindowProperty(roi_selector_window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)

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


def bestMatch(
  input_image_to_search: np.ndarray,
  template_to_match: np.ndarray,
  sub_pixel_search_increment: float=None,
  sub_pixel_refinement_radius: int=None
) -> (float, [float, float]):
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
  input_image = cv.cvtColor(input_image_to_search, cv.COLOR_BGR2GRAY).astype(np.float64)
  template = cv.cvtColor(template_to_match, cv.COLOR_BGR2GRAY).astype(np.float64)

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
  match_measure, match_sub_coordinates  = bestSubPixelMatch(
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
) -> (float, [float, float]):
    '''
      Computes the coordinates of the best sub pixel match between the input image and template.
    '''
    # the shift function turns the input image into a float64 dtype, and
    # computeECC requires both template & input to be the same type
    error_msg = ""
    if input_image_to_search.dtype != np.float64:
      error_msg += "input_image_to_search must be of type np.float64\n"
    if template_to_match.dtype != np.float64:
      error_msg += "template_to_match must be of type np.float64\n"
    if len(error_msg) > 0:
      error_msg += "ERROR. In function bestSubPixelMatch:\n" + error_msg
      raise TypeError(error_msg)

    input_dim_y, input_dim_x = input_image_to_search.shape
    template_dim_y, template_dim_x = template_to_match.shape
    search_length_y = input_dim_y - template_dim_y
    search_length_x = input_dim_x - template_dim_x
    max_ecc = 0.0
    max_coordinates = [0.0, 0.0]
    shifted_input = np.ndarray(shape=[template_dim_y + 1, template_dim_x + 1], dtype=np.float64)
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

        ecc = cv.computeECC(templateImage=template_to_match, inputImage=input_to_match)
        if ecc > max_ecc:
          max_ecc = ecc
          max_coordinates = [x_origin, y_origin]

    return (max_ecc, max_coordinates)
