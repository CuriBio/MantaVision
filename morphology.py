#! /usr/bin/env python


import numpy as np
from cv2 import cv2 as cv # pip install --user opencv-python
from typing import Tuple, List, Dict
from track_template import bestMatch
from mantavision import getFilePathViaGUI
from skimage import filters as skimagefilters

from matplotlib import pyplot as plt


def roiInfoFromTemplates(
  search_image: np.ndarray,
  template_image_paths: List[str],
  sub_pixel_search_increment: float=None,
  sub_pixel_refinement_radius: float=None
) -> List[Dict]:
  ''' Finds the best match ROI for templates within search_image,
      and passes back the location information for all ROIs found.
  '''
  rois = []
  for template_image_path in template_image_paths:
    template_image_rgb = cv.imread(template_image_path)
    if template_image_rgb is None:
      print(f'ERROR. Could not open the template image pointed to by the path provided: {template_image_path}. Exiting.')
      return None
    # template_image = cv.cvtColor(template_image_rgb, cv.COLOR_BGR2GRAY)
    template_image = template_image_rgb

    _, match_coordinates = bestMatch(
      input_image_to_search=search_image,
      template_to_match=template_image,
      sub_pixel_search_increment=sub_pixel_search_increment,
      sub_pixel_refinement_radius=sub_pixel_refinement_radius
    )
    roi_origin_x, roi_origin_y = match_coordinates
    roi_height = template_image.shape[0]
    roi_width = template_image.shape[1]  
    roi_info = {
      'ORIGIN_X': roi_origin_x,
      'ORIGIN_Y': roi_origin_y,
      'WIDTH':  roi_width,
      'HEIGHT': roi_height
    }
    rois.append(roi_info)

  return rois


def roiInfoFromUserDrawings(input_image: np.ndarray) -> List[Dict]:
  '''
  Show the user a window with an image they can draw a ROI on.
  Args:
    input_image: the image to show the user.
  Returns:
    ROIs selected by the user from the input image.
  '''
  # create a window that can be resized
  roi_selector_window_name = "DRAW RECTANGULAR ROI"
  roi_gui_flags = cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL  # can resize the window
  cv.namedWindow(roi_selector_window_name, flags=roi_gui_flags)

  # open a roi selector in the resizeable window we just created
  roi_selections = cv.selectROIs(roi_selector_window_name, input_image, showCrosshair=False)
  cv.destroyAllWindows()
  print()

  rois = []
  for roi_selection in roi_selections:
    x_start = roi_selection[0]
    x_end = x_start + roi_selection[2]
    if x_end - x_start <= 0:
      return None
    y_start = roi_selection[1]
    y_end = y_start + roi_selection[3]
    if y_end - y_start <= 0:
      return None

    roi_info = {
      'ORIGIN_X': roi_selection[0],
      'ORIGIN_Y': roi_selection[1],
      'WIDTH':  roi_selection[2],
      'HEIGHT': roi_selection[3]
    }
    rois.append(roi_info)

  return rois


def morphologyMetrics(
  search_image_path: str=None,
  template_image_paths: List[str]=None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  microns_per_pixel: float=None,
  background_is_white: bool=True
):

  if microns_per_pixel is None:
    microns_per_pixel = 1.0

  if search_image_path is None or search_image_path.lower() == 'select':
    search_image_path = getFilePathViaGUI('image to search path')
  search_image_rgb = cv.imread(search_image_path)
  if search_image_rgb is None:
    print(f'ERROR. Could not open the search image pointed to by the path provided: {search_image_path}. Exiting.')
    return None
  # search_image = cv.cvtColor(search_image_rgb, cv.COLOR_BGR2GRAY)
  search_image = search_image_rgb

  if template_image_paths == 'draw':
    rois_info = roiInfoFromUserDrawings(search_image)
  else:
    if template_image_paths is None or template_image_paths == 'select':
      template_1_image_path = getFilePathViaGUI('template to find path')  
      template_2_image_path = getFilePathViaGUI('template to find path')  
      template_image_paths = [template_1_image_path, template_2_image_path],
    rois_info = roiInfoFromTemplates(
      search_image=search_image,
      template_image_paths=template_image_paths,
      sub_pixel_search_increment=None,
      sub_pixel_refinement_radius=None  
    )
  
  left_roi = rois_info.pop()
  right_roi = rois_info.pop()
  if right_roi['ORIGIN_X'] < left_roi['ORIGIN_X']:
    roi_to_swap = left_roi
    left_roi = right_roi
    right_roi = roi_to_swap
  # TODO: deal with more than 2 roi's being drawn??? maybe not.

  left_distance_marker_x = left_roi['ORIGIN_X'] + left_roi['WIDTH']
  right_distance_marker_x = right_roi['ORIGIN_X']
  pixel_distance_between_rois = right_distance_marker_x - left_distance_marker_x
  distance_between_rois = microns_per_pixel*pixel_distance_between_rois
  horizontal_midpoint = left_distance_marker_x + int(pixel_distance_between_rois/2)

  search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)
  horizontal_midpoint_vertical_intensities = search_image_gray[:, horizontal_midpoint]
  edges = outerEdges(horizontal_midpoint_vertical_intensities)
  lower_edge_pos = edges['lower_edge_pos']
  upper_edge_pos = edges['upper_edge_pos']
  distance_between_edges = microns_per_pixel*(edges['lower_edge_pos'] - edges['upper_edge_pos'])
  # Note: can do abs of distance_between_edges if need to but shouldn't need to

  # cut out a sub region around the area of interest and segmentation that
  distance_between_edges_radius = float(distance_between_edges)/2.0
  sub_region_vertical_start = int(max(0, upper_edge_pos - distance_between_edges_radius))
  sub_region_vertical_end = int(min(len(search_image_gray), lower_edge_pos + distance_between_edges_radius))
  sub_region = search_image_gray[
    sub_region_vertical_start:sub_region_vertical_end, left_distance_marker_x:right_distance_marker_x
  ]
  sub_region_threshold, _ = cv.threshold(src=sub_region, thresh=None, maxval=None, type=cv.THRESH_TRIANGLE)
  foreground_value = 1
  if background_is_white:
    threshold_type = cv.THRESH_BINARY_INV
  else:
    threshold_type = cv.THRESH_BINARY  
  _, sub_region_segmented = cv.threshold(sub_region, sub_region_threshold, foreground_value, threshold_type)

  # create a version of the input that has the results drawn on it
  # first we overlay the segmentaion results
  results_image_overlay = np.zeros(search_image_rgb.shape, np.int8)
  channel_to_overlay = 2  # blue channel
  results_image_overlay[
      sub_region_vertical_start:sub_region_vertical_end, left_distance_marker_x:right_distance_marker_x, channel_to_overlay
  ] = sub_region_segmented
  results_image = search_image_rgb.copy()
  colour_to_overlay = [0, 0, 128]
  results_image[results_image_overlay[:,:,channel_to_overlay] > 0] = colour_to_overlay
  # then we draw the horzontal and vertical measurements
  
  plt.imshow(results_image)
  plt.show()



  # find the average distance between the 'edges' of the thresholded object of interest
  # where the average is +/- some small horizontal region around the horizontal midpoint 
  sub_sub_region_horizontal_radius = 5
  sub_region_midpoint = pixel_distance_between_rois/2
  sub_sub_region_start_x = int(max(0, sub_region_midpoint - sub_sub_region_horizontal_radius))
  sub_sub_region_end_x = int(min(pixel_distance_between_rois, sub_region_midpoint + sub_sub_region_horizontal_radius))
  sub_sub_region_segmented = sub_region_segmented[:, sub_sub_region_start_x:sub_sub_region_end_x]  
  first_occurances = np.argmax(sub_sub_region_segmented, axis=0)
  sub_sub_region_segmented_height = sub_sub_region_segmented.shape[0]
  last_occurances = np.argmax(
    np.flip(sub_sub_region_segmented, axis=0),
    axis=0
  )
  last_occurances = sub_sub_region_segmented_height - last_occurances
  sub_sub_region_vertical_thicknesses = last_occurances - first_occurances
  midpoint_thickness_avg = np.median(sub_sub_region_vertical_thicknesses)

  # compute an estimate of the 'area' of the object of interest
  area_between_rois = microns_per_pixel*np.sum(sub_region_segmented)
  metrics = {
    'distance_between_rois': distance_between_rois,
    'midpoint_thickness_avg': midpoint_thickness_avg,
    'area_between_rois': area_between_rois 
  }
  return metrics


def outerEdges(input_array: np.ndarray) -> int:
  '''
  Presumes there are two main edges on the outside of a
  horizontally aligned "tube/rectangle" like structure
  i.e. 
  ------------

  ------------
  The input should be a vertical cross section 1 pixel thick
  so a 1D array of intensities (either raw or gradmag)
  '''

  # normalize the input
  input_array_min = np.min(input_array)
  input_array_max = np.max(input_array)
  input_array_range = input_array_max - input_array_min
  input_array_new_range = 512
  input_array_normalized = input_array_new_range * (
    (input_array - input_array_min)/float(input_array_range)
  )

  # compute a moving average (ma) smoothed gradmag of the input
  input_array_gradient = np.gradient(input_array_normalized, edge_order=2)
  input_array_gradmag = np.square(input_array_gradient)
  # compute a moving averge
  ma_radius = 5
  len_of_average = 2*ma_radius + 1
  input_array_gradmag_ma = np.convolve(
    input_array_gradmag,
    np.ones(len_of_average)/len_of_average,
    mode='valid'
  )

  # # PLOTS
  # plt.plot(input_array_gradmag_ma)
  # plt.show()

  # gradient_cumulative_sum = np.cumsum(input_array_gradient)
  # plt.plot(gradient_cumulative_sum, label='Cummulative Distribution of Gradient', color = 'b')
  # plt.show()

  # find what we presume is the edge (one of two) with the highest gradmag intensity
  max_intensity_pos = np.argmax(input_array_gradmag_ma)
  max_intensity = input_array_gradmag_ma[max_intensity_pos]
  half_max_intensity = max_intensity/2

  # find what we presume is the second edge, with lower intensity than the first one we found
  # if we know the correct direction, we can basically just keep looking for a peak
  # and when the intensity has dropped to half the value of the current max of a proper local max
  # we bail out and say we must have found the peak.
  input_array_gradmag_ma_length = len(input_array_gradmag_ma)
  upper_cumulative_sum = np.sum(input_array_gradmag_ma[:max_intensity_pos])
  lower_cumulative_sum = np.sum(input_array_gradmag_ma[max_intensity_pos:])
  if lower_cumulative_sum < upper_cumulative_sum:
    start_pos = max_intensity_pos + 1
    end_pos = input_array_gradmag_ma_length
    increment = 1
  else:
    # traverse 'backwards'
    start_pos = input_array_gradmag_ma_length - 1
    end_pos = max_intensity_pos
    increment = -1

  search_radius = 5
  other_peak_max_pos = start_pos
  other_peak_max_value = -np.inf
  for pos in range(start_pos, end_pos, increment):
    min_pos = max(0, pos - search_radius)
    max_pos = min(pos + search_radius + 1, input_array_gradmag_ma_length)

    search_sub_region_max_value = input_array_gradmag_ma[min_pos]
    search_sub_region_max_pos = min_pos
    for search_sub_region_pos in range(min_pos, max_pos):
      current_value = input_array_gradmag_ma[search_sub_region_pos]
      if current_value >= search_sub_region_max_value:
        search_sub_region_max_value = current_value
        search_sub_region_max_pos = search_sub_region_pos

    if search_sub_region_max_pos == pos:
      # local max so check it's a 'global' max with what we've checked so far
      if search_sub_region_max_value >= other_peak_max_value:
        other_peak_max_pos = search_sub_region_max_pos
        other_peak_max_value = search_sub_region_max_value

    if other_peak_max_value > half_max_intensity:
      # we arbitrarily decide the other peak must be at least half the global max
      if search_sub_region_max_value < other_peak_max_value/2:
        # if the current search values have dropped by half from the max 
        # we must be walking down the descending side of a peak and therefore 
        # we presume we have found the other peak so bail out of the search
        break  

  if max_intensity_pos < other_peak_max_pos:
    upper_edge_pos = max_intensity_pos
    lower_edge_pos = other_peak_max_pos
  else:
    upper_edge_pos = other_peak_max_pos
    lower_edge_pos = max_intensity_pos

  # adjust for the moving average clipping the ends off
  upper_edge_pos += ma_radius
  lower_edge_pos += ma_radius

  return {
      'lower_edge_pos': lower_edge_pos,
      'upper_edge_pos': upper_edge_pos,
  }


if __name__ == '__main__':
  
  metrics = morphologyMetrics(
    search_image_path=None,
    template_image_paths=None,
    sub_pixel_search_increment=None,
    sub_pixel_refinement_radius=None
  )

  print(f"horizontal inner distance between rois: {round(metrics['distance_between_rois'], 2)} (microns)")
  print(f"vertical thickness at midpoint between rois: {round(metrics['midpoint_thickness_avg'], 2)} (microns)")
  print(f"area between rois: {round(metrics['area_between_rois'], 2)} (square microns)")
