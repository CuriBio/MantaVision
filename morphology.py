#! /usr/bin/env python


import numpy as np
from cv2 import cv2 as cv # pip install --user opencv-python
from typing import Tuple, List, Dict
from track_template import bestMatch
from mantavision import getFilePathViaGUI
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


def computeMorphologyMetrics(
  search_image_path: str=None,
  template_image_paths: List[str]=None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  microns_per_pixel: float=None
):

  if microns_per_pixel is None:
    microns_per_pixel = 1.0

  if search_image_path is None:
    search_image_path = getFilePathViaGUI('image to search path')
  search_image_rgb = cv.imread(search_image_path)
  if search_image_rgb is None:
    print(f'ERROR. Could not open the search image pointed to by the path provided: {search_image_path}. Exiting.')
    return None
  # search_image = cv.cvtColor(search_image_rgb, cv.COLOR_BGR2GRAY)
  search_image = search_image_rgb

  if template_image_paths is None:
    rois_info = roiInfoFromUserDrawings(search_image)
  else:
    if template_image_paths == 'select':
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
  right_distance_marker = right_roi['ORIGIN_X']
  pixel_distance_between_rois = right_distance_marker - left_distance_marker_x
  distance_between_rois = microns_per_pixel*pixel_distance_between_rois
  midpoint_marker = left_distance_marker_x + int(pixel_distance_between_rois/2)

  search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)
  vertical_intensities = search_image_gray[:, midpoint_marker]
  edges = findEdges(vertical_intensities)
  computed_thickness = microns_per_pixel*(edges['lower_edge_pos'] - edges['upper_edge_pos'])
  # Note: can do abs of computed_thickness if need to but shouldn't need to

  lower_edge_pos = edges['lower_edge_pos']
  lower_edge_pos_value = edges['lower_edge_pos_value']
  upper_edge_pos = edges['upper_edge_pos']
  upper_edge_pos_value = edges['upper_edge_pos_value']
  print(f'upper_edge_pos: {upper_edge_pos} (value: {upper_edge_pos_value})')
  print(f'lower_edge_pos: {lower_edge_pos} (value: {lower_edge_pos_value})')
  print(f'inner distance (microns) between rois: {distance_between_rois}')
  print(f'computed thickness (microns) at midpoint: {computed_thickness}')

  # find max and max_position for actual peak i.e. the value is max for it's pixels left and right of it.
  # then find max and max_position (again, of actual peak) when searcing from left of image and from right of max
  # so in other words, we need to find a position that is a local max, AND has a value that is higher than any other peak we find.
  # that gives us 2 peaks and we take the second peak as the one that isn't the same as the first one we found

  # maybe we just use the cumsum of the original intensities
  # then we find can compute a piecewise smooth version which would have 
  # approx 3 pieces, 2 very steep and 1 close to horizontal. 
  # the close to horizontal region is the part we're interested in.
  # given how steep the edges are, could use a very crude set threshold
  # of the first place (from the left and the right) where the gradmag is > some fixed value
  # because we're normalizing, this could actually work.

  # TODO: implement and run segmentation and related functions
  #       from the locations, determine the horizontal separation, the midpoint and the thickness at the midpoint
  #       the thickness at the midpoint could be determined in a similar manner to the readcoor fiducial finder,
  #       or we could take the region between the rois, and using a horizontal line, use the intensity of that
  #       horizontal line as a guide to perform a segmentation, where the midline pixel intensities become the foreground
  #       draw a vertical line from the central point and find the darkest pixel on that line which becomes the background,
  #       then we run a k-means or 2 gaussian mixture model to segment. we could of course also just run otsu
  #       on some small region between the rois only (and probably some fixed vertical height)
  #       we could also just use gradients i.e. compute the gradient along the vertical line thorugh the 
  #       horizontal midpoint of the rois. then where the gradient becomes significantly more than some epsilon
  #       we consider that to be the edges.

  # I think the easiest thing to do is determine the intensity derivative and look up an down from the centra point
  # and choose the highest point above and the highest point below that central point as the edges
  # and calculate the length between each edge.
  # we can then also accumulate the intensities within and without for foreground/background
  # then segment the rest of the region between the rois based on those intensity values for the foreground


def findEdges(input_array: np.ndarray) -> int:
  '''
  Presumes there are two edges i.e. ______|__|_______
  '''

  input_array_min = np.min(input_array)
  input_array_max = np.max(input_array)
  input_array_range = input_array_max - input_array_min
  input_array_new_range = 512
  input_array_normalized = input_array_new_range * (
    (input_array - input_array_min)/float(input_array_range)
  )
  input_array_gradient = np.gradient(input_array_normalized, edge_order=2)
  input_array_gradmag = np.square(input_array_gradient)
  # compute a moving averge
  len_of_average = 10
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

  # gradmag_cumulative_sum = np.cumsum(input_array_gradmag)
  # plt.plot(gradmag_cumulative_sum, label='Cummulative Distribution of Gradmag', color = 'b')
  # plt.show()

  # input_array_2nd_gradient = np.gradient(input_array_gradient, edge_order=2)
  # plt.plot(input_array_2nd_gradient, label='input_array_2nd_gradient', color = 'b')
  # plt.show()




  # find what we presume is the edge (one of two) with the highest gradmag intensity
  max_intensity_pos = np.argmax(input_array_gradmag_ma)
  max_intensity = input_array_gradmag_ma[max_intensity_pos]
  half_max_intensity = max_intensity/2

  print(f'half_max_intensity: {half_max_intensity}')

  # find what we presume is the second edge, with lower intensity that the other one
  # if we know the correct direction, we can basically just keep looking for a peak
  # and when the intensity has dropped to half the value of the current max of a proper local max
  # we bail out and say we must have found the peak.
  # but how do we deal with the initial finding of a local max peak while still checking if we've dropped?
  # do we just make the initial peak max intensity None? or -ve?

  input_array_gradmag_ma_length = len(input_array_gradmag_ma)

  upper_cumulative_sum = np.sum(input_array_gradmag_ma[:max_intensity_pos])
  lower_cumulative_sum = np.sum(input_array_gradmag_ma[max_intensity_pos:])
  if lower_cumulative_sum < upper_cumulative_sum:
    print('searching from left to right')
    start_pos = max_intensity_pos + 1
    end_pos = input_array_gradmag_ma_length
    increment = 1
  else:
    print('searching from right to left')
    # traverse 'backwards'
    start_pos = input_array_gradmag_ma_length - 1
    end_pos = max_intensity_pos
    increment = -1

# TODO: the gradmag is smaller than the original image
#       so we need to figure out how it shrinks 
#       and account for that (i'm guessing if the radius of the ddx operator is 5, then we loose 10 pixel in total, 5 at each end)

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
        print('other peak found')
        break  # we must have found the other peak

  if max_intensity_pos < other_peak_max_pos:
    upper_edge_pos = max_intensity_pos
    upper_edge_pos_value = max_intensity
    lower_edge_pos = other_peak_max_pos
    lower_edge_pos_value = other_peak_max_value
  else:
    upper_edge_pos = other_peak_max_pos
    upper_edge_pos_value = other_peak_max_value
    lower_edge_pos = max_intensity_pos
    lower_edge_pos_value = max_intensity
  return {
      'lower_edge_pos': lower_edge_pos,
      'lower_edge_pos_value': lower_edge_pos_value,
      'upper_edge_pos': upper_edge_pos,
      'upper_edge_pos_value': upper_edge_pos_value,
  }


if __name__ == '__main__':
  computeMorphologyMetrics(
    search_image_path=None,
    template_image_paths=None,
    sub_pixel_search_increment=None,
    sub_pixel_refinement_radius=None
  )
