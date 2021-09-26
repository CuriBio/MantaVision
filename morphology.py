#! /usr/bin/env python


import numpy as np
from cv2 import cv2 as cv # pip install --user opencv-python
from typing import Tuple, List, Dict
from track_template import bestMatch
from mantavision import getFilePathViaGUI
from skimage import filters as skimagefilters
from track_template import contrastAdjusted
from numpy.polynomial.polynomial import polyfit, Polynomial

from matplotlib import pyplot as plt


# TODO:

# - when doing multiple images, if we manually select ROIs, we use the same templates for all images

# - change the way we find the left magnets right edge, and right posts left edge
#   so that the initial templates can be roughly drawn just to find the magnet and fixed post
#   and then use another set of templates that have hard edges on the right of the magnet
#   and left of the fixed post that we use to refine those locations edges to compute the
#   length measurement end points.

# figure out a way to deal with images that have crap images i.e. tissue not being connected

# - compute a measure of waviness. need to compute the edge
# then fit splines to those edges and compute some measure of 
# squared error from the actual edges to the spline



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
    template_image = cv.imread(template_image_path)
    if template_image is None:
      print(f'ERROR. Could not open the template image pointed to by the path provided: {template_image_path}. Exiting.')
      return None

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
  search_image = cv.imread(search_image_path)
  if search_image is None:
    print(f'ERROR. Could not open the search image pointed to by the path provided: {search_image_path}. Exiting.')
    return None

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

  search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)
  search_image_gray = contrastAdjusted(search_image_gray) 

  # compute an estimate of the 'area' of the object of interest
  # first find the points we think area at the upper/lower edges  
  points_to_find_edges_at = np.asarray(range(left_distance_marker_x, right_distance_marker_x))
  points_to_fit_poly_at = np.asarray(range(pixel_distance_between_rois))  
  upper_edge_points = np.empty(pixel_distance_between_rois)
  lower_edge_points = np.empty(pixel_distance_between_rois)
  # TODO: remove edge_points and just use the upper and lower edge_points np arrays
  edge_points = []
  for index, point_to_find_edges_at in enumerate(points_to_find_edges_at):
    edge_point = outerEdges(search_image_gray[:, point_to_find_edges_at])    
    edge_points.append(edge_point)
    upper_edge_points[index] = edge_point['upper_edge_pos']
    lower_edge_points[index] = edge_point['lower_edge_pos']

  # TODO: maybe try to limit the extreme points by computing the median and x percentiles
  #       and then pulling in anything above the x percentile

  # then fit curves to those upper/lower "edge points" because it can be very noisy
  polyfit_deg = 4
  upper_edge_points_polyfit = polyfit(points_to_fit_poly_at, upper_edge_points, polyfit_deg)
  upper_edge_points_poly = Polynomial(upper_edge_points_polyfit)
  upper_edge_points = upper_edge_points_poly(points_to_fit_poly_at)
  lower_edge_points_polyfit = polyfit(points_to_fit_poly_at, lower_edge_points, polyfit_deg)
  lower_edge_points_poly = Polynomial(lower_edge_points_polyfit)
  lower_edge_points = lower_edge_points_poly(points_to_fit_poly_at)
  
  # now compute the actual area between the fitted curves
  edge_point_diffs = lower_edge_points - upper_edge_points
  area_between_rois = microns_per_pixel * np.sum(edge_point_diffs)

  # find distance between edges at the midpoint
  half_pixel_distance_between_rois = round(pixel_distance_between_rois/2)
  midpoint_upper_edge_position = lower_edge_points[half_pixel_distance_between_rois]
  midpoint_thickness = microns_per_pixel*(
    midpoint_upper_edge_position - upper_edge_points[half_pixel_distance_between_rois]
  )

  # find thickness at left roi inner edge
  left_end_point_thickness = microns_per_pixel*(lower_edge_points[0] - upper_edge_points[0])

  # find thickness at right roi inner edge
  right_end_point_thickness = microns_per_pixel*(lower_edge_points[-1] - upper_edge_points[-1])

  metrics = {
    'distance_between_rois': distance_between_rois,
    'midpoint_thickness': midpoint_thickness,
    'left_end_point_thickness': left_end_point_thickness,
    'right_end_point_thickness': right_end_point_thickness,
    'area_between_rois': area_between_rois 
  }

  # create a version of the input that has the results drawn on it
  # TODO: figure out if I have to make the results image a uint8
  #       becuase i don't think it's necessary.
  #       other than for adding colours.
  results_image = search_image.copy().astype(np.uint8)

  # draw the results metrics on the results image
  # draw the horizontal line between left and right ROI inner edges
  horizontal_line_position_colour_bgr = (255, 0, 0)
  upper_midpoint_pos_y = upper_edge_points[half_pixel_distance_between_rois]
  lower_midpoint_pos_y = lower_edge_points[half_pixel_distance_between_rois]
  horizontal_line_position_y = round(
    upper_midpoint_pos_y + (lower_midpoint_pos_y - upper_midpoint_pos_y)*0.5
  )
  cv.line(
    results_image,
    pt1=(left_distance_marker_x, horizontal_line_position_y),
    pt2=(right_distance_marker_x, horizontal_line_position_y),
    color=horizontal_line_position_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )

  # draw the upper and lower edges of object
  edge_contour_colour_bgr = (0, 0, 255)
  lower_edge_points_to_draw = np.dstack((points_to_find_edges_at.astype(np.int32), lower_edge_points.astype(np.int32)))[0]
  lower_edge_points_to_draw = lower_edge_points_to_draw.reshape((-1, 1, 2))
  cv.polylines(
    results_image,
    pts=[lower_edge_points_to_draw],
    isClosed=False,    
    color=edge_contour_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )
  upper_edge_points_to_draw = np.dstack((points_to_find_edges_at.astype(np.int32), upper_edge_points.astype(np.int32)))[0]
  upper_edge_points_to_draw = upper_edge_points_to_draw.reshape((-1, 1, 2))
  cv.polylines(
    results_image,
    pts=[upper_edge_points_to_draw],
    isClosed=False,
    color=edge_contour_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )

  # draw the left ROI inner edge object vertical thickness line
  edge_width_lines_colour_bgr = (0, 255, 0)
  left_end_point_upper_edge_pos = round(upper_edge_points[0])
  left_end_point_lower_edge_pos = round(lower_edge_points[0])
  cv.line(
    results_image,
    pt1=(left_distance_marker_x, left_end_point_lower_edge_pos),
    pt2=(left_distance_marker_x, left_end_point_upper_edge_pos),
    color=edge_width_lines_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )
  # draw the right ROI inner edge object vertical thickness line
  right_end_point_upper_edge_pos = round(upper_edge_points[-1])
  right_end_point_lower_edge_pos = round(lower_edge_points[-1]) 
  cv.line(
    results_image,
    pt1=(right_distance_marker_x, right_end_point_lower_edge_pos),
    pt2=(right_distance_marker_x, right_end_point_upper_edge_pos),
    color=edge_width_lines_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )
  # draw the horizontal midpoint object vertical thickness line
  midpoint_point_upper_edge_pos = round(upper_edge_points[half_pixel_distance_between_rois])
  midpoint_point_lower_edge_pos = round(lower_edge_points[half_pixel_distance_between_rois]) 
  cv.line(
    results_image,
    pt1=(left_distance_marker_x + half_pixel_distance_between_rois, midpoint_point_upper_edge_pos),
    pt2=(left_distance_marker_x + half_pixel_distance_between_rois, midpoint_point_lower_edge_pos),
    color=edge_width_lines_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )  
  
  return results_image, metrics 


def outerEdges(input_array: np.ndarray, show_plots: bool=False) -> int:
  '''
  Presumes there are two main edges on the outside of a
  horizontally aligned "tube/rectangle" like structure
  i.e. 
  ------------

  ------------
  If horizontal_pos is None, input_array should be a 1D array of intensities 
  (either raw or gradmag) i.e. a vertical cross section 1 pixel thick
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

  if show_plots:
    plt.plot(input_array_gradmag_ma, label='Gradient Magnitude of vertical line at midpoint', color = 'g')
    plt.show()

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
    # look for the second highest peak scaning left to right 
    # moving from left end right towards the max peak. 
    start_pos = 0
    end_pos = max_intensity_pos - 1
    increment = 1
  else:
    # look for the second highest peak scanning right to left 
    # moving from right end left towards the max peak. 
    start_pos = input_array_gradmag_ma_length + 1
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
  print(f"vertical thickness at midpoint between rois: {round(metrics['midpoint_thickness'], 2)} (microns)")
  print(f"area between rois: {round(metrics['area_between_rois'], 2)} (square microns)")
