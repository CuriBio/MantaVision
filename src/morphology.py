#! /usr/bin/env python


import numpy as np
from cv2 import cv2 as cv # pip install --user opencv-python
from typing import Tuple, List, Dict
from track_template import matchResults
from mantavision import getFilePathViaGUI
from skimage import filters as skimagefilters
from track_template import intensityAdjusted
from numpy.polynomial.polynomial import polyfit, Polynomial

from matplotlib import pyplot as plt


# TODO: what we need is some form of background subtraction?
#       if we can remove the post section and it's rings?
#       or at least account for it.
#       but again the problem is, the images aren't consistent 
#       so how to do we know when those rings are there and 
#       how strong they are and how much to "remove" etc.???

# TODO: we could walk out along the outer edges from the vertical centreline and
#       if the next pixels edge is "continuous" with the current pixel
#       i.e. within +/- 1 or max 2 pixels, it's allowed otherwise we leave it empty
#       so we really need to be checking that the current pixels is within x_diff * (+/- tolerance) 
#       but the problem is that this isn't really any different from doing a poly fit of order 2
#       and using that as the guide, because if the edges get messed up at the point where the curve
#       starts bending, our polyfit won't pick that up and we'll just get a straight line


# TODO:
# do a polynomial fit with 1, 2, 3, or 4 segments, and degree 2, 3, or 4.
# then for each edge pixel, compute the abs difference between the detected edge position
# and the position in the polyfit. we then remove any pixels with an edge position tha tis
# > x% away for the polyfit position at that pixel (where x% is determined by the height/dims
# of the image and converted to a specific pixel amount. should probably also have some default
# min and max pixels.). So then, we go and fill back in each of the removed pixels by:
# - searching left and right of the end points that are still there, picking 
#   a small number of pixels on either side (2-5 i guess) and fitting a poly to that and
#   filling in the missing pixels.
# - the same as above start at the left end point and for every missing pixel, 
#   only search in the direction of whichever end point we are closest to, 
#   and only use pixels on that side to fit a poly. we probably want to use the specific
#   position in the curve to decide where to search from i.e. if we're really close to the 
#   left then search left and extend right, if we're left but close to the middle, maybe
#   use the middle section right hand side and extend left. if the gap is not very big, 
#   then perhaps use both left and right. the sections shouldn't be too wide. maybe
#   only 1/10 the entire length of the horizontal midline length.

# - when doing multiple images, if we manually select ROIs, we use the same templates for all images

# - change the way we find the left magnets right edge, and right posts left edge
#   so that the initial templates can be roughly drawn just to find the magnet and fixed post
#   and then use another set of templates that have hard edges on the right of the magnet
#   and left of the fixed post that we use to refine those locations edges to compute the
#   length measurement end points.


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
    template_image = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
    template_image = intensityAdjusted(template_image)
    _, match_coordinates = matchResults(
      image_to_search=search_image,
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
  search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)
  search_image_gray = intensityAdjusted(search_image_gray) 

  # search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2RGB).astype(np.uint8)
  # metrics = {
  #   'distance_between_rois': 1,
  #   'midpoint_thickness': 1,
  #   'left_end_point_thickness': 1,
  #   'right_end_point_thickness': 1,
  #   'area_between_rois': 1 
  # }
  # return meanShiftSegmentation(search_image), metrics

  if template_image_paths == 'draw':
    rois_info = roiInfoFromUserDrawings(search_image)
  else:
    if template_image_paths is None or template_image_paths == 'select':
      template_1_image_path = getFilePathViaGUI('left template to find path')  
      template_2_image_path = getFilePathViaGUI('right template to find path')
      template_image_paths = [template_1_image_path, template_2_image_path]

    rois_info = roiInfoFromTemplates(
      search_image=search_image_gray,
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

  # find the points we think are at the upper/lower edges  
  points_to_find_edges_at = np.asarray(range(left_distance_marker_x, right_distance_marker_x))
  points_to_fit_poly_at = np.asarray(range(pixel_distance_between_rois))  
  upper_edge_points = np.empty(pixel_distance_between_rois)
  lower_edge_points = np.empty(pixel_distance_between_rois)

  # TODO: 
  # compute a measure of variance for the gradmag and if there is too much variance..., or
  # take the two estimated peaks and if they're not significantly larger (10+ times) 
  # than 98/99% of all the other points...
  # then we know we have crap edges and we can compute the  
  # cumulative sum of the gradmag moving average, 
  # chop off the top and bottom approx 50-100 pixels (possibly make it a parameter?) 
  # and then fit an "S" shaped poly
  #   _
  # _/
  #
  # so fit a logistic style function?
  # then we can use the fitted parameters to "draw" lines at the 
  # bottom and top horizontal sections and middle slope section
  # and then work out where the horizontal sections intersect with the sloped section
  # and those become the points where the "edges" are

  show_plots = False
  for index, point_to_find_edges_at in enumerate(points_to_find_edges_at):
    # if point_to_find_edges_at >= 700 and point_to_find_edges_at <= 730:
    #   show_plots = True
    # else:
    #   show_plots = False
    edge_point = outerEdges(search_image_gray[:, point_to_find_edges_at], show_plots=show_plots)    
    upper_edge_points[index] = edge_point['upper_edge_pos']
    lower_edge_points[index] = edge_point['lower_edge_pos']
 
  # # fit curves to those upper/lower "edge points" because it can be very noisy
  # # compute a single polyfit
  # polyfit_deg = 4
  # upper_edge_points_poly = Polynomial.fit(points_to_fit_poly_at, upper_edge_points, polyfit_deg)
  # upper_edge_points_from_poly = upper_edge_points_poly(points_to_fit_poly_at)
  # lower_edge_points_poly = Polynomial.fit(points_to_fit_poly_at, lower_edge_points, polyfit_deg)
  # lower_edge_points_from_poly = lower_edge_points_poly(points_to_fit_poly_at)

  # # compute a piecewise smooth fit 
  # polyfit_deg = 4
  # num_sub_regions = 8
  # split_upper_edge_points = np.array_split(upper_edge_points, num_sub_regions)
  # split_lower_edge_points = np.array_split(lower_edge_points, num_sub_regions)
  # for split_num in range(num_sub_regions):
  #   sub_region_upper_edge_points = split_upper_edge_points[split_num]
  #   upper_edge_sub_region_range = np.asarray(range(len(sub_region_upper_edge_points)))
  #   upper_edge_points_poly = Polynomial.fit(
  #     upper_edge_sub_region_range,
  #     sub_region_upper_edge_points, 
  #     polyfit_deg
  #   )
  #   split_upper_edge_points[split_num] = upper_edge_points_poly(upper_edge_sub_region_range)
  #   sub_region_lower_edge_points = split_lower_edge_points[split_num]
  #   lower_edge_sub_region_range = np.asarray(range(len(sub_region_lower_edge_points)))
  #   lower_edge_points_poly = Polynomial.fit(
  #     lower_edge_sub_region_range,
  #     sub_region_lower_edge_points, 
  #     polyfit_deg
  #   )
  #   split_lower_edge_points[split_num] = lower_edge_points_poly(lower_edge_sub_region_range)
  # upper_edge_points = np.concatenate(split_upper_edge_points, axis=0)
  # lower_edge_points = np.concatenate(split_lower_edge_points, axis=0)

  # compute the area between the fitted curves
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


def outerEdges(input_array: np.ndarray, show_plots: bool=True) -> int:
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
    # plt.plot(input_array_gradmag_ma, label='Gradient Magnitude of vertical line at midpoint', color = 'r')
    # plt.show()
    
    gradient_cumulative_sum = np.cumsum(input_array_gradmag)
    gradient_cumulative_sum_ma = np.convolve(
      gradient_cumulative_sum,
      np.ones(len_of_average)/len_of_average,
      mode='valid'
    )
    plt.plot(gradient_cumulative_sum_ma, label='Cummulative Distribution of Gradient', color = 'g')
    plt.show()

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
