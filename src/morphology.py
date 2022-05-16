#! /usr/bin/env python

import os
from pathlib import Path
from platform import system as os_name
from distutils.log import warn
from datetime import datetime
import openpyxl
import numpy as np
import cv2 as cv
from typing import Tuple, List, Dict
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from scipy import ndimage as ndifilters
from skimage import filters as skimagefilters
from mantavision import getFilePathViaGUI, getDirPathViaGUI, contentsOfDir
from track_template import matchResults, intensityAdjusted, userDrawnROI


def roiInfoFromTemplates(
  search_image: np.ndarray,
  template_image_paths: Dict,
  sub_pixel_search_increment: float=None,
  sub_pixel_refinement_radius: float=None
) -> Dict:
  ''' Finds the best match ROI for templates within search_image,
      and passes back the location information for all ROIs found.
  '''
  rois_info = {}
  for template_position, template_path_details in template_image_paths.items():
    outer_template_path = template_path_details['outer']
    outer_template_image = cv.imread(outer_template_path)
    if outer_template_image is None:
      print(f'ERROR. Could not open the template image at path provided: {outer_template_path}. Exiting.')
      return None
    outer_template_image = cv.cvtColor(outer_template_image, cv.COLOR_BGR2GRAY)
    outer_template_image = intensityAdjusted(outer_template_image)
    if template_path_details['inner'] is not None:
      inner_template_image_path = template_path_details['inner']
      inner_template_image = cv.imread(inner_template_image_path)
      inner_template_image = cv.cvtColor(inner_template_image, cv.COLOR_BGR2GRAY)
      inner_template_image = intensityAdjusted(inner_template_image)
      if sub_pixel_search_increment is None:
        sub_pixel_search_increment = 1.0
      if sub_pixel_refinement_radius is None:
        sub_pixel_refinement_radius = 5
    else:
      inner_template_image = None
    if template_position == 'left':
      sub_pixel_search_offset_right = True
    else:
      sub_pixel_search_offset_right = False
    _, match_coordinates = matchResults(
      image_to_search=search_image,
      template_to_match=outer_template_image,
      sub_pixel_search_increment=sub_pixel_search_increment,
      sub_pixel_refinement_radius=sub_pixel_refinement_radius,
      sub_pixel_search_template=inner_template_image,
      sub_pixel_search_offset_right=sub_pixel_search_offset_right
    )
    roi_origin_x, roi_origin_y = match_coordinates
    roi_height = outer_template_image.shape[0]
    roi_width = outer_template_image.shape[1]  
    roi_parameters = {
      'ORIGIN_X': int(roi_origin_x),
      'ORIGIN_Y': int(roi_origin_y),
      'WIDTH':  int(roi_width),
      'HEIGHT': int(roi_height)
    }
    rois_info[template_position] = roi_parameters
  return rois_info


def computeMorphologyMetrics(
  search_image_path: str,
  left_template_image_path: str,
  right_template_image_path: str,
  left_sub_template_image_path: str=None,
  right_sub_template_image_path: str=None,  
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  template_refinement_radius: int = 40,
  edge_finding_smoothing_radius: int = 1,
  microns_per_pixel: float=None,
  write_result_images: bool=False,
  display_result_images: bool=True
):

  if template_refinement_radius < 1:
    raise RuntimeError("template_refinement_radius cannot be < 1")
  if edge_finding_smoothing_radius < 1:
    raise RuntimeError("edge_finding_smoothing_radius cannot be < 1")

  # if 'windows' in os_name().lower():
  #   import codecs
  #   search_image_path = codecs.escape_decode(bytes(win_path, "utf-8"))[0].decode("utf-8")    

  if search_image_path.lower() == 'select_file':
    search_image_path = getFilePathViaGUI('Select File To Analyize')
  elif search_image_path.lower() == 'select_dir':
    search_image_path = getDirPathViaGUI('Select Directory With Images To Analyze')
  base_dir, file_names = contentsOfDir(dir_path=search_image_path, search_terms=['.tif', '.tiff', '.jpg', '.png'])
  
  results_dir_name = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  results_dir = os.path.join(base_dir, results_dir_name)
  os.mkdir(results_dir)
  if write_result_images:
    results_image_dir = os.path.join(results_dir, 'result_images')
    os.mkdir(results_image_dir)

  template_paths = {
    'left': {'outer': left_template_image_path, 'inner': left_sub_template_image_path},
    'right': {'outer': right_template_image_path, 'inner': right_sub_template_image_path}    
  }
  for template_position, template_details in template_paths.items():
    for template_type, template_path in template_details.items():
      if template_path == 'select_file':
        if template_paths[template_position]['inner'] is None:
          template_selection_title = f'Select File For {template_position} template'
        else:
          template_selection_title = f'Select File For {template_position} {template_type} template'          
        template_paths[template_position][template_type] = getFilePathViaGUI(template_selection_title)
      elif template_path == 'draw_roi':
        drawn_roi_templates_dir = os.path.join(results_dir, 'drawn_template_images')
        if not os.path.exists(drawn_roi_templates_dir):
          os.mkdir(drawn_roi_templates_dir)
        file_name, file_extension = file_names[0]  # NOTE: this could be a selected file too
        search_image_for_template = cv.imread(os.path.join(base_dir, file_name + file_extension))
        if template_paths[template_position]['inner'] is None:
          draw_roi_title = f'Draw ROI For {template_position} template'
        else:
          draw_roi_title = f'Draw ROI For {template_position} {template_type} template'          
        roi_info = userDrawnROI(search_image_for_template, draw_roi_title)
        template_image = search_image_for_template[
          roi_info['y_start'] : roi_info['y_end'],
          roi_info['x_start'] : roi_info['x_end']
        ]
        template_image_path = os.path.join(drawn_roi_templates_dir, template_type + '_template_image.tif')
        cv.imwrite(template_image_path, template_image)
        template_paths[template_position][template_type] = template_image_path

  all_metrics = []
  image_analyzed_id = 0
  for file_name, file_extension in file_names:
      file_to_analyze = os.path.join(base_dir, file_name + file_extension)
      results_image, metrics = morphologyMetricsForImage(
        search_image_path=file_to_analyze,
        template_image_paths=template_paths,
        sub_pixel_search_increment=sub_pixel_search_increment,
        sub_pixel_refinement_radius=sub_pixel_refinement_radius,
        microns_per_pixel=microns_per_pixel,
        template_refinement_radius=template_refinement_radius,
        edge_finding_smoothing_radius=edge_finding_smoothing_radius
      )
      all_metrics.append(
        {
          'file': file_to_analyze,
          'horizontal_length': round(metrics['distance_between_rois'], 2),
          'left_edge_vertical_length': round(metrics['left_end_point_thickness'], 2),
          'mid_point_vertical_length': round(metrics['midpoint_thickness'], 2),
          'right_edge_vertical_length': round(metrics['right_end_point_thickness'], 2),
          'tissue_area': round(metrics['area_between_rois'], 2),
          'orientation': round(metrics['orientation'], 2),
          'warning_flags': metrics['warning_flags']
        }        
      )

      # write the results image to file if requested
      if write_result_images:
        analyzed_file_results_image_name = file_name + '_results.jpg'
        analyzed_file_results_image_path = os.path.join(results_image_dir, analyzed_file_results_image_name)
        cv.imwrite(analyzed_file_results_image_path, results_image)

      # display the metrics and results image here if requested
      if display_result_images:
        print(f'image no: {image_analyzed_id} - {file_to_analyze}')
        print(f"horizontal length between ROIs: {round(metrics['distance_between_rois'], 2)} (microns)")
        print(f"vertical length at left ROI : {round(metrics['left_end_point_thickness'], 2)} (microns)")
        print(f"vertical length at midpoint between ROIs: {round(metrics['midpoint_thickness'], 2)} (microns)")
        print(f"vertical length at right ROI : {round(metrics['right_end_point_thickness'], 2)} (microns)")
        print(f"area between ROIs: {round(metrics['area_between_rois'], 2)} (microns\u00b2)")
        print(f"orientation of horizontal line between ROIs: {round(metrics['orientation'], 2)}\u00b0")        
        
        if len(metrics['warning_flags']) > 0:
          print('WARNING!:')
          for warning_msg in metrics['warning_flags']:
            print(warning_msg)
        plt.imshow(cv.cvtColor(results_image, cv.COLOR_BGR2RGB))
        plt.show()
        print()
      image_analyzed_id += 1

  results_xlsx_name = 'results.xlsx'
  results_xlsx_path = os.path.join(results_dir, results_xlsx_name)  
  runtime_parameters = {
    'search_image_path' : base_dir,
    'template_image_paths': template_paths,
    'sub_pixel_search_increment' : sub_pixel_search_increment,
    'sub_pixel_refinement_radius' : sub_pixel_refinement_radius,    
    'microns_per_pixel': microns_per_pixel,
    'template_refinement_radius': template_refinement_radius,
    'edge_finding_smoothing_radius': edge_finding_smoothing_radius

  }
  resultsToCSV(all_metrics, results_xlsx_path, runtime_parameters)


def smoothedHorizontally(input_image: np.ndarray, sigma: float) -> np.ndarray:
  return ndifilters.gaussian_filter1d(
      input_image,
      sigma=sigma,
      mode='constant',
      axis=1
  )


def morphologyMetricsForImage(
  search_image_path: str,
  template_image_paths: Dict,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  microns_per_pixel: float=None,
  template_refinement_radius: int = 40,
  edge_finding_smoothing_radius: int = 1
):
  ''' '''
  if microns_per_pixel is None:
    microns_per_pixel = 1.0

  # load the image
  if search_image_path is None or search_image_path.lower() == 'select_file':
    search_image_path = getFilePathViaGUI('image to search path')
  search_image = cv.imread(search_image_path)
  if search_image is None:
    print(f'ERROR. Could not open the search image pointed to by the path provided: {search_image_path}. Exiting.')
    return None
  search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)
  search_image_gray_adjusted = intensityAdjusted(search_image_gray) 

  # get the ROIs
  rois_info = roiInfoFromTemplates(
    search_image=search_image_gray_adjusted,
    template_image_paths=template_image_paths,
    sub_pixel_search_increment=None,
    sub_pixel_refinement_radius=None
  )
  left_roi = rois_info['left']
  right_roi = rois_info['right']

  left_vertical_midpoint = left_roi['ORIGIN_Y'] + left_roi['HEIGHT']/2
  right_vertical_midpoint = right_roi['ORIGIN_Y'] + right_roi['HEIGHT']/2
  vertical_midpoint = (left_vertical_midpoint + right_vertical_midpoint)/2

  # compute the distance between the inner sides of each ROI
  left_distance_marker_x = left_roi['ORIGIN_X'] + left_roi['WIDTH']
  right_distance_marker_x = right_roi['ORIGIN_X']
  if left_distance_marker_x >= right_distance_marker_x - template_refinement_radius:
    # the analysis is garbage and we'll report that but 
    # we separate the rois to allow the process to keep working
    left_distance_marker_x = right_distance_marker_x - template_refinement_radius - 1
  else:
    median_smoothed_image = ndifilters.median_filter(
      smoothed(input_image=search_image_gray_adjusted, sigma=5),
      size=5
    )
    # adjust the left and right distance markers based on max edge detection
    left_distance_marker_x = verticalEdge(
      median_smoothed_image[int(left_vertical_midpoint), :],
      horizontal_midpoint=int(left_distance_marker_x),
      search_radius=template_refinement_radius
    )
    right_distance_marker_x = verticalEdge(
      median_smoothed_image[int(right_vertical_midpoint), :],
      horizontal_midpoint=int(right_distance_marker_x),
      search_radius=template_refinement_radius
    )
  pixel_distance_between_rois = right_distance_marker_x - left_distance_marker_x
  distance_between_rois = microns_per_pixel*pixel_distance_between_rois
  horizontal_midpoint = left_distance_marker_x + pixel_distance_between_rois/2

  # find the points we think are at the upper/lower edges for the far left, mid and far right horizontal positions
  points_to_find_edges_at = np.asarray([int(left_distance_marker_x), int(horizontal_midpoint), int(right_distance_marker_x)])
  key_upper_edge_points = np.empty(len(points_to_find_edges_at))
  key_lower_edge_points = np.empty(len(points_to_find_edges_at))

  # smooth the image horizontally only so that the vertical edges remain as sharp as possible 
  median_image = ndifilters.median_filter(
    smoothedHorizontally(input_image=search_image_gray, sigma=5),
    size=5
  )
  show_plots = False
  # find the edges at the horizontal centre and end points
  for index, point_to_find_edges_at in enumerate(points_to_find_edges_at):
    edge_point = outerEdges(
      median_image[:, point_to_find_edges_at],
      vertical_midpoint=vertical_midpoint,
      show_plots=show_plots
    )
    key_upper_edge_points[index] = edge_point['upper_edge_pos']
    key_lower_edge_points[index] = edge_point['lower_edge_pos']

  # find the edge points in a region near the horizontal midpoint
  midpoint_search_radius = 5
  left_midpoint_edge = int(horizontal_midpoint) - midpoint_search_radius
  right_midpoint_edge = int(horizontal_midpoint) + midpoint_search_radius
  points_to_find_edges = np.asarray([point for point in range(left_midpoint_edge, right_midpoint_edge + 1)])
  top_edge_points = np.empty(len(points_to_find_edges))
  bottom_edge_points = np.empty(len(points_to_find_edges))
  show_plots = False
  for index, point_to_find_edges in enumerate(points_to_find_edges):
    edge_points = outerEdges(
      median_image[:, point_to_find_edges],
      vertical_midpoint=vertical_midpoint,
      show_plots=show_plots
    )
    top_edge_points[index] = edge_points['upper_edge_pos']
    bottom_edge_points[index] = edge_points['lower_edge_pos']
  # ensure we don't randomly choose a bad edge midpoint by
  # forcing it to be within some range of the median for the midpoint region
  edge_point_variance = 5
  top_edge_points_median = np.median(top_edge_points)
  top_edge_midpoint = top_edge_points[midpoint_search_radius]
  if top_edge_midpoint < top_edge_points_median - edge_point_variance:
    top_edge_midpoint = top_edge_points_median
  if top_edge_midpoint > top_edge_points_median + edge_point_variance:    
    top_edge_midpoint = top_edge_points_median
  bottom_edge_points_median = np.median(bottom_edge_points)
  bottom_edge_midpoint = bottom_edge_points[midpoint_search_radius]
  if bottom_edge_midpoint < bottom_edge_points_median - edge_point_variance:
    bottom_edge_midpoint = bottom_edge_points_median
  if bottom_edge_midpoint > bottom_edge_points_median + edge_point_variance:    
    bottom_edge_midpoint = bottom_edge_points_median

  # now find ALL edge points of tissue between templates by walking out from the central points
  # and only looking at max edges within a narrow horizontal range of the adjacent inner pixel
  all_points_to_find_edges = np.asarray([point for point in range(left_distance_marker_x, right_distance_marker_x)])
  edge_values_middle = int(len(all_points_to_find_edges)/2)
  top_edge_values = np.zeros(len(all_points_to_find_edges))
  top_edge_values[edge_values_middle] = top_edge_midpoint
  bottom_edge_values = np.zeros(len(all_points_to_find_edges))
  bottom_edge_values[edge_values_middle] = bottom_edge_midpoint
  
  right_points = np.asarray([point for point in range(int(horizontal_midpoint) + 1, right_distance_marker_x)])
  left_points = np.asarray([point for point in range(int(horizontal_midpoint) - 1, left_distance_marker_x  - 1, -1)])
  all_points = [(left_points, edge_values_middle - 1, -1), (right_points, edge_values_middle + 1, 1)]
  edge_point_details = [(top_edge_midpoint, top_edge_values), (bottom_edge_midpoint, bottom_edge_values)]
  for points_to_find_edges, start, direction in all_points:
    if direction < 1:
      pass  # moving averge should be with points ot the right up to the midpoint
    else:
      pass  # moving averge should be with points to the left up to the midpoint      
    for edge_midpoint, edge_values in edge_point_details:
      prev_edge_value = edge_midpoint
      for index, point_to_find_edges in enumerate(points_to_find_edges):
        edge_value = outerEdge(
          median_image[:, point_to_find_edges],
          vertical_midpoint=prev_edge_value, 
          search_radius=edge_point_variance
        )
        edge_value_index = start + direction*index
        edge_values[edge_value_index] = edge_value
        # compute a moving average for the value to use as the next vertical search position
        if direction < 1:  # moving average should be to the right
          start_point_for_avg = edge_value_index
          end_point_for_avg = min(edge_values_middle, edge_value_index - direction*edge_finding_smoothing_radius)
        else:
          end_point_for_avg = edge_value_index
          start_point_for_avg = max(edge_values_middle, edge_value_index - direction*edge_finding_smoothing_radius)
        prev_edge_value_avg = np.average(edge_values[start_point_for_avg:end_point_for_avg])
        prev_edge_value = round(prev_edge_value_avg)

  # compute the vertical distance between "edges" all edge points estimated by walking outward
  left_end_point_thickness = microns_per_pixel*(bottom_edge_values[0] - top_edge_values[0])
  midpoint_thickness = microns_per_pixel*(bottom_edge_points_median - top_edge_points_median)
  right_end_point_thickness = microns_per_pixel*(bottom_edge_values[-1] - top_edge_values[-1])

  # compute the area between the fitted curves
  edge_point_diffs = bottom_edge_values - top_edge_values
  area_between_rois = microns_per_pixel * np.sum(edge_point_diffs)

  # compute WARNING FLAGS indicating bad measurements
  warning_flags = []

  # sanity check on area computed
  image_area = search_image_gray.shape[0] * search_image_gray.shape[1]
  if area_between_rois > image_area:
    warning_flags.append(
      "Area indicates measurements may be inaccurate"
    )
  
  # edges too close to image border  
  vertical_edge_pos_limit = 20
  min_vertical_edge_pos = vertical_edge_pos_limit
  upper_edge_too_close = top_edge_values < min_vertical_edge_pos
  if np.any(upper_edge_too_close):
    warning_flags.append(
      "Closeness of upper edge to the image border indicates measurements may be inaccuratee"
    )
  max_vertical_edge_pos = search_image_gray.shape[0] - vertical_edge_pos_limit
  lower_edge_too_close = bottom_edge_values > max_vertical_edge_pos
  if np.any(lower_edge_too_close):
    warning_flags.append(
      "Closeness of lower edge to the image border indicates measurements may be inaccurate"
    )

  # left or right end points are on the wrong side of midline
  image_mid_point = search_image_gray.shape[1]/2
  min_distance_from_midpoint = 20
  if left_distance_marker_x > image_mid_point - min_distance_from_midpoint:
    warning_flags.append(
      "Position of left end key point indicates measurements may be inaccurate"
    )
  if right_distance_marker_x < image_mid_point + min_distance_from_midpoint:
    warning_flags.append(
      "Position of right end key point indicates measurements may be inaccurate"
    )

  # left and right end points are too close
  distance_between_end_points = right_distance_marker_x - left_distance_marker_x
  min_distance_between_end_points = 0.25*search_image_gray.shape[1]
  if distance_between_end_points < min_distance_between_end_points:
    warning_flags.append(
      "Distance between left and right end points indicates measurements may be inaccurate"
    )

  # length of left and right portions is too different
  left_side_length = horizontal_midpoint - left_distance_marker_x
  right_side_length = right_distance_marker_x - horizontal_midpoint
  # TODO: check for div by zero
  # np.any(np.isclose(
  if np.isclose(left_side_length, 0.0):
    warning_flags.append(
      "distance between left end and center indicates measurements may be inaccurate"
    )
  elif np.isclose(right_side_length, 0.0):
    warning_flags.append(
      "distance between right end and center indicates measurements may be inaccurate"
    )
  else:
    side_length_ratio = left_side_length/right_side_length
    if side_length_ratio < 1.0:
      side_length_ratio = 1.0/side_length_ratio
    acceptable_side_length_ratio = 1.3
    if side_length_ratio > acceptable_side_length_ratio:
      warning_flags.append(
        "Ratio between left and right portion lengths indicates measurements may be inaccurate"
      )     

  # vertical thickness at left and right key points is too different
  if np.isclose(right_end_point_thickness, 0.0) or right_end_point_thickness is None:
    warning_flags.append(
      "Missing right vertical end point indicates measurements may be inaccurate"
    )
  elif np.isclose(left_end_point_thickness, 0.0) or left_end_point_thickness is None:
    warning_flags.append(
      "Missing left vertical end point indicates measurements may be inaccurate"
    )    
  else:
    end_point_thickness_ratio = left_end_point_thickness / right_end_point_thickness
    if end_point_thickness_ratio < 1.0:
      end_point_thickness_ratio = 1.0/end_point_thickness_ratio
    acceptable_end_point_thickness_ratio = 1.3
    if end_point_thickness_ratio > acceptable_end_point_thickness_ratio:
      warning_flags.append(
        "Vertical length between upper and lower edges at left and right key points indicates measurements may be inaccurate"
      )

  # edge points are too close to the horizontal midline
  horizontal_line_grad = (right_vertical_midpoint - left_vertical_midpoint) / (right_distance_marker_x - left_distance_marker_x)
  num_edge_points = len(all_points_to_find_edges)
  horizontal_line_y_points = np.zeros(num_edge_points, dtype=np.uint32)
  for x_pos in range(num_edge_points):
    horizontal_line_y_points[x_pos] = left_vertical_midpoint + horizontal_line_grad*x_pos
  upper_edge_distance_to_midline = horizontal_line_y_points - top_edge_values
  min_acceptable_edge_distance = 40
  upper_edge_too_close_to_midline = upper_edge_distance_to_midline < min_acceptable_edge_distance
  if np.any(upper_edge_too_close_to_midline):
    warning_flags.append(
      "Closeness of upper edge to the horizontal midline indicates measurements may be inaccurate"
    )
  lower_edge_distance_to_midline = bottom_edge_values - horizontal_line_y_points
  lower_edge_too_close_to_midline = lower_edge_distance_to_midline < min_acceptable_edge_distance
  if np.any(lower_edge_too_close_to_midline):
    warning_flags.append(
      "Closeness of lower edge to the horizontal midline indicates measurements may be inaccurate"
    )
  # upper and lower edge point distances to horizontal midline are too great
  upper_edge_distance_to_midline = np.abs(upper_edge_distance_to_midline)
  lower_edge_distance_to_midline = np.abs(lower_edge_distance_to_midline)

  if np.any(np.isclose(upper_edge_distance_to_midline, 0.0)) or upper_edge_distance_to_midline is None:
    warning_flags.append(
        "upper edge distance to horizontal midline indicates measurements may be inaccurate"
    )
  elif np.any(np.isclose(lower_edge_distance_to_midline, 0.0)) or lower_edge_distance_to_midline is None:
    warning_flags.append(
        "lower edge distance to horizontal midline indicates measurements may be inaccurate"
    )    
  else:
    edge_distance_to_midline_ratio = upper_edge_distance_to_midline/lower_edge_distance_to_midline
    acceptable_edge_distance_to_midline_ratio = 2.0
    for edge_distance_ratio in edge_distance_to_midline_ratio:
      if edge_distance_ratio < 1.0:
        edge_distance_ratio = 1.0/edge_distance_ratio
      if edge_distance_ratio > acceptable_edge_distance_to_midline_ratio:
        warning_flags.append(
          "Ratio of upper and lower edge distance to horizontal midline indicates measurements may be inaccurate"
        )      
        break

  angle_of_midline = -1.0*np.rad2deg(
    np.arctan(horizontal_line_grad)
  )

  metrics = {
    'distance_between_rois': distance_between_rois,
    'midpoint_thickness': midpoint_thickness,
    'left_end_point_thickness': left_end_point_thickness,
    'right_end_point_thickness': right_end_point_thickness,
    'area_between_rois': area_between_rois,
    'orientation': angle_of_midline,
    'warning_flags': warning_flags
  }

  # draw the results metrics on a results image
  results_image = search_image.copy().astype(np.uint8)

  # draw the horizontal line between left and right ROI inner edges
  blue_bgr = (255, 0, 0)
  cv.line(
    results_image,
    pt1=(left_distance_marker_x, int(left_vertical_midpoint)),
    pt2=(right_distance_marker_x, int(right_vertical_midpoint)),
    color=blue_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )

  # draw the upper edges
  red_bgr = (0, 0, 128)
  lower_edge_points_to_draw = zip(all_points_to_find_edges.astype(np.int32), bottom_edge_values.astype(np.int32))
  for x_pos, y_pos in lower_edge_points_to_draw:
    cv.circle(
      results_image,
      center=(x_pos, y_pos),
      radius=2,    
      color=red_bgr,
      thickness=3,
      lineType=cv.LINE_AA
    )
  # draw the lower edges    
  upper_edge_points_to_draw = zip(all_points_to_find_edges.astype(np.int32), top_edge_values.astype(np.int32))
  for x_pos, y_pos in upper_edge_points_to_draw:
    cv.circle(
      results_image,
      center=(x_pos, y_pos),
      radius=2,    
      color=red_bgr,
      thickness=3,
      lineType=cv.LINE_AA
    )

  # draw the left ROI inner edge object vertical thickness line
  green_bgr = (0, 255, 0)
  left_end_point_upper_edge_pos = round(top_edge_values[0])
  left_end_point_lower_edge_pos = round(bottom_edge_values[0])
  cv.line(
    results_image,
    pt1=(left_distance_marker_x, left_end_point_lower_edge_pos),
    pt2=(left_distance_marker_x, left_end_point_upper_edge_pos),
    color=green_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )
  # draw the right ROI inner edge object vertical thickness line
  right_end_point_upper_edge_pos = round(top_edge_values[-1])
  right_end_point_lower_edge_pos = round(bottom_edge_values[-1]) 
  cv.line(
    results_image,
    pt1=(right_distance_marker_x, right_end_point_lower_edge_pos),
    pt2=(right_distance_marker_x, right_end_point_upper_edge_pos),
    color=green_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )
  # draw the horizontal midpoint object vertical thickness line
  midpoint_point_upper_edge_pos = round(top_edge_points_median)
  midpoint_point_lower_edge_pos = round(bottom_edge_points_median) 
  cv.line(
    results_image,
    pt1=(int(horizontal_midpoint), midpoint_point_upper_edge_pos),
    pt2=(int(horizontal_midpoint), midpoint_point_lower_edge_pos),
    color=green_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )

  return results_image, metrics 


def normalized(input_image: np.ndarray, new_range: float = 512.0) -> np.ndarray:
  input_array_min: float = np.min(input_image)
  input_array_max: float = np.max(input_image)
  input_array_zero_origin: np.ndarray = input_image - input_array_min
  input_array_current_range: float = float(input_array_max - input_array_min)
  if np.isclose(input_array_current_range, 0.0):
    return input_array_zero_origin
  input_array_normalizer: float = new_range/input_array_current_range
  return input_array_zero_origin * input_array_normalizer


def yGradmag(input_array: np.ndarray) -> np.ndarray:
  # NOTE: this only returns the 0th x element because 
  # it's only used for a single vertical strip of pixels
  return np.square(
    cv.Sobel(input_array, -1, 0, 1, ksize=5)[:, 0]
  )


def xGradmag(input_array: np.ndarray) -> np.ndarray:
  # NOTE: this only returns the 0th x element because 
  # it's only used for a single vertical strip of pixels
  return np.square(
    cv.Sobel(input_array, -1, 1, 0, ksize=5)[0, :]
  )


def movingAverage(input_array: np.ndarray, filter_radius: int = 5) -> np.ndarray:
  len_of_average = 2*filter_radius + 1
  ma_operator = np.ones(len_of_average)/len_of_average
  return np.convolve(input_array, ma_operator, mode='valid')


def verticalEdge(input_array: np.ndarray, horizontal_midpoint: int, search_radius: int) -> int:
  # TODO: restrict the input_array to +/- some region that is larger than the
  #       radii of the gradmag and moving average functions so we don't have to 
  #       compute them for the entire set that we never look at
  input_array_normalized = normalized(input_array)
  input_array_gradmag = yGradmag(input_array_normalized)
  input_array_gradmag_ma = movingAverage(input_array_gradmag)
  search_range_start = max(0, horizontal_midpoint - search_radius)
  search_range_end = min(len(input_array), horizontal_midpoint + search_radius)
  return search_range_start + np.argmax(
    input_array_gradmag_ma[int(search_range_start) : int(search_range_end)]
  )


def outerEdge(input_array: np.ndarray, vertical_midpoint: int, search_radius: int) -> int:
  '''
  '''
  # TODO: restrict the input_array to +/- some region that is larger than the
  #       radii of the gradmag and moving average functions so we don't have to 
  #       compute them for the entire set that we never look at
  input_array_normalized = normalized(input_array)
  input_array_gradmag = yGradmag(input_array_normalized)
  input_array_gradmag_ma = movingAverage(input_array_gradmag)
  search_range_start = max(0, vertical_midpoint - search_radius)
  search_range_end = min(len(input_array), vertical_midpoint + search_radius)
  return search_range_start + np.argmax(
    input_array_gradmag_ma[int(search_range_start) : int(search_range_end)]
  )


def outerEdges(input_array: np.ndarray, vertical_midpoint: int, show_plots: bool=False) -> int:
  '''
  Presumes there are two main edges on the outside of a
  horizontally aligned "tube/rectangle" like structure
  i.e.
  ------------

  ------------
  If horizontal_pos is None, input_array should be a 1D array of intensities 
  (either raw or gradmag) i.e. a vertical cross section 1 pixel thick
  '''

  input_array_normalized = normalized(input_array)
  input_array_gradmag = yGradmag(input_array_normalized)
  input_array_gradmag_ma = movingAverage(input_array_gradmag)

  if show_plots:
    plt.plot(input_array_gradmag_ma, label='Gradient Magnitude of vertical line at midpoint', color = 'r')
    gradient_cumulative_sum = np.cumsum(input_array_gradmag)
    plt.show()
    # gradient_cumulative_sum_ma = np.convolve(
    #   gradient_cumulative_sum,
    #   np.ones(len_of_average)/len_of_average,
    #   mode='valid'
    # )
    plt.plot(gradient_cumulative_sum, label='Cummulative Distribution of Gradient', color = 'g')
    plt.show()

  # first find what we presume is one of two tissue edges using the highest intensity gradmag 
  max_intensity_pos = np.argmax(input_array_gradmag_ma)
  # then find what we presume is the other edge on the opposite side of the horizontal midpoint
  # we presume the tissue is mostly symmetric and so look for the other edge in the vacinity 
  # of the first edge mirrored on the opposite side of the horizontal midline 
  half_width = max_intensity_pos - vertical_midpoint
  second_edge_candidate_pos: int = int(vertical_midpoint - half_width)
  # create a buffer of +/- 20% of image height to look for the other edge
  vertical_height = input_array_gradmag_ma.shape[0]
  candidate_range_buffer = int(vertical_height * 0.2)
  candidate_start_pos = max(0, second_edge_candidate_pos - candidate_range_buffer)
  candidate_end_pos = min(second_edge_candidate_pos + candidate_range_buffer, vertical_height)
  other_peak_max_pos = np.argmax(input_array_gradmag_ma[candidate_start_pos:candidate_end_pos]) + candidate_start_pos
  if max_intensity_pos < other_peak_max_pos:
    return {
        'lower_edge_pos': other_peak_max_pos,
        'upper_edge_pos': max_intensity_pos,
    }
  else:
    return {
        'lower_edge_pos': max_intensity_pos,
        'upper_edge_pos': other_peak_max_pos,
    }


def resultsToCSV(
  analysis_results: List[Dict],
  path_to_output_file: str,
  runtime_parameters: Dict
):    

  workbook = openpyxl.Workbook()
  sheet = workbook.active

  file_column = 'A'
  horizontal_column = 'B'
  mid_point_column = 'C'
  left_edge_column = 'D'
  right_edge_column = 'E'
  area_column = 'F'
  orientation_column = 'G'
  warning_column = 'H'
  
  heading_row = '1'
  sheet[file_column + heading_row] = 'File'
  sheet[horizontal_column + heading_row] = 'Horizontal Length (microns)'
  sheet[left_edge_column + heading_row] = 'Left Edge Length (microns)'
  sheet[mid_point_column + heading_row] = 'Mid Point Length (microns)'
  sheet[right_edge_column + heading_row] = 'Right Edge Length (microns)'
  sheet[area_column + heading_row] = 'Area (square-microns)'
  sheet[orientation_column + heading_row] = 'Orientation (degrees)'  
  sheet[warning_column + heading_row] = 'WARNINGS'

  data_row = 2
  num_rows_to_write = len(analysis_results)
  for results_row in range(num_rows_to_write):
      metrics = analysis_results[results_row]
      sheet_row = str(results_row + data_row)
      sheet[file_column + sheet_row] = metrics['file']
      sheet[horizontal_column + sheet_row] = float(metrics['horizontal_length'])
      sheet[left_edge_column + sheet_row] = float(metrics['left_edge_vertical_length'])
      sheet[mid_point_column + sheet_row] = float(metrics['mid_point_vertical_length'])
      sheet[right_edge_column + sheet_row] = float(metrics['right_edge_vertical_length'])
      sheet[area_column + sheet_row] = float(metrics['tissue_area'])
      sheet[orientation_column + sheet_row] = float(metrics['orientation'])

      warnings = metrics['warning_flags']
      if len(warnings) > 0:
        warning_text = warnings.pop()
        for warning_msg in warnings:
          next_warning_msg = ', ' + warning_msg
          warning_text += next_warning_msg
      else:
        warning_text = "NONE"
      sheet[warning_column + sheet_row] = warning_text

  # add the runtime parameters
  runtime_config_lables_column = 'J'
  runtime_config_data_column = 'K'
  sheet[runtime_config_lables_column + heading_row] = 'runtime parameters'
  sheet[runtime_config_lables_column + str(data_row + 0)] = 'input images'
  sheet[runtime_config_data_column + str(data_row + 0)] = runtime_parameters['search_image_path']
  sheet[runtime_config_lables_column + str(data_row + 1)] = 'input templates'
  template_paths = ''
  template_image_paths = runtime_parameters['template_image_paths']
  for template_position, template_path_details in template_image_paths.items():
    for template_type, template_path in template_path_details.items():
      if template_path is not None:
        template_paths += template_path + ', '
  template_paths = template_paths[:-2]  # remove the last ', '
  sheet[runtime_config_data_column + str(data_row + 1)] = template_paths 
  sheet[runtime_config_lables_column + str(data_row + 2)] = 'microns per pixel'
  sheet[runtime_config_data_column + str(data_row + 2)] = runtime_parameters['microns_per_pixel']    
  sheet[runtime_config_lables_column + str(data_row + 3)] = 'sub pixel refinement search increment'
  if runtime_parameters['sub_pixel_search_increment'] == '' or runtime_parameters['sub_pixel_search_increment'] is None:
    sheet[runtime_config_data_column + str(data_row + 3)] = 'None'
  else:
    sheet[runtime_config_data_column + str(data_row + 3)] = runtime_parameters['sub_pixel_search_increment']
  sheet[runtime_config_lables_column + str(data_row + 4)] = 'sub pixel refinement radius'
  if runtime_parameters['sub_pixel_refinement_radius'] == '' or runtime_parameters['sub_pixel_refinement_radius'] is None:
   sheet[runtime_config_data_column + str(data_row + 4)] = 'None'
  else:
    sheet[runtime_config_data_column + str(data_row + 4)] = runtime_parameters['sub_pixel_refinement_radius']
  sheet[runtime_config_lables_column + str(data_row + 5)] = 'template refinement radius'
  sheet[runtime_config_data_column + str(data_row + 5)] = runtime_parameters['template_refinement_radius']
  sheet[runtime_config_lables_column + str(data_row + 6)] = 'edge finding smoothing radius'
  sheet[runtime_config_data_column + str(data_row + 6)] = runtime_parameters['edge_finding_smoothing_radius']
  workbook.save(filename=path_to_output_file)


def verticalVariance(input_array: np.ndarray, var_rad: int, smoothing_sigma: float = None) -> np.ndarray:
  def varApprox(input_array: np.ndarray, var_rad: int, pos: Tuple[int]) -> float:
    centre_value = input_array[pos]
    diff_square_sum = 0
    for y_offset in range(-var_rad, var_rad + 1):
        diff = centre_value - input_array[pos[0] + y_offset, pos[1]]
        diff_square_sum += diff**2
    return diff_square_sum

  variability_image = np.zeros(input_array.shape, dtype=np.float32)
  grid_points_y = range(var_rad, input_array.shape[0] - var_rad)
  grid_points_x = range(var_rad, input_array.shape[1] - var_rad)
  grid_coordinates = tuple(
    np.meshgrid(
      grid_points_y,
      grid_points_x,
      indexing='ij',
    )
  )
  if smoothing_sigma is not None:
    input_to_compute = smoothedHorizontally(input_array, sigma=6.0)
  else: 
    input_to_compute = input_array
  variability_image[grid_coordinates] = varApprox(
    input_array=input_to_compute,
    var_rad=var_rad,
    pos=grid_coordinates
  )
  return variability_image


def verticalVariance1D(input_array: np.ndarray, var_rad: int, smoothing_sigma: float = None) -> np.ndarray:
  variability_image = np.zeros(input_array.shape, dtype=np.float32)
  grid_points_y = range(var_rad, input_array.shape[0] - var_rad)
  for pos in grid_points_y:
    centre_value = input_array[pos]
    diff_square_sum = 0
    for y_offset in range(-var_rad, var_rad + 1):
        diff = centre_value - input_array[pos + y_offset]
        diff_square_sum += diff**2
    variability_image[pos] = diff_square_sum
  return variability_image


def thresholded(input_image: np.ndarray, method: str, binarize: bool=False) -> np.ndarray:
  # https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_isodata
  foreground_image_mask = input_image.copy().astype(np.float32)
  if method == 'otsu':
    input_image_threshold = skimagefilters.threshold_otsu(
      foreground_image_mask
    )
  elif method == 'multi':
    input_image_threshold = skimagefilters.threshold_multiotsu(
      foreground_image_mask,
      classes=4
    )
  elif method == 'adaptive':
    input_image_threshold = skimagefilters.threshold_local(
      foreground_image_mask,
      block_size=3,
      method='mean'  # 'gaussian', 'median'
    )    
  elif method == 'li':
    input_image_threshold = skimagefilters.threshold_li(
      foreground_image_mask
    )
  elif method == 'triangle':
    input_image_threshold = skimagefilters.threshold_triangle(
      foreground_image_mask
    )
  elif method == 'mean':
    input_image_threshold = skimagefilters.threshold_mean(
      foreground_image_mask
    )    
  else:
    raise RuntimeError(f"threshold method {method} is not supported.")
  if method == 'multi':
    input_image_threshold = input_image_threshold.tolist()
    max_val = np.max(input_image)
    input_image_threshold = [0] + input_image_threshold + [max_val]
    num_segments = len(input_image_threshold)
    for segment in range(1, num_segments):
      prev_thresh = input_image_threshold[segment - 1]
      curr_thresh = input_image_threshold[segment]
      foreground_image_mask[
          (foreground_image_mask >= prev_thresh) & (foreground_image_mask < curr_thresh) 
      ] = round(prev_thresh)
    highest_thresh = input_image_threshold[-1]
    foreground_image_mask[
        foreground_image_mask >= highest_thresh 
    ] = round(highest_thresh)
  else:
    foreground_image_mask[
        foreground_image_mask < input_image_threshold
    ] = 0.0
    if binarize:
      foreground_image_mask[foreground_image_mask > 0.0] = 1.0
  # foreground_image_mask = foreground_image_mask.astype(np.uint8)
  return foreground_image_mask


def smoothed(input_image: np.ndarray, sigma: float) -> np.ndarray:
  return ndifilters.gaussian_filter(
      input_image,
      sigma=sigma,
      mode='constant',
  )


def gradmag(input_array: np.ndarray, order: int = 2) -> np.ndarray:
  input_array_gradient = np.gradient(input_array, edge_order=order)
  return np.square(input_array_gradient)


if __name__ == '__main__':
  
  computeMorphologyMetrics(
    search_image_path=None,
    template_image_paths=None,
    sub_pixel_search_increment=None,
    sub_pixel_refinement_radius=None
  )
