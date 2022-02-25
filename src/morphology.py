#! /usr/bin/env python


import os
# import glob
from datetime import datetime
# import shutil
import openpyxl
import numpy as np
import cv2 as cv
from typing import Tuple, List, Dict
from track_template import matchResults
from mantavision import getFilePathViaGUI, getDirPathViaGUI
# from skimage import filters as skimagefilters
from track_template import intensityAdjusted
from numpy.polynomial.polynomial import Polynomial  # , polyfit
from mantavision import getDirPathViaGUI, getFilePathViaGUI, contentsOfDir
from matplotlib import pyplot as plt
from scipy import ndimage as ndifilters
from skimage import filters as skimagefilters

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
) -> Dict:
  ''' Finds the best match ROI for templates within search_image,
      and passes back the location information for all ROIs found.
  '''
  rois = {}
  num_rois = 0
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
    roi_parameters = {
      'ORIGIN_X': roi_origin_x,
      'ORIGIN_Y': roi_origin_y,
      'WIDTH':  roi_width,
      'HEIGHT': roi_height
    }
    rois[num_rois] = roi_parameters
    num_rois += 1

  # return a ordered and labelled set or ROI's
  roi_info = {}
  if rois[0]['ORIGIN_X'] < rois[1]['ORIGIN_X']:
    roi_info['left'] = rois[0]
    roi_info['right'] = rois[1]
  else:
    roi_info['left'] = rois[1]
    roi_info['right'] = rois[0]
  return roi_info


def roiInfoFromUserDrawings(input_image: np.ndarray) -> Dict:
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

  rois = {}
  num_rois = 0
  for roi_selection in roi_selections:
    x_start = roi_selection[0]
    x_end = x_start + roi_selection[2]
    if x_end - x_start <= 0:
      return None
    y_start = roi_selection[1]
    y_end = y_start + roi_selection[3]
    if y_end - y_start <= 0:
      return None

    roi_parameters = {
      'ORIGIN_X': roi_selection[0],
      'ORIGIN_Y': roi_selection[1],
      'WIDTH':  roi_selection[2],
      'HEIGHT': roi_selection[3]
    }
    rois[num_rois] = roi_parameters
    num_rois += 1

  # return a ordered and labelled set or ROI's
  roi_info = {}
  if rois[0]['ORIGIN_X'] < rois[1]['ORIGIN_X']:
    roi_info['left'] = rois[0]
    roi_info['right'] = rois[1]
  else:
    roi_info['left'] = rois[1]
    roi_info['right'] = rois[0]
  return roi_info


def computeMorphologyMetrics(
  search_image_path: str=None,
  template_image_paths: List[str]=None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  microns_per_pixel: float=None,
  use_midline_background: bool=True,
  write_result_images: bool=False,
  display_result_images: bool=True
):

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

  if isinstance(template_image_paths, str):
    template_image_paths = [template_image_paths]    
  if template_image_paths[0].lower() == 'select_files':
    left_template_image_path = getFilePathViaGUI('Select Left ROI Template File')    
    right_template_image_path = getFilePathViaGUI('Select Right ROI Template File')
    template_image_paths = [left_template_image_path, right_template_image_path]
  elif template_image_paths[0].lower() == 'draw_rois_once':
    drawn_roi_templates_dir = os.path.join(results_dir, 'drawn_template_images')
    os.mkdir(drawn_roi_templates_dir)
    file_name, file_extension = file_names[0]  # NOTE: this could be a selected file too
    search_image_for_template = cv.imread(os.path.join(base_dir, file_name + file_extension))
    rois_info = roiInfoFromUserDrawings(search_image_for_template)
    left_roi = rois_info['left']
    left_template_image = search_image_for_template[
      left_roi['ORIGIN_Y'] : left_roi['ORIGIN_Y'] + left_roi['HEIGHT'],
      left_roi['ORIGIN_X'] : left_roi['ORIGIN_X'] + left_roi['WIDTH']
    ]
    left_template_image_path = os.path.join(drawn_roi_templates_dir, 'template_image_1.tif')
    cv.imwrite(left_template_image_path, left_template_image)
    right_roi = rois_info['right']
    right_template_image = search_image_for_template[
      right_roi['ORIGIN_Y'] : right_roi['ORIGIN_Y'] + right_roi['HEIGHT'],
      right_roi['ORIGIN_X'] : right_roi['ORIGIN_X'] + right_roi['WIDTH']
    ]
    right_template_image_path = os.path.join(drawn_roi_templates_dir, 'template_image_2.tif')
    cv.imwrite(right_template_image_path, right_template_image)
    template_image_paths = [left_template_image_path, right_template_image_path]

  all_metrics = []
  image_analyzed_id = 0
  for file_name, file_extension in file_names:
      file_to_analyze = os.path.join(base_dir, file_name + file_extension)
      results_image, metrics = morphologyMetricsForImage(
        search_image_path=file_to_analyze,
        template_image_paths=template_image_paths,
        sub_pixel_search_increment=sub_pixel_search_increment,
        sub_pixel_refinement_radius=sub_pixel_refinement_radius,
        microns_per_pixel=microns_per_pixel
      )
      all_metrics.append(
        {
          'file': file_to_analyze,
          'horizontal_length': round(metrics['distance_between_rois'], 2),
          'left_edge_vertical_length': round(metrics['left_end_point_thickness'], 2),
          'mid_point_vertical_length': round(metrics['midpoint_thickness'], 2),
          'right_edge_vertical_length': round(metrics['right_end_point_thickness'], 2),
          'tissue_area': round(metrics['area_between_rois'], 2)
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
        print(f"horizontal inner distance between rois: {round(metrics['distance_between_rois'], 2)} (microns)")
        print(f"vertical thickness at inner edge of left ROI : {round(metrics['left_end_point_thickness'], 2)} (microns)")
        print(f"vertical thickness at midpoint between rois: {round(metrics['midpoint_thickness'], 2)} (microns)")
        print(f"vertical thickness at inner edge of right ROI : {round(metrics['right_end_point_thickness'], 2)} (microns)")
        print(f"area between rois: {round(metrics['area_between_rois'], 2)} (microns)")
        plt.imshow(cv.cvtColor(results_image, cv.COLOR_BGR2RGB))
        plt.show()
        print()

      image_analyzed_id += 1

  results_xlsx_name = 'results.xlsx'
  results_xlsx_path = os.path.join(results_dir, results_xlsx_name)  
  runtime_parameters = {
    'search_image_path' : base_dir,
    'template_image_paths': template_image_paths,
    'sub_pixel_search_increment' : sub_pixel_search_increment,
    'sub_pixel_refinement_radius' : sub_pixel_refinement_radius,    
    'microns_per_pixel': microns_per_pixel,
    'use_midline_background' : use_midline_background  
  }
  resultsToCSV(all_metrics, results_xlsx_path, runtime_parameters)


# TODO:
# we can find the edge of the points near the middle by:
# - segmenting the image with the triangle method and walk out from the midpoint in both up and down directions
# until the first backround pixel is found, this is the edge at the mid point.
# and we can find the edge for the end points by also walking out from the vertical midpoint, both up and down,
# - computing a measure of 2D smoothed local variance, then
# compute a moving average of say 3-5 pixels. when this measure of variance increases by more than say, 25%?
# we presume we've walked over the edge. This variance method might also work for the middle points.
# we could compute a measure of variance near the midpoint and presume the whole tissue has that intensity
# +/- the variance (stddev actually) and just walk out from the center line until the change is greater than 1 stddev
# (maybe that needs to be true for 3 of the following5 pixels to say that the 1st pixel that was > variance threshold is the edge)
# - and we can also combine these methods or use some as a guide for the others i.e. if one is very robust
# but not accurate, we can use it to limit where we look, then use one of the other methods that is more accurate.

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


def smoothedHorizontally(input_image: np.ndarray, sigma: float) -> np.ndarray:
  return ndifilters.gaussian_filter1d(
      input_image,
      sigma=sigma,
      mode='constant',
      axis=1
  )


def morphologyMetricsForImage(
  search_image_path: str=None,
  template_image_paths: List[str]=None,
  sub_pixel_search_increment: float = None,
  sub_pixel_refinement_radius: float = None,
  microns_per_pixel: float=None
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

  # load or draw the ROIs
  if isinstance(template_image_paths, str):
    template_image_paths = [template_image_paths]
  if template_image_paths[0].lower() == 'draw_rois':
    rois_info = roiInfoFromUserDrawings(search_image)
  else:
    if template_image_paths[0].lower() == 'select_files':
      template_1_image_path = getFilePathViaGUI('left template to find path')  
      template_2_image_path = getFilePathViaGUI('right template to find path')
      template_image_paths = [template_1_image_path, template_2_image_path]

    rois_info = roiInfoFromTemplates(
      search_image=search_image_gray_adjusted,
      template_image_paths=template_image_paths,
      sub_pixel_search_increment=None,
      sub_pixel_refinement_radius=None  
    )
  left_roi = rois_info['left']
  right_roi = rois_info['right']
  # TODO: deal with more than 2 roi's being drawn??? maybe not.

  # compute the distance between the inner sides of each ROI
  right_distance_marker_x = right_roi['ORIGIN_X']
  left_distance_marker_x = left_roi['ORIGIN_X'] + left_roi['WIDTH']
  if left_distance_marker_x >= right_distance_marker_x:
    # there was a problem and just so as not to kill the process
    # we separate the rois to allow things to keep working
    left_distance_marker_x = right_distance_marker_x - 1
  pixel_distance_between_rois = right_distance_marker_x - left_distance_marker_x
  distance_between_rois = microns_per_pixel*pixel_distance_between_rois
  horizontal_midpoint = left_distance_marker_x + pixel_distance_between_rois/2
  left_vertical_midpoint = left_roi['ORIGIN_Y'] + left_roi['HEIGHT']/2
  right_vertical_midpoint = right_roi['ORIGIN_Y'] + right_roi['HEIGHT']/2
  vertical_midpoint = (left_vertical_midpoint + right_vertical_midpoint)/2

  # find the points we think are at the upper/lower edges for the far left, mid and far right horizontal positions
  points_to_find_edges_at = np.asarray([int(left_distance_marker_x), int(horizontal_midpoint), int(right_distance_marker_x)])
  upper_edge_points = np.empty(len(points_to_find_edges_at))
  lower_edge_points = np.empty(len(points_to_find_edges_at))

  show_plots = False
  for index, point_to_find_edges_at in enumerate(points_to_find_edges_at):
    edge_point = outerEdges(
      search_image_gray[:, point_to_find_edges_at],
      vertical_midpoint=vertical_midpoint,
      show_plots=show_plots
    )
    upper_edge_points[index] = edge_point['upper_edge_pos']
    lower_edge_points[index] = edge_point['lower_edge_pos']

  # find ALL edge points of tissue between templates
  median_image = ndifilters.median_filter(search_image_gray, size=10)
  # plt.imshow(median_image)
  # plt.show()
  # median_image_path = '/storage/data/home/positimothy/dev/repo/curibio/optical-tracking/test_data/morphology/failing/images/out/median.tif'
  # cv.imwrite(median_image_path, median_image)
  points_to_find_edges = np.asarray([point for point in range(left_distance_marker_x, right_distance_marker_x)])
  top_edge_points = np.empty(len(points_to_find_edges))
  bottom_edge_points = np.empty(len(points_to_find_edges))
  show_plots = False
  
  for index, point_to_find_edges in enumerate(points_to_find_edges):
    edge_point = outerEdges(
      median_image[:, point_to_find_edges],
      vertical_midpoint=vertical_midpoint,
      show_plots=show_plots
    )
    top_edge_points[index] = edge_point['upper_edge_pos']
    bottom_edge_points[index] = edge_point['lower_edge_pos']

  # attempt to segment the image so we can find some edges we think are reasonable locations
  # which we can then use as a guide to determine if the previous edge points need to be adjusted
  # the mean thresholding method works best for a single threshold, (li works too but not as well)
  # multi also works but then you have to pick the phase (from whatever the phase is at the midpoint)
  segmented_image = smoothed(search_image_gray, sigma=10.0)
  segmented_image = ndifilters.median_filter(segmented_image, size=5)
  segmented_image = thresholded(segmented_image, method='mean', binarize=True)
  # plt.imshow(segmented_image)
  # plt.show()
  # segmented_image_image_path = '/storage/data/home/positimothy/dev/repo/curibio/optical-tracking/test_data/morphology/failing/images/out/segmented.tif'
  # cv.imwrite(segmented_image_image_path, segmented_image)

  # y-gradient
  compute_ddx = False
  compute_ddy = True
  y_grad_image = np.square(
    cv.Sobel(segmented_image, -1, int(compute_ddx), int(compute_ddy), ksize=5)
  )
  # plt.imshow(y_grad_image)
  # plt.show()

  # original + amount * (original - blurred)
  sharpened_image = median_image + 0.25*y_grad_image
  # plt.imshow(sharpened_image)
  # plt.show()

  # y_grad_image_path = '/storage/data/home/positimothy/dev/repo/curibio/optical-tracking/test_data/morphology/failing/images/out/y_grad.tif'
  # cv.imwrite(y_grad_image_path, y_grad_image)

  # using the edge found at the midpoint, walk along that edge in the segmented_image
  # i.e. look up and down +/- 1 pixel at the horizontally next "edge" pixel if it's too far from the segmented edge, adjust it
  # until we get to the far left/right edge point we found earlier
  # (Li method on the smoothed image might work to obtain just the tissue and separate it from the post background)

  # y_grad_image_normalized = normalized(y_grad_image)
  # search_image_gray_normalized = normalized(median_image)
  # enhanced_image = search_image_gray_normalized + y_grad_image_normalized
  # enhanced_image = normalized(enhanced_image, new_range=255.0).astype(np.uint8)
  # enhanced_image = smoothed(enhanced_image, sigma=6.0)
  # plt.imshow(enhanced_image)
  # plt.show()
  # enhanced_image_path = '/storage/data/home/positimothy/dev/repo/curibio/optical-tracking/test_data/morphology/failing/images/out/enhanced.tif'
  # cv.imwrite(enhanced_image_path, enhanced_image)
  # ##############################################################################
  # ##############################################################################


  # compute the area between the fitted curves
  edge_point_diffs = lower_edge_points - upper_edge_points
  area_between_rois = microns_per_pixel * np.sum(edge_point_diffs)

  # compute the vertical distance beteen "edges" for the left, mid, and right points
  left_end_point_thickness = microns_per_pixel*(lower_edge_points[0] - upper_edge_points[0])
  midpoint_thickness = microns_per_pixel*(lower_edge_points[1] - upper_edge_points[1])  
  right_end_point_thickness = microns_per_pixel*(lower_edge_points[2] - upper_edge_points[2])

  metrics = {
    'distance_between_rois': distance_between_rois,
    'midpoint_thickness': midpoint_thickness,
    'left_end_point_thickness': left_end_point_thickness,
    'right_end_point_thickness': right_end_point_thickness,
    'area_between_rois': area_between_rois
  }

  # create a version of the input that has the results drawn on it
  results_image = search_image.copy().astype(np.uint8)

  # draw the results metrics on the results image
  # draw the horizontal line between left and right ROI inner edges
  horizontal_line_position_colour_bgr = (255, 0, 0)
  cv.line(
    results_image,
    pt1=(left_distance_marker_x, int(left_vertical_midpoint)),
    pt2=(right_distance_marker_x, int(right_vertical_midpoint)),
    color=horizontal_line_position_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )

  # draw the upper and lower edges of object
  edge_contour_colour_bgr = (0, 0, 128)
  lower_edge_points_to_draw = zip(points_to_find_edges.astype(np.int32), bottom_edge_points.astype(np.int32))
  for x_pos, y_pos in lower_edge_points_to_draw:
    cv.circle(
      results_image,
      center=(x_pos, y_pos),
      radius=2,    
      color=edge_contour_colour_bgr,
      thickness=3,
      lineType=cv.LINE_AA
    )
  upper_edge_points_to_draw = zip(points_to_find_edges.astype(np.int32), top_edge_points.astype(np.int32))
  for x_pos, y_pos in upper_edge_points_to_draw:
    cv.circle(
      results_image,
      center=(x_pos, y_pos),
      radius=2,    
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
  midpoint_point_upper_edge_pos = round(upper_edge_points[1])
  midpoint_point_lower_edge_pos = round(lower_edge_points[1]) 
  cv.line(
    results_image,
    pt1=(int(horizontal_midpoint), midpoint_point_upper_edge_pos),
    pt2=(int(horizontal_midpoint), midpoint_point_lower_edge_pos),
    color=edge_width_lines_colour_bgr,
    thickness=3,
    lineType=cv.LINE_AA
  )  
  
  return results_image, metrics 


def normalized(input_image: np.ndarray, new_range: float = 512.0) -> np.ndarray:
  input_array_min: float = np.min(input_image)
  input_array_max: float = np.max(input_image)
  input_array_zero_origin: np.ndarray = input_image - input_array_min
  input_array_current_range: float = float(input_array_max - input_array_min)
  input_array_normalizer: float = new_range/input_array_current_range
  return input_array_zero_origin * input_array_normalizer


def gradmag(input_array: np.ndarray, order: int = 2) -> np.ndarray:
  input_array_gradient = np.gradient(input_array, edge_order=order)
  return np.square(input_array_gradient)


def yGradmag(input_array: np.ndarray) -> np.ndarray:
  # NOTE: this only returns the 0th x element because 
  # it's only used for a single vertical strip of pixels
  return np.square(
    cv.Sobel(input_array, -1, 0, 1, ksize=5)[:, 0]
  )


def movingAverage(input_array: np.ndarray, filter_radius: int = 5) -> np.ndarray:
  len_of_average = 2*filter_radius + 1
  ma_operator = np.ones(len_of_average)/len_of_average
  return np.convolve(input_array, ma_operator, mode='valid')


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
  # create a buffer of +/- 10% of image height to look for the other edge
  vertical_height = input_array_gradmag_ma.shape[0]
  candidate_range_buffer = int(vertical_height * 0.1)
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
  
  heading_row = '1'
  sheet[file_column + heading_row] = 'File'
  sheet[horizontal_column + heading_row] = 'Horizontal Length'
  sheet[left_edge_column + heading_row] = 'Left Edge Length'
  sheet[mid_point_column + heading_row] = 'Mid Point Length'
  sheet[right_edge_column + heading_row] = 'Right Edge Length'
  sheet[area_column + heading_row] = 'Area'

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

  # add the runtime parameters
  runtime_config_lables_column = 'H'
  runtime_config_data_column = 'I'
  sheet[runtime_config_lables_column + heading_row] = 'runtime parameters'
  sheet[runtime_config_lables_column + str(data_row + 0)] = 'input images'
  sheet[runtime_config_data_column + str(data_row + 0)] = runtime_parameters['search_image_path']
  sheet[runtime_config_lables_column + str(data_row + 1)] = 'input templates'
  template_paths = runtime_parameters['template_image_paths'][0] + ', ' + runtime_parameters['template_image_paths'][1]
  sheet[runtime_config_data_column + str(data_row + 1)] = template_paths 
  sheet[runtime_config_lables_column + str(data_row + 2)] = 'microns per pixel'
  sheet[runtime_config_data_column + str(data_row + 2)] = runtime_parameters['microns_per_pixel']
  sheet[runtime_config_lables_column + str(data_row + 3)] = 'sub pixel refinement search increment'
  sheet[runtime_config_data_column + str(data_row + 3)] = runtime_parameters['sub_pixel_search_increment']
  sheet[runtime_config_lables_column + str(data_row + 4)] = 'sub pixel refinement radius'
  sheet[runtime_config_data_column + str(data_row + 4)] = runtime_parameters['sub_pixel_refinement_radius']

  workbook.save(filename=path_to_output_file)


if __name__ == '__main__':
  
  computeMorphologyMetrics(
    search_image_path=None,
    template_image_paths=None,
    sub_pixel_search_increment=None,
    sub_pixel_refinement_radius=None
  )
