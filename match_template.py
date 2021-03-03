import argparse
import glob
import json
import os
import sys
import cv2 as cv # pip install --user opencv-python
import numpy as np # pip install --user numpy


def template_match(input_path: str = None, template_path: str = None) -> {}:
  '''
  Finds a region in the input image/s where the template image matches best.

  Args:
    input_path: full path to an image or directory of images to search for a match.
    template_path: full path to the template image used for matching.
  Returns:
    a dictionary with results of the matching process.
  '''
  # check that all the input parameters have been provided and are valid
  if template_path is None:
    error_message = "ERROR. No path provided to a template file for matching. Nothing to do. "
    return {'STATUS': 'ERROR', 'STATUS_DESCRIPTION': error_message}
  if input_path is None:
    error_message = "ERROR. No path provided to an input file or dir of files. Nothing to do. "
    return {'STATUS': 'ERROR', 'STATUS_DESCRIPTION': error_message}

  # collect all the input files to search
  if not os.path.isdir(input_path):
    input_files = [input_path]  # just one file to search
  else:  # a directory of files to search
    input_files = []
    files_in_dir = glob.glob(os.path.join(input_path, '*'))
    for filename in files_in_dir:
        input_files.append(filename)
    input_files.sort() # there's no reason for this sorting other than safety i.e.
                       # in case anybody presumed alpha numeric order which glob doesn't guarantee.

  # load the template as a gray scale image
  template = cv.cvtColor(cv.imread(template_path), cv.COLOR_BGR2GRAY)
  
  # search all the input files and record the match results
  best_match_measure = 0.0
  best_match_coordinates = ''
  best_match_file = ''
  for input_file in input_files:
    image_to_search = cv.cvtColor(cv.imread(input_file), cv.COLOR_BGR2GRAY)
    match_results = cv.matchTemplate(image_to_search, template, cv.TM_CCOEFF_NORMED)
    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)
    if match_measure > best_match_measure:
      best_match_measure = match_measure
      best_match_coordinates = match_coordinates
      best_match_file = input_file
  results = {
    'BEST_MATCH_FILE': best_match_file,    
    'MATCH_MEASURE': best_match_measure,
    'TEMPLATE_MATCH_ORIGIN_X': best_match_coordinates[0],
    'TEMPLATE_MATCH_ORIGIN_Y': best_match_coordinates[1], 
    'TEMPLATE_WIDTH': template.shape[::-1][0],
    'TEMPLATE_HEIGHT': template.shape[::-1][1]   
  }

  # sanity check the results
  min_match_measure = 0.5  # TODO: arbitrary value. need to determine something sensible or user defined
  if best_match_measure < min_match_measure:
    results['STATUS'] = 'FAIL'
    results['STATUS_DESCRIPTION'] = 'no reliable match found'    
  else:
    results['STATUS'] = 'SUCCESS' 
    results['STATUS_DESCRIPTION'] = 'match found'

  return results


def run_template_match(input_path: str, template_path: str, results_json_path: str, visualize_match: bool):
  '''
  Run template_match() and then perform various user specified operations like writing results to json,
  writing out an image with the results of the template matching visible.

  Args:
    input_path: path to an image (or dir of images) to search for the location of a template match.
    template_path: path to a template image used to search for a matching location withing the input image/s.
    results_json_path: path to write a json file with the results of the template matching.
    visualize_match: switch to control writing a version of the best match file with the best match ROI drawn on it.
  '''

  results = template_match(input_path, template_path)

  # print the results or dump them to a json as requested
  if results_json_path is None:
    # print the results to std out
    for key, value in results.items():
      print(f'{key}: {value}')
  else:
      with open(results_json_path, 'w') as outfile:
          json.dump(results, outfile, indent=4)

  if results['STATUS'] != 'SUCCESS':
    sys.exit(1)

  if visualize_match:
    # load the best match file as a gray scale image
    file_path = results['BEST_MATCH_FILE']
    matched_file = cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2GRAY)
    # draw a rectangle around the area we matched with the template
    template_origin_x = results['TEMPLATE_MATCH_ORIGIN_X']
    template_width = results['TEMPLATE_WIDTH']
    template_bottom_right_x = template_origin_x + template_width
    template_origin_y = results['TEMPLATE_MATCH_ORIGIN_Y']
    template_height = results['TEMPLATE_HEIGHT']
    template_bottom_right_y = template_origin_y + template_height    
    cv.rectangle(
      matched_file,
      (template_origin_x, template_origin_y),
      (template_bottom_right_x, template_bottom_right_y),
      color=125,
      thickness=1
    )
    # write a version of the best matched file 
    # with the region of the matched drawn in
    file_name = os.path.basename(file_path)      
    matched_file_name = './template_matched_file-' + file_name + '.jpg'
    cv.imwrite(matched_file_name, matched_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='find the best match position of a template in a single or directory of images.',
    )
    parser.add_argument(
        'input_path',
        default=None,
        help='path to an image (or dir of images) to search for the location of a template match.',
    )
    parser.add_argument(
        'template_path',
        default=None,
        help='path to a template image used to search for a matching location withing the input image/s',
    )
    parser.add_argument(
        '--results_json_path',
        default=None,
        help='path to write a json file with the results of the template matching.',
    )
    parser.add_argument(
        '-visualize_match',
        action='store_true',
        help='switch to control writing a version of the best match file with the best match ROI drawn on it.',
    )
    args = parser.parse_args()
    run_template_match(args.input_path, args.template_path, args.results_json_path, args.visualize_match)
