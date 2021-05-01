#! /usr/bin/env python

import argparse
import os
import glob
import shutil
import sys
import json
import numpy as np
import math
import numbers
import pathlib
import zipfile
import openpyxl # pip install --user openpyxl
import cv2 as cv # pip install --user opencv-python
from skimage import filters as skimage_filters # pip install --user scikit-image
from skimage import exposure as skimage_exposure
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename, askdirectory

from video2jpgs import video_to_jpgs

# TODO: If accuracy isn't acceptable, try 
#       - sub pixel version of current method, 
#       - different match measures like mutual information and/or grad angle,
#       - adding in rotation of the template +/- some small angle for each position in the video frame.
# TODO: option to add fixed grid lines in say a red colour


def contrast_enhanced(image_to_adjust):
  '''
  Performs an automatic adjustment of the input intensity range to enhance contrast.
  
  Args:
    image_to_adjust: the image to adjust the contrast of. 
  '''
  # optimal_threshold = skimage_filters.threshold_yen(image_to_adjust)
  optimal_threshold = skimage_filters.threshold_minimum(image_to_adjust)
  uint8_range = (0, 255)
  return skimage_exposure.rescale_intensity(image_to_adjust, in_range=(0, optimal_threshold), out_range=uint8_range)


def best_template_match_ccoef(video_to_search, template_to_find, max_frames_to_check) -> np.ndarray:
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
      print("Error. Unexpected problem during video frame capture. Exiting.")

    # track the template
    frame = contrast_enhanced(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).astype(np.uint8)
    match_results = cv.matchTemplate(frame, template_to_find, cv.TM_CCOEFF)
    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)
    if match_measure > best_match_measure:
      best_match_measure = match_measure
      best_match_coordinates = match_coordinates
      best_match_frame = frame

  # reset the video to where it was initially
  video_to_search.set(cv.CAP_PROP_POS_FRAMES, initial_frame_pos)

  # cut out a new best match template from the best match frame
  template_height = template_to_find.shape[0]
  template_width = template_to_find.shape[1]
  new_template_start_x = best_match_coordinates[0]
  new_template_end_x = new_template_start_x + template_width
  new_template_start_y = best_match_coordinates[1]
  new_template_end_y = new_template_start_y + template_height
  new_template = best_match_frame[new_template_start_y:new_template_end_y, new_template_start_x:new_template_end_x]
  return new_template


def best_template_match(video_to_search, template_to_find, max_frames_to_check) -> np.ndarray:
  return best_template_match_ccoef(video_to_search, template_to_find, max_frames_to_check)


def run_track_template(
  input_video_path: str,
  template_guide_image_path: str,
  output_video_path: str = None,
  guide_match_search_seconds: float = None,
  microns_per_pixel: float = None,
  output_conversion_factor: float = None
) -> (str, [{}], float):
  '''
  Tracks a template image through each frame of a video.

  Args:
    input_video_path:           path of the input video to track the template.
    template_guide_image_path:        path to an image that will be used as a template to match.
    output_video_path:          path to write a video with the tracking results visualized. 
    guide_match_search_seconds:  approximate number of seconds for template to complete a full period of movement.',
  Returns:
    and error string (or None if no errors occurred), a list of per frame tracking results, & the frame rate.
  '''
  error_msg = None
  frames_per_second = float(0.0)

  if input_video_path is None:
    error_msg = "ERROR. No path provided to an input video. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second)

  # open a video reader stream
  input_video_stream = cv.VideoCapture(input_video_path)
  if not input_video_stream.isOpened():
    error_msg = "Error. Can't open videos stream for capture. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second)
  frame_width  = int(input_video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
  frame_height = int(input_video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
  frame_size = (frame_width, frame_height)
  frames_per_second = input_video_stream.get(cv.CAP_PROP_FPS)

  # open the template image
  template = cv.imread(template_guide_image_path)
  if template is None:
    error_msg = "ERROR. The path provided for template does not point to an image file. Nothing has been tracked."
    return (error_msg, [{}], frames_per_second)
  template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
  if guide_match_search_seconds is None:
    max_frames_to_check = None
  else:
    max_frames_to_check = int(math.ceil(frames_per_second*float(guide_match_search_seconds)))
    template = best_template_match(input_video_stream, template, max_frames_to_check)
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
      return (error_msg, [{}], frames_per_second)

    output_video_codec = cv.VideoWriter_fourcc(*'mp4v') #cv.VideoWriter_fourcc(*'DIVX') #     
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
    frame_returned, raw_frame = input_video_stream.read()
    if not frame_returned:
      error_msg = "Error. Unexpected problem during video frame capture. Exiting."
      return (error_msg, [{}], frames_per_second)

    frame = contrast_enhanced(cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)).astype(np.uint8)

    # find the best match for the template in the current frame
    match_results = cv.matchTemplate(frame, template, cv.TM_CCOEFF)

    _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)  # opencv returns in x, y order
    template_origin_x = match_coordinates[0]
    template_origin_y = match_coordinates[1]

    tracking_results.append(
      {
        'FRAME_NUMBER': frame_number,
        'ELAPSED_TIME': frame_number/frames_per_second,
        'MATCH_MEASURE': match_measure,
        'Y_DISPLACEMENT': 0,
        'X_DISPLACEMENT': 0,
        'XY_DISPLACEMENT': 0,
        'TEMPLATE_MATCH_ORIGIN_X': template_origin_x,
        'TEMPLATE_MATCH_ORIGIN_Y': template_origin_y,
      }
    )
    # update the min and max positions of the template origin for ALL frames
    if template_origin_y < min_template_origin_y: 
      min_template_origin_y = template_origin_y
      min_template_origin_x = template_origin_x
    if template_origin_y > max_template_origin_y:
      max_template_origin_y = template_origin_y
      max_template_origin_x = template_origin_x

    # draw a grid over the region in the frame where the template matched
    if output_video_stream is not None:
      grid_colour_bgr = (255, 128, 0)      
      grid_square_diameter = 10
      frame = raw_frame

      # mark the template ROI on each frame
      # first draw a rectangle around the template border
      grid_width = int(template_width/grid_square_diameter)*grid_square_diameter
      grid_height = int(template_height/grid_square_diameter)*grid_square_diameter
      template_bottom_right_x = min(frame_width, template_origin_x + grid_width)
      template_bottom_right_y = min(frame_height, template_origin_y + grid_height)
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
      for line_pos_y in range(template_origin_y + grid_square_diameter, template_bottom_right_y, grid_square_diameter):
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
      for line_pos_x in range(template_origin_x + grid_square_diameter, template_bottom_right_x, grid_square_diameter):
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
    y_displacement*= float(output_conversion_factor)
    frame_info['Y_DISPLACEMENT'] = y_displacement
    if positive_x_movement_slope:
      x_displacement = (max_template_origin_x - frame_info['TEMPLATE_MATCH_ORIGIN_X'])*float(microns_per_pixel)
    else:
      x_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_X'] - min_template_origin_x)*float(microns_per_pixel)
      x_displacement*= float(output_conversion_factor)
    frame_info['X_DISPLACEMENT'] = x_displacement 
    frame_info['XY_DISPLACEMENT'] = math.sqrt(x_displacement*x_displacement + y_displacement*y_displacement)
    adjusted_tracking_results.append(frame_info)

  return (error_msg, adjusted_tracking_results, frames_per_second)


def results_to_csv(
  tracking_results: [{}],
  path_to_template_file: str,
  path_to_output_file,
  frames_per_second: float,
  well_name: str = None, 
  date_stamp: str = '1010-01-01'
):    
  if path_to_template_file is None:
    workbook = openpyxl.Workbook()  # open a blank workbook
  else:  # open the template workbook
    shutil.copyfile(path_to_template_file, path_to_output_file)
    workbook = openpyxl.load_workbook(filename=path_to_output_file)
  sheet = workbook.active

  if well_name is None:
    well_name = 'Z01'
  sheet['E2'] = well_name
  sheet['E3'] = date_stamp + ' 00:00:00'
  sheet['E4'] = 'NA'  # plate barcode
  sheet['E5'] = frames_per_second
  sheet['E6'] = 'y'   # do twiches point up
  sheet['E7'] = 'NA'  # microscope name

  # set the time and post displacement fields
  template_start_row = 2
  time_column = 'A'
  displacement_column = 'B'
  num_rows_to_write = len(tracking_results)
  for results_row in range(num_rows_to_write):
      tracking_result = tracking_results[results_row]
      elapsed_time = float(tracking_result['ELAPSED_TIME'])
      post_displacement = float(tracking_result['XY_DISPLACEMENT'])
      sheet_row = results_row + template_start_row
      sheet[time_column + str(sheet_row)] = elapsed_time
      sheet[displacement_column + str(sheet_row)] = post_displacement

  workbook.save(filename=path_to_output_file)


def contents_of_dir(dir_path: str, search_terms: [str]) -> ([str], [('str', 'str')]):
  if os.path.isdir(dir_path):
    base_dir = dir_path
    for search_term in search_terms:
      glob_search_term = '*' + search_term + '*'
      file_paths = glob.glob(os.path.join(dir_path, glob_search_term))
      if len(file_paths) > 0:
        # we've found videos so don't bother searching for more
        break
  else:
    # presume it's actually a single file path
    base_dir = os.path.dirname(dir_path)
    file_paths = [dir_path]
  files = []
  for file_path in file_paths:
      file_name, file_extension = os.path.splitext(os.path.basename(file_path))
      files.append((file_name, file_extension))
  return base_dir, files


def get_dir_path_via_gui() -> str:
  # pop up a dialog for directory selection
  tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
  return askdirectory() # show an "Open" dialog box and return the path to the selected dir


def get_file_path_via_gui() -> str:
  # pop up a dialog for file selection
  tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
  return askopenfilename() # show an "Open" dialog box and return the path to the selected file


def verified_inputs(config: {}) -> (str, [{}]):
  '''

  '''
  error_msgs = []
  # check the dir path to input videos
  open_video_dir_dialog = False
  if 'input_video_path' in config:
    if config['input_video_path'] is None:
      open_video_dir_dialog = True
    else:
      if config['input_video_path'].lower() == 'ask' or config['input_video_path'].lower() == '':
        open_video_dir_dialog = True
      elif not os.path.isdir(config['input_video_path']):
        error_msgs.append('Input path to video/s does not exist.')      
  else:
      open_video_dir_dialog = True
  # pop up a dialog to select the dir for videos if required
  if open_video_dir_dialog:
    print()
    print("waiting for user input (video dir) via pop up dialog box...")
    dir_path_via_gui = get_dir_path_via_gui()
    if dir_path_via_gui == () or dir_path_via_gui == '':
      error_msgs.append('No input path to video/s was provided.')
    else:
      config['input_video_path'] = dir_path_via_gui
    print("...user input obtained from pop up dialog box.")
    print()

  # check the file path to the template image
  open_template_dir_dialog = False
  if 'template_guide_image_path' in config:
    if config['template_guide_image_path'] is None:
      open_template_dir_dialog = True
    else:
      if config['template_guide_image_path'].lower() == 'ask' or config['template_guide_image_path'].lower() == '':
        open_template_dir_dialog = True
      elif not os.path.isfile(config['template_guide_image_path']):
        error_msgs.append('Input template image file does not exist.')      
  else:
      open_template_dir_dialog = True
  # pop up a dialog to select the template file if required
  if open_template_dir_dialog:
    print()
    print("waiting for user input (template to use) via pop up dialog box...")    
    file_path_via_gui = get_file_path_via_gui()
    if file_path_via_gui == () or file_path_via_gui == '':
      error_msgs.append('No input template image path was provided.')
    else:
      config['template_guide_image_path'] = file_path_via_gui
    print("...user input obtained from pop up dialog box.")
    print()
  
  # barf if there was an error with either the input video dir path or template file path
  if len(error_msgs) > 0:
    error_msg = 'ERROR.'
    for error_string in error_msgs:
      error_msg = error_msg + ' ' + error_string
    error_msg += ' Nothing to do. Exiting.'
    print(error_msg)
    sys.exit(1)
  
  template_guide_image_path = config['template_guide_image_path']
  file_extensions = ['mp4', 'avi']
  base_dir, video_files = contents_of_dir(dir_path=config['input_video_path'], search_terms=file_extensions)
  
  results_dir_path = os.path.join(base_dir, 'results')
  results_json_dir_path = os.path.join(results_dir_path, 'json')
  results_xlsx_dir_path = os.path.join(results_dir_path, 'xlsx')
  results_video_dir_path = os.path.join(results_dir_path, 'video')
  results_video_frames_dir_path = os.path.join(results_video_dir_path, 'frames')  

  dirs = {
    'base_dir': base_dir,
    'results_dir_path': results_dir_path,
    'results_json_dir_path': results_json_dir_path,
    'results_xlsx_dir_path': results_xlsx_dir_path,
    'results_video_dir_path': results_video_dir_path,
    'results_video_frames_dir_path': results_video_frames_dir_path
  }

  if 'path_to_excel_template' not in config:
    path_to_excel_template = None
  else:
    path_to_excel_template = config['path_to_excel_template']

  if 'guide_match_search_seconds' not in config:
    guide_match_search_seconds = None
  else:
    guide_match_search_seconds = config['guide_match_search_seconds']
  
  if 'microns_per_pixel' not in config:
    microns_per_pixel = None
  else:
    microns_per_pixel = config['microns_per_pixel']

  if 'output_conversion_factor' not in config:
    output_conversion_factor = 1.0
  else:
    output_conversion_factor = config['output_conversion_factor']

  # set all the values needed to run template matching on each input video
  configs = []
  for file_name, file_extension in video_files:

    # check the file name conforms to minimum requirements
    file_name_parsed = file_name.split("_")
    min_num_words_in_file_name = 3
    if len(file_name_parsed) < min_num_words_in_file_name:
      print(f'ERROR. the input video {file_name} does not have a valid name.')
      print('The pattern must conform to "datestamp_x_wellname" i.e. 1010-01-01_any_other_words_A001')
      sys.exit(1)

    # make sure a valid well name can be extracted from the file name
    well_name_position = -1
    well_name = file_name_parsed[well_name_position]
    well_name_valid = True
    well_name_length = 4
    if not len(well_name) == well_name_length:
      well_name_valid = False
    well_name_letter_part = well_name[0]
    if not well_name_letter_part.isalpha():
      well_name_valid = False    
    well_name_number_part = well_name[1:]
    if not well_name_number_part.isdigit():
      well_name_valid = False
    if not well_name_valid:
      print("ERROR. An input video file does not contain a valid well name as expected as the last word.")
      print(f"The last word of the filename is: {well_name}")
      print("The last word of the file name must be a letter followed by a zero padded 3 digit number i.e. A001 or D006")
      sys.exit(1)

    # make sure a valid date stamp can be extacted from the file name
    date_stamp_position = 0
    date_stamp = file_name_parsed[date_stamp_position]
    date_stamp_parsed = date_stamp.split("-")
    num_parts_in_date = 3
    date_stamp_valid = True
    if len(date_stamp_parsed) != num_parts_in_date:
      date_stamp_valid = False
    else:
      for date_part in date_stamp_parsed:
        if not date_part.isdigit():
          date_stamp_valid = False
    if not date_stamp_valid:
      print("ERROR. An input video file does not contain a valid date stamp as expected for the first word.")
      print(f"The first word of the filename is: {date_stamp}")
      print("The first word of the file name must be of the form yyyy-mm-dd i.e. 1010-01-01")
      sys.exit(1)

    # set all the required path values
    input_video_path = os.path.join(base_dir, file_name + file_extension)
    output_video_frames_dir_path = os.path.join(results_video_frames_dir_path, file_name)
    output_json_path = os.path.join(results_json_dir_path, file_name + '-results.json')
    path_to_excel_results = os.path.join(results_xlsx_dir_path, file_name + '-reslts.xlsx')
    output_video_path = os.path.join(results_video_dir_path, file_name + '-results' + file_extension)

    configs.append({
      'input_video_path': input_video_path,
      'template_guide_image_path': template_guide_image_path,
      'output_video_path': output_video_path,
      'output_video_frames_dir_path': output_video_frames_dir_path,
      'output_json_path': output_json_path,
      'path_to_excel_template': path_to_excel_template,
      'path_to_excel_results': path_to_excel_results,
      'guide_match_search_seconds': guide_match_search_seconds,
      'microns_per_pixel': microns_per_pixel,
      'output_conversion_factor': output_conversion_factor,
      'well_name': well_name,
      'date_stamp': date_stamp,      
    })

  return (dirs, configs)


def track_templates(config: {}):
  dirs, args = verified_inputs(config)
   
  # make all the dirs needed for writing the results 
  # unless they already exist in which case we need to barf
  dirs_exist_error_message = ''
  if os.path.isdir(dirs['results_dir_path']):
    dirs_exist_error_message += "results dir already exists. Cannot overwrite.\n"
  if os.path.isdir(dirs['results_json_dir_path']):
    dirs_exist_error_message += "json results dir already exists. Cannot overwrite.\n"
  if os.path.isdir(dirs['results_xlsx_dir_path']):
    dirs_exist_error_message += "xlsx results dir already exists. Cannot overwrite.\n"
  if os.path.isdir(dirs['results_video_dir_path']):
    dirs_exist_error_message += "video results dir already exists. Cannot overwrite.\n"
  if len(dirs_exist_error_message) > 0:
    dirs_exist_error_message = "ERROR.\n" + dirs_exist_error_message + "Nothing Tracked."
    print(dirs_exist_error_message)
    sys.exit(1)
  os.mkdir(dirs['results_dir_path'])
  os.mkdir(dirs['results_json_dir_path'])
  os.mkdir(dirs['results_xlsx_dir_path'])
  os.mkdir(dirs['results_video_dir_path'])
  os.mkdir(dirs['results_video_frames_dir_path'])

  # run the tracking routine on each input video
  # and write out the results
  print("\nTemplate Tracker running...\n") 
  for input_args in args:
    error_msg, tracking_results, frames_per_second = run_track_template(
      input_args['input_video_path'],
      input_args['template_guide_image_path'],
      input_args['output_video_path'],
      input_args['guide_match_search_seconds'],
      input_args['microns_per_pixel'],
      input_args['output_conversion_factor']
    )

    if error_msg is not None:
      print(error_msg)
      sys.exit(1)

    # write out the results video as frames
    os.mkdir(input_args['output_video_frames_dir_path'])
    video_to_jpgs(input_args['output_video_path'], input_args['output_video_frames_dir_path'])

    # write the results as xlsx
    # TODO: extract the date part from the first part of the file name
    results_to_csv(
      tracking_results,
      input_args['path_to_excel_template'],
      input_args['path_to_excel_results'],
      frames_per_second,
      input_args['well_name'], 
      input_args['date_stamp']
    )

    # write the run config and results as json
    if input_args['output_json_path'] is not None:
      tracking_results_complete = {
        "INPUT_ARGS": input_args,
        "RESULTS": tracking_results
      }
      with open(input_args['output_json_path'], 'w') as outfile:
        json.dump(tracking_results_complete, outfile, indent=4)

  # create a zip archive and write all the xlsx files to it
  xlsx_archive_file_path = os.path.join(dirs['results_dir_path'], 'xlsx-results.zip')
  xlsx_archive = zipfile.ZipFile(xlsx_archive_file_path, 'w')
  for dir_name, _, file_names in os.walk(dirs['results_xlsx_dir_path']):
      for file_name in file_names:
          file_path = os.path.join(dir_name, file_name)
          xlsx_archive.write(file_path, os.path.basename(file_path))
  xlsx_archive.close()

  print("\n...Template Tracker complete.\n")


def config_from_json(json_config_path) -> (str, [{}]):
  json_file = open(json_config_path)
  config = json.load(json_file)
  return config


def config_from_cmdline(cmdline_args) -> dict:
  config = {}
  config['input_video_path'] = cmdline_args.input_video_path
  config['template_guide_image_path'] = cmdline_args.template_guide_image_path
  config['output_path'] = cmdline_args.output_video_path
  config['guide_match_search_seconds'] = cmdline_args.guide_match_search_seconds
  config['microns_per_pixel'] = cmdline_args.microns_per_pixel
  config['path_to_excel_template'] = cmdline_args.path_to_excel_template  
  return config


if __name__ == '__main__':
  # os.path.expanduser('~')
  # home_dir = pathlib.Path.home()

  # read in a config file & parse the input args
  parser = argparse.ArgumentParser(
      description='Tracks a template image through each frame of a video.',
  )
  parser.add_argument(
      '--input_video_path',
      default=None,
      help='Path of input video/s to track a template.',
  )
  parser.add_argument(
      '--template_guide_image_path',
      default=None,
      help='Path to an image that will be used as a template to match.',
  )
  parser.add_argument(
      '--path_to_excel_template',
      default=None,
      help='path to exel spread sheet used as a template to write the results into',
  )
  parser.add_argument(
    '--json_config_path',
    default=None,
    help='Path of a json file with run config parameters'      
  )
  parser.add_argument(
      '--output_path',
      default=None,
      help='Path to write tracking results.',
  )
  parser.add_argument(
      '--guide_match_search_seconds',
      default=None,
      help='number of seconds to search the video for the best match with the guide template.',
  )
  parser.add_argument(
      '--microns_per_pixel',
      default=None,
      help='conversion factor for pixel distances to microns',
  )
  raw_args = parser.parse_args()

  if raw_args.json_config_path is not None:
    config = config_from_json(raw_args.json_config_path)
  else:
    config = config_from_cmdline(raw_args)

  track_templates(config)


# TODO:
# the template we use in the algo should be called the roi_template and the one we get from the user 
# should be called guide_template
# so we get the guide_template and use it to find an roi_template in the video we're searching.
# we could get opencv to ask the user to define the roi since apparently it has a gui to do that.

# TODO:
# after each templateMatch step, we do a further brute force search +/- say 2 pixels around the result
# in increments of say 0.2 pixels (need to shift the image and use good interpolation) and perform
# cv.computeECC(). so for each frame there'd be 25x25 calls to computeECC() to find the best sub pixel match.
# 