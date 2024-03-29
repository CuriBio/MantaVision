#! /usr/bin/env python
import sys
import math
import numpy as np
from scipy.ndimage import shift
import cv2 as cv
from video_api import VideoReader, VideoWriter
from typing import Tuple, List, Dict


# TODO: change naming of dynamic post to flexible
#       change naming of fixed post to rigid or fixed post

# TODO: guard against having no contractions or relaxations or neither in the frequency estimation.
# TODO: turn the code that captures all ROI's up front into a separate function that can be called
#       and returns a struct with all the ROI info required. Start out doing exactly what Ca2+
#       analysis does so it can just be swapped straight in. and then change tracking to use it.

# TODO: change trackTemplate to return a dict

# TODO: parallelize the computation of matching for each frame. i.e. if we have 10 processors,
#       split up the search space into
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


def trackTemplate(
    input_video_path: str,
    template_guide_image_path: str,
    template_rgb: np.ndarray = None,
    template_gray: np.ndarray = None,
    output_video_path: str = None,
    guide_match_search_seconds: float = None,
    microns_per_pixel: float = None,
    output_conversion_factor: float = None,
    sub_pixel_search_increment: float = None,
    sub_pixel_refinement_radius: float = None,
    user_roi_selection: bool = True,
    max_translation_per_frame: Tuple[float] = None,
    max_rotation_per_frame: float = None,
    contraction_vector: Tuple[int] = None,
    blank_frame_stddev_cutoff: float = 100.0
) -> Tuple[Tuple[str, str], List[Dict], 'str', float, float, np.ndarray, int]:
    """
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
    """

    if microns_per_pixel is None:
        microns_per_pixel = 1.0
    if output_conversion_factor is None:
        output_conversion_factor = 1.0
    if max_rotation_per_frame is not None:
        rotation_increment = 0.5
    else:
        rotation_increment = 0.0

    error_msg = None
    warning_msg = None
    frames_per_second = float(0.0)
    if input_video_path is None:
        error_msg = "ERROR. No path provided to an input video. Nothing has been tracked."
        return error_msg, [{}], 0.0, frames_per_second, None, -1

    # open a video reader stream
    try:
        input_video_stream = VideoReader(input_video_path)
    except RuntimeError as runtime_error:
        print()
        print(runtime_error)
        print()
        sys.exit(1)

    if not input_video_stream.isOpened():
        error_msg = "Error. Can't open videos stream for capture. Nothing has been tracked."
        return error_msg, [{}], 0.0, frames_per_second, None, -1
    frame_width = int(input_video_stream.frameWidth())
    frame_height = int(input_video_stream.frameHeight())
    frames_per_second = input_video_stream.avgFPS()

    # get the template to track
    if template_rgb is not None:
        if template_gray is None:
            template_gray = cv.cvtColor(template_rgb, cv.COLOR_BGR2GRAY)
    else:
        template_gray = None
        if user_roi_selection or template_guide_image_path is None:
            # user needs to draw the ROI
            template_as_guide = None
            max_frames_to_check = None
        else:
            if guide_match_search_seconds is None:
                # user wants to use a template as is
                template_rgb = cv.imread(template_guide_image_path)
                if template_rgb is None:
                    error_msg = "\nERROR."
                    error_msg += " Template Path provided does not point to an image file."
                    error_msg += " Nothing has been tracked.\n"
                    return error_msg, [{}], 0.0, frames_per_second, None, -1
                template_gray = cv.cvtColor(template_rgb, cv.COLOR_BGR2GRAY)
            else:
                # user wants to use a template to find a match in the video to use as a template
                max_frames_to_check = int(math.ceil(frames_per_second * float(guide_match_search_seconds)))
                template_as_guide = intensityAdjusted(
                    cv.cvtColor(cv.imread(template_guide_image_path), cv.COLOR_BGR2GRAY)
                )
                if template_as_guide is None:
                    error_msg = "\nERROR."
                    error_msg += " Template Path provided does not point to an image file."
                    error_msg += " Nothing has been tracked.\n"
                    return error_msg, [{}], 0.0, frames_per_second, None, -1

        if template_gray is None:
            # create the new templates
            template_rgb, template_gray = templateFromInputROI(
                input_video_stream,
                template_as_guide,
                max_frames_to_check,
                blank_frame_stddev_cutoff
            )

    template = intensityAdjusted(template_gray)
    template_height = template.shape[0]
    template_half_height = template_height / 2.0
    template_width = template.shape[1]
    template_half_width = template_width / 2.0

    # open an output (writable) video stream if required
    if output_video_path is not None:
        video_writer = VideoWriter(
            path=output_video_path,
            width=input_video_stream.frameVideoWidth(),
            height=input_video_stream.frameVideoHeight(),
            time_base=input_video_stream.timeBase(),
            fps=input_video_stream.avgFPS(),
            bitrate=input_video_stream.bitRate()
        )
    else:
        video_writer = None

    # track the template in the video stream
    min_x_origin = (frame_width, frame_height)
    max_x_origin = (0, 0)
    min_x_frame = 0
    max_x_frame = 0
    min_y_origin = (frame_width, frame_height)
    max_y_origin = (0, 0)
    min_y_frame = 0
    max_y_frame = 0

    tracking_results: List[Dict] = []
    best_match_origin_x = None
    best_match_origin_y = None
    best_match_rotation = 0.0

    prev_frame_blank = True
    while input_video_stream.next():

        current_frame = input_video_stream.frameGray()
        frame_std_dev = np.std(current_frame)
        frame_number = input_video_stream.framePosition()
        if frame_std_dev < blank_frame_stddev_cutoff:
            # don't process blank frames
            prev_frame_blank = True  # only applies to the next frame
            print(f"WARNING. Frame {frame_number} appears to be blank. Ignoring.")
            match_measure = 0.0
            match_success = False
            match_coordinates = None
            best_match_origin_x = 0
            best_match_origin_y = 0
            best_match_rotation = 0.0
            flipped_image_best_match_rotation = 0.0
            if video_writer is not None:
                frame = input_video_stream.frameVideoRGB()
                video_writer.writeFrame(frame, input_video_stream.frameVideoPTS())
        else:
            if max_rotation_per_frame is None:
                search_set = [{'angle': 0.0, 'frame': current_frame}]
            else:
                search_set = []
                rotation_angle = best_match_rotation - max_rotation_per_frame
                max_rotation_angle = best_match_rotation + max_rotation_per_frame + rotation_increment
                while rotation_angle < max_rotation_angle:

                    pivot_point_x = best_match_origin_x
                    if pivot_point_x is not None:
                        pivot_point_x += template_half_width
                    pivot_point_y = best_match_origin_y
                    if pivot_point_y is not None:
                        pivot_point_y += template_half_height

                    search_set.append(
                        {
                            'angle': rotation_angle,
                            'frame': rotatedImage(current_frame, rotation_angle, pivot_point_x, pivot_point_y)
                        }
                    )
                    rotation_angle += rotation_increment

            # crop out a smaller subregion to search if required
            if max_translation_per_frame is None:
                sub_region_padding = None
            else:
                sub_region_padding = (
                    math.ceil(max_translation_per_frame[0] / microns_per_pixel),
                    math.ceil(max_translation_per_frame[1] / microns_per_pixel)
                )
            input_image_sub_region_origin = None
            for frame_details in search_set:
                if prev_frame_blank:
                    # need to search the entire image again from the origin
                    frame_details['frame'] = intensityAdjusted(frame_details['frame'])
                    input_image_sub_region_origin = (0, 0)
                else:
                    input_image_sub_region_to_search, input_image_sub_region_origin = inputImageSubRegion(
                        input_image=frame_details['frame'],
                        sub_region_base_shape=(template_width, template_height),
                        sub_region_origin=(best_match_origin_x, best_match_origin_y),
                        sub_region_padding=sub_region_padding
                    )
                    input_image_sub_region_to_search = intensityAdjusted(input_image_sub_region_to_search)
                    frame_details['frame'] = input_image_sub_region_to_search
            prev_frame_blank = False  # only applies to the next frame

            # TODO: make matchResults return a dict instead of a tuple?
            match_measure, match_coordinates, match_rotation = matchResults(
                search_set=search_set,
                template_to_match=template,
                sub_pixel_search_increment=sub_pixel_search_increment,
                sub_pixel_refinement_radius=sub_pixel_refinement_radius
            )
            if match_coordinates is None:
                match_success = False
                best_match_origin_x = 0
                best_match_origin_y = 0
                best_match_rotation = 0.0
                flipped_image_best_match_rotation = 0.0
            else:
                match_success = True
                best_match_origin_x = match_coordinates[0] + input_image_sub_region_origin[0]
                best_match_origin_y = match_coordinates[1] + input_image_sub_region_origin[1]
                best_match_rotation = match_rotation
                # because images are stored flipped ("upside down"),
                # in order for match rotations to appear as humans see them
                # they need to be relative to the flipped image
                flipped_image_best_match_rotation = -best_match_rotation

        original_time_stamp = input_video_stream.timeStamp()
        time_stamp_in_seconds = original_time_stamp
        tracking_results.append({
            'MATCH_SUCCESS': match_success,
            'FRAME_NUMBER': frame_number,
            'TIME_STAMP': time_stamp_in_seconds,
            'MATCH_MEASURE': match_measure,
            'Y_DISPLACEMENT': 0,
            'X_DISPLACEMENT': 0,
            'XY_DISPLACEMENT': 0,
            'TEMPLATE_MATCH_ORIGIN_X': best_match_origin_x,
            'TEMPLATE_MATCH_ORIGIN_Y': best_match_origin_y,
            'TEMPLATE_MATCH_ROTATION': flipped_image_best_match_rotation,
            'x_start': best_match_origin_x,
            'x_end': best_match_origin_x + template_width,
            'y_start': best_match_origin_y,
            'y_end': best_match_origin_y + template_height,
        })

        if match_coordinates is not None:
            # update the min and max positions of the template origin for ALL frames
            # using the position in the y dimension only as the reference measure
            if best_match_origin_y < min_y_origin[1]:
                min_y_origin = (best_match_origin_x, best_match_origin_y)
                min_y_frame = frame_number
            if best_match_origin_y > max_y_origin[1]:
                max_y_origin = (best_match_origin_x, best_match_origin_y)
                max_y_frame = frame_number
            if best_match_origin_x < min_x_origin[0]:
                min_x_origin = (best_match_origin_x, best_match_origin_y)
                min_x_frame = frame_number
            if best_match_origin_x > max_x_origin[0]:
                max_x_origin = (best_match_origin_x, best_match_origin_y)
                max_x_frame = frame_number

            # mark the ROI on the frame where the template matched
            if video_writer is not None:
                frame = input_video_stream.frameVideoRGB()

                roi_edges = boundingBoxEdges(
                    box_origin_x=best_match_origin_x,
                    box_origin_y=best_match_origin_y,
                    box_width=template_width,
                    box_height=template_height,
                    rotation_degrees=flipped_image_best_match_rotation
                )

                grid_colour_bgr = (0, 255, 0)
                for edge_point_a, edge_point_b in roi_edges:
                    cv.line(
                        img=frame,
                        pt1=edge_point_a,
                        pt2=edge_point_b,
                        color=grid_colour_bgr,
                        thickness=1,
                        lineType=cv.LINE_AA
                    )

                video_writer.writeFrame(frame, input_video_stream.frameVideoPTS())

    if video_writer is not None:
        video_writer.close()
    input_video_stream.close()

    # adjust match displacements, so they're relative to the match closest to the origin
    displacement_adjusted_results, min_contraction_frame_number, major_movement_direction = displacementAdjustedResults(
        results_to_adjust=tracking_results,
        microns_per_pixel=microns_per_pixel,
        output_conversion_factor=output_conversion_factor,
        extreme_points=[min_x_origin, min_y_origin, max_x_origin, max_y_origin],
        min_frame_numbers=(min_x_frame, min_y_frame),
        max_frame_numbers=(max_x_frame, max_y_frame),
        contraction_vector=contraction_vector,
        template_half_height=template_half_height,
        template_half_width=template_half_width
    )
    estimated_frequency = frequencyEstimate(displacement_adjusted_results)

    return (
        (warning_msg, error_msg),
        displacement_adjusted_results,
        major_movement_direction,
        estimated_frequency,
        frames_per_second,
        template_rgb,
        min_contraction_frame_number
    )


def frequencyEstimate(tracking_info: List[Dict]) -> float:
    """
       :param tracking_info: tracking results generated by trackTemplate()
       :return: estimated frequency of the tracked objects movement
    """
    # convert time stamp and displacement data from all frames into 2 numpy arrays
    time_stamps = []
    displacements = []
    for frame_info in tracking_info:
        time_stamps.append(frame_info['TIME_STAMP'])
        frame_displacement = frame_info['XY_DISPLACEMENT']
        displacements.append(frame_displacement)
    displacements = np.asarray(displacements)
    # estimate the midpoint in the displacements (which we presume to be moving periodically)
    displacement_midpoint = np.median(displacements)

    # compute the sign of each displacement from the midpoint
    distances_from_midpoint = displacements - displacement_midpoint
    distances_from_midpoint_signs = np.sign(distances_from_midpoint)
    previous_sign = distances_from_midpoint_signs[0]

    # record the times when the signs change from +ve to -ve and vice versa
    time_stamps = np.asarray(time_stamps)
    neg_to_pos_crossing_times = []
    pos_to_neg_crossing_times = []
    for index in range(1, len(distances_from_midpoint)):
        current_sign = distances_from_midpoint_signs[index]
        if current_sign != previous_sign:
            if current_sign == 0:
                continue
            time_stamp = time_stamps[index]
            if previous_sign < 0:
                neg_to_pos_crossing_times.append(time_stamp)
            else:
                pos_to_neg_crossing_times.append(time_stamp)
            previous_sign = current_sign

    num_neg_to_pos_crossing_times = len(neg_to_pos_crossing_times)
    num_pos_to_neg_crossing_times = len(pos_to_neg_crossing_times)
    if num_neg_to_pos_crossing_times < 1 or num_pos_to_neg_crossing_times < 1:
        # no periodicity detected
        return 0.0

    # compute the frequency of the -ve to +ve change points
    num_neg_to_pos_periods = len(neg_to_pos_crossing_times) - 1
    neg_to_pos_time_diff = neg_to_pos_crossing_times[-1] - neg_to_pos_crossing_times[0]
    neg_to_pos_frequency = num_neg_to_pos_periods/neg_to_pos_time_diff

    # compute the frequency of the +ve to -ve change points
    num_pos_to_neg_periods = len(pos_to_neg_crossing_times) - 1
    pos_to_neg_time_diff = pos_to_neg_crossing_times[-1] - pos_to_neg_crossing_times[0]
    pos_to_neg_frequency = num_pos_to_neg_periods/pos_to_neg_time_diff
    # TODO: need to guard against having no contractions or relaxations or neither.

    # compute the average of the +ve to +ve and +ve to -ve frequency estimates
    estimated_frequency = (pos_to_neg_frequency + neg_to_pos_frequency)/2.0

    return estimated_frequency


def boundingBoxEdges(box_origin_x, box_origin_y, box_width, box_height, rotation_degrees) -> List[Tuple]:
    """ returns a list of (x, y) coordinate points for the corners of a box after rotation """
    # define the x & y points at the box corners
    box_origin_x = box_origin_x
    box_end_x = box_origin_x + box_width
    box_origin_y = box_origin_y
    box_end_y = box_origin_y + box_height

    # define the x, y coordinates of the box corners
    top_left_point = (box_origin_x, box_origin_y)
    top_right_point = (box_end_x, box_origin_y)
    bottom_left_point = (box_origin_x, box_end_y)
    bottom_right_point = (box_end_x, box_end_y)

    # rotate the box corner points around the midpoint of the box
    box_half_width = int(box_width / 2.0)
    box_half_height = int(box_height / 2.0)
    box_origin_point = (
        box_origin_x + box_half_width,
        box_origin_y + box_half_height
    )
    rotation_radians = math.radians(rotation_degrees)
    top_left_point = rotatedPoint(
        top_left_point, rotation_radians, box_origin_point
    )
    top_right_point = rotatedPoint(
        top_right_point, rotation_radians, box_origin_point
    )
    bottom_left_point = rotatedPoint(
        bottom_left_point, rotation_radians, box_origin_point
    )
    bottom_right_point = rotatedPoint(
        bottom_right_point, rotation_radians, box_origin_point
    )

    # define the end points of the 4 edges
    top_edge = (top_left_point, top_right_point)
    bottom_edge = (bottom_left_point, bottom_right_point)
    left_edge = (top_left_point, bottom_left_point)
    right_edge = (bottom_right_point, top_right_point)

    return [top_edge, left_edge, right_edge, bottom_edge]


def rotatedPoint(xy_point_to_rotate, angle_radians, rotation_center=(0, 0), round_result=True) -> Tuple:
    """ Rotate an (x, y) point around rotation_center """
    x_point_to_rotate, y_point_to_rotate = xy_point_to_rotate
    rotation_center_x, rotation_center_y = rotation_center
    x_point_to_rotate_centered = (x_point_to_rotate - rotation_center_x)
    y_point_to_rotate_centered = (y_point_to_rotate - rotation_center_y)
    cos_rad = math.cos(angle_radians)
    sin_rad = math.sin(angle_radians)
    x_point_rotated = rotation_center_x + cos_rad * x_point_to_rotate_centered + sin_rad * y_point_to_rotate_centered
    y_point_rotated = rotation_center_y - sin_rad * x_point_to_rotate_centered + cos_rad * y_point_to_rotate_centered

    if round_result:
        return round(x_point_rotated), round(y_point_rotated)
    return x_point_rotated, y_point_rotated


def rotatedImage(image_to_rotate, rotation_degrees, pivot_x, pivot_y):
    """ returns the input image rotated by rotation_degrees around the point (pivot_x, pivot_y) """
    if pivot_x is None or pivot_y is None:
        return image_to_rotate
    scale_factor = 1.0
    warp_matrix = cv.getRotationMatrix2D(
        (pivot_x, pivot_y),
        rotation_degrees,
        scale_factor
    )
    rotated_image = cv.warpAffine(
        image_to_rotate,
        warp_matrix,
        (image_to_rotate.shape[1], image_to_rotate.shape[0])
    )
    return rotated_image


def displacementAdjustedResults(
    results_to_adjust: List[Dict],
    microns_per_pixel: float,
    output_conversion_factor: float,
    extreme_points: List[Tuple[float, float]],
    min_frame_numbers: Tuple[int, int],
    max_frame_numbers: Tuple[int, int],
    contraction_vector: Tuple[int, int],
    template_half_height: float,
    template_half_width: float
) -> Tuple[List[Dict], int, str]:
    """ Adjust displacement values for matches in all frames so they are relative to the
        point in the video stream where the template (roi) match is closest to the origin
        instead of the being relative to the actual video frame origin. """

    if contraction_vector is None:
        contraction_vector = (1, -1)

    # TODO: change to a dict or many dicts so we're not referring to random tuple
    #       positions that have zero meaning and can easily be passed in the wrong order
    contraction_moves_right = contraction_vector[0] > 0
    contraction_moves_down = contraction_vector[1] < 0  # image orientation is flipped in y
    min_x_origin, min_y_origin, max_x_origin, max_y_origin = extreme_points
    range_of_x_movement = abs(max_x_origin[0] - min_x_origin[0])
    range_of_y_movement = abs(max_y_origin[1] - min_y_origin[1])
    if range_of_x_movement > range_of_y_movement:
        major_movement_direction = 'x'
        if contraction_moves_right:
            min_frame_number = min_frame_numbers[0]
            min_template_origin_x = min_x_origin[0]
            min_template_origin_y = min_x_origin[1]
        else:
            min_frame_number = max_frame_numbers[0]
            min_template_origin_x = max_x_origin[0]
            min_template_origin_y = max_x_origin[1]
    else:
        major_movement_direction = 'y'
        if contraction_moves_down:
            min_frame_number = min_frame_numbers[1]
            min_template_origin_y = min_y_origin[1]
            min_template_origin_x = min_y_origin[0]
        else:
            min_frame_number = max_frame_numbers[1]
            min_template_origin_y = max_y_origin[1]
            min_template_origin_x = max_y_origin[0]

    adjusted_tracking_results = []
    for frame_info in results_to_adjust:
        if frame_info['MATCH_SUCCESS']:
            # compute the x, y and xy displacements relative to the min of the main axis of movement
            x_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_X'] - min_template_origin_x) * float(microns_per_pixel)
            x_displacement *= float(output_conversion_factor)
            frame_info['X_DISPLACEMENT'] = x_displacement
            y_displacement = (frame_info['TEMPLATE_MATCH_ORIGIN_Y'] - min_template_origin_y) * float(microns_per_pixel)
            y_displacement *= float(output_conversion_factor)
            frame_info['Y_DISPLACEMENT'] = y_displacement
            frame_info['XY_DISPLACEMENT'] = math.sqrt(x_displacement * x_displacement + y_displacement * y_displacement)
            # adjust x and y match positions so they're relative to the center of the template
            frame_info['TEMPLATE_MATCH_ORIGIN_X'] += template_half_width
            frame_info['TEMPLATE_MATCH_ORIGIN_Y'] += template_half_height
        else:
            frame_info['X_DISPLACEMENT'] = 0.0
            frame_info['Y_DISPLACEMENT'] = 0.0
            frame_info['XY_DISPLACEMENT'] = 0.0
            # adjust x and y match positions so they're relative to the center of the template
            frame_info['TEMPLATE_MATCH_ORIGIN_X'] = min_template_origin_x + template_half_width
            frame_info['TEMPLATE_MATCH_ORIGIN_Y'] = min_template_origin_y + template_half_height
        adjusted_tracking_results.append(frame_info)
    return adjusted_tracking_results, min_frame_number, major_movement_direction


def inputImageSubRegion(
    input_image: np.ndarray,
    sub_region_base_shape: Tuple[int, int],
    sub_region_origin: Tuple[int, int],
    sub_region_padding: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """ Returns a subregion of the input image and the x, y coordinates of its origin """
    sub_region_origin_x, sub_region_origin_y = sub_region_origin
    if sub_region_origin_x is None or sub_region_origin_y is None:
        return input_image, (0, 0)

    if sub_region_padding is None:
        return input_image, (0, 0)
    sub_region_padding_x, sub_region_padding_y = sub_region_padding

    input_shape_x, input_shape_y = (input_image.shape[1], input_image.shape[0])
    sub_region_base_width, sub_region_base_height = sub_region_base_shape

    # define the subregion end points
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
    # crop out the new subregion
    sub_region = input_image[sub_region_start_y:sub_region_end_y, sub_region_start_x:sub_region_end_x]

    # return the new subregion, and it's origin relative to the input_image
    return sub_region, (sub_region_start_x, sub_region_start_y)


def templateFromInputROI(
    video_to_search,
    template_to_find: np.ndarray,
    max_frames_to_check: int = None,
    blank_frame_stddev_cutoff: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """ Return a ROI from the input video that will be tracked """

    initial_frame_pos = video_to_search.framePosition()
    video_to_search.initialiseStream()

    best_match_measure = 0.0
    best_match_coordinates = None
    best_match_frame = None
    best_match_frame_gray = None
    number_of_frames = video_to_search.numFrames()
    if max_frames_to_check is None:
        number_of_frames_to_check = number_of_frames
    else:
        number_of_frames_to_check = min(number_of_frames, max_frames_to_check)

    num_frames_compared = 0
    while video_to_search.next():
        frame_gray = video_to_search.frameGray()
        if frame_gray is None:
            error_msg = "Error. No Frame returned during video capture in templateFromInputROI function. Exiting."
            raise RuntimeError(error_msg)

        frame_std_dev = np.std(frame_gray)
        if frame_std_dev < blank_frame_stddev_cutoff:
            frame_number = video_to_search.framePosition()
            print(f"WARNING. Frame {frame_number} appears to be blank. Ignoring in template from ROI.")
            video_to_search.next()
            continue  # ignore blank frames since user can't see anything to draw an ROI

        frame = video_to_search.frameRGB()
        if template_to_find is None:  # return the user drawn roi from the first video frame
            print("Waiting on user to manually select ROI ...")
            roi = userDrawnROI(frame)
            if roi is not None:
                print("...ROI selection complete")
                # return the selected roi as a new template
                new_template = frame[roi['y_start']:roi['y_end'], roi['x_start']:roi['x_end']]
                new_template_grey = frame_gray[roi['y_start']:roi['y_end'], roi['x_start']:roi['x_end']]
                # reset the video to where it was initially before we return
                video_to_search.setFramePosition(initial_frame_pos)
                return new_template, new_template_grey
            print("...No ROI selected. Presenting the next frame...")
        else:
            # find the best match in the input video to the template passed in
            frame_adjusted = intensityAdjusted(video_to_search.frameGray())
            match_results = cv.matchTemplate(frame_adjusted, template_to_find, cv.TM_CCOEFF)
            _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)
            if match_measure > best_match_measure:
                best_match_measure = match_measure
                best_match_coordinates = match_coordinates
                best_match_frame = frame
                best_match_frame_gray = video_to_search.frameGray()

        if num_frames_compared > number_of_frames_to_check:
            break
        else:
            num_frames_compared += 1

    # reset the video to where it was initially
    video_to_search.setFramePosition(initial_frame_pos)
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
    new_template_gray = best_match_frame_gray[
        new_template_start_y:new_template_end_y,
        new_template_start_x:new_template_end_x
    ]
    return new_template, new_template_gray


def userDrawnROI(input_image: np.ndarray, title_text: str = None) -> Dict:
    """
    Show the user a window with an image they can draw a ROI on.
    Args:
      input_image: the image to show the user.
      title_text: text to use for dialog window title
    Returns:
      ROI selected by the user from the input image.
    """

    # create a resizeable window big enough to draw roi's and see the heading text
    if title_text is None:
        roi_selector_window_name = "DRAW RECTANGULAR ROI"
    else:
        roi_selector_window_name = title_text
    roi_gui_flags = cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL
    cv.namedWindow(roi_selector_window_name, flags=roi_gui_flags)
    # we resize so the user can easily select ROIs and read the title instructions
    cv.resizeWindow(roi_selector_window_name, width=1280, height=720)
    # we move to arbitrary coordinates closer to the screen center
    cv.moveWindow(roi_selector_window_name, x=200, y=200)
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

    return {
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
    }


def intensityAdjusted(image_to_adjust: np.ndarray, adjust_gamma: bool = False) -> np.ndarray:
    """
    Performs an automatic adjustment of the input intensity range to enhance contrast.
    Args:
      image_to_adjust:    a gray scale image to adjust the intensity of.
      adjust_gamma:  use gamma rescaling to adjust intensity.
    Returns:
      float32 version of input image with intensity adjusted
    """
    if adjust_gamma:
        image_stddev = np.std(image_to_adjust)
        gamma_value = 1.0/np.sqrt(np.log2(image_stddev))
        image_to_adjust = gammaAdjusted(
            intensity=image_to_adjust,
            gamma=gamma_value
        ).astype(np.float32)
    current_image_min: float = np.min(image_to_adjust)
    current_image_max: float = np.max(image_to_adjust)
    current_image_range: float = max(1.0, current_image_max - current_image_min)
    adjusted_image = rescaled(
        intensity=image_to_adjust,
        intensity_min=current_image_min,
        intensity_range=current_image_range,
        new_scale=2.0**16
    ).astype(np.float32)
    return adjusted_image


def gammaAdjusted(intensity: float, gamma: float) -> float:
    return intensity ** gamma


def rescaled(intensity: float, intensity_min: float, intensity_range: float, new_scale: float = 1.0) -> float:
    return new_scale * (intensity - intensity_min) / intensity_range


def matchResults(
    search_set: List[dict],
    template_to_match: np.ndarray,
    sub_pixel_search_increment: float = None,
    sub_pixel_refinement_radius: int = None,
    sub_pixel_search_offset_right: bool = False,
    sub_pixel_search_template_alternative: np.ndarray = None,
) -> Tuple[float, List[float], float]:
    """
        Computes the coordinates of the best match between the input image and template_to_match.
        Accuracy is +/-1 pixel if sub_pixel_search_increment is None or >= 1.0.
        Accuracy is +/-sub_pixel_search_increment if |sub_pixel_search_increment| < 1.0 and not None.
    """

    best_match_measure = -1.0
    best_match_coordinates = None
    best_match_rotation = None
    best_match_frame = None

    for frame_details in search_set:
        image_to_search = frame_details['frame']
        frame_rotation = frame_details['angle']
        # find the best match for template_to_match in image_to_search
        match_results = cv.matchTemplate(image_to_search, template_to_match, cv.TM_CCOEFF_NORMED)
        _, match_measure, _, match_coordinates = cv.minMaxLoc(match_results)
        if match_measure is None:
            continue
        if match_measure < best_match_measure:
            continue
        if math.isclose(match_measure, best_match_measure, rel_tol=1e-6):
            if math.fabs(frame_rotation) > math.fabs(best_match_rotation):
                continue
        best_match_measure = match_measure
        best_match_coordinates = match_coordinates
        best_match_rotation = frame_rotation
        best_match_frame = image_to_search

    if sub_pixel_search_increment is None:
        return best_match_measure, best_match_coordinates, best_match_rotation

    # refine the results with a sub pixel search
    if sub_pixel_refinement_radius is None:
        sub_pixel_search_radius = 1
    else:
        sub_pixel_search_radius = sub_pixel_refinement_radius
    match_coordinates_origin_x = best_match_coordinates[0]
    match_coordinates_origin_y = best_match_coordinates[1]
    if sub_pixel_search_offset_right:
        match_coordinates_origin_x += template_to_match.shape[1]
    if sub_pixel_search_template_alternative is not None:
        template_to_match = sub_pixel_search_template_alternative  # must go after the offset
        if sub_pixel_search_offset_right:
            match_coordinates_origin_x -= template_to_match.shape[1]

    sub_region_y_start = max(match_coordinates_origin_y - sub_pixel_search_radius, 0)
    sub_region_y_end = min(
        match_coordinates_origin_y + template_to_match.shape[0] + sub_pixel_search_radius, best_match_frame.shape[0]
    )
    sub_region_x_start = max(match_coordinates_origin_x - sub_pixel_search_radius, 0)
    sub_region_x_end = min(
        match_coordinates_origin_x + template_to_match.shape[1] + sub_pixel_search_radius, best_match_frame.shape[1]
    )
    best_match_measure, best_match_sub_coordinates = bestSubPixelMatch(
        image_to_search=best_match_frame[
            int(math.floor(sub_region_y_start)):int(math.ceil(sub_region_y_end)),
            int(math.floor(sub_region_x_start)):int(math.ceil(sub_region_x_end))
        ],
        template_to_match=template_to_match,
        search_increment=sub_pixel_search_increment
    )
    best_match_sub_coordinates = [
        sub_region_x_start + best_match_sub_coordinates[0],
        sub_region_y_start + best_match_sub_coordinates[1]
    ]

    return best_match_measure, best_match_sub_coordinates, best_match_rotation


def roiInfoFromTemplate(
    search_image_gray: np.ndarray,
    template_image_gray: np.ndarray,
    sub_pixel_search_increment: float = None,
    sub_pixel_refinement_radius: int = None,
    sub_pixel_search_offset_right: bool = False
) -> Dict:
    """ Finds the best match ROI for a template within search_image """
    template_image = intensityAdjusted(template_image_gray)
    search_set = [
        {
            'angle': 0.0,
            'frame': intensityAdjusted(search_image_gray)
        }
    ]
    _, match_coordinates, _ = matchResults(
        search_set=search_set,
        template_to_match=template_image,
        sub_pixel_search_increment=sub_pixel_search_increment,
        sub_pixel_refinement_radius=sub_pixel_refinement_radius,
        sub_pixel_search_offset_right=sub_pixel_search_offset_right
    )
    roi_origin_x, roi_origin_y = match_coordinates
    roi_height = template_image.shape[0]
    roi_width = template_image.shape[1]
    return {
        'x_start': int(roi_origin_x),
        'x_end': int(roi_origin_x + roi_width),
        'y_start': int(roi_origin_y),
        'y_end': int(roi_origin_y + roi_height),
    }


def bestSubPixelMatch(
    image_to_search: np.ndarray,
    template_to_match: np.ndarray,
    search_increment: float
) -> Tuple[float, List[float]]:
    """ Computes the coordinates of the best sub pixel match between the input image and template_to_match. """
    input_dim_y, input_dim_x = image_to_search.shape
    template_dim_y, template_dim_x = template_to_match.shape
    search_length_y = input_dim_y - template_dim_y
    search_length_x = input_dim_x - template_dim_x
    best_match_measure = -1.0
    best_match_coordinates = None
    shifted_input = np.ndarray(shape=[template_dim_y + 1, template_dim_x + 1], dtype=np.float32)
    for y_origin in np.arange(search_increment, search_length_y, search_increment):
        for x_origin in np.arange(search_increment, search_length_x, search_increment):
            sub_region_start_y = math.floor(y_origin)
            sub_region_end_y = sub_region_start_y + template_dim_y + 1
            sub_region_start_x = math.floor(x_origin)
            sub_region_end_x = sub_region_start_x + template_dim_x + 1
            sub_image_to_shift = image_to_search[
                sub_region_start_y:sub_region_end_y,
                sub_region_start_x:sub_region_end_x,
            ]
            shift(input=sub_image_to_shift, shift=[-y_origin, -x_origin], output=shifted_input)
            input_to_match = shifted_input[:template_dim_y, :template_dim_x]
            match_results = cv.matchTemplate(input_to_match, template_to_match, cv.TM_CCOEFF_NORMED)
            _, match_measure, _, _ = cv.minMaxLoc(match_results)
            if match_measure > best_match_measure:
                best_match_measure = match_measure
                best_match_coordinates = [x_origin, y_origin]

    return best_match_measure, best_match_coordinates


def fourccNum(c1, c2, c3, c4) -> int:
    return (
        (ord(c1) & 255) + ((ord(c2) & 255) << 8) + ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)
    )


def fourccChars(codec_num: int) -> List[str]:
    return [
        chr(codec_num & 255),
        chr((codec_num >> 8) & 255),
        chr((codec_num >> 16) & 255),
        chr((codec_num >> 24) & 255)
    ]
    # avc1 is the codec returned by videos from CuriBios microscope
