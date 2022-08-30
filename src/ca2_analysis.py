import os
from sys import exit as sys_exit
from math import floor
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
from io_utils import contentsOfDir
from waveform_analysis import waveFormAnalysis
from video_api import VideoReader, VideoWriter, supported_file_extensions
from morphology import morphologyMetricsForImage
from track_template import trackTemplate, roiInfoFromTemplate, userDrawnROI
from io_utils import fileNameParametersForSDK, zipDir
from xlsx_utils import (
    metadataRequiredForXLSX,
    trackingResultsToXSLX,
    trackingResultsToXLSXforSDK,
    ca2SignalDataToXLSXforSDK,
    ca2AnalysisToXLSX
)

import matplotlib.pyplot as plt
pd.set_option("display.precision", 2)
pd.set_option("display.expand_frame_repr", False)


# TODO: in order to allow for the ca2+ videos to contract in any direction,
#       we'll need to use the "main axis" of movement to determine if we adjust the
#       horizontal or vertical positions of the tissue roi.
#       we can do this using major_movement_direction returned by the dynamic_roi tracking

# TODO: parallelize the analysis of each frame. i.e.
#       if we have 10 processors, split up the frames into 10 contiguous sections
#       and have each section processed independently by one process
#       the final result is obtained by simple concatenation of each sections results in order
#  OR
#       just parallelize the processing of videos i.e. have more than 1 video processed at a time

# TODO: attempt to determine if the video has low S/N and run morphology in low_s/n mode?
#       it might be enough to just determine if the variance of the first frame is > blah, or
#       even if the mean and median are more than x apart etc

# TODO: check there are no double peaks or troughs.
#       would need to use the peak and trough indices
#       find if the peak or trough is first
#       then oscillate between peaks and troughs ensuring
#       the values in time_stamps of the alternating sequence
#       of peak and trough indices forms a strictly monotonically increase sequence
#       i.e. from one to the next the time is always greater
#       because if there is ever a double peak, the alternating sequence wont be monotonically increasing
#       the alternative is to just throw a tuple ('peak or trough', index) into a list
#       and sort on the index then step through the list and ensure we alternate

# TODO: IF the current method of estimate the various metrics where we
#       use a polynomial fit and then find the roots of the poly FAILS,
#       numpy warnings such as the rank is blah or the fit is ill-conditioned etc ...
#       we will have to use a method like:
#       take the polynomial coefficients and perform a search of the function
#       between the start and end times
#       could do this with a bisection and stop when the step size < some tol in seconds say 0.001
#       or we could do a multi grid search i.e step size of 0.1 second
#       then 0.01 second between the points we narrowed it to
#       then 0.001 between the subsequent points we narrow it to, etc
#       if the fit still fails, we'd have to find the two closest time points that the
#       metric parameter lives between, and linear interpolate between those two points
#       to find the point the metric is for. this would clearly be inferior since
#       we know most of these signals have exponential or polynomial shape and
#       a linear interpolation between two points (even if close) isn't a great fit.
#  OR
#       the alternative is to use splines which do essentially the same thing
#       without having to explicitly code the linear interpolation, but since we have to
#       compute enough splines to for the resolution we need computationally very expensive


def analyzeCa2Data(
    path_to_data: str,
    expected_frequency_hz: None,
    analysis_method: str = 'None',
    low_signal_to_noise: bool = False,
    save_result_plots: bool = False,
    dynamic_roi_template_path: str = None,
    reference_roi_template_path: str = None,
    display_results: bool = False,
    select_background_once: bool = False,
    expected_min_peak_width: int = None,
    expected_min_peak_height: float = None,
    microns_per_pixel: float = None,
    max_translation_per_frame: Tuple[float] = None,
    max_rotation_per_frame: float = None,
    contraction_vector: Tuple[int] = None
):
    """ """
    if 'auto' not in analysis_method.lower():
        if expected_frequency_hz is None:
            print("ERROR!\nExpected Frequency Hint must be provided for non automatic methods\nExiting")
            sys_exit(1)

    # get paths to all the video files being processed and set up the results' directory structure
    base_dir, files_to_analyze = contentsOfDir(dir_path=path_to_data, search_terms=supported_file_extensions)
    if files_to_analyze is None or len(files_to_analyze) < 1:
        print("ERROR!\nNo video files to analyze in the directory specified\nExiting")
        sys_exit(1)
    dir_paths = outputDirPaths(base_dir)

    # collect bg & tissue or dynamic & reference ROIs for all the videos to be analyzed, up front, so that
    # all the videos can then be processed automatically without any user interaction
    all_videos_rois = {}
    background_roi = None
    for file_name, file_extension in files_to_analyze:
        well_name, date_stamp = fileNameParametersForSDK(file_name)
        input_video_file_path = os.path.join(base_dir, file_name + file_extension)
        video_rois = videoROIs(
            input_video_file_path,
            analysis_method,
            dynamic_roi_template_path,
            reference_roi_template_path,
            background_roi
        )
        all_videos_rois[file_name] = video_rois
        if select_background_once:
            background_roi = video_rois['background']

    num_files_to_analyze = len(files_to_analyze)
    file_num_being_analyzed = 0
    print()
    for file_name, file_extension in files_to_analyze:
        file_num_being_analyzed += 1
        print(f"\nAnalyzing {file_name} ({file_num_being_analyzed} of {num_files_to_analyze})...")

        if all_videos_rois is None:
            video_roi_info = None
            output_video_path = None
        else:
            video_roi_info = all_videos_rois[file_name]
            if "nd2" in file_extension.lower():
                output_file_extension = ".avi"
            else:
                output_file_extension = file_extension
            output_video_file_name = file_name + "-with_rois" + output_file_extension

            output_video_path = os.path.join(dir_paths['video_dir'], output_video_file_name)
        input_video_file_path = os.path.join(base_dir, file_name + file_extension)
        signal_data = signalDataFromVideo(
            input_video_file_path,
            output_video_path,
            low_signal_to_noise,
            analysis_method,
            video_roi_info,
            microns_per_pixel,
            max_translation_per_frame,
            max_rotation_per_frame,
            contraction_vector
        )
        if signal_data is None:
            raise RuntimeError("Error. Signal from video could not be extracted")

        # write out the analysis results
        time_stamps = signal_data['time_stamps']
        input_signal = signal_data['signal_values']
        if signal_data['estimated_frequency'] is not None:
            expected_frequency_hz = signal_data['estimated_frequency']
            # TODO: should we be setting expected_min_peak_width & expected_min_peak_height to None in this case?
        ca2_analysis = waveFormAnalysis(
            input_signal,
            time_stamps,
            expected_frequency_hz=expected_frequency_hz,
            expected_min_peak_width=expected_min_peak_width,
            expected_min_peak_height=expected_min_peak_height
        )
        if ca2_analysis is None:
            print("\n")
            print("Error. Could not extract sensible peaks/troughs from data")
            print("Was expected frequency set to within +/- 0.5Hz of actual frequency?")
            print("No analysis results were written")
            print("\n")
            continue
        else:
            tissue_means = signal_data['tissue_means']
            background_means = signal_data['background_means']
            path_to_ca2_analysis_results_file = os.path.join(
                dir_paths['xlsx_dir'], file_name + '-ca2_analysis_results.xlsx'
            )
            ca2AnalysisToXLSX(
                time_stamps,
                input_signal,
                tissue_means,
                background_means,
                ca2_analysis,
                path_to_ca2_analysis_results_file
            )

        if display_results:
            print()
            print(f'metrics for {file_name}')
            for metrics in ca2_analysis['metrics']:
                p2p_order = metrics['p2p_order']
                average_metrics = metrics['mean_metric_data']
                metric_failure_proportions = metrics['metric_failure_proportions']
                if display_results:
                    print(f'{p2p_order} average metrics (failure %): {average_metrics} ({metric_failure_proportions})')

        if display_results or save_result_plots is not None:
            if save_result_plots:
                plot_file_path = os.path.join(dir_paths['plot_dir'], file_name + '-plot.png')
            else:
                plot_file_path = None
            plotCa2Signals(
                time_stamps,
                input_signal,
                ca2_analysis['peak_indices'],
                ca2_analysis['trough_indices'],
                file_name,
                display_results=display_results,
                plot_file_path=plot_file_path
            )

        # write out Ca2 signal data for SDK
        signal_data_for_sdk_file_path = os.path.join(
            dir_paths['sdk_results_xlsx_ca2_dir_path'],
            file_name + '-ca2_signal_data_for_sdk.xlsx'
        )
        frames_per_second = signal_data['frames_per_second']
        if frames_per_second is None:
            input_video_stream = VideoReader(input_video_file_path)
            frames_per_second = input_video_stream.avgFPS()
            input_video_stream.close()
        video_meta_data = {
            'well_name': well_name,
            'video_date': date_stamp,
            'frames_per_second': frames_per_second
        }
        ca2SignalDataToXLSXforSDK(
            time_stamps,
            input_signal,
            signal_data_for_sdk_file_path,
            video_meta_data
        )

        # write out the contraction (dynamic ROI tracking) results if necessary
        if 'auto' in analysis_method.lower():
            contraction_data = signal_data['contraction_results']
            user_roi_selection = dynamic_roi_template_path is None
            meta_data = metadataRequiredForXLSX(
                well_name=well_name,
                date_stamp=date_stamp,
                frames_per_second=frames_per_second,
                user_roi_selection=user_roi_selection,
                max_translation_per_frame=max_translation_per_frame,
                max_rotation_per_frame=max_rotation_per_frame,
                contraction_vector=contraction_vector,
                microns_per_pixel=microns_per_pixel,
                output_conversion_factor=None,
                sub_pixel_search_increment=None,
                sub_pixel_refinement_radius=None,
                estimated_frequency=signal_data['estimated_frequency'],
            )
            contraction_results_file_path = os.path.join(
                dir_paths['xlsx_dir'], file_name + '-contraction_results.xlsx'
            )
            trackingResultsToXSLX(contraction_data, meta_data, contraction_results_file_path)
            contraction_results_for_sdk_file_path = os.path.join(
                dir_paths['sdk_results_xlsx_contraction_dir_path'], file_name + '-contraction_data_for_sdk.xlsx'
            )
            trackingResultsToXLSXforSDK(contraction_data, meta_data, contraction_results_for_sdk_file_path)

    # create a zip archive of the Ca2+ and contraction (dynamic ROI tracking) results for SDK
    ca2_signal_zip_file_path = os.path.join(dir_paths['sdk_results_zip_dir_path'], 'ca2_signal_data_for_sdk.zip')
    zipDir(
        input_dir_path=dir_paths['sdk_results_xlsx_ca2_dir_path'],
        zip_file_path=ca2_signal_zip_file_path,
        sdk_files_only=True
    )
    if 'auto' in analysis_method.lower():
        contraction_zip_file_path = os.path.join(dir_paths['sdk_results_zip_dir_path'], 'contraction_data_for_sdk.zip')
        zipDir(
            input_dir_path=dir_paths['sdk_results_xlsx_contraction_dir_path'],
            zip_file_path=contraction_zip_file_path,
            sdk_files_only=True
        )


def outputDirPaths(base_dir: str) -> Dict:
    results_dir_name = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir_path = os.path.join(base_dir, results_dir_name)
    sdk_results_dir_path = os.path.join(results_dir_path, 'sdk')
    sdk_results_xlsx_dir_path = os.path.join(sdk_results_dir_path, 'xlsx')
    # define the dir structure
    dir_paths = {
        'results_dir': results_dir_path,
        'video_dir': os.path.join(results_dir_path, 'video'),
        'xlsx_dir': os.path.join(results_dir_path, 'xlsx'),
        'plot_dir': os.path.join(results_dir_path, 'plot'),
        'sdk_results_dir': sdk_results_dir_path,
        'sdk_results_zip_dir_path': os.path.join(sdk_results_dir_path, 'zip'),
        'sdk_results_xlsx_dir_path': sdk_results_xlsx_dir_path,
        'sdk_results_xlsx_ca2_dir_path': os.path.join(sdk_results_xlsx_dir_path, 'ca2'),
        'sdk_results_xlsx_contraction_dir_path': os.path.join(sdk_results_xlsx_dir_path, 'contraction')
    }
    # create the dir structure
    for _, dir_path in dir_paths.items():
        os.mkdir(dir_path)
    return dir_paths


def videoROIsInfo(
    video_path: str,
    template_info: Dict,
    reference_roi_captures_tissue: bool,
    microns_per_pixel: float = None,
    max_translation_per_frame: Tuple[float] = None,
    max_rotation_per_frame: float = None,
    contraction_vector: Tuple[int] = None
) -> Tuple[List[Dict], List[Dict], str, float, float]:
    """ """
    dynamic_roi_template_image = template_info['dynamic']['image']
    _, dynamic_roi, major_movement_direction, estimated_frequency, frames_per_second, _, _ = trackTemplate(
        input_video_path=video_path,
        template_guide_image_path=None,
        template_rgb=dynamic_roi_template_image,
        microns_per_pixel=microns_per_pixel,
        max_translation_per_frame=max_translation_per_frame,
        max_rotation_per_frame=max_rotation_per_frame,
        contraction_vector=contraction_vector
    )
    num_frames = len(dynamic_roi)
    rois_info = []

    if reference_roi_captures_tissue:
        # we will only be using the reference roi, and, the top, bottom and rhs don't change,
        # only the dynamic roi side changes with the results of the tracking of the dynamic roi

        # to make the tissue roi that the user drew have its lhs adjusted with tracking, rather than
        # just making the tissue roi lhs the dynamic roi rhs, we compute the distance between the
        # dynamic & reference roi for the initial frame which == the amount to shift the dynamic roi to
        # in order to place it where the tissue roi would be with the same movement as the dynamic roi
        # dynamic_roi_reference_roi_diff = template_info['reference']['roi']['x_start'] - dynamic_roi[0]['x_end']
        dynamic_roi_reference_roi_diff = 0
        for frame_num in range(num_frames):
            rois_info.append(
                {
                    'reference': {
                        'x_start': int(dynamic_roi[frame_num]['x_end'] + dynamic_roi_reference_roi_diff),
                        'x_end': template_info['reference']['roi']['x_end'],
                        'y_start': template_info['reference']['roi']['y_start'],
                        'y_end': template_info['reference']['roi']['y_end']
                    }
                }
            )
    else:
        reference_roi_template_image = template_info['reference']['image']
        _, reference_roi, _, _, _, _, _ = trackTemplate(
            input_video_path=video_path,
            template_guide_image_path=None,
            template_rgb=reference_roi_template_image,
            microns_per_pixel=microns_per_pixel,
            max_translation_per_frame=max_translation_per_frame,
            max_rotation_per_frame=max_rotation_per_frame,
            contraction_vector=contraction_vector
        )
        for frame_num in range(num_frames):
            rois_info.append(
                {
                    'dynamic': {
                        'x_start': int(dynamic_roi[frame_num]['x_start']),
                        'x_end': int(dynamic_roi[frame_num]['x_end']),
                        'y_start': int(dynamic_roi[frame_num]['y_start']),
                        'y_end': int(dynamic_roi[frame_num]['y_end']),
                    },
                    'reference': {
                        'x_start': int(reference_roi[frame_num]['x_start']),
                        'x_end': int(reference_roi[frame_num]['x_end']),
                        'y_start': int(reference_roi[frame_num]['y_start']),
                        'y_end': int(reference_roi[frame_num]['y_end']),
                    }
                }
            )

    return rois_info, dynamic_roi, major_movement_direction, estimated_frequency, frames_per_second


def signalDataFromVideo(
    input_video_path: str,
    output_video_path: str = None,
    low_signal_to_noise: bool = False,
    analysis_method: str = 'None',
    video_roi_info: Dict = None,
    microns_per_pixel: float = None,
    max_translation_per_frame: Tuple[float] = None,
    max_rotation_per_frame: float = None,
    contraction_vector: Tuple[int] = None
) -> Dict:
    """ """
    major_movement_direction = None
    frames_per_second = None
    estimated_frequency = None
    contraction_results = None
    if 'auto' in analysis_method.lower():
        video_frames_rois, contraction_results, major_movement_direction, estimated_frequency, frames_per_second = videoROIsInfo(
            video_path=input_video_path,
            template_info=video_roi_info['template_info'],
            reference_roi_captures_tissue=video_roi_info['reference_roi_captures_tissue'],
            microns_per_pixel=microns_per_pixel,
            max_translation_per_frame=max_translation_per_frame,
            max_rotation_per_frame=max_rotation_per_frame,
            contraction_vector=contraction_vector
        )
        video_rois = iter(video_frames_rois)

    input_video_stream = VideoReader(input_video_path)
    if output_video_path is not None:
        video_writer = VideoWriter(
            path=output_video_path,
            width=input_video_stream.frameVideoWidth(),
            height=input_video_stream.frameVideoHeight(),
            time_base=input_video_stream.timeBase(),
            fps=input_video_stream.avgFPS(),
            bitrate=input_video_stream.bitRate(),
            pixel_format=input_video_stream.pixelFormat()
        )
    else:
        video_writer = None

    time_stamps = []
    signal_values = []
    fg_means = []
    bg_means = []
    while input_video_stream.next():
        time_stamps.append(input_video_stream.timeStamp())
        frame_gray = input_video_stream.frameGray()
        frame_rgb = None
        bg_roi = frame_gray[
            video_roi_info['background']['y_start']:video_roi_info['background']['y_end'],
            video_roi_info['background']['x_start']:video_roi_info['background']['x_end']
        ]
        background_subtractor = np.mean(bg_roi)
        if 'auto' in analysis_method.lower():
            frame_roi_info = next(video_rois)
            if 'morphology' in analysis_method.lower():
                # get the tissue only roi from morphology function
                frame_rgb = input_video_stream.frameVideoRGB()
                frame_rgb, morphology = morphologyMetricsForImage(
                    search_image=frame_rgb,
                    rois_info=frame_roi_info,
                    template_refinement_radius=0,
                    edge_finding_smoothing_radius=10,
                    draw_tissue_roi_only=True,
                    low_signal_to_noise=low_signal_to_noise,
                    microns_per_pixel=microns_per_pixel
                )
                # compute the mean of the tissue only region
                # rotate the input image so the roi description of vertical sections becomes horizontal sections
                # this is just much simpler to define and faster to run than working with the vertical sections
                frame_gray_rotated = np.rot90(frame_gray, k=-1)
                frame_gray_rotated_width = frame_gray_rotated.shape[1]
                y_pos = list(
                    range(
                        int(floor(morphology['x_start_position'])),
                        int(floor(morphology['x_end_position'])) + 1
                    )
                )
                x_start_pos = frame_gray_rotated_width - morphology['y_end_positions']
                x_end_pos = frame_gray_rotated_width - morphology['y_start_positions']
                num_rows = len(x_start_pos)
                row_sums = 0
                row_counts = 0
                for row_num in range(0, num_rows):
                    x_row_start = int(x_start_pos[row_num])
                    x_row_end = int(x_end_pos[row_num])
                    if x_row_start == x_row_end:
                        # edge finding failed, so sample from the entire line
                        x_row_start = 0
                        x_row_end = frame_gray_rotated_width
                    frame_row = frame_gray_rotated[y_pos[row_num], x_row_start:x_row_end]
                    row_sums += np.sum(frame_row)
                    row_counts += len(frame_row)
                tissue_mean = float(row_sums)/float(row_counts)
                # append the tissue mean
                fg_means.append(tissue_mean)
                # append the background subtracted tissue mean
                signal_values.append(tissue_mean - background_subtractor)
            else:
                reference_roi = frame_roi_info['reference']
                frame_gray = frame_gray[
                    reference_roi['y_start']:reference_roi['y_end'],
                    reference_roi['x_start']:reference_roi['x_end'],
                ]
        else:  # we're using fixed tissue roi which was already set by the user
            # NOTE: cutting out a new frame_gray from the tissue roi has to go AFTER background roi extraction
            frame_gray = frame_gray[
                video_roi_info['tissue']['y_start']:video_roi_info['tissue']['y_end'],
                video_roi_info['tissue']['x_start']:video_roi_info['tissue']['x_end']
            ]

        # append the background mean
        bg_means.append(background_subtractor)
        # append the mean of the fixed tissue roi for all methods other than fully auto
        if 'morphology' not in analysis_method.lower():
            fg_means.append(np.mean(frame_gray))
            adjusted_frame = frame_gray - background_subtractor
            adjusted_frame[adjusted_frame < 0] = 0
            adjusted_frame_mean = np.mean(adjusted_frame)
            signal_values.append(adjusted_frame_mean)

        # write out the video frame with the rois drawn
        if video_writer is not None:
            # mark the bg (& maybe tissue) ROIs on the output video frame
            if frame_rgb is None:
                frame_rgb = input_video_stream.frameVideoRGB()
            rois_to_draw = [video_roi_info['background']]
            # tissue roi is drawn by the morphology function when auto method is run
            # all other methods require drawing the reference roi here
            if 'auto' not in analysis_method.lower():
                rois_to_draw.append(video_roi_info['tissue'])
            elif 'adjusted' in analysis_method.lower():
                rois_to_draw.append(reference_roi)
            for roi in rois_to_draw:
                top_left_point = (roi['x_start'], roi['y_start'])
                top_right_point = (roi['x_end'], roi['y_start'])
                bottom_left_point = (roi['x_start'], roi['y_end'])
                bottom_right_point = (roi['x_end'], roi['y_end'])
                roi_edges = [
                    (top_left_point, top_right_point),
                    (bottom_left_point, bottom_right_point),
                    (top_left_point, bottom_left_point),
                    (bottom_right_point, top_right_point),
                ]

                roi_outline_colour = (0, 0, 255)
                for edge_point_a, edge_point_b in roi_edges:
                    cv.line(
                        img=frame_rgb,
                        pt1=edge_point_a,
                        pt2=edge_point_b,
                        color=roi_outline_colour,
                        thickness=1,
                        lineType=cv.LINE_AA
                    )
            video_writer.writeFrame(frame_rgb, input_video_stream.frameVideoPTS())

    if video_writer is not None:
        video_writer.close()
    input_video_stream.close()

    return {
        'time_stamps': np.array(time_stamps, dtype=float),
        'signal_values': np.array(signal_values, dtype=float),
        'tissue_means': np.array(fg_means, dtype=float),
        'background_means': np.array(bg_means, dtype=float),
        'contraction_results': contraction_results,
        'major_movement_direction': major_movement_direction,
        'estimated_frequency': estimated_frequency,
        'frames_per_second': frames_per_second
    }


def videoROIs(
    video_file_path: str,
    roi_method: str = None,
    dynamic_roi_template_path: str = None,
    reference_roi_template_path: str = None,
    background_roi: Dict = None
) -> Dict:
    input_video_stream = VideoReader(video_file_path)
    if not input_video_stream.isOpened():
        return None, None

    input_video_duration = input_video_stream.duration()
    max_seconds_to_search: float = 2.0
    if max_seconds_to_search > input_video_duration:
        max_seconds_to_search = input_video_duration

    frame_for_roi_extraction: np.ndarray = None
    frame_for_roi_extraction_gray: np.ndarray = None
    max_intensity_sum: float = 0.0
    while input_video_stream.next():
        frame_gray = input_video_stream.frameGray()
        frame_gray_intensity_sum: float = np.sum(frame_gray)
        if frame_gray_intensity_sum > max_intensity_sum:
            max_intensity_sum = frame_gray_intensity_sum
            frame_for_roi_extraction = input_video_stream.frameRGB()
            frame_for_roi_extraction_gray = frame_gray
        if input_video_stream.timeStamp() > max_seconds_to_search:
            break

    if 'morphology' not in roi_method.lower() and roi_method.lower() != 'none':
        reference_roi_captures_tissue = True
    else:
        reference_roi_captures_tissue = False
    if background_roi is None:
        background_roi = userDrawnROI(frame_for_roi_extraction, "Select the Background ROI")

    rois = {
        'background': background_roi,
        'reference_roi_captures_tissue': reference_roi_captures_tissue
    }
    if 'fixed' in roi_method.lower():
        if reference_roi_template_path is None:
            rois['tissue'] = userDrawnROI(frame_for_roi_extraction, "Select the Tissue/Signal ROI")
        else:
            reference_roi_image = cv.imread(reference_roi_template_path)
            rois['tissue'] = roiInfoFromTemplate(
                frame_for_roi_extraction_gray,
                cv.cvtColor(reference_roi_image, cv.COLOR_BGR2GRAY)
            )
    else:
        rois['tissue'] = {}
        if dynamic_roi_template_path is None:
            dynamic_roi = userDrawnROI(frame_for_roi_extraction, "Select the dynamic ROI to Track")
            dynamic_roi_image = frame_for_roi_extraction[
                dynamic_roi['y_start']:dynamic_roi['y_end'],
                dynamic_roi['x_start']:dynamic_roi['x_end'],
            ]
        else:
            dynamic_roi_image = cv.imread(dynamic_roi_template_path)
            dynamic_roi = roiInfoFromTemplate(
                frame_for_roi_extraction_gray,
                cv.cvtColor(dynamic_roi_image, cv.COLOR_BGR2GRAY)
            )
        if reference_roi_template_path is None:
            if reference_roi_captures_tissue:
                roi_capture_heading = "Select the tissue ROI that will be auto adjusted"
            else:
                roi_capture_heading = "Select the reference ROI to track for morphological analysis"
            reference_roi = userDrawnROI(frame_for_roi_extraction, roi_capture_heading)
            reference_roi_image = frame_for_roi_extraction[
                reference_roi['y_start']:reference_roi['y_end'],
                reference_roi['x_start']:reference_roi['x_end'],
            ]
        else:
            reference_roi_image = cv.imread(reference_roi_template_path)
            reference_roi = roiInfoFromTemplate(
                frame_for_roi_extraction_gray,
                cv.cvtColor(reference_roi_image, cv.COLOR_BGR2GRAY)
            )

        rois['template_info'] = {
            'dynamic': {
                'path': '',
                'image': dynamic_roi_image,
                'roi': dynamic_roi
            },
            'reference': {
                'path': '',
                'image': reference_roi_image,
                'roi': reference_roi
            }
        }

    return rois


def plotCa2Signals(
    time_stamps: np.ndarray,
    signal: np.ndarray,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    plot_title: str = '',
    plot_smoothed_signal: bool = True,
    display_results: bool = True,
    plot_file_path: str = None
):
    plt.suptitle('Ca2+ Activity')
    plt.title(plot_title)
    if plot_smoothed_signal:
        plt.plot(time_stamps, signal)
    else:
        plt.scatter(time_stamps, signal, s=2, facecolors='none', edgecolors='g')
    plt.scatter(time_stamps[peak_indices], signal[peak_indices], s=80, facecolors='none', edgecolors='b')
    plt.scatter(time_stamps[trough_indices], signal[trough_indices], s=80, facecolors='none', edgecolors='r')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Intensity')
    if plot_file_path is not None:
        # NOTE: do NOT change the order of saving and showing the plot
        #       the plots will be blank if save is not performed first
        plt.savefig(fname=plot_file_path, format='png', facecolor='white', transparent=False, dpi=400.0)
    if display_results:
        plt.show()
    plt.close()


def signalDataFromXLSX(path_to_data: str) -> Tuple[np.ndarray]:
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


def dataMode(input_data: np.ndarray) -> float:
    data_min = np.floor(np.min(input_data))
    data_max = np.ceil(np.max(input_data))
    histogram_range = range(int(data_min) + 1, int(data_max) + 1)
    data_histogram = np.histogram(input_data, bins=histogram_range, range=(data_min, data_max))[0]
    histogram_peak = np.argmax(data_histogram)
    return data_min + histogram_peak


if __name__ == '__main__':
    analyzeCa2Data(
        path_to_data='select_dir',
        expected_frequency_hz=1.5,
        analysis_method=None,
        save_result_plots=True,
        display_results=False
    )
