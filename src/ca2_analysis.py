import os
import cv2 as cv
import openpyxl
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter
from track_template import userDrawnROI
from os_functions import contentsOfDir
from waveform_analysis import waveFormAnalysis
from video_api import VideoReader, VideoWriter, supported_file_extensions
from morphology import roiInfoFromTemplates, morphologyMetricsForImage
from track_template import trackTemplate
from math import floor, ceil
import matplotlib.pyplot as plt
pd.set_option("display.precision", 2)
pd.set_option("display.expand_frame_repr", False)


# TODO: instead of using an expected frequency parameters,
#       we could allow users to select a "shape" parameter i.e. sinusoid, exponential, sawtooth, square etc
#       and then perform regression to determine the parameters of those shapes that best fits the data points
#       that we extract from a contraction (calcium or regular) video

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
#       or, 
#       the alternative is to use splines which do essentially the same thing
#       without having to explicitly code the linear interpolation, but since we have to 
#       compute enough splines to for the resolution we need computationally very expensive


def videoROIsInfo(video_path: str, template_info: Dict) -> Dict:
    """ """
    template_image_rgb_left = template_info['left']['image']
    _, left_template_results, _, _, _ = trackTemplate(
        input_video_path=video_path,
        template_guide_image_path=None,
        template_rgb=template_image_rgb_left,
        max_translation_per_frame=[50, 50]
    )
    right_template_image_rgb = template_info['right']['image']
    _, right_template_results, _, _, _ = trackTemplate(
        input_video_path=video_path,
        template_guide_image_path=None,
        template_rgb=right_template_image_rgb,
        max_translation_per_frame=[50, 50]
    )

    rois_info = []
    num_frames = len(right_template_results)
    for frame_num in range(num_frames):
        rois_info.append(
            {
                'left': {
                    'x_start': int(left_template_results[frame_num]['x_start']),
                    'x_end': int(left_template_results[frame_num]['x_end']),
                    'y_start': int(left_template_results[frame_num]['y_start']),
                    'y_end': int(left_template_results[frame_num]['y_end']),
                },
                'right': {
                    'x_start': int(right_template_results[frame_num]['x_start']),
                    'x_end': int(right_template_results[frame_num]['x_end']),
                    'y_start': int(right_template_results[frame_num]['y_start']),
                    'y_end': int(right_template_results[frame_num]['y_end']),
                }
            }
        )
    return rois_info


def videoROIsInfoPerFrame(video_path: str, template_info: Dict) -> Dict:
    rois_info = []
    video_stream = VideoReader(video_path)
    while video_stream.next():
        rois_info.append(
            roiInfoFromTemplates(
                search_image=video_stream.frameRGB(),
                template_info=template_info
            )
        )
    return rois_info


def signalDataFromVideo(
    input_video_path: str,
    output_video_path: str = None,
    bg_subtraction_method: str = 'None',
    bg_subtraction_rois: Dict = None
) -> Dict:
    """ """

    if 'auto' in bg_subtraction_method.lower():
        rois_info = iter(
            videoROIsInfo(
                video_path=input_video_path,
                template_info=bg_subtraction_rois['template_info']
            )
        )

    input_video_stream = VideoReader(input_video_path)
    if output_video_path is not None:
        video_writer = VideoWriter(
            path=output_video_path,
            width=input_video_stream.frameVideoWidth(),
            height=input_video_stream.frameVideoHeight(),
            time_base=input_video_stream.timeBase(),
            fps=input_video_stream.avgFPS(),
            bitrate=input_video_stream.bitRate(),

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
        if bg_subtraction_method.lower() == 'none':
            signal_values.append(np.sum(frame_gray))
            continue
        if 'rois' in bg_subtraction_method.lower():
            bg_roi = frame_gray[
                bg_subtraction_rois['background']['y_start']:bg_subtraction_rois['background']['y_end'],
                bg_subtraction_rois['background']['x_start']:bg_subtraction_rois['background']['x_end']
            ]
            background_subtractor = np.mean(bg_roi)
            if 'auto' in bg_subtraction_method.lower():
                # get the tissue only roi from morphology function
                frame_rgb = input_video_stream.frameVideoRGB()
                frame_rois_info = next(rois_info)
                frame_rgb, morphology = morphologyMetricsForImage(
                    search_image=frame_rgb,
                    rois_info=frame_rois_info,
                    template_refinement_radius=0,
                    edge_finding_smoothing_radius=10,
                    draw_tissue_roi_only=True
                )
                # compute the mean of the tissue only region
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
                    frame_row = frame_gray_rotated[y_pos[row_num], int(x_start_pos[row_num]):int(x_end_pos[row_num])]
                    row_sums += np.sum(frame_row)
                    row_counts += len(frame_row)
                tissue_mean = float(row_sums)/float(row_counts)
                # append the tissue mean
                fg_means.append(tissue_mean)
                # append the background subtracted tissue mean
                signal_values.append(tissue_mean - background_subtractor)
            else:  # we're using fixed tissue roi which was already set by the user
                # NOTE: cutting out a new frame_gray from the tissue roi has to go AFTER background roi extraction
                frame_gray = frame_gray[
                    bg_subtraction_rois['tissue']['y_start']:bg_subtraction_rois['tissue']['y_end'],
                    bg_subtraction_rois['tissue']['x_start']:bg_subtraction_rois['tissue']['x_end']
                ]
        elif bg_subtraction_method.lower() == 'lowpass':
            background_subtractor = gaussian_filter(frame_gray.astype(float), sigma=8, mode='reflect')
        else:  # bg_subtraction_method.lower() == 'frame_mean':
            background_subtractor = np.mean(frame_gray)

        # append the background mean
        bg_means.append(background_subtractor)
        # append the mean of the fixed tissue roi for all methods other than auto
        if 'auto' not in bg_subtraction_method.lower():
            fg_means.append(np.mean(frame_gray))
            adjusted_frame = frame_gray - background_subtractor
            adjusted_frame[adjusted_frame < 0] = 0
            adjusted_frame_mean = np.mean(adjusted_frame)
            signal_values.append(adjusted_frame_mean)

        # write out the video frame with the rois drawn
        if video_writer is not None:
            # mark the bg (& maybe fg) ROIs on the output video frame
            if frame_rgb is None:
                frame_rgb = input_video_stream.frameVideoRGB()
            rois_to_draw = [bg_subtraction_rois['background']]
            # tissue roi is drawn by the morphology function when auto method is run
            # all other methods require drawing the background roi here
            if 'auto' not in bg_subtraction_method.lower():
                rois_to_draw.append(bg_subtraction_rois['tissue'])
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

                grid_colour_bgr = (0, 255, 0)
                for edge_point_a, edge_point_b in roi_edges:
                    cv.line(
                        img=frame_rgb,
                        pt1=edge_point_a,
                        pt2=edge_point_b,
                        color=grid_colour_bgr,
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
        'background_means': np.array(bg_means, dtype=float)
    }


def videoBGSubtractionROIs(
    video_file_path: str,
    max_seconds_to_search: float = None,
    roi_method: str = None
) -> Dict:
    input_video_stream = VideoReader(video_file_path)
    if not input_video_stream.isOpened():
        return None, None

    input_video_duration = input_video_stream.duration()
    if max_seconds_to_search is None:
        max_seconds_to_search = input_video_duration
    if max_seconds_to_search < 0 or max_seconds_to_search > input_video_duration:
        max_seconds_to_search = input_video_duration

    max_intensity_sum: float = 0.0
    max_intensity_sum_frame: np.ndarray = None
    while input_video_stream.next():
        frame_gray = input_video_stream.frameGray()
        frame_gray_intensity_sum: float = np.sum(frame_gray)
        if frame_gray_intensity_sum > max_intensity_sum:
            max_intensity_sum = frame_gray_intensity_sum
            max_intensity_sum_frame = input_video_stream.frameRGB()  # frame_gray
        if input_video_stream.timeStamp() > max_seconds_to_search:
            break
    rois = {
        'background': userDrawnROI(max_intensity_sum_frame, "Select the Background ROI")
    }
    if 'fixed' in roi_method.lower():
        rois['tissue'] = userDrawnROI(max_intensity_sum_frame, "Select the Tissue/Signal ROI")
    else:
        rois['tissue'] = {}
        left_roi = userDrawnROI(max_intensity_sum_frame, "Select the Left ROI to Track")
        right_roi = userDrawnROI(max_intensity_sum_frame, "Select the Right ROI to Track")
        rois['template_info'] = {
            'left': {
                'path': '',
                'image': max_intensity_sum_frame[
                    left_roi['y_start']:left_roi['y_end'],
                    left_roi['x_start']:left_roi['x_end'],
                ]
            },
            'right': {
                'path': '',
                'image': max_intensity_sum_frame[
                    right_roi['y_start']:right_roi['y_end'],
                    right_roi['x_start']:right_roi['x_end'],
                ]
            }
        }

    return rois


def analyzeCa2Data(
    path_to_data: str,
    expected_frequency_hz: float,
    bg_subtraction_method: str = 'None',
    save_result_plots: bool = False,
    display_results: bool = False,
    expected_min_peak_width: int = None,
    expected_min_peak_height: float = None
):
    """ """
    base_dir, files_to_analyze = contentsOfDir(dir_path=path_to_data, search_terms=supported_file_extensions)
    results_dir_name = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_dir, results_dir_name)
    os.mkdir(results_dir)

    if 'rois' in bg_subtraction_method.lower():
        # collect bg & fg or left/right ROIs for all the videos to be analyzed, up front, so that
        # all the videos can then be processed automatically without any user interaction
        video_rois = {}
        expected_frequency_hz_tolerance = 0.5
        frequency_epsilon = 0.01
        seconds_for_period = 1.0/(expected_frequency_hz - expected_frequency_hz_tolerance + frequency_epsilon)
        bg_subtraction_roi_search_seconds = 2.0*seconds_for_period
        for file_name, file_extension in files_to_analyze:
            input_video_file_path = os.path.join(base_dir, file_name + file_extension)
            video_rois[file_name] = videoBGSubtractionROIs(
                input_video_file_path,
                bg_subtraction_roi_search_seconds,
                roi_method=bg_subtraction_method
            )
    else:
        video_rois = None

    for file_name, file_extension in files_to_analyze:
        if video_rois is None:
            bg_subtraction_rois = None
            output_video_path = None
        else:
            bg_subtraction_rois = video_rois[file_name]
            if "nd2" in file_extension.lower():
                output_file_extension = ".avi"
            else:
                output_file_extension = file_extension
            output_video_file_name = file_name + "-with_rois" + output_file_extension

            output_video_path = os.path.join(results_dir, output_video_file_name)
        input_video_file_path = os.path.join(base_dir, file_name + file_extension)
        # TODO: merge bg_method and rois into a single dictionary called method_details
        signal_data = signalDataFromVideo(
            input_video_file_path,
            output_video_path,
            bg_subtraction_method,
            bg_subtraction_rois
        )
        if signal_data is None:
            raise RuntimeError("Error. Signal from video could not be extracted")
        time_stamps = signal_data['time_stamps']
        input_signal = signal_data['signal_values']
        tissue_means = signal_data['tissue_means']
        background_means = signal_data['background_means']

        signal_data_file_path = os.path.join(results_dir, file_name + '-signal_data.xlsx')
        signalDataToCSV(
            time_stamps,
            input_signal,
            tissue_means,
            background_means,
            signal_data_file_path
        )

        signal_data_for_sdk_file_path = os.path.join(results_dir, file_name + '-signal_data_for_sdk.xlsx')
        signalDataForSDK(
            time_stamps,
            input_signal,
            signal_data_for_sdk_file_path
        )

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
            print("Was expected frequency set to +/- 0.5Hz of expected frequency?")
            print("No analysis results were written")
            print("\n")
            continue

        path_to_ca2_analysis_results_file = os.path.join(results_dir, file_name + '-results.xlsx')
        ca2AnalysisResultsToCSV(
            ca2_analysis,
            path_to_ca2_analysis_results_file,
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
                plot_file_path = os.path.join(results_dir, file_name + '-plot.png')
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


def signalDataFromXLSX(path_to_data: str) -> Tuple[np.ndarray]:
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


def plotCa2Signals(
    time_stamps: np.ndarray,
    signal: np.ndarray,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    plot_title: str='',
    plot_smoothed_signal: bool=True,
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


def signalDataForSDK(
    time_stamps: np.ndarray,
    input_signal: np.ndarray,
    sdk_signal_data_file_path: str
):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # set the column headings
    heading_row = 1
    time_stamp_column = 1
    sheet.cell(heading_row, time_stamp_column).value = 'time'
    signal_value_column = time_stamp_column + 1
    sheet.cell(heading_row, signal_value_column).value = 'signal'

    # set the data values
    data_row_start = heading_row + 1
    num_data_points = len(time_stamps)
    for data_point_index in range(0, num_data_points):
        row_num = data_point_index + data_row_start
        sheet.cell(row_num, time_stamp_column).value = time_stamps[data_point_index]
        sheet.cell(row_num, signal_value_column).value = input_signal[data_point_index]

    workbook.save(filename=sdk_signal_data_file_path)


def signalDataToCSV(
    time_stamps: np.ndarray,
    input_signal: np.ndarray,
    tissue_means: np.ndarray,
    background_means: np.ndarray,
    output_data_file_path: str
):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # set the column headings
    heading_row = 1
    frame_number_column = 1
    sheet.cell(heading_row, frame_number_column).value = 'frame number'
    time_stamp_column = frame_number_column + 1
    sheet.cell(heading_row, time_stamp_column).value = 'time'
    signal_value_column = time_stamp_column + 1
    sheet.cell(heading_row, signal_value_column).value = 'signal'
    fg_mean_value_column = signal_value_column + 1
    sheet.cell(heading_row, fg_mean_value_column).value = 'fg_mean'
    bg_mean_value_column = fg_mean_value_column + 1
    sheet.cell(heading_row, bg_mean_value_column).value = 'bg_mean'

    # set the data values
    data_row_start = heading_row + 1
    num_data_points = len(time_stamps)
    for data_point_index in range(0, num_data_points):
        row_num = data_point_index + data_row_start
        sheet.cell(row_num, frame_number_column).value = data_point_index
        sheet.cell(row_num, time_stamp_column).value = time_stamps[data_point_index]
        sheet.cell(row_num, signal_value_column).value = input_signal[data_point_index]
        sheet.cell(row_num, fg_mean_value_column).value = tissue_means[data_point_index]
        sheet.cell(row_num, bg_mean_value_column).value = background_means[data_point_index]

    workbook.save(filename=output_data_file_path)


def ca2AnalysisResultsToCSV(
        ca2_analysis: Dict,
        path_to_results_file: str,
):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # set the column headings
    heading_row = 1
    metric_type_column = 1
    sheet.cell(heading_row, metric_type_column).value = 'metric type'
    metric_value_column = metric_type_column + 1
    sheet.cell(heading_row, metric_value_column).value = 'metric average'
    normalized_metric_value_column = metric_value_column + 1
    sheet.cell(heading_row, normalized_metric_value_column).value = 'normalized metric average'
    num_points_column = normalized_metric_value_column + 1
    sheet.cell(heading_row, num_points_column).value = 'num points'
    num_failed_points_column = num_points_column + 1
    sheet.cell(heading_row, num_failed_points_column).value = 'num failed points'
    percent_failed_points_column = num_failed_points_column + 1
    sheet.cell(heading_row, percent_failed_points_column).value = '% failed points'

    # set the column values for each metric
    row_num = heading_row + 1
    num_p2p_types = len(ca2_analysis['metrics'])
    for p2p_type_num in range(num_p2p_types):
        metrics = ca2_analysis['metrics'][p2p_type_num]
        metric_labels = metrics['metrics_labels']
        metric_values = metrics['mean_metric_data']
        normalized_metric_values = metrics['normalized_metric_data']
        num_points = metrics['num_metric_points']
        num_failed_points = metrics['num_metric_failures']
        failure_percentages = metrics['metric_failure_proportions']

        num_metrics = len(metric_values)
        for metric_num in range(num_metrics):
            sheet.cell(row_num, metric_type_column).value = metric_labels[metric_num]
            sheet.cell(row_num, metric_value_column).value = metric_values[metric_num]
            sheet.cell(row_num, normalized_metric_value_column).value = normalized_metric_values[metric_num]
            sheet.cell(row_num, num_points_column).value = num_points
            sheet.cell(row_num, num_failed_points_column).value = num_failed_points[metric_num]
            sheet.cell(row_num, percent_failed_points_column).value = failure_percentages[metric_num]
            row_num += 1

    # add a measure of average frequency
    row_num += 1
    avg_frequency = ca2_analysis['avg_frequency']
    sheet.cell(row_num, metric_type_column).value = 'frequency'
    sheet.cell(row_num, metric_value_column).value = avg_frequency

    workbook.save(filename=path_to_results_file)


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
        bg_subtraction_method=None,
        save_result_plots=True,
        display_results=False
    )
