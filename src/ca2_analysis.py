import os
import openpyxl
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
from numpy.polynomial.polynomial import polyfit, Polynomial
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, sosfilt
from video_api import VideoReader, VideoWriter, supported_file_extensions
from track_template import userDrawnROI
from optical_tracking import getDirPathViaGUI, getFilePathViaGUI, contentsOfDir
import cv2 as cv

import matplotlib.pyplot as plt
pd.set_option("display.precision", 2)
pd.set_option("display.expand_frame_repr", False)


# TODO: for contraction measurements of regular videos
#       we can use tracking to determine the smallest min and largest max positions
#       that way we'd have a good approximation of where the relaxation and contractions should be
#       and we can then pick out the time stamps of frames where there is a local min/max that is
#       within some small margin of error (say +/-10% of the difference in (largest max - smallest min)
#       using these time stamps for known min/max, we can plot the peaks and troughs and estimate
#       the frequency etc.

# TODO: we could also allow people to select a "shape" parameter i.e. sinusoid, exponential, sawtooth, square etc
#       and then perform regression to determine the parameters of those shapes that best fits the data points
#       we extract from a contraction (calcium or regular) video

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


def dataMode(input_data: np.ndarray) -> float:
    data_min = np.floor(np.min(input_data))
    data_max = np.ceil(np.max(input_data))
    histogram_range = range(int(data_min) + 1, int(data_max) + 1)
    data_histogram = np.histogram(input_data, bins=histogram_range, range=(data_min, data_max))[0]
    histogram_peak = np.argmax(data_histogram)
    return data_min + histogram_peak


def signalDataFromVideo(
    input_video_path: str,
    output_video_path: str = None,
    bg_subtraction_method: str = 'None',
    bg_subtraction_rois: Dict = None
) -> Dict:
    """ """
    input_video_stream = VideoReader(input_video_path)
    if not input_video_stream.isOpened():
        return None

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

    time_stamps = []
    signal_values = []
    fg_means = []
    bg_means = []
    while input_video_stream.next():
        time_stamps.append(input_video_stream.timeStamp())
        current_frame = input_video_stream.frameGray()
        if bg_subtraction_method.lower() == 'none':
            signal_values.append(np.sum(current_frame))
            continue
        if bg_subtraction_method.lower() == 'roi':
            bg_roi = current_frame[
                bg_subtraction_rois['bg_roi']['y_start']:bg_subtraction_rois['bg_roi']['y_end'],
                bg_subtraction_rois['bg_roi']['x_start']:bg_subtraction_rois['bg_roi']['x_end']
            ]
            background_subtractor = np.mean(bg_roi)
            # NOTE: cutting out a new current_frame from the fg_roi has to go AFTER bg_roi extraction
            current_frame = current_frame[
                bg_subtraction_rois['fg_roi']['y_start']:bg_subtraction_rois['fg_roi']['y_end'],
                bg_subtraction_rois['fg_roi']['x_start']:bg_subtraction_rois['fg_roi']['x_end']
            ]
        elif bg_subtraction_method.lower() == 'lowpass':
            background_subtractor = gaussian_filter(current_frame.astype(float), sigma=8, mode='reflect')
        else:  # bg_subtraction_method.lower() == 'mean':
            background_subtractor = np.mean(current_frame)
        bg_means.append(background_subtractor)
        fg_means.append(np.mean(current_frame))
        adjusted_frame = current_frame - background_subtractor

        # clip pixels to > 0 and use the mean of all pixels as the signal
        adjusted_frame[adjusted_frame < 0] = 0
        adjusted_frame_mean = np.mean(adjusted_frame)
        signal_values.append(adjusted_frame_mean)

        if video_writer is not None:
            # mark the fg & bg ROIs on the output video frame
            frame = input_video_stream.frameVideoRGB()
            for roi in [bg_subtraction_rois['bg_roi'], bg_subtraction_rois['fg_roi']]:
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

    return {
        'time_stamps': np.array(time_stamps, dtype=float),
        'signal_values': np.array(signal_values, dtype=float),
        'fg_means': np.array(fg_means, dtype=float),
        'bg_means': np.array(bg_means, dtype=float)
    }


def videoBGSubtractionROIs(
    video_file_path: str,
    max_seconds_to_search: float = None
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
        current_frame = input_video_stream.frameGray()
        current_frame_intensity_sum: float = np.sum(current_frame)
        if current_frame_intensity_sum > max_intensity_sum:
            max_intensity_sum = current_frame_intensity_sum
            max_intensity_sum_frame = current_frame
        if input_video_stream.timeStamp() > max_seconds_to_search:
            break
    return {
        'bg_roi': userDrawnROI(max_intensity_sum_frame, "Select the Background ROI"),
        'fg_roi': userDrawnROI(max_intensity_sum_frame, "Select the Foreground/Signal ROI"),
    }


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

    if bg_subtraction_method.lower() == 'roi':
        # collect fg & bg ROIs for all the videos to be analyzed, up front, so that
        # all the videos can then be processed automatically without any user interaction
        video_rois = {}
        expected_frequency_hz_tolerance = 0.5
        frequency_epsilon = 0.01
        seconds_for_period = 1.0/(expected_frequency_hz - expected_frequency_hz_tolerance + frequency_epsilon)
        bg_subtraction_roi_search_seconds = 2.0*seconds_for_period
        for file_name, file_extension in files_to_analyze:
            input_video_file_path = os.path.join(base_dir, file_name + file_extension)
            video_rois[file_name] = videoBGSubtractionROIs(input_video_file_path, bg_subtraction_roi_search_seconds)
    else:
        video_rois = None

    for file_name, file_extension in files_to_analyze:
        if video_rois is None:
            bg_subtraction_rois = None
            output_video_path = None
        else:
            bg_subtraction_rois = video_rois[file_name]
            output_video_file_name = file_name + "-with_rois" + file_extension
            output_video_path = os.path.join(results_dir, output_video_file_name)
        input_video_file_path = os.path.join(base_dir, file_name + file_extension)
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
        fg_means = signal_data['fg_means']
        bg_means = signal_data['bg_means']

        signal_data_file_path = os.path.join(results_dir, file_name + '-signal_data.xlsx')
        signalDataToCSV(
            time_stamps,
            input_signal,
            fg_means,
            bg_means,
            signal_data_file_path
        )

        signal_data_for_sdk_file_path = os.path.join(results_dir, file_name + '-signal_data_for_sdk.xlsx')
        signalDataForSDK(
            time_stamps,
            input_signal,
            signal_data_for_sdk_file_path
        )

        ca2_analysis = ca2Analysis(
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


def ca2Analysis(
    value_data: np.ndarray,
    time_stamps: np.ndarray=None,
    expected_frequency_hz: float=None,
    expected_min_peak_width: int=None,
    expected_min_peak_height: float=None    
) -> Dict:
    """ """
    peak_indices, trough_indices = peakAndTroughIndices(
        value_data,
        time_stamps,
        expected_frequency_hz=expected_frequency_hz,
        expected_min_peak_width=expected_min_peak_width,
        expected_min_peak_height=expected_min_peak_height
    )

    first_peak_time = time_stamps[peak_indices[0]]
    first_trough_time = time_stamps[trough_indices[0]]

    # compute the peak to trough metrics
    p2t_value_fractions = np.asarray([0.5, 0.9], dtype=np.float32)  
    peak_sequence_start = 0
    if first_peak_time < first_trough_time:
        trough_sequence_start = 0
    else:
        trough_sequence_start = 1
    num_peaks = len(peak_indices)
    num_useable_troughs = len(trough_indices) - trough_sequence_start
    num_peaks_to_use = min(num_peaks, num_useable_troughs)
    try:
        peak_to_trough_metrics = pointToPointMetrics(
            start_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_peaks_to_use],
            end_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_peaks_to_use],
            point_values=value_data,
            point_times=time_stamps,
            endpoint_value_fractions=p2t_value_fractions
        )
        peak_to_trough_metrics['p2p_order'] = 'peak_to_trough'
        peak_to_trough_metrics['metrics_labels'] = ['T50R', 'T90R', 'T100R']
    except IndexError:
        return None

    # compute the trough to peak metrics
    t2p_value_fractions = np.zeros(0, dtype=np.float32)
    trough_sequence_start = 0
    if first_trough_time < first_peak_time:
        peak_sequence_start = 0
    else:
        peak_sequence_start = 1
    num_troughs = len(trough_indices)
    num_useable_peaks = len(peak_indices) - peak_sequence_start
    num_troughs_to_use = min(num_troughs, num_useable_peaks)
    try:
        trough_to_peak_metrics = pointToPointMetrics(
            start_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_troughs_to_use],
            end_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_troughs_to_use],
            point_values=value_data,
            point_times=time_stamps,
            endpoint_value_fractions=t2p_value_fractions
        )
        trough_to_peak_metrics['p2p_order'] = 'trough_to_peak'
        trough_to_peak_metrics['metrics_labels'] = ['Tpeak']
    except IndexError:
        return None

    # compute a measure of average frequency
    peak_times = time_stamps[peak_indices]
    avg_frequency = 1.0/np.mean(np.diff(peak_times))

    return {
        'peak_indices': peak_indices,
        'trough_indices': trough_indices, 
        'metrics': [trough_to_peak_metrics, peak_to_trough_metrics],
        'avg_frequency': avg_frequency
    }


def pointToPointMetrics(
    start_point_indices: np.ndarray,
    end_point_indices: np.ndarray,
    point_values: np.ndarray,
    point_times: np.ndarray,
    endpoint_value_fractions: np.ndarray  # don't include a 1.0 case since we always perform this
) -> Dict:
    num_metrics_to_compute = len(endpoint_value_fractions) + 1  # +1 for 100% case we always perform
    num_point_values = len(start_point_indices)
    metrics = np.zeros(shape=(num_metrics_to_compute, num_point_values), dtype=np.float32)
    normalized_metrics = metrics.copy()
    metric_failure_counter = np.zeros(shape=(num_metrics_to_compute), dtype=np.float32)

    for point_index in range(len(start_point_indices)):
        start_point_index = start_point_indices[point_index]
        end_point_index = end_point_indices[point_index] + 1
        # shift times and values to 0 start/reference
        points_to_fit_poly_at = point_times[start_point_index:end_point_index] - point_times[start_point_index]
        value_of_points_to_fit = point_values[start_point_index:end_point_index] - point_values[start_point_index]

        # the time to 100% (endpoint_value_fractions = 1.0) can just be read from the data.
        # a polynomial fit to estimate this time can be wrong because end points for the fit
        # don't always go through the real end points. plus it's a waste of compute time.
        start_point_time = points_to_fit_poly_at[0]
        end_point_time = points_to_fit_poly_at[-1]
        metrics[-1, point_index] = end_point_time
        end_point_value = value_of_points_to_fit[-1]
        normalized_metrics[-1, point_index] = end_point_time/np.abs(end_point_value)

        num_points_for_fit = len(points_to_fit_poly_at)
        min_points_for_3rd_deg_poly = 6  # arbitrary value. empirically determine on a small data set.
        if num_points_for_fit >= min_points_for_3rd_deg_poly:
            polyfit_deg = 3
        else:
            polyfit_deg = 2

        polyfit_of_values = Polynomial.fit(
            points_to_fit_poly_at,
            value_of_points_to_fit,
            polyfit_deg,
            domain=[start_point_time, end_point_time],
            window=[start_point_time, end_point_time]
        )
        poly = Polynomial(polyfit_of_values.convert().coef)
        
        for fraction_id_to_add in range(len(endpoint_value_fractions)):
            fraction_of_value = endpoint_value_fractions[fraction_id_to_add]
            fraction_of_p2p_diff = fraction_of_value*end_point_value
            roots = Polynomial.roots(poly - fraction_of_p2p_diff)
            failure_count = 1.0
            for fraction_point_time in roots:
                if np.iscomplex(fraction_point_time):
                    continue  # imaginary part must be non zero so we can't use it 
                if fraction_point_time < start_point_time or fraction_point_time > end_point_time:
                    continue
                # could still have a complex num obj with imaginary part 0 so force to be real
                fraction_point_time = np.real(fraction_point_time)
                metrics[fraction_id_to_add, point_index] = fraction_point_time
                normalized_metrics[fraction_id_to_add, point_index] = fraction_point_time/np.abs(fraction_of_p2p_diff)
                failure_count = 0.0
                break
            metric_failure_counter[fraction_id_to_add] += failure_count

    metrics_counters = np.abs(metric_failure_counter - num_point_values)
    metrics_sums = np.sum(metrics, axis=-1)
    metrics_means = metrics_sums/metrics_counters
    normalized_metrics_sums = np.sum(normalized_metrics, axis=-1)
    normalized_metrics_sums_means = normalized_metrics_sums/metrics_counters
    metrics_failure_proportions = metric_failure_counter/num_point_values
    return {
        'metric_fractions':             np.append(endpoint_value_fractions, [1.0]),  # add the 100% case
        'p2p_metric_data':              metrics,
        'mean_metric_data':             metrics_means,
        'normalized_metric_data':       normalized_metrics_sums_means,
        'num_metric_failures':          metric_failure_counter,
        'num_metric_points':            num_point_values,
        'metric_failure_proportions':   metrics_failure_proportions
    }


def peakAndTroughIndices(
    input_data: np.ndarray,
    time_stamps: np.ndarray = None,
    expected_frequency_hz: float = None,
    expected_min_peak_width: int = None,
    expected_min_peak_height: float = None
) -> Tuple[np.ndarray]:
    """ Returns the indices of peaks and troughs found in the 1D input data """
    if expected_min_peak_height is None:
        expected_min_peak_height = 1.0  # was 5.0
    if expected_min_peak_width is None:
        expected_min_peak_width = 1  # was 5
    
    if expected_frequency_hz is not None and time_stamps is not None:
        expected_frequency_hz = expected_frequency_hz
        expected_frequency_tolerance_hz = 0.5
        pacing_frequency_min_hz = expected_frequency_hz - expected_frequency_tolerance_hz
        pacing_frequency_max_hz = expected_frequency_hz + expected_frequency_tolerance_hz

        # compute the width (in samples) from peak to peak or trough to trough that we expect the
        # signal to contain so we can eliminate noise components we presume will be shorter than this
        duration = time_stamps[-1] - time_stamps[0]
        num_samples = len(time_stamps)
        sampling_rate = float(num_samples)/duration
        expected_min_peak_width = sampling_rate/pacing_frequency_max_hz

        # compute the height from trough to peak or peak to trough that we expect the signal to contain.
        # we use this to eliminate noise components we presume will be smaller than this value.
        # NOTE: it is probably not necessary to pass this parameter to the peak finder; as in,
        # it will likely work without this, and since it is much harder to estimate than the expected width,
        # and be a problem with for instance highly decaying signals and/or signals with significant noise,
        # it should be the first thing to consider changing (not using) if we're failing to pick all peaks/troughs.
        # also the use of HALF the calculated height of the middle-ish peak is entirely arbitrary and could
        # be replaced with some other fraction of a trough to peak height, or from a different place in the signal.
        middle_sample = int(num_samples/2)
        signal_sample_1_cycle_start = middle_sample 
        signal_sample_1_cycle_end = signal_sample_1_cycle_start + int(expected_min_peak_width)
        signal_sample_1_cycle = input_data[signal_sample_1_cycle_start:signal_sample_1_cycle_end]
        signal_sample_1_cycle_min = np.min(signal_sample_1_cycle)
        signal_sample_1_cycle_max = np.max(signal_sample_1_cycle)
        middle_peak_height = np.abs(signal_sample_1_cycle_max - signal_sample_1_cycle_min)
        min_height_scale_factor = 0.5
        expected_min_peak_height = min_height_scale_factor * middle_peak_height

    peaks, _ = find_peaks(input_data, prominence=expected_min_peak_height, distance=expected_min_peak_width)
    troughs, _ = find_peaks(-input_data, prominence=expected_min_peak_height, distance=expected_min_peak_width)

    return peaks, troughs


def extremePointIndices(signal: np.ndarray) -> Tuple[np.ndarray]:
    peaks, troughs = peakAndTroughIndices(signal)
    peaks_and_troughs = np.concatenate((peaks, troughs), axis=0)
    peaks_and_troughs_sorted = np.sort(peaks_and_troughs)
    return (peaks, troughs, peaks_and_troughs_sorted)


def signalDataFromXLSX(path_to_data: str) -> Tuple[np.ndarray]:
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


def lowPassFiltered(input_signal: np.ndarray, time_stamps: np.ndarray) -> np.ndarray:
    filter_order = 5  # how sharply the filter cuts off the larger the sharper it bends
    # frequency_range_hz = [1.0, 2.0]  # cut off frequency [start, stop] for bandpass/bandstop
    frequency_hz = 1.5
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = len(time_stamps)
    sampleing_frequency = float(num_samples)/duration
    sos = butter(filter_order, frequency_hz, 'lowpass', fs=sampleing_frequency, output='sos')
    return sosfilt(sos, input_signal)


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
    fg_signal_means: np.ndarray,
    bg_signal_means: np.ndarray,
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
        sheet.cell(row_num, fg_mean_value_column).value = fg_signal_means[data_point_index]
        sheet.cell(row_num, bg_mean_value_column).value = bg_signal_means[data_point_index]

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


if __name__ == '__main__':
    analyzeCa2Data(
        path_to_data='select_dir',
        expected_frequency_hz=1.5,
        bg_subtraction_method=None,
        save_result_plots=True,
        display_results=False
    )
