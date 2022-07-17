import math
import os
import glob
import openpyxl
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
from numpy.polynomial.polynomial import polyfit, Polynomial
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, sosfilt
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename, askdirectory
from video_api import VideoReader

import matplotlib.pyplot as plt
pd.set_option("display.precision", 2)
pd.set_option("display.expand_frame_repr", False)


# TODO: figure out why stream initialisation doesn't work

# TODO: figure out why T50 is larger than T90 which makes no sense

# TODO: create a function that will iterate over each frame and compute the sum of all pixel values
#       then the frame with the highest count will be the frame with max contraction
#       (although given there is decay, I presume we'll get the very first peak max contraction)
#       we use that frame to present a multi ROI selector, the user selects 2 ROIs
#       we then compute the sum of the intensity values of each ROI, the one with the smallest sum
#       must be the background ROI and the other must be the foreground ROI. Then, we
#       compute the mean "background adjusted" foreground intensity of the foreground ROI by
#       first computing the mean intensity value of the background ROI,
#       then subtracting the mean background intensity value from each pixel in the foreground ROI, and then
#       computing the mean intensity value of the foreground ROI from the "background adjusted" foreground ROI.

# TODO: if the user selects to do background subtraction using a ROI,
#       then let them select the ROI's for all the videos up front
#       and then save the details and use them to perform the analysis

# TODO: check there are no double peaks or troughs.
#       would need to use the peak and trough indicies
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
#       metric parameter lives between, and linear iteropolate between those two points
#       to find the point the metric is for. this would clearly be inferiour since
#       we know most of these signals have exponential or polynomial shape and
#       a linear interpolation between two points (even if close) isn't a great fit.
#       or, 
#       the alternative is to use splines which do essentially the same thing
#       without having to explicitly code the linear interpolation, but since we have to 
#       compute enough splines to for the resolution we need computationally very expensive


def signalDataFromVideo(input_video_path: str, background_subtraction: str = 'None') -> Tuple[np.ndarray, np.ndarray]:
    """ """
    input_video_stream = VideoReader(input_video_path)
    if not input_video_stream.isOpened():
        return None, None

    time_stamps = []
    signal_values = []
    while input_video_stream.next():
        time_stamps.append(input_video_stream.timeStamp())
        current_frame = input_video_stream.frameGray()
        if background_subtraction.lower() == 'none':
            signal_values.append(np.sum(current_frame))
            continue
        if background_subtraction.lower() == 'lowpass':
            background_subtractor = gaussian_filter(current_frame.astype(float), sigma=8, mode='reflect')
        else:  # background_subtraction.lower() == 'mean':
            background_subtractor = np.mean(current_frame)
        adjusted_frame = current_frame - background_subtractor

        # clip pixels to > 0 and use the mean of all pixels as the signal
        adjusted_frame[adjusted_frame < 0] = 0
        adjusted_frame_mean = np.mean(adjusted_frame)
        signal_values.append(adjusted_frame_mean)

    input_video_stream.close()
    return np.array(time_stamps, dtype=float), np.array(signal_values, dtype=float)


def analyzeCa2Data(
    path_to_data: str,
    expected_frequency_hz: float,
    background_subtraction: str = 'None',
    save_result_plots: bool = False,
    display_results: bool = False,
    expected_min_peak_width: int = None,
    expected_min_peak_height: float = None
):
    """ """

    if path_to_data.lower() == 'select_file':
        path_to_data = getFilePathViaGUI(window_title='Select the file to analyze')
    elif path_to_data.lower() == 'select_dir':
        path_to_data = getDirPathViaGUI(window_title='Select the directory with files to analyze')

    # base_dir, files_to_analyze = contentsOfDir(dir_path=path_to_data, search_terms=['.xlsx'])
    base_dir, files_to_analyze = contentsOfDir(dir_path=path_to_data, search_terms=['*.*'])
    results_dir_name = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_dir, results_dir_name) 
    os.mkdir(results_dir)

    for file_name, file_extension in files_to_analyze:
        # signal_data_file_path = os.path.join(base_dir, file_name + file_extension)
        # time_stamps, input_signal = signalDataFromXLSX(signal_data_file_path)
        input_video_file_path = os.path.join(base_dir, file_name + file_extension)
        time_stamps, input_signal = signalDataFromVideo(input_video_file_path, background_subtraction)
        if input_signal is None or time_stamps is None:
            raise RuntimeError("Error. Signal from video could not be extracted")

        ca2_analysis = ca2Analysis(
            input_signal,
            time_stamps,
            expected_frequency_hz=expected_frequency_hz,
            expected_min_peak_width=expected_min_peak_width,
            expected_min_peak_height=expected_min_peak_height
        )

        path_to_results_file = os.path.join(results_dir, file_name + '-results.xlsx')
        resultsToCSV(
            ca2_analysis,
            path_to_results_file,
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
    peak_to_trough_metrics = pointToPointMetrics(
        start_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_peaks_to_use],
        end_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_peaks_to_use],
        point_values=value_data,
        point_times=time_stamps,
        endpoint_value_fractions=p2t_value_fractions
    )
    peak_to_trough_metrics['p2p_order'] = 'peak_to_trough'
    peak_to_trough_metrics['metrics_labels'] = ['T50R', 'T90R', 'T100R']

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
    trough_to_peak_metrics = pointToPointMetrics(
        start_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_troughs_to_use],
        end_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_troughs_to_use],
        point_values=value_data,
        point_times=time_stamps,
        endpoint_value_fractions=t2p_value_fractions        
    )
    trough_to_peak_metrics['p2p_order'] = 'trough_to_peak'
    trough_to_peak_metrics['metrics_labels'] = ['Tpeak']

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


# below are copies of functions in mantavision and a separate function with these 
# utility functions should be made in a separate repo, or these repos combined

def contentsOfDir(
        dir_path: str,
        search_terms: List[str],
        search_extension_only: bool=True
) -> Tuple[List[str], List[Tuple[str]]]:
    all_files_found = []
    if os.path.isdir(dir_path):
        base_dir = dir_path
    for search_term in search_terms:
        glob_search_term = '*' + search_term
        if not search_extension_only:
            glob_search_term += '*'
        files_found = glob.glob(os.path.join(dir_path, glob_search_term))
        if len(files_found) > 0:
            all_files_found.extend(files_found)
        else:
            # presume it's actually a single file path
            base_dir = os.path.dirname(dir_path)
            all_files_found = [dir_path]
        if len(all_files_found) < 1:
            return None, None

    files = []
    for file_path in all_files_found:
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        files.append((file_name, file_extension))
    return base_dir, files


def getDirPathViaGUI(window_title: str='') -> str:
    # show an "Open" dialog box and return the path to the selected dir
    window=tk()
    window.withdraw()
    window.lift()
    window.overrideredirect(True)
    window.call('wm', 'attributes', '.', '-topmost', True)
    return askdirectory(
        initialdir='./',
        title=window_title
    )


def getFilePathViaGUI(window_title: str='') -> str:
  # show an "Open" dialog box and return the path to the selected file
  window=tk()
  window.withdraw()
  window.lift()  
  window.overrideredirect(True)
  window.call('wm', 'attributes', '.', '-topmost', True)
  return askopenfilename(
    initialdir='./',
    title=window_title    
  )



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


def resultsToCSV(
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
        background_subtraction=None,
        save_result_plots=True,
        display_results=False
    )
