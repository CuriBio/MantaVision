import numpy as np
from typing import Tuple, Dict
from scipy.signal import find_peaks, butter, sosfilt
from numpy.polynomial.polynomial import Polynomial


def waveFormAnalysis(
    value_data: np.ndarray,
    time_stamps: np.ndarray = None,
    expected_frequency_hz: float = None,
    expected_min_peak_width: int = None,
    expected_min_peak_height: float = None
) -> Dict:
    """ """
    extreme_point_indices = peakAndTroughIndices(
        value_data,
        time_stamps,
        expected_frequency_hz=expected_frequency_hz,
        expected_min_peak_width=expected_min_peak_width,
        expected_min_peak_height=expected_min_peak_height
    )
    peak_indices, trough_indices = extreme_point_indices
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
    metric_failure_counter = np.zeros(shape=num_metrics_to_compute, dtype=np.float32)

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
        expected_min_peak_height = 1.0
    if expected_min_peak_width is None:
        expected_min_peak_width = 1

    if expected_frequency_hz is not None and time_stamps is not None:
        expected_frequency_hz = expected_frequency_hz
        expected_frequency_tolerance_hz = 0.5
        pacing_frequency_min_hz = expected_frequency_hz - expected_frequency_tolerance_hz
        pacing_frequency_max_hz = expected_frequency_hz + expected_frequency_tolerance_hz

        # compute the width (in samples) from peak to peak or trough to trough that we expect the
        # signal to contain, so we can eliminate noise components we presume will be shorter than this
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
    return peaks, troughs, peaks_and_troughs_sorted


def lowPassFiltered(input_signal: np.ndarray, time_stamps: np.ndarray) -> np.ndarray:
    filter_order = 5  # how sharply the filter cuts off the larger the sharper it bends
    # frequency_range_hz = [1.0, 2.0]  # cut off frequency [start, stop] for bandpass/bandstop
    frequency_hz = 1.5
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = len(time_stamps)
    sampleing_frequency = float(num_samples)/duration
    sos = butter(filter_order, frequency_hz, 'lowpass', fs=sampleing_frequency, output='sos')
    return sosfilt(sos, input_signal)
