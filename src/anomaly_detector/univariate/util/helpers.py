# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pandas as pd

import numpy as np
from seasonal import periodogram_peaks
from seasonal import trend as trend_detector
import os
import sys
from anomaly_detector.univariate._anomaly_kernel_cython import median_filter

import json
from anomaly_detector.univariate._anomaly_kernel_cython import calculate_esd_values
from statsmodels import robust
from anomaly_detector.univariate.util.date_utils import get_date_difference
from anomaly_detector.univariate.util.fields import DEFAULT_ALPHA, YEAR_SECOND, MONTH_SECOND, WEEK_SECOND, \
    DAY_SECOND, HOUR_SECOND, MINUTE_SECOND, Granularity, SECOND, MICRO_SECOND
from anomaly_detector.univariate.resource.error_message import ValueOverflow
from anomaly_detector.univariate.util.critical_table_values import critical_table


def leastsq(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(np.multiply(x, x))
    sum_xy = np.sum(np.multiply(x, y))

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_xx * sum_y - sum_x * sum_xy) / (n * sum_xx - sum_x * sum_x)

    return a, b


def average_filter(values, n=3, fill_to_n=False):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """
    fill_n = n
    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        if fill_to_n:
            res[i] = (res[i] + ((res[i] - values[i]) / i) * (fill_n - i - 1)) / fill_n
        else:
            res[i] /= (i + 1)

    return res


def interp(values):
    value_array = np.array(values)
    nans = np.isnan(value_array)
    nans_index = nans.nonzero()[0]
    non_nans_index = (~np.isnan(value_array)).nonzero()[0]
    value_array[nans_index] = np.interp(nans_index, non_nans_index, value_array[non_nans_index])
    return value_array


def trend_detection(series, trend_type='spline', period=None):
    if len(series) < 6:
        mean = np.mean(series)
        return np.full(len(series), mean)
    return fit_trend(np.array(series), kind=trend_type, period=period)


def fit_trend(data, kind="spline", period=None, ptimes=2):
    if kind is None:
        return np.zeros(len(data)) + np.mean(data)
    if period is None:
        period = guess_trended_period(data)
    if period <= 1:
        window = min(len(data) // 3, 512)
    else:
        window = (int(period * ptimes) // 2) * 2 - 1  # odd window
    if kind == "median":
        filtered = trend_detector.aglet(median_filter(data, window), window)
    elif kind == "mean":
        filtered = trend_detector.aglet(trend_detector.mean_filter(data, window), window)
    elif kind == "line":
        filtered = trend_detector.line_filter(data, window)
    elif kind == "spline":
        n_segments = len(data) // (window * 2) + 1
        filtered = trend_detector.aglet(trend_detector.spline_filter(data, n_segments), window)
    else:
        raise Exception("adjust_trend: unknown filter type {}".format(kind))
    return filtered


def guess_trended_period(data):
    max_period = min(len(data) // 3, 512)
    broad = fit_trend(data, kind="median", period=max_period)
    if np.any(~np.isfinite(broad)):
        raise Exception(ValueOverflow)

    peaks = periodogram_peaks(data - broad)
    if peaks is None:
        return max_period
    periods, scores, _, _ = zip(*peaks)
    period = int(round(np.average(periods, weights=scores)))
    return period


def get_verified_majority_value(sorted_series, num_obs):
    if np.isclose(robust.mad(sorted_series), 0):
        majority_value = sorted_series.iloc[(num_obs - 1) // 2]
        # this type of series should not be calculated by AnomalyDetectorMad
        if np.isclose(majority_value, sorted_series.iloc[0]) or np.isclose(majority_value, sorted_series.iloc[-1]):
            return None
        return majority_value

    return None


def get_critical(alpha, num_obs, max_outliers):
    if np.isclose(alpha, DEFAULT_ALPHA) and num_obs <= len(critical_table):
        begin = len(critical_table) - num_obs
        return critical_table[begin: begin + max_outliers]
    else:
        return [calculate_esd_values(x, alpha, num_obs) for x in range(1, max_outliers + 1)]


def get_second_by_granularity(granularity, custom_interval):
    if custom_interval is None:
        custom_interval = 1
    if granularity == Granularity.secondly:
        return SECOND * custom_interval
    elif granularity == Granularity.minutely:
        return MINUTE_SECOND * custom_interval
    elif granularity == Granularity.hourly:
        return HOUR_SECOND * custom_interval
    elif granularity == Granularity.daily:
        return DAY_SECOND * custom_interval
    elif granularity == Granularity.weekly:
        return WEEK_SECOND * custom_interval
    elif granularity == Granularity.monthly:
        return MONTH_SECOND * custom_interval
    elif granularity == Granularity.yearly:
        return YEAR_SECOND * custom_interval
    elif granularity == Granularity.microsecond:
        return MICRO_SECOND * custom_interval
    elif granularity == Granularity.none:
        return 0
    else:
        raise AttributeError('Invalid granularity')


def get_microsecond_by_granularity(granularity, custom_interval):
    return int(get_second_by_granularity(granularity, custom_interval) * 1000)


def get_pattern_by_second(second):
    if second >= YEAR_SECOND:
        return Granularity.yearly
    elif second >= MONTH_SECOND:
        return Granularity.monthly
    elif second >= WEEK_SECOND:
        return Granularity.weekly
    elif second >= DAY_SECOND:
        return Granularity.daily
    elif second >= HOUR_SECOND:
        return Granularity.hourly
    elif second >= MINUTE_SECOND:
        return Granularity.minutely
    elif second >= SECOND:
        return Granularity.secondly
    elif second >= MICRO_SECOND:
        return Granularity.microsecond
    else:
        return Granularity.none


def get_period_pattern(period, granularity, custom_interval):
    if custom_interval is None:
        custom_interval = 1
    second = get_second_by_granularity(granularity, custom_interval)
    return get_pattern_by_second(second * period)


def get_indices_from_timestamps(granularity, custom_interval, timestamps):
    """
    Transform the timestamps to indices. Assuming the timestamps list is in ascending order.
    e.g. timestamps is ['2019-01-01', '2019-01-02', '2019-01-04'],  granularity is Granularity.daily,
    custom_interval is 1. So the indices should be [0, 1, 3].

    :param granularity: Granularity.
    :param custom_interval: int.
    :param timestamps: list, list of timestamps
    :return: indices: list, list of index.
    :return: first_invalid_index: int, the first index of the timestamp which is invalid according to granularity,
        custom_interval and the first timestamp.
    """

    custom_interval = max(1 if custom_interval is None else custom_interval, 1)
    if timestamps is None or len(timestamps) == 0:
        return [], None
    start_timestamp = timestamps[0]
    timestamps_len = len(timestamps)

    if granularity != Granularity.monthly and granularity != Granularity.yearly and granularity != Granularity.microsecond:
        interval_in_seconds = get_second_by_granularity(granularity, custom_interval)
        start_timestamp_in_second = start_timestamp.timestamp()
        deltas = [t.timestamp() - start_timestamp_in_second for t in timestamps]
        indices = list([x / interval_in_seconds for x in deltas])

    if granularity == Granularity.microsecond:
        interval_in_microseconds = get_microsecond_by_granularity(granularity, custom_interval)
        start_timestamp_in_microsecond = int(start_timestamp.timestamp() * 1000)
        deltas = [int(t.timestamp() * 1000) - start_timestamp_in_microsecond for t in timestamps]
        indices = list([x / interval_in_microseconds for x in deltas])

    elif granularity == Granularity.monthly:
        deltas = list([get_date_difference(x, start_timestamp) for x in timestamps])
        for i in range(timestamps_len):
            if deltas[i].days != 0:
                return None, i
        indices = list([(x.months + x.years * 12) / custom_interval for x in deltas])

    elif granularity == Granularity.yearly:
        deltas = list([get_date_difference(x, start_timestamp) for x in timestamps])
        for i in range(timestamps_len):
            if deltas[i].days != 0 or deltas[i].months != 0:
                return None, i
        indices = list([x.years / custom_interval for x in deltas])

    for i in range(timestamps_len):
        if not isinstance(indices[i], int):
            if not indices[i].is_integer():
                return None, i
            else:
                indices[i] = int(indices[i])
    return indices, None


def normalize(values, min_max=False):
    min_val = values.min()
    max_val = values.max()
    if min_val != max_val:
        values = (values - min_val) / (max_val - min_val)
    if min_max:
        return values, min_val, max_val
    return values


def smooth_spikes(data):
    std = data.std() + 1e-8
    mean = data.mean()
    anomaly_index = [] if std == 0 else np.where(np.abs(data - mean) / std >= 3)[0]
    series = pd.Series(data, copy=True)
    series[anomaly_index] = np.nan
    return series.interpolate(method='linear', limit_direction='both').values


def get_delta(delta, values):
    d_values = [values[i] - values[i - 1] for i in range(1, len(values))]
    d_values = [d_values[delta]] * (delta + 1) + d_values[delta:]
    return d_values


def reverse_delta(first_value, delta, d_values):
    rd_values = [first_value] * delta
    for d in d_values[delta:]:
        rd_values.append(rd_values[-1] + d)
    return rd_values