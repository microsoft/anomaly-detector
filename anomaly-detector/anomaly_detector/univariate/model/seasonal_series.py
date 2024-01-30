# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import math
import numpy as np
import pandas as pd

from anomaly_detector.univariate.model.detect import detect
from anomaly_detector.univariate.util import Value, Direction, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, Trend, \
    get_verified_majority_value, ModelType, normalize
from anomaly_detector.univariate.util import stl, stl_adjust_trend, interp, de_outlier_stl, MAPE_LB, MAPE_UB
from anomaly_detector.univariate.detectors import ESD
from anomaly_detector.univariate.detectors import ZScoreDetector


def seasonal_series_detection(series, period, max_anomaly_ratio, alpha, adjust_trend=False, need_trend=False, last_value=None):
    return detect_ts(series, period, max_anomaly_ratio=max_anomaly_ratio, alpha=alpha, adjust_trend=adjust_trend,
                     need_trend=need_trend, last_value=last_value)


def detect_ts(series, num_obs_per_period, max_anomaly_ratio=0.10, alpha=0.05, adjust_trend=False, need_trend=False, last_value=None):
    if max_anomaly_ratio > 0.49:
        length = len(series)
        raise ValueError(
            ("max_anomaly_ratio must be less than 50% of "
             "the data points (max_anomaly_ratio =%f data_points =%s).")
            % (round(max_anomaly_ratio * length, 0), length))

    num_obs = len(series)

    clamp = (1 / float(num_obs))
    if max_anomaly_ratio < clamp:
        max_anomaly_ratio = clamp

    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if num_obs < num_obs_per_period * 2 + 1:
        raise ValueError("Anomaly detection needs at least 2 periods worth of data")

    input_series = series.copy()
    stl_func = stl_adjust_trend if adjust_trend else stl
    decompose_result = de_outlier_stl(input_series, stl_func=stl_func, period=num_obs_per_period, log_transform=False)

    mape = np.mean(np.abs(decompose_result['remainder'] / input_series))

    if mape > MAPE_UB:
        decompose_result_log = de_outlier_stl(input_series, stl_func=stl_func, period=num_obs_per_period,
                                              log_transform=True)
        mape_log = np.mean(np.abs(decompose_result_log['remainder'] / input_series))
        if mape_log < MAPE_LB:
            decompose_result = decompose_result_log

    decompose_trend = decompose_result['trend']
    decompose_season = decompose_result['seasonal']

    de_seasoned_value = series - decompose_season
    remainder = de_seasoned_value - decompose_trend

    directions = [Direction.upper_tail, Direction.lower_tail]
    data = pd.Series(normalize(de_seasoned_value))
    anomalies, model_id = detect_anomaly(data=data, alpha=alpha, ratio=max_anomaly_ratio,
                                         directions=directions,
                                         remainder_values=pd.Series(normalize(remainder)),
                                         last_value=last_value)

    if len(anomalies) != 0:
        decompose_trend[anomalies.index] = np.nan
        nan_window = num_obs_per_period // 2
        if np.sum(anomalies.index >= len(data) - nan_window) >= 0.5 * nan_window:
            decompose_trend[-nan_window:] = np.nan
        decompose_trend = interp(decompose_trend)

    p = {
        ExpectedValue: decompose_trend + decompose_season,
        Value: series
    }

    if need_trend:
        p[Trend] = decompose_trend

    expected_values = pd.DataFrame(p)
    if len(anomalies) != 0:
        anomalies = anomalies.join(expected_values, how='inner')
        anomalies[IsPositiveAnomaly], anomalies[IsNegativeAnomaly] = \
            zip(*anomalies.apply(lambda x: modify_anomaly_status(x), axis=1))

    anomalies = anomalies[[IsAnomaly, IsPositiveAnomaly,
                           IsNegativeAnomaly]]

    merged_result = expected_values.join(anomalies, how='left')
    merged_result.fillna(False, inplace=True)

    return merged_result, model_id


def detect_anomaly(data, alpha, ratio, directions, remainder_values, last_value=None):
    sorted_data = data.sort_values(ascending=True)
    sorted_reminder = remainder_values.sort_values(ascending=True)
    num_obs = len(sorted_data)
    max_outliers = min(max(math.ceil(num_obs * ratio), 1), len(data) // 2 - 1)
    majority_value = get_verified_majority_value(sorted_data, num_obs)
    reminder_majority_value = get_verified_majority_value(sorted_reminder, num_obs)

    detectors = [ESD(series=sorted_data, alpha=alpha, max_outliers=max_outliers, majority_value=majority_value),
                 ZScoreDetector(series=sorted_data, max_outliers=max_outliers),
                 ESD(series=sorted_reminder, alpha=alpha, max_outliers=max_outliers,
                     majority_value=reminder_majority_value)]

    model_id = ModelType.AnomalyDetector

    if majority_value is not None or reminder_majority_value is not None:
        model_id = ModelType.AnomalyDetectorMad

    return detect(directions=directions, detectors=detectors, max_outliers=max_outliers, num_obs=num_obs,
                  last_value=last_value), model_id


def modify_anomaly_status(x):
    if not x[IsAnomaly]:
        return False, False
    if x[ExpectedValue] > x[Value]:
        return False, True
    else:
        return True, False
