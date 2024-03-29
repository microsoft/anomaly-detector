# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from anomaly_detector.univariate.util import stl, stl_adjust_trend, trend_detection, interp, de_outlier_stl, normalize, MAPE_LB, MAPE_UB
from anomaly_detector.univariate.util import Value, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, AnomalyScore, \
    ModelType, Trend


def hbos_detection_seasonal(series, period, outlier_fraction, threshold, adjust_trend=False, need_trend=False, last_value=None):
    if outlier_fraction > 0.49:
        length = len(series)
        raise ValueError(
            ("max_anomaly_ratio must be less than 50% of "
             "the data points (max_anomaly_ratio =%f data_points =%s).")
            % (round(outlier_fraction * length, 0), length))

    num_obs = len(series)

    clamp = (1 / float(num_obs))
    if outlier_fraction < clamp:
        outlier_fraction = clamp

    if num_obs < period * 2 + 1:
        raise ValueError("Anomaly detection needs at least 2 periods worth of data")

    input_series = series.copy()
    stl_func = stl_adjust_trend if adjust_trend else stl
    decompose_result = de_outlier_stl(input_series, stl_func=stl_func, period=period, log_transform=False)

    mape = np.mean(np.abs(decompose_result['remainder'] / input_series))

    if mape > MAPE_UB:
        decompose_result_log = de_outlier_stl(input_series, stl_func=stl_func, period=period,
                                              log_transform=True)
        mape_log = np.mean(np.abs(decompose_result_log['remainder'] / input_series))
        if mape_log < MAPE_LB:
            decompose_result = decompose_result_log

    decompose_trend = decompose_result['trend'].values
    decompose_season = decompose_result['seasonal'].values

    values_to_detect = normalize(series - decompose_trend - decompose_season).reshape(-1, 1)

    model = HBOS(contamination=outlier_fraction)
    model.fit(values_to_detect)
    scores = model.predict_proba(values_to_detect)[:, 1]
    isAnomaly = scores > threshold

    if np.any(isAnomaly) and np.sum(isAnomaly) < len(series):
        decompose_trend[isAnomaly] = np.nan
        decompose_trend = interp(decompose_trend)

    p = {
        ExpectedValue: decompose_trend + decompose_season,
        Value: series,
        IsAnomaly: isAnomaly
    }

    if need_trend:
        p[Trend] = decompose_trend

    results = pd.DataFrame(p)
    if np.any(isAnomaly):
        results[IsPositiveAnomaly], results[IsNegativeAnomaly] = \
            zip(*results.apply(lambda x: modify_anomaly_status(x), axis=1))

    results.fillna(False, inplace=True)

    return results, ModelType.HbosSeasonal


def hbos_detection_nonseasonal(series, threshold, outlier_fraction, need_trend=False, last_value=None):
    series = np.asarray(series)
    num_obs = len(series)

    clamp = (1 / float(num_obs))
    if outlier_fraction < clamp:
        outlier_fraction = clamp

    values = series.reshape(-1, 1)
    model = HBOS(contamination=outlier_fraction)
    model.fit(values)
    model.predict_proba(values)
    scores = model.predict_proba(values)[:, 1]
    isAnomaly = scores > threshold

    result = pd.DataFrame({
        Value: series,
        AnomalyScore: scores,
        IsAnomaly: isAnomaly
    })

    if np.any(isAnomaly) and np.sum(isAnomaly) < np.sum(isAnomaly):
        de_anomaly_series = pd.Series(series)
        de_anomaly_series[de_anomaly_series[isAnomaly]] = np.nan
        trend_values = trend_detection(interp(de_anomaly_series))
    else:
        trend_values = trend_detection(series)

    result[ExpectedValue] = trend_values

    if need_trend:
        result[Trend] = trend_values

    if np.any(isAnomaly):
        result[IsPositiveAnomaly], result[IsNegativeAnomaly] = \
            zip(*result.apply(lambda x: modify_anomaly_status(x), axis=1))

    result.fillna(False, inplace=True)

    return result, ModelType.HbosNonseasonal


def hbos_detection(series, period, threshold, outlier_fraction, need_trend=False, last_value=None):
    if period > 0:
        adjust_trend = True if last_value is not None else False
        return hbos_detection_seasonal(series, period,
                                       outlier_fraction=outlier_fraction, threshold=threshold, adjust_trend=adjust_trend,
                                       need_trend=need_trend, last_value=last_value)
    else:
        return hbos_detection_nonseasonal(series, threshold=threshold, outlier_fraction=outlier_fraction,
                                          need_trend=need_trend, last_value=last_value)


def modify_anomaly_status(x):
    if not x[IsAnomaly]:
        return False, False
    if x[ExpectedValue] > x[Value]:
        return False, True
    else:
        return True, False
