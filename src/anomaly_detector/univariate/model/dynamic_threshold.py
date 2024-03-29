# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import math

from anomaly_detector.univariate.detectors import ESD
from anomaly_detector.univariate.detectors import ZScoreDetector
from anomaly_detector.univariate.model.detect import detect
from anomaly_detector.univariate.util import Value, Direction, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, Trend, \
    ModelType, get_verified_majority_value, normalize
from anomaly_detector.univariate.util import interp, trend_detection


def dynamic_threshold_detection(series, trend_values, alpha, max_anomaly_ratio, need_trend, last_value=None):
    directions = [Direction.upper_tail, Direction.lower_tail]
    _series = pd.Series(series)
    data = pd.Series(normalize(_series))
    anomalies, model_id = detect_anomaly(data=data, alpha=alpha, ratio=max_anomaly_ratio,
                                         directions=directions, last_value=last_value)
    if len(anomalies) != 0:
        de_anomaly_series = pd.Series(series)
        de_anomaly_series[anomalies.index] = np.nan
        trend_values = trend_detection(interp(de_anomaly_series))

    expected_values = _series.to_frame(name=Value)
    expected_values[ExpectedValue] = trend_values

    if need_trend:
        expected_values[Trend] = trend_values

    if len(anomalies) != 0:
        anomalies = anomalies.join(expected_values, how='inner')
        anomalies[IsPositiveAnomaly], anomalies[IsNegativeAnomaly] = \
            zip(*anomalies.apply(lambda x: modify_anomaly_status(x), axis=1))

    anomalies = anomalies[[IsAnomaly, IsPositiveAnomaly, IsNegativeAnomaly]]

    merged_result = expected_values.join(anomalies, how='left')
    merged_result.fillna(False, inplace=True)

    return merged_result, model_id


def detect_anomaly(data, alpha, ratio, directions, last_value=None):
    sorted_data = data.sort_values(ascending=True)
    num_obs = len(data)
    max_outliers = min(max(math.ceil(num_obs * ratio), 1), len(data) // 2 - 1)
    majority_value = get_verified_majority_value(sorted_data, num_obs)
    detectors = [ESD(series=sorted_data,  alpha=alpha, max_outliers=max_outliers, majority_value=majority_value),
                 ZScoreDetector(series=sorted_data, max_outliers=max_outliers)]

    model_id = ModelType.DynamicThreshold
    if majority_value is not None:
        model_id = ModelType.DynamicThresholdMad

    return detect(directions=directions, detectors=detectors, max_outliers=max_outliers, num_obs=num_obs,
                  last_value=last_value), model_id


def modify_anomaly_status(x):
    if not x[IsAnomaly]:
        return False, False
    if x[ExpectedValue] > x[Value]:
        return False, True
    else:
        return True, False
