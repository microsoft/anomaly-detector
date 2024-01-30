# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.detectors.spectral_residual import SpectralResidual
from anomaly_detector.univariate.util import average_filter, Value, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, Trend, \
    AnomalyScore, MIN_SR_RAW_SCORE, MAX_SR_RAW_SCORE, SKELETON_POINT_SCORE_THRESHOLD
from anomaly_detector.univariate.util import interp, trend_detection
import numpy as np
import pandas as pd


def spectral_residual_detection(series, threshold, max_anomaly_ratio, need_trend, last_value=None):
    num_obs = len(series)
    max_outliers = max(int(num_obs * max_anomaly_ratio), 1)
    detector = SpectralResidual(series=series, max_outliers=max_outliers, threshold=threshold)
    anomalies, model_id = detector.detect(last_value=last_value)
    merged_result = pd.DataFrame({Value: series})
    merged_result[AnomalyScore] = anomalies[AnomalyScore]

    anomalies_index = anomalies[anomalies[IsAnomaly]].index
    anomalies = anomalies.loc[anomalies_index] if len(anomalies_index) != 0 else \
        pd.DataFrame(columns=[IsAnomaly, IsPositiveAnomaly, IsNegativeAnomaly])

    # calculate expected value for each point not belong to skeleton points
    skeleton_points_bool_index = merged_result[AnomalyScore] <= SKELETON_POINT_SCORE_THRESHOLD
    expected_values = np.copy(series)

    if last_value is not None:
        skeleton_points = merged_result[skeleton_points_bool_index]
        expected_values[-1] = np.mean(skeleton_points[len(skeleton_points)//2:][Value])
    else:
        skeleton_points_partial_cnt = np.cumsum(skeleton_points_bool_index)
        skeleton_points_partial_cnt = np.concatenate(([0], skeleton_points_partial_cnt), axis=0)
        skeleton_points_partial_sum = np.cumsum(np.multiply(merged_result[Value].values, skeleton_points_bool_index))
        skeleton_points_partial_sum = np.concatenate(([0], skeleton_points_partial_sum), axis=0)
        for i, v in enumerate(skeleton_points_bool_index):
            if v is not True:
                skeleton_point_cnt = skeleton_points_partial_cnt[i+1] - skeleton_points_partial_cnt[i//2]
                if skeleton_point_cnt == 0:
                    expected_values[i] = np.mean(series[:i+1])
                else:
                    expected_values[i] = (skeleton_points_partial_sum[i+1] - skeleton_points_partial_sum[i//2]) / \
                                        skeleton_point_cnt

        expected_values = average_filter(expected_values, 5)

    merged_result[ExpectedValue] = expected_values

    if need_trend:
        merged_result[Trend] = trend_detection(expected_values)

    # normailize the anomaly score to be in region [0,1] and 0 represents normal points
    merged_result[AnomalyScore] = np.clip(merged_result[AnomalyScore].values - MIN_SR_RAW_SCORE /
                                          (MAX_SR_RAW_SCORE - MIN_SR_RAW_SCORE), 0.0, 1.0)

    if len(anomalies) != 0:
        anomalies = anomalies.sort_values(by=AnomalyScore, ascending=False)
        anomalies = anomalies[[IsAnomaly]].head(min(max_outliers, len(anomalies)))
        anomalies = anomalies.join(merged_result, how='inner')
        anomalies[IsPositiveAnomaly], anomalies[IsNegativeAnomaly] = \
            zip(*anomalies.apply(lambda x: modify_anomaly_status(x), axis=1))

    anomalies = anomalies[[IsAnomaly, IsPositiveAnomaly, IsNegativeAnomaly]]

    merged_result = merged_result.join(anomalies, how='left')
    merged_result.fillna(False, inplace=True)

    return merged_result, model_id


def modify_anomaly_status(x):
    if not x[IsAnomaly]:
        return False, False
    if x[ExpectedValue] > x[Value]:
        return False, True
    else:
        return True, False
