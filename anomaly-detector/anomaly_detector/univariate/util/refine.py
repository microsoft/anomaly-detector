# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
from anomaly_detector.univariate.util import Value, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, \
    AnomalyScore, Trend, EPS
from anomaly_detector.univariate.util import boundary_utils, ModelType, BoundaryVersion

def refine_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity, upper_margins,
                lower_margins):
    upper_boundaries = expected_values + upper_margins
    lower_boundaries = expected_values - lower_margins

    # tight the boundary
    upper_boundaries = np.clip(upper_boundaries, np.min(upper_boundaries), max(np.max(actual_values), np.max(expected_values)))
    lower_boundaries = np.clip(lower_boundaries, min(np.min(actual_values), np.min(expected_values)), np.max(lower_boundaries))

    upper_margins = upper_boundaries - expected_values
    lower_margins = expected_values - lower_boundaries

    anomaly_refine = np.where(np.logical_and(is_anomaly,
                            np.logical_and(upper_boundaries >= actual_values, actual_values >= lower_boundaries)
                                            ))

    upper_refine = np.where(np.logical_and(actual_values > upper_boundaries,
                                        np.logical_not(is_anomaly)))

    upper_margins[upper_refine] = np.subtract(actual_values[upper_refine], expected_values[upper_refine]) * 1.01
    lower_margins[upper_refine] = upper_margins[upper_refine]

    lower_refine = np.where(np.logical_and(actual_values < lower_boundaries,
                                        np.logical_not(is_anomaly)))

    lower_margins[lower_refine] = np.subtract(expected_values[lower_refine], actual_values[lower_refine]) * 1.01
    upper_margins[lower_refine] = lower_margins[lower_refine]

    if sensitivity == 100:
        upper_margins[anomaly_refine] = 0.0
        lower_margins[anomaly_refine] = 0.0
    else:
        is_anomaly[anomaly_refine] = False
        anomaly_neg[anomaly_refine] = False
        anomaly_pos[anomaly_refine] = False

    severity = [boundary_utils.calculate_severity_v1(av, ev, anomaly) for av, ev, anomaly in zip(actual_values, expected_values, is_anomaly)]

    return expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity


def get_spectral_residual_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
                                anomaly_scores):
    """
    this method generates margin and refine the anomaly detection result based on the anomaly score.
    :param actual_values: actual value of each data point
    :param expected_values: estimated expected value of each data point
    :param anomaly_scores: the anomaly score of each data point
    :param is_anomaly: boolean value indicating each data point is anomaly or not
    :param anomaly_neg: boolean value indicating if each data point is a negative anomaly or not
    :param anomaly_pos: boolean value indicating if each data point is a positive anomaly or not
    :param sensitivity: percentage of anomaly points will be marked anomaly after refinement
    """

    elements_count = len(actual_values)
    margins = np.zeros(elements_count, dtype=np.float64)

    # for normal points, the margin is 1 percent of max_normal_value - min_normal_value
    normal_point_bool_index = np.less_equal(anomaly_scores, EPS)
    normal_values = actual_values[normal_point_bool_index]
    if len(normal_values) > 0:
        min_normal_value, max_normal_value = min(normal_values), max(normal_values)
        normal_margin = (max_normal_value - min_normal_value) * 0.01
        margins = np.ones(elements_count, dtype=np.float64) * normal_margin

    bar = 1 - sensitivity / 100.0

    margins[~normal_point_bool_index] = \
        np.abs(actual_values[~normal_point_bool_index] - expected_values[~normal_point_bool_index]) / \
        anomaly_scores[~normal_point_bool_index] * bar

    return refine_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
                        margins, np.copy(margins))


def get_anomaly_detector_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity):

    upper_margins = abs(expected_values) * (100 - sensitivity) / 100
    lower_margins = np.array(upper_margins)

    return refine_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
                        upper_margins, lower_margins)


def get_margins_v1(results, sensitivity, model_id):
    if model_id == ModelType.SpectralResidual:
        return get_spectral_residual_margins(actual_values=results[Value].values,
                                            expected_values=results[ExpectedValue].values,
                                            is_anomaly=results[IsAnomaly].values,
                                            anomaly_neg=results[IsNegativeAnomaly].values,
                                            anomaly_pos=results[IsPositiveAnomaly].values,
                                            sensitivity=sensitivity, anomaly_scores=results[AnomalyScore].values)
    else:
        return get_anomaly_detector_margins(actual_values=results[Value].values,
                                            expected_values=results[ExpectedValue].values,
                                            is_anomaly=results[IsAnomaly].values,
                                            anomaly_neg=results[IsNegativeAnomaly].values,
                                            anomaly_pos=results[IsPositiveAnomaly].values, sensitivity=sensitivity)


def get_margins_v2(results, sensitivity, last=False):
    values = results[Value].values
    expected_values = results[ExpectedValue].values
    is_anomaly = results[IsAnomaly].values

    boundary_units = boundary_utils.calculate_boundary_units(results[Trend].values, is_anomaly)

    if last:
        # calculate for only the last point
        value, expected_value, anomaly, unit = values[-1], expected_values[-1], is_anomaly[-1], boundary_units[-1]
        anomaly_score = boundary_utils.calculate_anomaly_score(value, expected_value, unit, anomaly)
        severity = boundary_utils.calculate_severity_v2(anomaly_score, anomaly)
        upper_margin, lower_margin = boundary_utils.calculate_margin(unit, sensitivity, value, expected_value, anomaly)
        anomaly_pos = value > expected_value + upper_margin and anomaly
        anomaly_neg = value < expected_value - lower_margin and anomaly
        anomaly = anomaly_pos or anomaly_neg

        return expected_value, upper_margin, lower_margin, bool(anomaly_neg), bool(anomaly_pos), bool(anomaly), severity, \
            unit, anomaly_score
    else:
        # calculate for the entire series
        anomaly_scores = boundary_utils.calculate_anomaly_scores(values, expected_values, boundary_units, is_anomaly)
        boundaries = [boundary_utils.calculate_margin(u, sensitivity, v, ev, a) for u, v, ev, a
                    in zip(boundary_units, values, expected_values, is_anomaly)]
        upper_margins, lower_margins = list(zip(*boundaries))
        upper_boundaries = expected_values + upper_margins
        lower_boundaries = expected_values - lower_margins
        anomaly_pos = np.logical_and(is_anomaly, values > upper_boundaries)
        anomaly_neg = np.logical_and(is_anomaly, values < lower_boundaries)
        is_anomaly = np.logical_or(anomaly_neg, anomaly_pos)
        severity = [boundary_utils.calculate_severity_v2(s, a) for s, a in zip(anomaly_scores, is_anomaly)]

        return expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity, \
            boundary_units, anomaly_scores

def get_margins(results, sensitivity, model_id, boundary_version, last=False):
    if boundary_version == BoundaryVersion.V1:
        expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity \
            = get_margins_v1(results, sensitivity, model_id)
        if last:
            return expected_values[-1], upper_margins[-1], lower_margins[-1], bool(anomaly_neg[-1]), bool(anomaly_pos[-1]), bool(is_anomaly[-1]), severity[-1], None, None
        else:
            return expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity, None, None
    else:
        return get_margins_v2(results, sensitivity, last)
