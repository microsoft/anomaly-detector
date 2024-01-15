import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import json
from univariate.resource.error_message import *
from univariate.util import Granularity, get_indices_from_timestamps, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion
from univariate.util.enum import default_gran_window
from common.exception import AnomalyDetectionRequestError

from univariate import AnomalyDetectionModel
from univariate.util.fields import DEFAULT_PERIOD, DEFAULT_GRANULARITY_NONE, DEFAULT_MAX_RATIO, DEFAULT_ALPHA, DEFAULT_THRESHOLD, DEFAULT_MARGIN_FACTOR, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion, VALUE_LOWER_BOUND, VALUE_UPPER_BOUND, DetectType
from univariate.util import boundary_utils, BoundaryVersion
from univariate.util import ModelType, Value, ExpectedValue, IsAnomaly, IsNegativeAnomaly, IsPositiveAnomaly, \
    AnomalyScore, Trend, EPS

class UnivariateAnomalyDetector:
    def __init__(self):

        self.error_msg = None
        self.error_code = 'BadArgument'

    def is_timestamp_ascending(self, series):
        count = len(series)
        if count <= 1:
            return 0

        for i in range(0, count - 1):
            if series[i]['timestamp'] > series[i + 1]['timestamp']:
                return -1
            elif series[i]['timestamp'] == series[i + 1]['timestamp']:
                return -2
        return 0
    
    def series_validator(self, series, params):
        try:
            if series is None:
                return 'BadArgument', InvalidSeries
            if not isinstance(series, list):
                return 'BadArgument', InvalidSeriesType
            if len(series) < 12:
                return 'InvalidSeries', NotEnoughPoints.format(12)
            if len(series) > 8640:
                return 'InvalidSeries', TooManyPoints

            values = [float(x['value']) for x in series]
            if params.get("granularity", DEFAULT_GRANULARITY_NONE) != Granularity.none:
                timestamps = pd.to_datetime([x['timestamp'] for x in series]).tolist()
                series = [{'timestamp': t, 'value': v} for t, v in zip(timestamps, values)]
            else:
                series = [{'value': v} for v in values]
        except Exception:
            return 'BadArgument', InvalidSeriesFormat

        if np.any(np.less(values, VALUE_LOWER_BOUND)) or np.any(np.greater(values, VALUE_UPPER_BOUND)):
            return 'InvalidSeries', ValueOverflow

        if len([1 for x in series if np.isnan(x['value'])]) > 0:
            return 'BadArgument', InvalidSeriesValue

        # check the order of timestamps
        if params.get("granularity", DEFAULT_GRANULARITY_NONE) != Granularity.none:
            ascending = self.is_timestamp_ascending(series)
            if ascending == -1:
                return 'InvalidSeries', InvalidSeriesOrder
            if ascending == -2:
                return 'InvalidSeries', DuplicateSeriesTimestamp

            params["indices"], first_invalid_index = get_indices_from_timestamps(params.get("granularity", DEFAULT_GRANULARITY_NONE), params.get("customInterval", None),
                                                                            timestamps)
            if first_invalid_index is not None:
                return 'InvalidSeries', InvalidSeriesTimestamp.format(first_invalid_index, params['granularity'].name,
                                                                      1 if params.get("customInterval", None) is None else params.get("customInterval", None))

            if params.get("period",None) is not None and \
                    (params.get("indices",None)[-1] + 1 < params.get("period",None) * 2 + 1
                     or (params.get("fillUpMode",DEFAULT_FILL_UP_MODE) == FillUpMode.no or params.get("fillUpMode",DEFAULT_FILL_UP_MODE) == FillUpMode.notFill) and len(series) < params.get("period",None) * 2 + 1):
                return 'InvalidModelArgument', InsufficientPoints
        else:
            params["indices"] = [i for i in range(len(values))]

        return None, None

    def parse_arg(self, data, params):

        model_params = {}
        # check the whole request body
        if data is None or data.empty or not isinstance(data, pd.DataFrame):
            self.error_msg = InvalidInputFormat
            return
        
        # print(data)
        # check granularity filed.
        # If it is not present or none, then customInterval, 'timestamp' in series will be ignored.
        # And fil up strategy is set 'nofill'.
        if 'granularity' in params:
            if params['granularity'] not in Granularity.__members__.keys():
                self.error_code = 'InvalidGranularity'
                self.error_msg = InvalidGranularity.format(list(default_gran_window))
                return
            model_params['granularity'] = params['granularity'] = Granularity[params['granularity']]
        else:
            params['granularity'] = DEFAULT_GRANULARITY_NONE

        ### check for optional parameters

        if 'customInterval' in params:
            model_params['interval'] = params['customInterval']
            if not isinstance(params["customInterval"], int) or params["customInterval"] <= 0:
                self.error_code = 'InvalidCustomInterval'
                self.error_msg = InvalidCustomInterval
                return
        else:
            model_params['interval'] = params['customInterval']= None

        if 'period' in params:
            if not isinstance(params["period"], int) or params["period"] < 0:
                self.error_code = 'InvalidPeriod'
                self.error_msg = InvalidPeriod
                return

        # 'alpha' is a parameter for dynamic threshold and seasonal series.
        if "alpha" in params:
            model_params["alpha"] = params["alpha"]
            if not (isinstance(params["alpha"], int) or isinstance(params["alpha"], float)) or params["alpha"] <= 0:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAlpha
                return

        if "maxAnomalyRatio" in params:
            model_params["max_anomaly_ratio"] = params["maxAnomalyRatio"]
            if not isinstance(params["maxAnomalyRatio"], float) or params["maxAnomalyRatio"] <= 0 or params["maxAnomalyRatio"] > 0.49:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAnomalyRatio
                return

        if "sensitivity" in params:
            if not isinstance(params["sensitivity"], int) or params["sensitivity"] < 0 or params["sensitivity"] > 100:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidSensitivity
                return
        else:
            params["sensitivity"] = DEFAULT_MARGIN_FACTOR

        if "threshold" in params:
            model_params["threshold"] = params["threshold"]
            if not isinstance(params["threshold"], float):
                self.error_code = "InvalidModelArgument"
                self.error_msg = InvalidThreshold
                return

        params["needDetectorId"] = params.get("needDetectorId", False)
        
        if "imputeMode" in params and params["imputeMode"] is not None:
            if params["imputeMode"] not in [x.value for x in FillUpMode]:
                self.error_code = "InvalidImputeMode"
                self.error_msg = InvalidImputeMode.format([x.value for x in FillUpMode])
                return
            
            model_params["fill_up_mode"] = FillUpMode(params["imputeMode"])

            if model_params["fill_up_mode"] == FillUpMode.fixed:
                if "imputeFixedValue" not in params \
                    or ((not isinstance(params["imputeFixedValue"], int))
                        and (not isinstance(params["imputeFixedValue"], float))):
                    self.error_code = "InvalidImputeFixedValue"
                    self.error_msg = InvalidImputeValue
                    return

            elif model_params["fill_up_mode"] == FillUpMode.zero:
                model_params["fill_up_mode"] = FillUpMode.fixed
                model_params["fixed_value_to_fill"] = 0

        # To be compatible with old contract
        elif "fillUpMode" in params and params["fillUpMode"] is not None:
            if params["fillUpMode"] not in [x.value for x in FillUpMode]:
                self.error_code = "InvalidFillUpMode"
                self.error_msg = InvalidImputeMode.format([x.value for x in FillUpMode])
                return

            model_params["fill_up_mode"] = FillUpMode(params["fillUpMode"])

            if model_params["fill_up_mode"] == FillUpMode.fixed:
                if "fixedValue" not in params or params["fixedValue"] is None or \
                        (not (isinstance(params["fixedValue"], int) or isinstance(params["fixedValue"], float))):
                    self.error_code = "InvalidFixedValue"
                    self.error_msg = InvalidImputeValue
                    return

                model_params["fixed_value_to_fill"] = float(params["fixedValue"])

        params["needFillUpConfirm"] = params.get("needFillUpConfirm", False)
        model_params['need_spectrum_period'] = params["needSpectrumPeriod"] = params.get("needSpectrumPeriod", False)
        
        if "boundaryVersion" in params and params['boundaryVersion'] is not None:
            if params['boundaryVersion'] not in BoundaryVersion.__members__.keys():
                self.error_code = "InvalidBoundaryVersion"
                self.error_msg = InvalidBoundaryVersion.format(BoundaryVersion.__members__.keys())
                return

            params['boundaryVersion'] = BoundaryVersion[params['boundaryVersion']]
        else:
            params['boundaryVersion'] = BoundaryVersion.V1

        model_params['need_trend'] = params['need_trend'] = params['boundaryVersion'] != BoundaryVersion.V1

        if "detector" in params and isinstance(params['detector'], dict):
            # validation parameters

            if "parameters" not in params['detector']:
                self.error_code = 'MissingDetectorParameters'
                self.error_msg = MissingDetectorParameters
                return

            if not isinstance(params['detector']['parameters'], dict):
                self.error_code = 'InvalidDetectorParameters'
                self.error_msg = InvalidDetectorParameters
                return

            if "name" not in params['detector']:
                self.error_code = 'MissingDetectorName'
                self.error_msg = MissingDetectorName
                return

            if params['detector']['name'].lower() not in ['spectral_residual', 'hbos', 'seasonal_series', 'dynamic_threshold']:
                self.error_code = 'InvalidDetector'
                self.error_msg = InvalidDetectorName
                return
            
            model_params['detector'] = params['detector']

        # convert to list of dict
        data = data.to_dict(orient='records')

        self.error_code, self.error_msg = self.series_validator(data, params)
        
        model_params['indices'] = params['indices']
        return data, model_params

        
    def refine_margins(self, actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity, upper_margins,
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


    def get_spectral_residual_margins(self, actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
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

        return self.refine_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
                            margins, np.copy(margins))


    def get_anomaly_detector_margins(self, actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity):

        upper_margins = abs(expected_values) * (100 - sensitivity) / 100
        lower_margins = np.array(upper_margins)

        return self.refine_margins(actual_values, expected_values, is_anomaly, anomaly_neg, anomaly_pos, sensitivity,
                            upper_margins, lower_margins)


    def get_margins_v1(self, results, sensitivity, model_id):
        if model_id == ModelType.SpectralResidual:
            return self.get_spectral_residual_margins(actual_values=results[Value].values,
                                                expected_values=results[ExpectedValue].values,
                                                is_anomaly=results[IsAnomaly].values,
                                                anomaly_neg=results[IsNegativeAnomaly].values,
                                                anomaly_pos=results[IsPositiveAnomaly].values,
                                                sensitivity=sensitivity, anomaly_scores=results[AnomalyScore].values)
        else:
            return self.get_anomaly_detector_margins(actual_values=results[Value].values,
                                                expected_values=results[ExpectedValue].values,
                                                is_anomaly=results[IsAnomaly].values,
                                                anomaly_neg=results[IsNegativeAnomaly].values,
                                                anomaly_pos=results[IsPositiveAnomaly].values, sensitivity=sensitivity)


    def get_margins_v2(self, results, sensitivity, last=False):
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

    def get_margins(self, results, sensitivity, model_id, boundary_version, last=False):
        if boundary_version == BoundaryVersion.V1:
            expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity \
                = self.get_margins_v1(results, sensitivity, model_id)
            if last:
                return expected_values[-1], upper_margins[-1], lower_margins[-1], bool(anomaly_neg[-1]), bool(anomaly_pos[-1]), bool(is_anomaly[-1]), severity[-1], None, None
            else:
                return expected_values, upper_margins, lower_margins, anomaly_neg, anomaly_pos, is_anomaly, severity, None, None
        else:
            return self.get_margins_v2(results, sensitivity, last)
    
    def predict(self, data, params):

        data, model_params = self.parse_arg(data, params)
        
        if self.error_msg is not None:
            raise AnomalyDetectionRequestError(error_code=self.error_code, error_msg=self.error_msg)

        detector = AnomalyDetectionModel(series=data, **model_params)
        
        results, period, spectrum_period, model_id, do_fill_up = detector.detect(
            period=params.get('period',DEFAULT_PERIOD), last_value=data[-1] if params['detect_mode'] == DetectType.LATEST else None)

        return [
            {"params": params},
            {"results": results},
            {"period": period},
            {"spectrum_period": spectrum_period},
            {"model_id": model_id},
            {"do_fill_up": do_fill_up}
        ]

