import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import json
from univariate.resource.error_message import *
from univariate.util import Granularity, get_indices_from_timestamps, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion
from univariate.handlers.enum import default_gran_window
from univariate.util.exceptions import AnomalyDetectionRequestError

from univariate import AnomalyDetectionModel
from univariate.util.fields import DEFAULT_PERIOD, DEFAULT_GRANULARITY_NONE, DEFAULT_MAX_RATIO, DEFAULT_ALPHA, DEFAULT_THRESHOLD, DEFAULT_MARGIN_FACTOR, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion, VALUE_LOWER_BOUND, VALUE_UPPER_BOUND, DetectType

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


if __name__ == "__main__":

    # Read the JSON file
    with open("univariate\json_spectrum_period.entire_0.json", "r") as file:
        json_data = json.load(file)

    # Convert json["request"] to a dataframe
    data = pd.DataFrame(json_data["request"]["series"])
       
#     print(data)
#       timestamp     value
# 0    2018-05-01T00:00:00Z  0.000768
# 1    2018-05-02T00:00:00Z  0.001137
# 2    2018-05-03T00:00:00Z  0.001671
# 3    2018-05-04T00:00:00Z  0.002438
# 4    2018-05-05T00:00:00Z  0.003531
# ..                    ...       ...
# 367  2019-06-04T00:00:00Z  0.089706
# 368  2019-06-05T00:00:00Z  0.066446
# 369  2019-06-06T00:00:00Z  0.048863
# 370  2019-06-07T00:00:00Z  0.035675
# 371  2019-06-08T00:00:00Z  0.025860

# [372 rows x 2 columns]
    params = {
        "detect_mode": "entire",
        "granularity": "daily",
        "needSpectrumPeriod": True
    }

    detector = UnivariateAnomalyDetector()
    result = detector.predict(data, params)
    
    print(result)
    
    # { results: [{'ExpectedValues': 0.0007682854790509452, 'Value': 0.0007682854790278987, 'IsAnomaly': False, 'IsPositiveAnomaly': False, 'IsNegativeAnomaly': False}, {'ExpectedValues': 0.0011371349002278208, 'Value': 0.0011371349002072366, 'IsAnomaly': False, 'IsPositiveAnomaly': False, 'IsNegativeAnomaly': False},... }
    # { period: 101 }
    # { spectrum_period: None }
    # { model_id: ModelType.AnomalyDetector }
    # { do_fill_up: True }