import numpy as np
import pandas as pd

from univariate.handlers.enum import default_gran_window
from univariate.resource.error_message import *
from univariate.util import Granularity, get_indices_from_timestamps, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion
from univariate.util.fields import DEFAULT_GRANULARITY_NONE, DEFAULT_MAX_RATIO, DEFAULT_ALPHA, DEFAULT_THRESHOLD, \
    DEFAULT_MARGIN_FACTOR, VALUE_LOWER_BOUND, VALUE_UPPER_BOUND


def RequestParser_VX(params):
    
    params_copy = params.copy()
    del params_copy['series']
    return params_copy

class RequestParser(object):
    def __init__(self, json_request):
        self.series = None
        self.period = None
        self.granularity = DEFAULT_GRANULARITY_NONE
        self.custom_interval = None
        self.ratio = DEFAULT_MAX_RATIO
        self.alpha = DEFAULT_ALPHA
        self.sensitivity = DEFAULT_MARGIN_FACTOR
        self.threshold = DEFAULT_THRESHOLD
        self.need_detector_id = False
        self.indices = None
        self.fill_up_mode = DEFAULT_FILL_UP_MODE
        self.fixed_value_to_fill = None
        self.need_fill_up_confirm = False
        self.need_spectrum_period = False
        self.boundary_version = BoundaryVersion.V1
        self.detector = dict()
        self.error_msg = None
        self.error_code = 'BadArgument'
        self.parse_arg(json_request)

    def parse_arg(self, data):
        # check the whole request body
        if data is None or len(data) == 0 or not isinstance(data, dict):
            self.error_msg = InvalidJsonFormat
            return

        # check if 'series' field is present.
        if 'series' not in data:
            self.error_msg = RequiredSeries
            return

        # check granularity filed.
        # If it is not present or none, then customInterval, 'timestamp' in series will be ignored.
        # And fil up strategy is set 'nofill'.
        if 'granularity' in data:
            if data['granularity'] not in Granularity.__members__.keys():
                self.error_code = 'InvalidGranularity'
                self.error_msg = InvalidGranularity.format(list(default_gran_window))
                return
            self.granularity = Granularity[data['granularity']]

        ### check for optional parameters

        if 'customInterval' in data:
            self.custom_interval = data['customInterval']
            if not isinstance(self.custom_interval, int) or self.custom_interval <= 0:
                self.error_code = 'InvalidCustomInterval'
                self.error_msg = InvalidCustomInterval
                return

        if 'period' in data:
            self.period = data['period']
            if not isinstance(self.period, int) or self.period < 0:
                self.error_code = 'InvalidPeriod'
                self.error_msg = InvalidPeriod
                return

        # 'alpha' is a parameter for dynamic threshold and seasonal series.
        if "alpha" in data:
            self.alpha = data["alpha"]
            if not (isinstance(self.alpha, int) or isinstance(self.alpha, float)) or self.alpha <= 0:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAlpha
                return

        if "maxAnomalyRatio" in data:
            self.ratio = data["maxAnomalyRatio"]
            if not isinstance(self.ratio, float) or self.ratio <= 0 or self.ratio > 0.49:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAnomalyRatio
                return

        if "sensitivity" in data:
            self.sensitivity = data["sensitivity"]
            if not isinstance(self.sensitivity, int) or self.sensitivity < 0 or self.sensitivity > 100:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidSensitivity
                return

        if "threshold" in data:
            self.threshold = data["threshold"]
            if not isinstance(self.threshold, float):
                self.error_code = "InvalidModelArgument"
                self.error_msg = InvalidThreshold
                return

        if "needDetectorId" in data:
            self.need_detector_id = data['needDetectorId']

        if "imputeMode" in data and data["imputeMode"] is not None:
            if data["imputeMode"] not in [x.value for x in FillUpMode]:
                self.error_code = "InvalidImputeMode"
                self.error_msg = InvalidImputeMode.format(['auto', 'previous', 'linear', 'zero', 'fixed', 'notFill'])
                return

            self.fill_up_mode = FillUpMode(data["imputeMode"])

            if self.fill_up_mode == FillUpMode.fixed:
                if "imputeFixedValue" not in data \
                    or ((not isinstance(data["imputeFixedValue"], int))
                        and (not isinstance(data["imputeFixedValue"], float))):
                    self.error_code = "InvalidImputeFixedValue"
                    self.error_msg = InvalidImputeValue
                    return

                self.fixed_value_to_fill = float(data["imputeFixedValue"])
            elif self.fill_up_mode == FillUpMode.zero:
                self.fill_up_mode = FillUpMode.fixed
                self.fixed_value_to_fill = 0

        # To be compatible with old contract
        elif "fillUpMode" in data and data["fillUpMode"] is not None:
            if data["fillUpMode"] not in [x.value for x in FillUpMode]:
                self.error_code = "InvalidFillUpMode"
                self.error_msg = InvalidImputeMode.format([x.value for x in FillUpMode])
                return

            self.fill_up_mode = FillUpMode(data["fillUpMode"])
            if self.fill_up_mode == FillUpMode.fixed:
                if "fixedValue" not in data or data["fixedValue"] is None or \
                        (not (isinstance(data["fixedValue"], int) or isinstance(data["fixedValue"], float))):
                    self.error_code = "InvalidFixedValue"
                    self.error_msg = InvalidImputeValue
                    return

                self.fixed_value_to_fill = float(data["fixedValue"])

        # flag for internal use
        if "needFillUpConfirm" in data and data['needFillUpConfirm'] is not None:
            self.need_fill_up_confirm = data['needFillUpConfirm']

        # flag for internal use
        if "needSpectrumPeriod" in data and data["needSpectrumPeriod"] is not None:
            self.need_spectrum_period = data["needSpectrumPeriod"]

        # flag for interval use
        if "boundaryVersion" in data and data['boundaryVersion'] is not None:
            if data['boundaryVersion'] not in BoundaryVersion.__members__.keys():
                self.error_code = "InvalidBoundaryVersion"
                self.error_msg = InvalidBoundaryVersion.format(BoundaryVersion.__members__.keys())
                return

            self.boundary_version = BoundaryVersion[data['boundaryVersion']]

        if "detector" in data and isinstance(data['detector'], dict):
            # validation parameters

            if "parameters" not in data['detector']:
                self.error_code = 'MissingDetectorParameters'
                self.error_msg = MissingDetectorParameters
                return

            if not isinstance(data['detector']['parameters'], dict):
                self.error_code = 'InvalidDetectorParameters'
                self.error_msg = InvalidDetectorParameters
                return

            if "name" not in data['detector']:
                self.error_code = 'MissingDetectorName'
                self.error_msg = MissingDetectorName
                return

            if data['detector']['name'].lower() not in ['spectral_residual', 'hbos', 'seasonal_series', 'dynamic_threshold']:
                self.error_code = 'InvalidDetector'
                self.error_msg = InvalidDetectorName
                return

            self.detector = data['detector']

        self.series = data['series']
        self.error_code, self.error_msg = self.series_validator()
        return

    def is_timestamp_ascending(self):
        count = len(self.series)
        if count <= 1:
            return 0

        for i in range(0, count - 1):
            if self.series[i]['timestamp'] > self.series[i + 1]['timestamp']:
                return -1
            elif self.series[i]['timestamp'] == self.series[i + 1]['timestamp']:
                return -2
        return 0

    def series_validator(self):
        try:
            if self.series is None:
                return 'BadArgument', InvalidSeries
            if not isinstance(self.series, list):
                return 'BadArgument', InvalidSeriesType
            if len(self.series) < 12:
                return 'InvalidSeries', NotEnoughPoints.format(12)
            if len(self.series) > 8640:
                return 'InvalidSeries', TooManyPoints

            values = [float(x['value']) for x in self.series]
            if self.granularity != Granularity.none:
                timestamps = pd.to_datetime([x['timestamp'] for x in self.series]).tolist()
                self.series = [{'timestamp': t, 'value': v} for t, v in zip(timestamps, values)]
            else:
                self.series = [{'value': v} for v in values]
        except Exception:
            return 'BadArgument', InvalidSeriesFormat

        if np.any(np.less(values, VALUE_LOWER_BOUND)) or np.any(np.greater(values, VALUE_UPPER_BOUND)):
            return 'InvalidSeries', ValueOverflow

        if len([1 for x in self.series if np.isnan(x['value'])]) > 0:
            return 'BadArgument', InvalidSeriesValue

        # check the order of timestamps
        if self.granularity != Granularity.none:
            ascending = self.is_timestamp_ascending()
            if ascending == -1:
                return 'InvalidSeries', InvalidSeriesOrder
            if ascending == -2:
                return 'InvalidSeries', DuplicateSeriesTimestamp

            self.indices, first_invalid_index = get_indices_from_timestamps(self.granularity, self.custom_interval,
                                                                            timestamps)
            if first_invalid_index is not None:
                return 'InvalidSeries', InvalidSeriesTimestamp.format(first_invalid_index, self.granularity.name,
                                                                      1 if self.custom_interval is None else self.custom_interval)

            if self.period is not None and \
                    (self.indices[-1] + 1 < self.period * 2 + 1
                     or (self.fill_up_mode == FillUpMode.no or self.fill_up_mode == FillUpMode.notFill) and len(self.series) < self.period * 2 + 1):
                return 'InvalidModelArgument', InsufficientPoints
        else:
            self.indices = [i for i in range(len(values))]

        return None, None
