import numpy as np
import pandas as pd

from univariate.handlers.enum import default_gran_window
from univariate.resource.error_message_v1 import *
from univariate.util import Granularity, get_indices_from_timestamps, DEFAULT_FILL_UP_MODE, FillUpMode, BoundaryVersion
from univariate.util.fields import DEFAULT_MAX_RATIO, DEFAULT_ALPHA, DEFAULT_THRESHOLD, \
    DEFAULT_MARGIN_FACTOR


class RequestParserV1(object):
    def __init__(self, json_request):
        self.points = None
        self.period = None
        self.granularity = None
        self.custom_interval = None
        self.ratio = DEFAULT_MAX_RATIO
        self.alpha = DEFAULT_ALPHA
        self.factor = DEFAULT_MARGIN_FACTOR
        self.threshold = DEFAULT_THRESHOLD
        self.need_detector_id = False
        self.indices = None
        self.fill_up_mode = DEFAULT_FILL_UP_MODE
        self.fixed_value_to_fill = None
        self.need_fill_up_confirm = False
        self.need_boundary_unit = False
        self.boundary_version = BoundaryVersion.V1
        self.error_msg = None
        self.error_code = 'BadArgument'
        self.parse_arg(json_request)
        self.detector = {}

    def parse_arg(self, data):
        if data is None or len(data) == 0 or not isinstance(data, dict):
            self.error_msg = InvalidJsonFormat
            return
        if 'Series' not in data:
            self.error_msg = RequiredSeries
            return
        if 'Granularity' not in data:
            self.error_msg = RequiredGranularity
            return
        if 'CustomInterval' in data:
            self.custom_interval = data['CustomInterval']
            if not isinstance(self.custom_interval, int) or self.custom_interval <= 0:
                self.error_code = 'InvalidCustomInterval'
                self.error_msg = InvalidCustomInterval
                return
        if data['Granularity'] is None or data['Granularity'] not in Granularity.__members__.keys():
            self.error_code = 'InvalidGranularity'
            self.error_msg = InvalidGranularity.format(list(default_gran_window))
            return
        self.granularity = Granularity[data['Granularity']]
        if self.granularity == Granularity.none:
            self.error_code = 'InvalidGranularity'
            self.error_msg = InvalidGranularity.format(list(default_gran_window))
            return

        if 'Period' in data:
            self.period = data['Period']
            if not isinstance(self.period, int) or self.period < 0:
                self.error_code = 'InvalidPeriod'
                self.error_msg = InvalidPeriod
                return

        if "Alpha" in data:
            self.alpha = data["Alpha"]
            if not (isinstance(self.alpha, int) or isinstance(self.alpha, float)) or self.alpha <= 0:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAlpha
                return

        if "MaxAnomalyRatio" in data:
            self.ratio = data["MaxAnomalyRatio"]
            if not isinstance(self.ratio, float) or self.ratio <= 0 or self.ratio > 0.49:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidAnomalyRatio
                return

        if "Sensitivity" in data:
            self.factor = data["Sensitivity"]
            if not isinstance(self.factor, int) or self.factor < 0 or self.factor > 100:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidSensitivity
                return

        if "Threshold" in data:
            self.threshold = data["Threshold"]
            if not isinstance(self.threshold, float):
                self.error_code = "InvalidModelArgument"
                self.error_msg = InvalidThreshold
                return

        if "needDetectorId" in data:
            self.need_detector_id = data['needDetectorId']

        if "fillUpMode" in data and data["fillUpMode"] is not None:
            if data["fillUpMode"] not in [x.value for x in FillUpMode]:
                self.error_code = "InvalidFillUpMode"
                self.error_msg = InvalidFillUpMode.format(['auto', 'previous', 'linear', 'zero', 'fixed', 'notFill'])
                return
            self.fill_up_mode = FillUpMode(data["fillUpMode"])
            if self.fill_up_mode == FillUpMode.fixed:
                if "fixedValue" not in data or data["fixedValue"] is None or \
                        (not (isinstance(data["fixedValue"], int) or isinstance(data["fixedValue"], float))):
                    self.error_code = "InvalidFixedValue"
                    self.error_msg = InvalidFixedValue
                self.fixed_value_to_fill = float(data["fixedValue"])

        if "needFillUpConfirm" in data and data['needFillUpConfirm'] is not None:
            self.need_fill_up_confirm = data['needFillUpConfirm']

        if "needBoundaryUnit" in data and data['needBoundaryUnit'] is not None:
            self.need_boundary_unit = bool(data['needBoundaryUnit'])

        self.points = data['Series']
        self.error_code, self.error_msg = self.points_validator()
        return

    def is_timestamp_ascending(self):
        count = len(self.points)
        if count <= 1:
            return 0

        for i in range(0, count - 1):
            if self.points[i]['timestamp'] > self.points[i + 1]['timestamp']:
                return -1
            elif self.points[i]['timestamp'] == self.points[i + 1]['timestamp']:
                return -2
        return 0

    def points_validator(self):
        try:
            if self.points is None:
                return 'BadArgument', InvalidSeries
            if not isinstance(self.points, list):
                return 'BadArgument', InvalidSeriesType
            if len(self.points) < 12:
                return 'InvalidSeries', NotEnoughPoints
            if len(self.points) > 8640:
                return 'InvalidSeries', TooManyPoints
            timestamps = pd.to_datetime([x['Timestamp'] for x in self.points]).tolist()
            self.points = [{'timestamp': timestamps[i], 'value': float(self.points[i]['Value'])}
                           for i in range(0, len(self.points))]
        except Exception as e:
            return 'BadArgument', InvalidSeriesFormat

        if len([1 for x in self.points if np.isnan(x['value'])]) > 0:
            return 'BadArgument', InvalidSeriesValue
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
                 or (self.fill_up_mode == FillUpMode.no or self.fill_up_mode == FillUpMode.notFill) and len(self.points) < self.period * 2 + 1):
            return 'InvalidModelArgument', InsufficientPoints

        return None, None
