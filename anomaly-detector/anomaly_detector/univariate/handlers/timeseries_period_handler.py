import math

from univariate import InvalidJsonFormat
from univariate.resource.error_message import RequiredSeriesValues, InvalidSeriesValues,\
    InvalidTrendType, InvalidPeriodThresh, InvalidMinVariance, InvalidGranularity, InvalidCustomInterval
from univariate.util import DEFAULT_TREND_TYPE, DEFAULT_PERIOD_THRESH, DEFAULT_MIN_VAR, \
    AnomalyDetectionRequestError, TREND_TYPES, DEFAULT_GRANULARITY, DEFAULT_INTERVAL
from univariate.period import period_detection
from univariate.util import Granularity
from univariate.handlers.enum import default_gran_window


class PeriodDetectRequest(object):
    def __init__(self, request):
        self.seriesValues = None
        self.trendType = DEFAULT_TREND_TYPE
        self.periodThresh = DEFAULT_PERIOD_THRESH
        self.minVariance = DEFAULT_MIN_VAR
        self.granularity = DEFAULT_GRANULARITY
        self.interval = DEFAULT_INTERVAL
        self.error_code = 'BadArgument'
        self.error_msg = None
        self.parse_arg(request)

    def parse_arg(self, data):
        if data is None or len(data) == 0 or not isinstance(data, dict):
            self.error_msg = InvalidJsonFormat
            return
        if 'seriesValues' not in data:
            self.error_msg = RequiredSeriesValues
            return

        if "trendType" in data and data["trendType"] is not None:
            self.trendType = data["trendType"]
            if self.trendType not in TREND_TYPES:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidTrendType.format(TREND_TYPES)
                return

        if "periodThresh" in data and data["periodThresh"] is not None:
            self.periodThresh = data["periodThresh"]
            if not isinstance(self.periodThresh, float) or self.periodThresh < 0.01 \
                    or self.periodThresh > 0.99:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidPeriodThresh
                return

        if "minVariance" in data and data["minVariance"] is not None:
            self.minVariance = data["minVariance"]
            if not isinstance(self.minVariance, float) or self.minVariance < 0.01 \
                    or self.minVariance > 0.99:
                self.error_code = 'InvalidModelArgument'
                self.error_msg = InvalidMinVariance
                return

        if "granularity" in data and data["granularity"] is not None:
            if data['granularity'] not in Granularity.__members__.keys():
                self.error_code = 'InvalidGranularity'
                self.error_msg = InvalidGranularity.format(list(default_gran_window))
                return
            self.granularity = Granularity[data["granularity"]]

        if "customInterval" in data and data["customInterval"] is not None:
            self.interval = data["customInterval"]
            if not isinstance(self.interval, int) or self.interval <= 0:
                self.error_code = 'InvalidCustomInterval'
                self.error_msg = InvalidCustomInterval
                return

        if data['seriesValues'] is None or len(data['seriesValues']) < 12 or len(data['seriesValues']) > 8640:
            self.error_msg = InvalidSeriesValues
            return
        self.seriesValues = data['seriesValues']

        return None, None


class PeriodDetectResponse(object):
    def __init__(self, request, period):
        self.period = period
        self.level = math.ceil(len(request.seriesValues) / 1000)
        self.response = {
            'period': period,
        }


def period_detect(request):
    request = PeriodDetectRequest(request)
    if request.error_msg is not None:
        raise AnomalyDetectionRequestError(error_code=request.error_code, error_msg=request.error_msg)
    period = period_detection(request.seriesValues, request.trendType, request.periodThresh, request.minVariance,
                              granularity=request.granularity, interval=request.interval)
    return PeriodDetectResponse(request, period)
