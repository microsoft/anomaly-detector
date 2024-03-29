# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from enum import Enum

IsAnomaly = "is_anomaly"
IsNegativeAnomaly = "is_negative_anomaly"
IsPositiveAnomaly = "is_positive_anomaly"
Period = "period"
ExpectedValue = "expected_value"
ESDRank = 'ESDRank'
StandardFilterRank = 'DeviationRank'
Reminder = 'Reminder'
Value = 'Value'
AnomalyId = "Id"
AnomalyScore = "Score"
Trend = "Trend"
Severity = 'Severity'
BoundaryUnit = 'BoundaryUnit'
UpperMargin = 'upper_margin'
LowerMargin = 'lower_margin'
SuggestedWindow = 'suggested_window'

DEFAULT_TREND_TYPE = "spline"
DEFAULT_PERIOD_THRESH = 0.9
DEFAULT_MIN_VAR = 0.20
DEFAULT_MAX_RATIO = 0.25
DEFAULT_ALPHA = 0.05
DEFAULT_MARGIN_FACTOR = 99
DEFAULT_NORMAL_MARGIN_FACTOR = 99
DEFAULT_MAXIMUM_FILLUP_LENGTH = 8640 * 2
DEFAULT_MAX_RATIO = 0.25
DEFAULT_THRESHOLD = 3.5

VALUE_LOWER_BOUND = -1.0e100
VALUE_UPPER_BOUND = 1.0e100

EPS = 1e-8

SKELETON_POINT_SCORE_THRESHOLD = 1.0
MIN_SR_RAW_SCORE = 3.5
MAX_SR_RAW_SCORE = 15.0


class Direction(Enum):
    upper_tail = 1
    lower_tail = 2


class ModelType(Enum):
    Unknown = 0
    AnomalyDetector = 10
    SpectralResidual = 20
    SpectralResidual_ZScore = 21
    Predict = 30
    DynamicThreshold = 40
    AnomalyDetectorMad = 11
    DynamicThresholdMad = 41
    PeriodDetection = 50
    ChangePointDetection = 60
    HbosNonseasonal = 70
    HbosSeasonal = 71
    PatternDetection = 80

class DetectType(Enum):
    ENTIRE = 0
    LATEST = 1

class Granularity(Enum):
    none = -1
    yearly = 0
    monthly = 1
    weekly = 2
    daily = 3
    hourly = 4
    minutely = 5
    secondly = 6
    microsecond = 7


DEFAULT_GRANULARITY_NONE = Granularity.none


class BoundaryVersion(Enum):
    V1 = 1
    V2 = 2
    V3 = 3


class FillUpMode(Enum):
    # these are to be compatible with current implementation
    # TODO: remove them
    use_last = 'last'
    no = 'no'

    auto = 'auto'
    previous = 'previous'
    linear = 'linear'
    zero = 'zero'
    fixed = 'fixed'
    notFill = 'notFill'


DEFAULT_FILL_UP_MODE = FillUpMode.auto

YEAR_SECOND = 12 * 4 * 7 * 24 * 60 * 60
MONTH_SECOND = 4 * 7 * 24 * 60 * 60
WEEK_SECOND = 7 * 24 * 60 * 60
DAY_SECOND = 24 * 60 * 60
HOUR_SECOND = 60 * 60
MINUTE_SECOND = 60
SECOND = 1
MICRO_SECOND = 0.001

DEFAULT_CHANGEPOINT_THRESHOLD = {
    "yearly": 0.9,
    "monthly": 0.9,
    "weekly": 0.9,
    "daily": 0.9,
    "hourly": 0.91,
    "minutely": 0.9,
    "secondly": 0.9,
    "microsecond": 0.9,
    "none": 0.9
}

DEFAULT_CHANGEPOINT_WINDOW = {
    "yearly": 3,
    "monthly": 6,
    "weekly": 3,
    "daily": 5,
    "hourly": 24,
    "minutely": 72,
    "secondly": 72,
    "microsecond": 100,
    "none": 100
}

DEFAULT_MIN_WINDOW = 3
DEFAULT_PERIOD = None

TREND_TYPES = ["median", "mean", "line", "spline"]

DEFAULT_DETECTOR_TYPE = "correlogram"
# DEFAULT_DETECTOR_TYPE = "periodogram"
DEFAULT_GRANULARITY = Granularity.minutely
DEFAULT_INTERVAL = 1
MIN_PERIOD = 4
