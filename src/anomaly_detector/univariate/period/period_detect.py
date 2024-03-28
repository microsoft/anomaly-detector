# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.util import DEFAULT_TREND_TYPE, DEFAULT_DETECTOR_TYPE, \
    DEFAULT_GRANULARITY, DEFAULT_PERIOD_THRESH, DEFAULT_MIN_VAR, DEFAULT_INTERVAL
from anomaly_detector.univariate.period import SimpleDetector, SpectrumDetector


def period_detection(series, trend_type=DEFAULT_TREND_TYPE, thresh=DEFAULT_PERIOD_THRESH,
                     min_var=DEFAULT_MIN_VAR, detector_type=DEFAULT_DETECTOR_TYPE,
                     granularity=DEFAULT_GRANULARITY, interval=DEFAULT_INTERVAL, skip_simple_detector = False, return_period_source = False):
    if not skip_simple_detector:
        period = SimpleDetector.detect(series, granularity, interval)
        if period:
            return [period, 0] if return_period_source else period
    specturm_period = SpectrumDetector.detect(series, trend_type=trend_type, thresh=thresh, min_var=min_var, detector_type=detector_type)
    return [specturm_period, 1] if return_period_source else specturm_period 
