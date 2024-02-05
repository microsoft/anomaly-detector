# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import math
from anomaly_detector.univariate._anomaly_kernel_cython import gcv, median_filter, remove_anomaly_in_bucket
from anomaly_detector.univariate.util import Granularity, normalize, smooth_spikes, MIN_PERIOD


class SimpleDetector:

    @staticmethod
    def detect(series, granularity, interval):
        if series is None:
            return None
        series_array = np.array(series)

        period = SimpleDetector.verify_period(series_array, granularity, interval)
        return period

    @staticmethod
    def verify_period(values, granularity, interval):
        periods = SimpleDetector.guess_period(granularity, interval)
        if len(periods) == 0:
            return None

        verified = None
        for period in periods:
            if len(values) <= period * 2 or period < MIN_PERIOD:
                continue
            if SimpleDetector.is_valid_period(values, period):
                verified = period
                break
        if verified:
            std_period = SimpleDetector.standard_period(granularity, interval)
            if len(values) <= std_period * 2 or std_period % verified != 0:
                return verified
            return std_period
        return None

    @staticmethod
    def is_valid_period(values, period):
        normed_values = normalize(values)
        removed_spike = smooth_spikes(normed_values)
        if np.isclose(removed_spike.var(), 0.0):
            return False
        if SimpleDetector.check_period(removed_spike, period, False):
            return True
        median_trend = median_filter(normed_values, period, True)
        detrended = normed_values - median_trend
        detrended = smooth_spikes(detrended)
        detrended = remove_anomaly_in_bucket(detrended, period)
        if np.isclose(detrended.var(), 0.0):
            return False
        return SimpleDetector.check_period(detrended, period, True)

    @staticmethod
    def check_period(values, period, detrend):
        config_mse = SimpleDetector.get_period_config(period) if not detrend else SimpleDetector.get_period_detrend_config(period)
        var = values.var()
        cv_mse, cv_seasons = gcv(values, period)
        if np.isclose(cv_mse, 0.0):
            mse = 1
        else:
            mse = 1 - cv_mse / var
        return mse > config_mse

    @staticmethod
    def get_period_config(period):
        if period <= 24:
            return 0.35
        if period <= 168:
            return 0.15
        return 0.1

    @staticmethod
    def get_period_detrend_config(period):
        if period <= 24:
            return 0.35
        if period <= 168:
            return 0.65
        return 0.7

    @staticmethod
    def guess_period(granularity, interval):
        interval = interval if interval else 1
        periods = {
            Granularity.yearly: [],
            Granularity.none: [],
            Granularity.daily: [7],
            Granularity.hourly: [168 // interval, 24 // interval],
            Granularity.minutely: [1440 * 7 // interval, 1440 // interval, 1440 * 2 // interval],
            Granularity.weekly: [4 * 3, 4],
            Granularity.monthly: [12],
            Granularity.secondly: [86400 * 7 // interval, 86400 // interval, 86400 * 2 // interval],
            Granularity.microsecond: [1000]
        }
        return periods[granularity]

    @staticmethod
    def standard_period(granularity, interval):
        interval = interval if interval else 1
        period = {
            Granularity.daily: 7,
            Granularity.hourly: 168 // interval,
            Granularity.minutely: 1440 * 7 // interval,
            Granularity.weekly: 12,
            Granularity.monthly: 12,
            Granularity.secondly: 86400 * 7 // interval,
            Granularity.yearly: 0,
            Granularity.none: 0
        }
        return period[granularity]
