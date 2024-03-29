# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
from anomaly_detector.univariate._anomaly_kernel_cython import max_gcv, gcv
from seasonal import periodogram_peaks
import statsmodels.api as sm
from anomaly_detector.univariate.util import fit_trend, normalize, smooth_spikes, MIN_PERIOD


class SpectrumDetector:

    @staticmethod
    def detect(series, trend_type, thresh, min_var, detector_type):
        if len(series) < 12:
            raise ValueError("Series length cannot be less than 12 for period detection.")

        series_array = np.array(series)

        series_array, s_max, s_min = normalize(series_array, min_max=True)

        period, seasons, trend = SpectrumDetector.calculate_period(
            series=series_array, trend_type=trend_type,
            thresh=thresh, min_var=min_var, detector_type=detector_type
        )

        if period == 0:
            return period

        verified_period = period
        while verified_period != 0:
            series_array = series_array[0::verified_period]
            verified_period, _, _ = SpectrumDetector.calculate_period(
                series_array, trend_type, thresh, min_var, detector_type)
            if verified_period != 0:
                period = period * verified_period

        return period

    @staticmethod
    def calculate_period(series, trend_type, thresh, min_var, detector_type):
        if len(series) < 12:
            return 0, None, None

        seasons, trend = SpectrumDetector.fit_seasons(
            series, trend_type=trend_type, period_gram_thresh=thresh,
            min_ev=min_var, detector_type=detector_type,
        )
        if seasons is None or len(seasons) == 0:
            return 0, seasons, trend

        period = len(seasons)

        cycles = len(series) / period + 1
        if cycles <= 3:
            return 0, seasons, trend

        return period, seasons, trend

    @staticmethod
    def is_valid_period(first, second):
        if first >= second:
            return first % second == 0

        return second % first == 0

    @staticmethod
    def fit_seasons(data, trend_type="spline", period=None, min_ev=0.05,
                    period_gram_thresh=0.5, detector_type="periodogram"):
        data = smooth_spikes(data)

        if trend_type is None:
            trend = np.zeros(len(data))
        elif not isinstance(trend_type, np.ndarray):
            trend = fit_trend(data, kind=trend_type, period=period)
        else:
            assert isinstance(trend_type, np.ndarray)
        data = data - trend
        var = data.var()
        if np.isclose(var, 0.0):
            return None, trend

        if period:
            cv_mse, cv_seasons = gcv(data, period)
            fev = 1 - cv_mse / var
            if np.isclose(cv_mse, 0.0) or fev >= min_ev:
                return cv_seasons, trend
            else:
                return None, trend

        if detector_type == "periodogram":
            periods = SpectrumDetector.periodogram_detector(data, period_gram_thresh)
        else:
            periods = SpectrumDetector.correlogram_detector(data)

        if len(periods) == 0:
            return None, trend
        cv_mse, cv_seasons = max_gcv(data, np.array(periods, dtype='i'))
        if np.isclose(cv_mse, 0.0) or min_ev <= 1 - cv_mse / var:
            return cv_seasons, trend
        else:
            return None, trend

    @staticmethod
    def periodogram_detector(data, period_gram_thresh):
        if period_gram_thresh:
            peaks = periodogram_peaks(data, thresh=period_gram_thresh)
            if peaks is None:
                return []
            peaks = sorted(peaks)
        else:
            peaks = [(0, 0, 4, len(data) // 2)]
        periods = []
        period = 0
        for peak in peaks:
            periods.extend(range(max(period, peak[2]), peak[3] + 1))
            period = peak[3] + 1
        return periods

    @staticmethod
    def correlogram_detector(data, min_period=MIN_PERIOD, max_period=None, corr_thresh=0.1):
        if max_period is None:
            max_period = int(min(len(data) / 3.0, 2880 * 2))

        acf, conf = sm.tsa.acf(data, nlags=max_period, fft=False, alpha=0.1)
        acf = acf[1:]
        conf = conf[1:]
        periods = []

        while True:
            peak_i = acf.argmax()
            ub = conf[peak_i, 1] - acf[peak_i]
            if acf[peak_i] < ub or acf[peak_i] < corr_thresh:
                break

            acf[peak_i] = 0
            if min_period < peak_i + 1 < max_period:
                periods.append(peak_i + 1)
        return periods
