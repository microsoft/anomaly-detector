# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.util import np, average_filter, AnomalyId, AnomalyScore, IsAnomaly, EPS, ModelType, SKELETON_POINT_SCORE_THRESHOLD
from anomaly_detector.univariate._anomaly_kernel_cython import spectral_residual_transform_core
import pandas as pd
import numpy as np


class SpectralResidual:
    def __init__(self, series, max_outliers, threshold):
        self.__series__ = series
        self.__max_outliers = max_outliers
        self.__threshold__ = threshold

    def detect(self, last_value=None):
        return self._detect(self.__threshold__, self.__series__, self.__max_outliers, last_detect_check=last_value is not None)

    @staticmethod
    def _detect(threshold, series, max_outliers, last_detect_check):
        model_id = ModelType.SpectralResidual
        if last_detect_check:
            length = len(series)
            anomaly_scores = SpectralResidual.generate_spectral_score(series=series)
            max_drop_num = min(max(max_outliers, int(length * 0.1)), int(length * 0.45))  # max number of points to drop out

            drop_threshold = 1.5  # points with score greater than this parameter will be dropped
            mean = np.mean(series)
            std = np.std(series)
            zscore = np.abs(series - mean) / std

            # filtered_series = series[:next(
            #     (i for i in range(length - 1, length - max_drop_num, -1) if zscore[i] < drop_threshold),
            #     length - 1)] + [series[-1]]

            filtered_series = series[:next(
                (i for i in range(length - max_drop_num, length) if anomaly_scores[i] >= drop_threshold),
                length - 1)] + [series[-1]]

            # dismiss these points in expected value calculation
            anomaly_scores[len(filtered_series)-1:-1] = SKELETON_POINT_SCORE_THRESHOLD * 2
            filtered_series_anomaly_scores = SpectralResidual.generate_spectral_score(series=filtered_series)
            anomaly_scores[-1] = filtered_series_anomaly_scores[-1]
            if zscore[-1] < drop_threshold:
                anomaly_scores[-1] = 0
                model_id = ModelType.SpectralResidual_ZScore
        else:
            anomaly_scores = SpectralResidual.generate_spectral_score(series=series, remove_outlier_in_extend = True)

        anomaly_frame = pd.DataFrame({AnomalyId: list(range(0, len(anomaly_scores))),
                                      AnomalyScore: anomaly_scores})
        anomaly_frame[IsAnomaly] = np.where(anomaly_frame[AnomalyScore] >= threshold, True, False)
        anomaly_frame.set_index(AnomalyId, inplace=True)
        return anomaly_frame, model_id

    @staticmethod
    def generate_spectral_score(series, remove_outlier_in_extend = False):
        extended_series = SpectralResidual.extend_series(series, remove_outlier_in_extend = remove_outlier_in_extend)
        mag = SpectralResidual.spectral_residual_transform(extended_series)[:len(series)]
        ave_mag = average_filter(mag, n=100, fill_to_n=True)
        ave_mag = [EPS if np.isclose(x, EPS) else x for x in ave_mag]

        return abs(mag - ave_mag) / ave_mag

    @staticmethod
    def spectral_residual_transform(values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        return np.asarray(spectral_residual_transform_core(values))


    @staticmethod
    def predict_next(values):
        """
        Predicts the next value by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j)
        :param values: list.
            a list of float numbers.
        :return : float.
            the predicted next value.
        """

        if len(values) <= 1:
            raise ValueError(f'data should contain at least 2 numbers')

        v_last = values[-1]
        n = len(values)

        slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

        return np.median(values) + sum(slopes) * 0.5 * (n+1) / (n-1)

    @staticmethod
    def extend_series(values, extend_num=5, look_ahead=5, remove_outlier_in_extend = False):
        """
        extend the array data by the predicted next value
        :param values: list.
            a list of float numbers.
        :param extend_num: int, default 5.
            number of values added to the back of data.
        :param look_ahead: int, default 5.
            number of previous values used in prediction.
        :return: list.
            The result array.
        """

        if look_ahead < 1:
            raise ValueError('look_ahead must be at least 1')

        if remove_outlier_in_extend:
            q75, q50, q25 = np.quantile(values, [0.75, 0.5, 0.25])
            maxv, minv = q75 + 1.5 * (q75 - q25), q25 - 1.5 * (q75 - q25)
            value_ahead = [q50] * (look_ahead + 1)
            i = look_ahead; j = len(values) - 1
            while i>=0 and j>=0:
                if values[j] >= minv and values[j] <= maxv:
                    value_ahead[i] = values[j]
                    i -= 1
                j -= 1
        else:
            value_ahead = values[-look_ahead - 2:-1]
        extension = [SpectralResidual.predict_next(value_ahead)] * extend_num
        return values + extension
