# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.detectors.detector import AnomalyDetector
from anomaly_detector.univariate.util import Direction, get_critical, EPS
from anomaly_detector.univariate._anomaly_kernel_cython import generalized_esd_test


class ESD(AnomalyDetector):
    def __init__(self, series, max_outliers, majority_value, alpha):
        self.__series__ = series
        self.__max_outliers = max_outliers
        self.__alpha = alpha
        self.__majority_value = majority_value
        self.__critical_values__ = get_critical(self.__alpha, len(series), max_outliers) \
            if majority_value is None else None

    def detect(self, direction, last_value=None):
        last_index = -1
        if last_value is not None:
            last_index = max(self.__series__.index)

        detect_data = self.__series__
        if direction == Direction.upper_tail:
            detect_data = self.__series__.iloc[::-1]
        if self.__majority_value is not None:
            detect_data = detect_data[:next(i for i in reversed(range(len(detect_data))) if
                                            abs(detect_data.iloc[i] - self.__majority_value) < EPS) + 1]

        if last_index != -1:
            if last_index not in detect_data.index:
                return []
            else:
                last_index = detect_data.index.get_loc(last_index)

        critical_values = get_critical(self.__alpha, len(detect_data), self.__max_outliers) \
            if self.__critical_values__ is None else self.__critical_values__

        selected_anomaly = generalized_esd_test(detect_data.tolist(), detect_data.index.tolist(),
                                                self.__max_outliers,
                                                critical_values,
                                                True if direction == Direction.upper_tail else False,
                                                last_index
                                                )

        if selected_anomaly is None or len(selected_anomaly) == 0:
            return []
        return selected_anomaly
