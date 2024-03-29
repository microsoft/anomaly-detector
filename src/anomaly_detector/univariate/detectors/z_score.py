# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.detectors.detector import AnomalyDetector
import numpy as np
from statsmodels import robust

from anomaly_detector.univariate.util import Direction


class ZScoreDetector(AnomalyDetector):
    def __init__(self, series, max_outliers):
        self.__series__ = series
        self.__max_outliers = max_outliers
        self.__median_value = np.median(series)
        self.__mad_value = robust.mad(self.__series__)
        if self.__mad_value == 0:
            self.__mad_value = np.std(self.__series__)
            self.__median_value = np.mean(self.__series__)

    def detect(self, direction, last_value=None):
        if self.__mad_value == 0:
            return []
        detect_data = self.__series__
        if direction == Direction.upper_tail:
            detect_data = self.__series__[::-1]
        selected_data = detect_data[:self.__max_outliers]
        selected_anomaly = selected_data[
            (abs(selected_data - self.__median_value) / self.__mad_value) > 3].index.tolist()
        if selected_anomaly is None or len(selected_anomaly) == 0:
            return []
        return selected_anomaly
