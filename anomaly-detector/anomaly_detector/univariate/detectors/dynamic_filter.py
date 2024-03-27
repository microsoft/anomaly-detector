# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.detectors.detector import AnomalyDetector
from anomaly_detector.univariate.util import Direction
from anomaly_detector.univariate._anomaly_kernel_cython import dynamic_threshold


class DynamicThreshold(AnomalyDetector):
    def __init__(self, series, max_outliers, threshold):
        self.__series__ = series
        self.__max_outliers = max_outliers
        self.__threshold__ = threshold

    def detect(self, direction, last_value=None):
        detect_data = self.__series__
        if direction == Direction.upper_tail:
            detect_data = self.__series__.iloc[::-1]
        last_index = -1
        if last_value is not None:
            last_index = max(detect_data.index)
            last_index = detect_data.index.get_loc(last_index)
        selected_anomaly = dynamic_threshold(detect_data.tolist(), detect_data.index.tolist(),
                                             self.__max_outliers,
                                             self.__threshold__,
                                             True if direction == Direction.upper_tail else False,
                                             last_index
                                             )

        if selected_anomaly is None or len(selected_anomaly) == 0:
            return []
        return selected_anomaly
