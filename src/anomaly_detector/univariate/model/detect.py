# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pandas as pd

from anomaly_detector.univariate.util import Direction, IsPositiveAnomaly, IsNegativeAnomaly, IsAnomaly


def detect(directions, detectors, max_outliers, num_obs, last_value=None):
    anomalies = {}
    for direction in directions:
        anomaly = {}
        for i in range(len(detectors)):
            selected_anomaly = detectors[i].detect(direction=direction, last_value=last_value)
            if selected_anomaly is None or len(selected_anomaly) == 0:
                continue
            for k in range(len(selected_anomaly)):
                index = selected_anomaly[k]
                if index in anomaly:
                    anomaly[index] += k
                else:
                    anomaly[index] = k + i * num_obs

        sorted_anomaly = sorted(anomaly, key=anomaly.get)
        anomalies[direction] = pd.DataFrame(sorted_anomaly[:min(max_outliers, len(sorted_anomaly))],
                                            columns=['Idx'])
        if direction == Direction.upper_tail:
            anomalies[direction][IsPositiveAnomaly] = True
        else:
            anomalies[direction][IsNegativeAnomaly] = True

        anomalies[direction].set_index('Idx', inplace=True)

    anomaly_frame = pd.DataFrame()
    for direction in anomalies:
        anomaly_frame = anomaly_frame.join(anomalies[direction], how='outer')

    if len(anomaly_frame) != 0:
        anomaly_frame[IsAnomaly] = True
    else:
        return pd.DataFrame(columns=[IsAnomaly, IsPositiveAnomaly,
                                     IsNegativeAnomaly,
                                     'Idx'])
    return anomaly_frame
