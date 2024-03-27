# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import bisect
import numpy as np
from anomaly_detector.univariate.util import EPS

ANOMALY_IGNORE_RATIO = 0.0001
MIN_UNIT = 0.3

# pseudo - code to generate the factors.
# factors = []
# for i in range(0, 30):
#     s = 0.8 * (i - 30) ** 2 + 32
#     factors.append(s)
#
# for i in range(30, 50):
#     s = -1.25 * i + 67.5
#     factors.append(s)
#
# for i in range(50, 60):
#     s = -0.4 * i + 25
#     factors.append(s)
#
# for i in range(60, 70):
#     s = -0.04 * i + 3.4
#     factors.append(s)
#
# for i in range(70, 80):
#     s = -0.03 * i + 2.7
#     factors.append(s)
#
# for i in range(80, 90):
#     s = -0.015 * i + 1.4999999999999998
#     factors.append(s)
#
# for i in range(90, 98):
#     s = -0.011818181818181818 * i + 1.2136363636363636
#     factors.append(s)
#
# factors.append(0.043636363636363695)
# factors.append(0.012)
# factors.append(0.0)

factors = [532.0, 492.8, 455.20000000000005, 419.20000000000005, 384.8, 352.0, 320.8, 291.2, 263.20000000000005,
           236.8, 212.0, 188.8, 167.20000000000002, 147.2, 128.8, 112.0, 96.8, 83.2, 71.2, 60.8, 52.0, 44.8, 39.2,
           35.2, 32.8, 30.0, 28.75, 27.5, 26.25, 25.0, 23.75, 22.5, 21.25, 20.0, 18.75, 17.5, 16.25, 15.0, 13.75,
           12.5, 11.25, 10.0, 8.75, 7.5, 6.25, 5.0, 4.599999999999998, 4.199999999999999, 3.799999999999997,
           3.3999999999999986, 3.0, 2.599999999999998, 2.1999999999999993, 1.7999999999999972, 1.3999999999999986,
           1.0, 0.96, 0.9199999999999999, 0.8799999999999999, 0.8399999999999999, 0.7999999999999998,
           0.7599999999999998, 0.7199999999999998, 0.6799999999999997, 0.6399999999999997, 0.6000000000000001,
           0.5700000000000003, 0.54, 0.5100000000000002, 0.4800000000000004, 0.4500000000000002, 0.4200000000000004,
           0.3900000000000001, 0.3600000000000003, 0.33000000000000007, 0.2999999999999998, 0.2849999999999999,
           0.2699999999999998, 0.2549999999999999, 0.23999999999999977, 0.22499999999999987, 0.20999999999999974,
           0.19499999999999984, 0.17999999999999994, 0.1649999999999998, 0.1499999999999999, 0.13818181818181818,
           0.12636363636363646, 0.1145454545454545, 0.10272727272727278, 0.09090909090909083, 0.0790909090909091,
           0.06727272727272737, 0.043636363636363695, 0.01200000000000001, 0.008, 0.0060750000000000005, 0.00415,
           0.0022249999999999995, 0.0002999999999999999, 0.0]


def calculate_boundary_units(trend, is_anomaly):
    if np.all([np.abs(u) < EPS for u in trend[~is_anomaly]]):
        return np.ones(len(trend)) * MIN_UNIT

    unit = np.mean(np.abs(trend[~is_anomaly]))
    units = np.abs(trend) * 0.5 + unit * 0.5
    units = np.clip(units, MIN_UNIT, max(MIN_UNIT, np.max(units)))

    return units


def calculate_margin(unit, sensitivity, value, expected_value, is_anomaly):
    def calculate_changed_margin(unit, sensitivity, value, expected_value, is_anomaly):
        percent = 0.5
        delta = unit * factors[int(sensitivity)]
        if not is_anomaly:
            # extend the margin to include the not anomaly point
            delta = np.abs(expected_value - value) + delta * percent
            if value > expected_value:
                return delta, delta / 3.0
            else:
                return delta / 3.0, delta
        else:  # is_anomaly
            # reduce the margin to show part of the anomalies with score above 99
            if delta * ANOMALY_IGNORE_RATIO < np.abs(value - expected_value) < delta and sensitivity == 99:
                delta = np.abs(expected_value - value) * percent

        return delta, delta

    def calculate_margin_core(unit, sensitivity, value, expected_value, is_anomaly):
        lb = int(sensitivity)
        margin1 = calculate_changed_margin(unit, lb, value, expected_value, is_anomaly)

        if lb == sensitivity:
            return margin1

        margin2 = calculate_changed_margin(unit, lb + 1, value, expected_value, is_anomaly)
        return margin2 + (1 - sensitivity + lb) * (margin1 - margin2)

    if 0 > sensitivity or sensitivity > 100:
        raise Exception('sensitivity should be integer in [0, 100]')

    if unit <= 0:
        raise Exception('unit should be a positive number')

    return calculate_margin_core(unit, sensitivity, value, expected_value, is_anomaly)


def calculate_anomaly_score(value, expected_value, unit, is_anomaly):
    dist = np.abs(expected_value - value) / unit
    margins = factors[::-1]
    lb = bisect.bisect_left(margins, dist)
    if lb == 0:
        score = 0
    elif lb >= 100:
        score = 100
    else:
        a, b = margins[lb - 1], margins[lb]
        score = lb - 1 + (dist - a) / (b - a)

    return score


def calculate_severity_v1(value, expected_value, is_anomaly):
    if not is_anomaly:
        return 0.0

    base = np.abs(expected_value)
    if base <= EPS:
        base = MIN_UNIT

    return np.min((np.abs(value - expected_value) / base, 1.0))


def calculate_severity_v2(anomaly_score, is_anomaly):
    if not is_anomaly:
        return 0.0

    return anomaly_score / 100.0


def calculate_anomaly_scores(values, expected_values, units, is_anomaly):
    scores = [calculate_anomaly_score(value, exp, unit, anomaly)
              for value, exp, unit, anomaly in zip(values, expected_values, units, is_anomaly)]
    return scores
