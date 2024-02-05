# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.util import FillUpMode
from anomaly_detector.univariate.period import period_detection


def fill_up_on_demand(filling_up_process, mode, fixed_value=None, period=None):
    if mode == FillUpMode.previous or mode == FillUpMode.use_last:
        return filling_up_process.fill_up(method='last')
    if mode == FillUpMode.fixed:
        return filling_up_process.fill_up(method='constant', number=fixed_value)
    if mode == FillUpMode.linear:
        return filling_up_process.fill_up(method='linear')
    if mode == FillUpMode.auto:
        return filling_up_process.fill_up(method='auto', period=period, if_exception='fill_with_last')
    return None, None


def period_detection_with_filled_values(filling_up_process, mode, fixed_value=None, **kwargs):
    if filling_up_process.need_fill_up:
        if mode == FillUpMode.auto:
            values_to_detect_period, _ = fill_up_on_demand(filling_up_process, FillUpMode.previous)
        else:
            values_to_detect_period, _ = fill_up_on_demand(filling_up_process, mode, fixed_value)
        if values_to_detect_period is not None:
            return period_detection(values_to_detect_period, **kwargs)

    values_to_detect_period = filling_up_process.init_values
    return period_detection(values_to_detect_period, **kwargs)
