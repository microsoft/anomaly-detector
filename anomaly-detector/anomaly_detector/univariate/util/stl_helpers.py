# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np

MAPE_UB = 0.10
MAPE_LB = 0.05


def get_outlier(values, period):
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    outlier_index = values[np.abs(values - mean) / std >= 3].index
    if len(outlier_index) == 0:
        return []
    period_bins = outlier_index % period
    unique, counts = np.unique(period_bins, return_counts=True)
    invalid_period_bin = unique[np.where(counts <= int((len(values) / period) / 2))]
    outlier_index = outlier_index[np.where(np.in1d(period_bins, invalid_period_bin))]
    return outlier_index


def de_outlier_stl(series, stl_func, period, log_transform):
    series_decompose = stl_func(np.asarray(series), np=period, log_transform=log_transform)
    series_de_trend = series_decompose['remainder'] + series_decompose['seasonal']
    outlier = get_outlier(series_de_trend, period)
    if len(outlier) == 0:
        return series_decompose

    series_de_trend[outlier] = np.nan
    series_de_trend = series_de_trend.fillna(method='ffill', limit=len(series_de_trend)).fillna(method='bfill')
    # series_de_trend = series_de_trend.groupby(series_de_trend.index % period).
    # ffill(limit=len(series_de_trend)).bfill()

    return stl_func(series_de_trend + series_decompose['trend'], np=period, log_transform=log_transform)
