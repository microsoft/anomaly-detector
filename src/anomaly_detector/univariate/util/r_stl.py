# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# -*- coding: utf-8 -*-

import pandas
import numpy as npy
from rstl import STL


def stl_core(data, np=None):
    """
    Seasonal-Trend decomposition procedure based on LOESS

    data : pandas.Series

    ns : int
        Length of the seasonal smoother.
        The value of  ns should be an odd integer greater than or equal to 3.
        A value ns>6 is recommended. As ns  increases  the  values  of  the
        seasonal component at a given point in the seasonal cycle (e.g., January
        values of a monthly series with  a  yearly cycle) become smoother.

    np : int
        Period of the seasonal component.
        For example, if  the  time series is monthly with a yearly cycle, then
        np=12.
        If no value is given, then the period will be determined from the
        ``data`` timeseries.
    """
    res_ts = STL(data, np, "periodic", robust=True)
    return pandas.DataFrame({"seasonal": res_ts.seasonal, "trend": res_ts.trend, "remainder": res_ts.remainder})


def stl_log(data, np=None):
    """
    Seasonal-Trend decomposition procedure based on LOESS for data with log transformation

    data : pandas.Series

    np : int
        Period of the seasonal component.
        For example, if  the  time series is monthly with a yearly cycle, then
        np=12.
        If no value is given, then the period will be determined from the
        ``data`` timeseries.
    """
    base = min(data)
    if base < 1:
        data = npy.subtract(data, base)
        data = data + 1  # add 1 along in case value scale in _data is extreme compared with 1

    result = STL(npy.log(data), np, "periodic", robust=True)
    trend_log = result.trend
    seasonal_log = result.seasonal

    trend = npy.exp(trend_log)
    seasonal = npy.exp(trend_log + seasonal_log) - trend
    remainder = data - trend - seasonal

    if base < 1:
        trend = trend - 1
        trend = trend + base

    try:
        res_ts = pandas.DataFrame({"seasonal": seasonal,
                                   "trend": trend,
                                   "remainder": remainder})
    except Exception as e:
        raise e

    return res_ts


def stl(data, np=None, log_transform=False):
    if log_transform:
        return stl_log(data.copy(), np)
    else:
        return stl_core(data.copy(), np)


def stl_adjust_trend(data, np, log_transform=False):
    """
    extend one point at the end of data to make the stl decompose result more robust
    :param data: pandas.Series
    :param np: period
    :return:
    """
    # make sure that data doesn't start or end with nan
    _data = data.copy()

    stl_func = stl_log if log_transform else stl_core

    # Append one point to the end of the array to make the stl decompose more robust.
    # The value of the point is the median of points in the same phrase in the cycle.
    extended_data = npy.append(_data, [npy.median(_data[-np::-np])])

    origin_stl_result = stl_func(_data, np=np)
    adjust_stl_result = stl_func(extended_data, np=np)

    origin_diff = npy.abs(origin_stl_result['remainder'].values[-1])
    adjust_diff = npy.abs(adjust_stl_result['remainder'].values[-2])

    if origin_diff <= adjust_diff:
        return origin_stl_result
    else:
        return pandas.DataFrame({"seasonal": adjust_stl_result['seasonal'][:len(data)],
                                 "trend": adjust_stl_result['trend'][:len(data)],
                                 "remainder": adjust_stl_result['remainder'][:len(data)]})
