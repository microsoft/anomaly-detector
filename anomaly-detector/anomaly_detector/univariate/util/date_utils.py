# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# -*- coding: utf-8 -*-
from datetime import datetime, date
from re import match
import pytz
from calendar import monthrange


def json_serial(obj):
    if isinstance(obj, (date, datetime)):
        return obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def datetime_from_ts(column):
    return column.map(
        lambda date_str: datetime.fromtimestamp(int(date_str), tz=pytz.utc))


def date_format(column, format_str):
    return column.map(lambda date_str: datetime.strptime(date_str, format_str))


def format_timestamp_str(date_str):
    if match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} \\+\\d{4}$",
             date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$", date_str):
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    elif match("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}$", date_str):
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$", date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}$", date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    elif match("^\\d{2}/\\d{2}/\\d{2}$", date_str):
        return datetime.strptime(date_str, "%m/%d/%y")
    elif match("^\\d{2}/\\d{2}/\\d{4}$", date_str):
        return datetime.strptime(date_str, "%Y%m%d")
    elif match("^\\d{4}\\d{2}\\d{2}$", date_str):
        return datetime.strptime(date_str, "%Y/%m/%d/%H")
    else:
        raise ValueError("timestamp format currently not supported.")


def format_timestamp(df, index=0):
    column = df.iloc[:, index]

    if match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} \\+\\d{4}$",
             column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$", column[0]):
        column = date_format(column, "%Y-%m-%dT%H:%M:%SZ")
    elif match("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}$", column[0]):
        column = date_format(column, "%Y-%m-%dT%H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$", column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}$", column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M")
    elif match("^\\d{2}/\\d{2}/\\d{2}$", column[0]):
        column = date_format(column, "%m/%d/%y")
    elif match("^\\d{2}/\\d{2}/\\d{4}$", column[0]):
        column = date_format(column, "%Y%m%d")
    elif match("^\\d{4}\\d{2}\\d{2}$", column[0]):
        column = date_format(column, "%Y/%m/%d/%H")
    elif match("^\\d{10}$", column[0]):
        column = datetime_from_ts(column)
    else:
        raise ValueError("timestamp format currently not supported.")

    df.iloc[:, index] = column

    return df


def format_timestamp_to_str(df):
    df['Timestamp'] = df['Timestamp'].map(lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return df


class DateDifference:
    def __init__(self, years=0, months=0, days=0):
        self.years = years
        self.months = months
        self.days = days

    def __str__(self):
        return f'years={self.years},months={self.months},days={self.days}'


def get_days_in_month(year, month):
    return monthrange(year, month)[1]


def get_date_difference(a, b):
    factor = 1
    if a < b:
        tmp = a
        a = b
        b = tmp
        factor = -1

    a_days_in_month = get_days_in_month(a.year, a.month)
    b_days_in_month = get_days_in_month(b.year, b.month)

    diff_day = 0
    diff_month = 0
    diff_year = 0

    if a.year == b.year and a.month == b.month:
        diff_day = a.day - b.day
    elif (a.day == b.day) or (a.day == a_days_in_month and b.day == b_days_in_month) \
            or (a_days_in_month != b_days_in_month and (
            a.day == a_days_in_month and b.day > a.day or b.day == b_days_in_month and a.day > b.day)):
        diff_month = a.month - b.month
    else:
        if a.day > b.day:
            diff_day = a.day - b.day
            diff_month = a.month - b.month
        else:
            diff_day = b_days_in_month - b.day + a.day
            diff_month = a.month - b.month - 1
    diff_year = a.year - b.year
    if diff_month < 0:
        diff_year -= 1
        diff_month += 12
    return DateDifference(years=factor * diff_year, months=factor * diff_month, days=factor * diff_day)
