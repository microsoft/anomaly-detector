# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import math

from dateutil import parser, tz

DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DT_FORMAT_WITH_MICROSECOND = "%Y-%m-%dT%H:%M:%S.%fZ"


def str_to_dt(s):
    """
    This method parses string to datetime. If the string has time zone information,
    transpose the datetime to utc time zone. If the string doesn't have time zone
    information, assume it's a time in utc time zone. The datetime is returned with
    tzinfo as None.
    :param s: str
    :return: datetime in utc time zone with tzinfo as None
    """
    parsed = parser.parse(s)
    if parsed.tzinfo is None:
        return parsed
    else:
        parsed = parsed.astimezone(tz.UTC)
        return parsed.replace(tzinfo=None)


def dt_to_str(dt, fmt=DT_FORMAT):
    """
    This method returns string format of datetime in utc time zone. If the datetime
    doesn't have time zone information, assume it's a time in utc time zone.
    :param dt: datetime
    :param fmt: format of str
    :return: str
    """
    if dt.tzinfo is not None and dt.tzinfo != tz.UTC:
        dt = dt.astimezone(tz.UTC)
    return dt.strftime(fmt)


def dt_to_str_with_microsecond(dt):
    return dt_to_str(dt, fmt=DT_FORMAT_WITH_MICROSECOND)
