from typing import List, Union

import pandas as pd
from anomaly_detector.common.constants import TIMESTAMP, FillNAMethod
from anomaly_detector.common.exception import *
from anomaly_detector.common.time_util import DT_FORMAT, dt_to_str


class MultiADDataProcessor:
    def __init__(
        self,
        *,
        fill_na_method: str = FillNAMethod.Linear.name,
        fill_na_value: float = 0.0,
        window: Union[int, float, str],
        start_time: str = None,
        end_time: str = None,
    ):
        if not hasattr(FillNAMethod, fill_na_method):
            raise InvalidParameterError(
                f"fill_na_method {fill_na_method} is not supported."
            )
        if not isinstance(fill_na_value, float):
            raise InvalidParameterError(f"fill_na_value must be a float number.")

        self.fill_na_method = FillNAMethod[fill_na_method]
        self.fill_na_value = fill_na_value
        try:
            self.window = int(window)
        except TypeError:
            raise InvalidParameterError(f"Cannot convert window to int.")
        try:
            self.start_time = pd.to_datetime(start_time).tz_localize(None)
            self.end_time = pd.to_datetime(end_time).tz_localize(None)
        except Exception as e:
            raise InvalidParameterError(
                f"Cannot convert start_time or end_time. {str(e)}."
            )
        if self.start_time > self.end_time:
            raise InvalidParameterError(f"start_time cannot be later than end_time.")

    def process(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise DataFormatError(f"data must be a pandas.DataFrame")
        if TIMESTAMP not in data.columns:
            raise DataFormatError(f"data must has a {TIMESTAMP} column.")
        data = data.set_index(TIMESTAMP, drop=True).sort_index()  # sort indices
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data[sorted(data.columns)]  # sort columns
        data = self.fill_na(data)
        data, effective_timestamps = self.truncate_data(data)
        effective_timestamps = [dt_to_str(x) for x in effective_timestamps]
        return data, effective_timestamps

    def fill_na(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise DataFormatError(f"data must be a pandas.DataFrame")
        try:
            data = data.astype(float)
        except Exception as e:
            raise DataFormatError(f"Cannot convert values to float. {str(e)}")
        if self.fill_na_method == FillNAMethod.Previous:
            output_series = data.fillna(method="ffill", limit=len(data)).fillna(
                method="bfill", limit=len(data)
            )
        elif self.fill_na_method == FillNAMethod.Subsequent:
            output_series = data.fillna(method="bfill", limit=len(data)).fillna(
                method="ffill", limit=len(data)
            )
        elif self.fill_na_method == FillNAMethod.Linear:
            output_series = data.interpolate(
                method="linear", limit_direction="both", axis=0, limit=len(data)
            )
        elif self.fill_na_method == FillNAMethod.Fixed:
            output_series = data.fillna(self.fill_merge_na_value)
        else:
            output_series = data
        return output_series.fillna(0)

    def truncate_data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise DataFormatError(f"data must be a pandas.DataFrame")
        effective_df = data.loc[self.start_time : self.end_time]
        if len(effective_df) == 0:
            raise DataFormatError(f"no effective data.")
        first_index = effective_df.index[0]
        start_loc = max(0, data.index.get_loc(first_index) - self.window + 1)
        start_index = data.index[start_loc]
        end_index = self.end_time
        data = data.loc[start_index:end_index]
        return data, effective_df.index.to_list()
