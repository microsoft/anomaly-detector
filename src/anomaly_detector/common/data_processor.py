# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

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
    ):
        if not hasattr(FillNAMethod, fill_na_method):
            raise InvalidParameterError(
                f"fill_na_method {fill_na_method} is not supported."
            )
        if not isinstance(fill_na_value, float):
            raise InvalidParameterError(f"fill_na_value must be a float number.")

        self.fill_na_method = FillNAMethod[fill_na_method]
        self.fill_na_value = fill_na_value

    def process(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise DataFormatError(f"data must be a pandas.DataFrame")
        data = data.sort_index()  # sort indices
        data = data[sorted(data.columns)]  # sort columns
        data = self.fill_na(data)
        return data

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
            output_series = data.fillna(self.fill_na_value)
        else:
            output_series = data
        return output_series.fillna(0)
