# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.util.helpers import get_indices_from_timestamps
from anomaly_detector.univariate.util.fields import FillUpMode, DEFAULT_FILL_UP_MODE
from anomaly_detector.univariate.model.series_compete_processor import fill_up_on_demand, period_detection_with_filled_values
from .fill_up import FillingUpProcess

FillingUpProcess.fill_up_on_demand = fill_up_on_demand
FillingUpProcess.period_detection_with_filled_values = period_detection_with_filled_values
