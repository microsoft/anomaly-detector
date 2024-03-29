# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from anomaly_detector.multivariate.util import minmax_normalize
from torch.utils.data import Dataset


class MultiADDataset(Dataset):
    """
    Dataset for Multi-AD model
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        interval: int,
        horizon: int,
        max_values: Union[np.ndarray, List[float]],
        min_values: Union[np.ndarray, List[float]],
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        """
        :param data: 2-d array, shape (num_timestamps, num_variables)
        :param window_size: size of the sliding window
        :param interval: step size of the sliding window
        :param horizon: horizon of the prediction
        :param max_values: max value of each variable
        :param min_values: min value of each variable
        :param clip_min: min value of the normalized data
        :param clip_max: max value of the normalized data
        """
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.max_values = np.array(max_values)
        self.min_values = np.array(min_values)
        self.data = data
        self.data_length = len(self.data)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.x_end_idx = self.get_x_end_idx()

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for getting data
        # range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.data_length - self.horizon + 1)
        x_end_idx = [
            x_index_set[j * self.interval]
            for j in range((len(x_index_set)) // self.interval)
        ]
        return x_end_idx

    def __len__(self):
        return len(self.x_end_idx)

    def __getitem__(self, idx):
        """
        :param idx: int, index of the sample
        :return x: 2-d array, shape (window_size, num_variables)
        :return y: 1-d array, shape (num_variables,)
        """
        hi = self.x_end_idx[idx]
        lo = hi - self.window_size
        train_data = self.data[lo:hi]
        target_data = self.data[hi - 1 + self.horizon]
        train_data = minmax_normalize(
            train_data, self.min_values, self.max_values, self.clip_min, self.clip_max
        )
        target_data = minmax_normalize(
            target_data, self.min_values, self.max_values, self.clip_min, self.clip_max
        )
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y
