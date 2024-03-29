# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import torch


class AverageMeter:
    def __init__(self):
        self.sum = None
        self.avg = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, num, is_sum=True):
        self.sum += val if is_sum else val * num
        self.count += num
        self.avg = self.sum / self.count


def get_threshold(scores, p=0.95):
    if len(scores) == 0:
        return 0
    data = np.sort(scores)
    return data[int(len(data) * p)]


def minmax_normalize(data, min_val, max_val, clip_min, clip_max):
    data_normalized = (data - min_val) / (max_val - min_val + 1e-8)
    if isinstance(data_normalized, np.ndarray):
        data_normalized = np.clip(data_normalized, clip_min, clip_max)
    elif isinstance(data_normalized, torch.Tensor):
        data_normalized = torch.clamp(data_normalized, clip_min, clip_max)
    else:
        raise NotImplementedError
    return data_normalized


def get_multiple_variables_pct_weight_score(data, window, timestamp_col="timestamp"):
    max_pct_weight = 1.9
    if isinstance(data, pd.DataFrame):
        if timestamp_col in data.columns:
            data = data.drop(timestamp_col, axis=1)
        data = data[sorted(data.columns)]
        data = data.values
    elif isinstance(data, np.ndarray):
        data = data
    else:
        raise TypeError(f"Unsupported type {type(data)}")
    variables_num = data.shape[1]
    pct_weight = np.empty(variables_num, float)
    i = 0
    for i in range(variables_num):
        s_data = pd.Series(data[:, i] + 0.0001)
        data1 = np.maximum(
            np.abs((s_data.shift(1) / s_data) - 1),
            np.abs((s_data.shift(-1) / s_data) - 1),
        )
        data2 = np.clip(data1, a_max=2, a_min=0)
        pct_weight[i] = data2.rolling(window).max().mean()
        i = i + 1
    reweight_num = np.count_nonzero(pct_weight > max_pct_weight)
    reweight_value = reweight_num / variables_num
    for i in range(variables_num):
        if pct_weight[i] > max_pct_weight:
            pct_weight[i] = reweight_value
        else:
            pct_weight[i] = 1
    return pct_weight.tolist()


def append_result(result, batch):
    if result is None:
        result = batch
    else:
        result = np.concatenate([result, batch])
    return result


def compute_anomaly_scores(rmse, prob, gamma):
    return (prob * gamma + rmse) / (1 + gamma)


def compute_severity(inference_scores):
    return inference_scores / (np.e - 1)
