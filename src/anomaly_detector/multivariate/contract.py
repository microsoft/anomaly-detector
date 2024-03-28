# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import List


class MultiADConstants:
    TRAIN_CLIP_MIN = 0.0
    TRAIN_CLIP_MAX = 1.0
    INFERENCE_CLIP_MIN = -1000.0
    INFERENCE_CLIP_MAX = 1000.0
    ANOMALY_UPPER_THRESHOLD = 0.5
    ANOMALY_LOWER_THRESHOLD = 0.3
    TOP_ATTENTION_COUNT = 10
    DEFAULT_INPUT_SIZE = 200
    DEFAULT_THRESHOLD_WINDOW = 200


@dataclass
class NormalizeBase:
    max_values: List[float]
    min_values: List[float]


@dataclass
class MultiADConfig:
    data_dim: int
    cnn_kernel_size: int = 7
    input_size: int = MultiADConstants.DEFAULT_INPUT_SIZE  # seq length for dataset
    gat_window_size: int = 100  # seq length for GAT
    hidden_size: int = 100
    epochs: int = 100
    batch_size: int = 128
    seed: int = 42
    z_dim: int = 100
    enc_dim: int = 100
    dec_dim: int = 100
    interval: int = 10
    lr: float = 1e-3
    beta: float = 1.0
    gamma: float = 1.0
    device: str = "cpu"
    horizon: int = 1
    dropout: float = 0.5
    save: str = "model"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    normalize_base: NormalizeBase = None
    threshold_window: int = MultiADConstants.DEFAULT_THRESHOLD_WINDOW
    level: float = 0.8
