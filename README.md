# Environment
Tested on
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)


# Getting Started

## Installing from pip

```bash
pip install time-series-anomaly-detector==0.4.0
```

## Installing from Source

```bash
git clone https://github.com/microsoft/anomaly-detector.git
pip install -e .
```

## Test

```bash
cd anomaly-detector/tests
python uvad_test.py
python test_demo.py
```

---

# Usage Guide

This library is imported as a Python package — there is no CLI. It supports two detection modes:

- **Univariate** — detects anomalies in a single time series
- **Multivariate** — detects anomalies across multiple correlated time series using a graph attention network

## Univariate Anomaly Detection

Import the detector and prepare a pandas DataFrame with `timestamp` and `value` columns:

```python
import pandas as pd
from anomaly_detector.univariate.univariate_anomaly_detection import UnivariateAnomalyDetector
from anomaly_detector.univariate.util.fields import DetectType
```

### Entire Series Detection

Scores every point in the series and flags which ones are anomalous:

```python
data = pd.DataFrame({
    "timestamp": ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z", ...],
    "value":     [100.1, 99.8, 140.5, ...]
})

params = {
    "granularity": "daily",
    "detect_mode": DetectType.ENTIRE,
    "sensitivity": 95,
}

detector = UnivariateAnomalyDetector()
results = detector.predict(None, data, params)

for r in results:
    if r["result"]["is_anomaly"]:
        print(r["timestamp"], r["result"]["expected_value"])
```

### Latest Point Detection

Only scores the most recent point — useful for real-time streaming where you append one new value and check it:

```python
params = {
    "granularity": "daily",
    "detect_mode": DetectType.LATEST,
    "sensitivity": 95,
}

results = UnivariateAnomalyDetector().predict(None, data, params)
last = results[0]["result"]
print(last["is_anomaly"], last["expected_value"])
```

> **Note:** `UnivariateAnomalyDetector` mutates the `params` dict you pass in (it converts string values like `"daily"` to internal enums). If you are calling `predict` multiple times with the same base params, pass a copy each time: `dict(params)`.

### Response Fields

Each element in the returned list has this structure:

```python
{
    "timestamp": "2023-01-01T00:00:00Z",
    "result": {
        "is_anomaly":          bool,   # True if this point is anomalous
        "is_positive_anomaly": bool,   # True if anomaly is above expected
        "is_negative_anomaly": bool,   # True if anomaly is below expected
        "expected_value":      float,  # What the model expected the value to be
        "upper_margin":        float,  # How far above expected_value is still normal
        "lower_margin":        float,  # How far below expected_value is still normal
        "period":              int,    # Detected seasonal period (0 = no seasonality)
        # Additional fields depending on params:
        "suggested_window":    int,    # Present in LATEST mode
        "severity":            float,  # Present in ENTIRE mode
    }
}
```

---

## Univariate Parameters Reference

### Required

| Parameter | Type | Description |
|---|---|---|
| `granularity` | `str` | Time interval between data points. One of: `"minutely"`, `"hourly"`, `"daily"`, `"weekly"`, `"monthly"`, `"yearly"`. |
| `detect_mode` | `DetectType` | `DetectType.ENTIRE` to score all points, `DetectType.LATEST` to score only the last point. |

### Optional

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sensitivity` | `int` (0–100) | `99` | Controls how tight the anomaly boundary is. Higher = more sensitive = more anomalies flagged. |
| `maxAnomalyRatio` | `float` (0–0.49) | `0.25` | Maximum proportion of points that can be flagged as anomalies. Caps the anomaly rate. |
| `period` | `int` >= 0 | auto-detected | The seasonal period of the series (e.g. `7` for weekly seasonality in daily data). Set to `0` to disable seasonality. |
| `customInterval` | `int` > 0 | `None` | Non-standard interval multiplier. Combined with `granularity` (e.g. `granularity="minutely"` + `customInterval=5` = every 5 minutes). |
| `imputeMode` | `str` | `"auto"` | How to fill missing values. One of: `"auto"`, `"previous"`, `"linear"`, `"zero"`, `"fixed"`, `"notFill"`. |
| `imputeFixedValue` | `int` or `float` | — | Required when `imputeMode="fixed"`. The constant value used to fill gaps. |
| `alpha` | `float` > 0 | — | Significance level for the dynamic threshold and seasonal series detectors. |
| `threshold` | `float` | — | Raw score threshold override for the spectral residual detector. |
| `boundaryVersion` | `str` | `"V1"` | Boundary calculation version. `"V1"`, `"V2"`, or `"V3"`. V2/V3 also return a trend component in results. |
| `detector` | `dict` | auto-selected | Force a specific detection algorithm. See [Choosing a Detection Algorithm](#choosing-a-detection-algorithm) below. |
| `needDetectorId` | `bool` | `False` | Include the internal detector ID in the response. |
| `needFillUpConfirm` | `bool` | `False` | Include confirmation of which points were imputed in the response. |
| `needSpectrumPeriod` | `bool` | `False` | Include the spectrum-based period estimate in the response. |

---

## Choosing a Detection Algorithm

By default the library auto-selects an algorithm. You can force a specific one via the `detector` parameter:

```python
params = {
    "granularity": "daily",
    "detect_mode": DetectType.ENTIRE,
    "detector": {
        "name": "seasonal_series",  # see options below
        "parameters": {}
    }
}
```

| Algorithm name | Best used when |
|---|---|
| `spectral_residual` | Data has no clear seasonality and you expect sudden, sharp spikes |
| `hbos` | Data has no clear seasonality; histogram-based scoring |
| `seasonal_series` | Data has a clear repeating seasonal pattern (e.g. daily, weekly cycles) |
| `dynamic_threshold` | Data has a trend or shifting baseline that changes over time |

---

## Multivariate Anomaly Detection

Multivariate detection finds anomalies that only appear when multiple variables are considered together. It requires a training step.

```python
import pandas as pd
import numpy as np
from anomaly_detector.multivariate.model import MultivariateAnomalyDetector

# Training data: DataFrame with one column per variable, indexed by timestamp
train_df = pd.DataFrame({
    "var_1": [...],
    "var_2": [...],
    "var_3": [...],
})
train_df.index = pd.date_range("2023-01-01", periods=len(train_df), freq="h")

model = MultivariateAnomalyDetector()
model.fit(train_df, params={"sliding_window": 200, "device": "cpu"})

# Evaluation data: same columns, no timestamp index required
eval_df = pd.DataFrame({"var_1": [...], "var_2": [...], "var_3": [...]})
results = model.predict(context=None, data=eval_df)
```

The evaluation data must have more rows than `sliding_window` (default 200).

### Multivariate Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sliding_window` | `int` | `200` | Number of time steps used as input context for each prediction. Eval data must have more rows than this. |
| `device` | `str` | `"cpu"` | Compute device. `"cpu"` or `"cuda"` (if a GPU is available). |
| `train_ratio` | `float` | — | Fraction of data used for training (remainder is validation). |
| `val_ratio` | `float` | — | Fraction of data used for validation. |

### Multivariate Response Fields

Each element in the returned list has this structure:

```python
{
    "index":          int,    # Position in the eval data
    "is_anomaly":     bool,   # True if this time step is anomalous
    "score":          float,  # Raw anomaly score
    "severity":       float,  # 0.0–1.0, where 1.0 is the most severe anomaly
    "interpretation": [       # Per-variable breakdown, sorted by contribution
        {
            "variable_name":     str,
            "contribution_score": float,
            "correlation_changes": {
                "changed_variables": [str, ...],
                "changed_values":    [float, ...]
            }
        },
        ...
    ]
}
```

---

## Worked Example

A complete runnable example covering all three scenarios (entire-series, latest-point, and multivariate) is provided in `demo.py`:

```bash
python demo.py
```

A Jupyter notebook with additional examples is also available:

```bash
jupyter notebook anomaly-detector-sample-code.ipynb
```

---

# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
