"""
Demo app for the anomaly-detector library.

Covers two scenarios:
  1. Univariate  – detect anomalies across an entire daily time series
  2. Univariate  – detect whether the latest point is anomalous
  3. Multivariate – train a model and score a short evaluation window

Run:
    python demo.py
"""

import numpy as np
import pandas as pd
from anomaly_detector.univariate.univariate_anomaly_detection import UnivariateAnomalyDetector
from anomaly_detector.univariate.util.fields import DetectType, IsAnomaly, ExpectedValue, UpperMargin, LowerMargin
from anomaly_detector.multivariate.model import MultivariateAnomalyDetector

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build synthetic univariate data
#    200 daily points that stay around 100, with a few injected spikes.
# ──────────────────────────────────────────────────────────────────────────────

def make_univariate_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="D")
    values = 100 + rng.normal(0, 1, n)

    # inject obvious anomalies
    spike_indices = [50, 100, 150, 180]
    for i in spike_indices:
        values[i] += rng.choice([-1, 1]) * 40   # ±40 spike

    return pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "value": values,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 2. Scenario A – entire-series detection
# ──────────────────────────────────────────────────────────────────────────────

def demo_entire(data: pd.DataFrame) -> None:
    section("Univariate – Entire Series Detection")

    detector = UnivariateAnomalyDetector()
    params = {
        "granularity": "daily",
        "sensitivity": 95,
        "detect_mode": DetectType.ENTIRE,
    }

    results = detector.predict(None, data, params)

    value_by_ts = dict(zip(data["timestamp"], data["value"]))
    anomalies = [
        (r["timestamp"], value_by_ts[r["timestamp"]], r["result"][ExpectedValue])
        for r in results
        if r["result"][IsAnomaly]
    ]

    print(f"Series length : {len(results)} points")
    print(f"Anomalies found: {len(anomalies)}")
    print()
    print(f"  {'Timestamp':<30} {'Actual':>10} {'Expected':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for ts, actual, expected in anomalies:
        print(f"  {str(ts):<30} {actual:>10.2f} {expected:>10.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Scenario B – latest-point detection
# ──────────────────────────────────────────────────────────────────────────────

def demo_latest(data: pd.DataFrame) -> None:
    section("Univariate – Latest Point Detection")

    base_params = {"granularity": "daily", "sensitivity": 95, "detect_mode": DetectType.LATEST}

    # Normal last point
    normal_data = data.copy()
    normal_data.iloc[-1, normal_data.columns.get_loc("value")] = 100.5
    results = UnivariateAnomalyDetector().predict(None, normal_data, dict(base_params))
    last = results[0]["result"]
    print(f"Normal last point  (value=100.5) -> is_anomaly={last[IsAnomaly]}, "
          f"expected={last[ExpectedValue]:.2f}")

    # Anomalous last point
    spike_data = data.copy()
    spike_data.iloc[-1, spike_data.columns.get_loc("value")] = 250.0
    results = UnivariateAnomalyDetector().predict(None, spike_data, dict(base_params))
    last = results[0]["result"]
    print(f"Spiked last point  (value=250.0) -> is_anomaly={last[IsAnomaly]}, "
          f"expected={last[ExpectedValue]:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Scenario C – multivariate detection
#    Requires training first; uses random data to keep the demo self-contained.
# ──────────────────────────────────────────────────────────────────────────────

def demo_multivariate() -> None:
    section("Multivariate Detection")

    N_TRAIN = 600
    N_EVAL = 210        # must be > sliding_window (default 200)
    N_VARS = 5
    SLIDING_WINDOW = 200

    rng = np.random.default_rng(0)
    columns = [f"var_{i}" for i in range(N_VARS)]

    # Training data – correlated Gaussian noise
    train_values = rng.standard_normal((N_TRAIN, N_VARS))
    timestamps = pd.date_range("2023-01-01", periods=N_TRAIN, freq="h")
    train_df = pd.DataFrame(train_values, columns=columns)
    train_df["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
    train_df = train_df.set_index("timestamp")

    print("Training model …")
    model = MultivariateAnomalyDetector()
    model.fit(train_df, params={"sliding_window": SLIDING_WINDOW, "device": "cpu"})
    print("Training complete.")

    # Evaluation data – last row is a massive spike (obvious anomaly)
    eval_values = rng.standard_normal((N_EVAL, N_VARS))
    eval_values[-1, :] += 50           # inject anomaly on the last point
    eval_df = pd.DataFrame(eval_values, columns=columns)

    results = model.predict(context=None, data=eval_df)

    anomalies = [r for r in results if r["is_anomaly"]]
    print(f"\nEval points : {len(results)}")
    print(f"Anomalies   : {len(anomalies)}")

    if anomalies:
        print()
        print(f"  {'Index':>6} {'Score':>8} {'Severity':>9}  Top contributing variable")
        print(f"  {'-'*6} {'-'*8} {'-'*9}  {'-'*26}")
        for r in anomalies[-5:]:   # show at most the last 5 anomalies
            top_var = r["interpretation"][0]["variable_name"] if r["interpretation"] else "—"
            print(f"  {r['index']:>6} {r['score']:>8.4f} {r['severity']:>9.4f}  {top_var}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = make_univariate_data()

    section("Univariate Input Data (200 daily points, ~100 baseline)")
    print(data.to_string(index=False))

    demo_entire(data)
    demo_latest(data)
    demo_multivariate()
    print("\nDone.")
