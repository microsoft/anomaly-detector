# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import mlflow
import numpy as np
import pandas as pd
from anomaly_detector import MultivariateAnomalyDetector
import json
from pprint import pprint

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def main():
    mlflow.set_experiment("MLflow Quickstart")
    training_data = np.random.randn(10000, 20)
    columns = [f"variable_{i}" for i in range(20)]
    training_data = pd.DataFrame(training_data, columns=columns)
    timestamps = pd.date_range(start="2023-01-03", periods=10000, freq="H")
    training_data["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
    training_data = training_data.set_index("timestamp", drop=True)
    params = {"sliding_window": 200, "device": "cpu", "abcd": 10}
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "Basic multivariate anomaly detector")

        model = MultivariateAnomalyDetector()

        model.fit(training_data, params=params)

        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="mvad_artifacts",
            registered_model_name="tracking-quickstart",
        )
    print(model_info.model_uri)
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    eval_data = np.random.randn(201, 20)
    eval_data[-1, :] += 100
    eval_data = pd.DataFrame(eval_data, columns=columns)
    timestamps = pd.date_range(start="2023-01-03", periods=201, freq="H")
    eval_data["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_data = eval_data.set_index("timestamp", drop=True)
    results = loaded_model.predict(data=eval_data)
    pprint(results)
    with open("result1.json", "w") as f:
        json.dump(results, f)

    eval_data = np.random.randn(201, 20)
    eval_data[-1, :] += 100
    eval_data = pd.DataFrame(eval_data, columns=columns)
    results = loaded_model.predict(data=eval_data)
    pprint(results)
    with open("result2.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
