import mlflow
import numpy as np
import pandas as pd
from anomaly_detector import MultivariateAnomalyDetector
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def main():
    mlflow.set_experiment("MLflow Quickstart")
    training_data = np.random.randn(10000, 20)
    columns = [f"variable_{i}" for i in range(20)]
    training_data = pd.DataFrame(training_data, columns=columns)
    timestamps = pd.date_range(start="2023-01-03", periods=10000, freq="H")
    training_data["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {"sliding_window": 200, "device": "cpu"}
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        model = MultivariateAnomalyDetector()

        model.fit(training_data, params=params)
        predict_params = {"start_time": "string", "end_time": "string"}
        signature = infer_signature(params=predict_params)

        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="mvad_artifacts",
            registered_model_name="tracking-quickstart",
            signature=signature,
        )
    print(model_info.model_uri)
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    eval_data = np.random.randn(201, 20)
    eval_data[-1, :] += 100
    eval_data = pd.DataFrame(eval_data, columns=columns)
    timestamps = pd.date_range(start="2023-01-03", periods=201, freq="H")
    eval_data["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
    results = loaded_model.predict(
        data=eval_data,
        params={
            "start_time": eval_data["timestamp"].iloc[-5],
            "end_time": eval_data["timestamp"].iloc[-1],
        },
    )
    print(results)


if __name__ == "__main__":
    main()
