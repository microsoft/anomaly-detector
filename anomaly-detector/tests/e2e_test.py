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
    params = {"sliding_window": 200, "device": "cuda"}
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        model = MultivariateAnomalyDetector()

        model.fit(training_data, params=params)
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="mvad_artifacts",
            registered_model_name="tracking-quickstart",
        )
    print(model_info.model_uri)
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    eval_data = np.random.randn(200, 20)
    is_anomalies, _, _, _, _ = loaded_model.predict(data=eval_data, params={})
    print(is_anomalies)


if __name__ == "__main__":
    main()
