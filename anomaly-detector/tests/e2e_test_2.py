import mlflow
import numpy as np
import pandas as pd
from anomaly_detector import UnivariateAnomalyDetector
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


def main():
    mlflow.set_experiment("MLflow Quickstart 2")

    params = {
        "detect_mode": "entire",
        "granularity": "monthly", 
        "maxAnomalyRatio": 0.25, 
        "sensitivity": 95, 
        "imputeMode": "auto"
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "Univariate Anomaly Detector")

        model = UnivariateAnomalyDetector()

        signature = infer_signature(params=params)
        print(model)
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="uvad_artifacts",
            registered_model_name="tracking-quickstart",
            signature=signature,
        )
    print(model_info.model_uri)
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    eval_data = np.ones(20)
    eval_data[-1] = 0
    eval_data = pd.DataFrame(eval_data, columns=["value"])
    timestamps = pd.date_range(start="1962-01-01", periods=20, freq="M")
    eval_data["timestamp"] = timestamps
    results = loaded_model.predict(
        data=eval_data,
        params=params,
    )
    print(results)


if __name__ == "__main__":
    main()
