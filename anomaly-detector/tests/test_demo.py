import pickle

import mlflow
import pytest
import pandas as pd
import yaml

from anomaly_detector import MultivariateAnomalyDetector
from anomaly_detector.common.exception import DataFormatError, InvalidParameterError

MODEL_URL = "runs:/bcfb06073804457388cb53b4107a8792/mvad_artifacts"
TEST_FILE_ROOT = "testCase/testCase_10000_20/"
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def read_yaml_config(path: str) -> dict:
    with open(path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data


class TestAnomalyDetector:
    def setup_method(self):
        self.model = MultivariateAnomalyDetector()

    def test_invalid_value(self):
        with pytest.raises(DataFormatError, match="Cannot convert values to float"):
            train_data = pd.read_csv(TEST_FILE_ROOT + "invalid_value.csv")
            params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["normal_config"]
            self.model.fit(train_data, params)

    def test_invalid_timestamp(self):
        with pytest.raises(ValueError, match="doesn't match format"):
            train_data = pd.read_csv(TEST_FILE_ROOT + "invalid_timestamp.csv")
            params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["normal_config"]
            self.model.fit(train_data, params)

    def test_timestamp_not_exist(self):
        with pytest.raises(DataFormatError, match="timestamp"):
            train_data = pd.read_csv(TEST_FILE_ROOT + "no_timestamp.csv")
            params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["normal_config"]
            self.model.fit(train_data, params)

    def test_bigger_start_time(self):
        with pytest.raises(InvalidParameterError, match="start_time cannot be later than end_time"):
            train_data = pd.read_csv(TEST_FILE_ROOT + "normal_data.csv")
            params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["bigger_start_time"]
            self.model.fit(train_data, params)

    def test_inference_data_smaller_than_window(self):
        with pytest.raises((TypeError, RuntimeError)):
            eval_data = pd.read_csv(TEST_FILE_ROOT + "inference_data_smaller_than_window.csv")
            params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["normal_predict_config"]
            with open(TEST_FILE_ROOT + 'model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
                loaded_model.predict(eval_data, params)

    def test_inference_start_end_time(self):
        with pytest.raises(InvalidParameterError, match="Cannot convert start_time or end_time"):
            eval_data = pd.read_csv("testCase/testCase_10000_20/normal_data.csv")
            params = read_yaml_config("testCase/testCase_10000_20/config.yaml")["no_predict_start_time"]
            with open(TEST_FILE_ROOT + 'model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
                loaded_model.predict(eval_data, params)


if __name__ == "__main__":
    pytest.main(["-s"])
