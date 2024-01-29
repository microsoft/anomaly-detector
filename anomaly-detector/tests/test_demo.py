import pickle
import pytest
import pandas as pd
import yaml
import numpy as np
from anomaly_detector.multivariate.model import MultivariateAnomalyDetector
from anomaly_detector.common.exception import DataFormatError, InvalidParameterError

TEST_FILE_ROOT = "testCase/testCase_10000_20/"


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

    def test_timestamp_not_exist(self):
        train_data = pd.read_csv(TEST_FILE_ROOT + "no_timestamp.csv")
        params = read_yaml_config(TEST_FILE_ROOT + "config.yaml")["normal_config"]
        self.model.fit(train_data, params)

    def test_normal_predict(self):
        eval_data = np.random.randn(201, 20)
        eval_data[-1, :] += 100
        columns = [f"variable_{i}" for i in range(20)]
        eval_data = pd.DataFrame(eval_data, columns=columns)
        with open(TEST_FILE_ROOT + 'model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
            loaded_model.predict(eval_data)

    def test_inference_data_smaller_than_window(self):
        with pytest.raises((TypeError, RuntimeError)):
            eval_data = pd.read_csv(TEST_FILE_ROOT + "inference_data_smaller_than_window.csv")
            with open(TEST_FILE_ROOT + 'model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
                loaded_model.predict(eval_data)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
