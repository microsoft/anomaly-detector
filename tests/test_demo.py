# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pickle

import pytest
import pandas as pd
import yaml
import numpy as np
from anomaly_detector.multivariate.model import MultivariateAnomalyDetector
from anomaly_detector.common.exception import DataFormatError, InvalidParameterError
import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_FILE_ROOT = "testCase/testCase_10000_20"


def read_yaml_config(path: str) -> dict:
    with open(path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data


class TestAnomalyDetector:
    def setup_method(self):
        self.model = MultivariateAnomalyDetector()

    def test_invalid_value(self):
        with pytest.raises(DataFormatError, match="Cannot convert values to float"):
            train_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "invalid_value.csv"))
            params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["normal_config"]
            self.model.fit(train_data, params)

    def test_timestamp_not_exist(self):
        train_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "no_timestamp.csv"))
        params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["normal_config"]
        self.model.fit(train_data, params)

    def test_normal_predict(self):
        eval_data = np.random.randn(201, 20)
        eval_data[-1, :] += 100
        columns = [f"variable_{i}" for i in range(20)]
        eval_data = pd.DataFrame(eval_data, columns=columns)
        with open(os.path.join(WORKING_DIR, TEST_FILE_ROOT, 'model.pkl'), 'rb') as f:
            loaded_model = pickle.load(f)
            loaded_model.predict(eval_data)

    def test_inference_data_smaller_than_window(self):
        with pytest.raises(ValueError):
            eval_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "inference_data_smaller_than_window.csv"))
            eval_data = eval_data.set_index("timestamp", drop=True)
            with open(os.path.join(WORKING_DIR, TEST_FILE_ROOT,'model.pkl'), 'rb') as f:
                loaded_model = pickle.load(f)
                loaded_model.predict(eval_data)

    def test_invalid_fillna_config(self):
        with pytest.raises(InvalidParameterError):
            train_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "normal_data.csv"))
            train_data = train_data.set_index("timestamp", drop=True)
            params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["invalid_fillna_config"]
            self.model.fit(train_data, params)

    def test_invalid_fillna_value(self):
        with pytest.raises(InvalidParameterError):
            train_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "normal_data.csv"))
            train_data = train_data.set_index("timestamp", drop=True)
            params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["invalid_fillna_value"]
            self.model.fit(train_data, params)

    def test_invalid_window_value(self):
        with pytest.raises(Exception):
            train_data = pd.read_csv(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "normal_data.csv"))
            train_data = train_data.set_index("timestamp", drop=True)
            params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["invalid_window_value"]
            self.model.fit(train_data, params)

    def test_response(self):
        data_lens = [500, 800, 1000, 2000, 5000]
        var_lens = [10, 20, 30, 50, 100]
        for data_len in data_lens:
            for var_len in var_lens:
                # 0. train model
                training_data = np.random.randn(data_len, var_len)
                columns = [f"variable_{i}" for i in range(var_len)]
                training_data = pd.DataFrame(training_data, columns=columns)
                timestamps = pd.date_range(start="2023-01-03", periods=data_len, freq="h")
                training_data["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
                training_data = training_data.set_index("timestamp", drop=True)

                params = read_yaml_config(os.path.join(WORKING_DIR, TEST_FILE_ROOT, "config.yaml"))["normal_config"]
                model = MultivariateAnomalyDetector()
                print(var_len, data_len)
                model.fit(training_data, params=params)

                eval_data = np.random.randn(210, var_len)
                eval_data[-1, :] += 100
                columns = [f"variable_{i}" for i in range(var_len)]
                eval_data = pd.DataFrame(eval_data, columns=columns)

                response = model.predict(context=None, data=eval_data)

                # 1. Check for missing fields
                for result in response:
                    if list(result.keys()) != ["index", "is_anomaly", "score", "severity", "interpretation"]:
                        raise Exception("missing fields in output")

                    for i in result["interpretation"]:
                        if list(i.keys()) != ["variable_name", "contribution_score", "correlation_changes"]:
                            raise Exception("missing fields in output")
                        if list(i["correlation_changes"].keys()) != ["changed_variables", "changed_values"]:
                            raise Exception("missing fields in output")

                # 2. Check to see if it is sorted by contribution_score
                for result in response:
                    contribution_score = [i["contribution_score"] for i in result["interpretation"]]
                    for i in range(1, len(contribution_score)):
                        if contribution_score[i] > contribution_score[i - 1]:
                            raise Exception("incorrect sorting by contribution score")

                # 3. Check severity is 0 if is anomaly is false
                for result in response:
                    if result["is_anomaly"] and result["severity"] == 0:
                        raise Exception("wrong severity")

                # 4. Check changed values are sorted by absolute value
                for result in response:
                    changed_values_of_all_vars = [i["correlation_changes"]["changed_values"] for i in
                                                  result["interpretation"]]
                    for changed_values in changed_values_of_all_vars:
                        for i in range(1, len(changed_values)):
                            if abs(changed_values[i]) > abs(changed_values[i - 1]):
                                raise Exception("incorrect sorting by changed values")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
