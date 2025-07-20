# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from operator import sub
import os
import json
import unittest
from enum import Enum
import pandas as pd
from collections import Counter
from anomaly_detector.univariate.util.enum import default_gran_window
from anomaly_detector.univariate.util.fields import DetectType, IsAnomaly, ExpectedValue, Severity, IsPositiveAnomaly, IsNegativeAnomaly, EPS
from anomaly_detector.univariate.univariate_anomaly_detection import UnivariateAnomalyDetector
from anomaly_detector.univariate.util.fields import DetectType, IsAnomaly, ExpectedValue, Severity, IsPositiveAnomaly, IsNegativeAnomaly, Period
def call_entire(content):
    
    # input: DataFrame
    data = pd.DataFrame(content["request"]["series"])
    # params: dict()
    params = {key: value for key, value in content["request"].items() if key != "series"}
    # add detect_mode
    params['detect_mode'] = DetectType.ENTIRE
    # init model
    detector = UnivariateAnomalyDetector()
    # predict
    response = detector.predict(None, data, params)
    # compare period
    if 'period' in content['response'] and response[0].get('result')[Period] != content['response']['period']:
        return False, 'Error period'

    # compare is anomaly
    if 'isAnomaly' in content['response']:
        if len(response) != len(content['response']['isAnomaly']):
            return False, 'isAnomaly length incorrect'
        else:
            for i in range(len(response)):
                if response[i]['result'][IsAnomaly] != content['response']['isAnomaly'][i]:
                    return False, 'isAnomaly not match'
    if 'expectedValues' in content['response']:
        tolerant_ratio = 0.05        
        for true_exp, new_exp in zip(content['response']['expectedValues'], [item["result"][ExpectedValue] for item in response]):
            upper = true_exp + max(EPS, tolerant_ratio * abs(true_exp))
            lower = true_exp - max(EPS, tolerant_ratio * abs(true_exp))
            if new_exp < lower or new_exp > upper:
                return False, 'expectedValues difference exceed 5 percents'
    
    return True, "Success"


def call_last(content):
    
    # input: DataFrame
    data = pd.DataFrame(content["request"]["series"])
    # params: dict()
    params = {key: value for key, value in content["request"].items() if key != "series"}
    # add detect_mode
    params['detect_mode'] = DetectType.LATEST
    # init model
    detector = UnivariateAnomalyDetector()
    # predict
    response = detector.predict(None, data, params)
    # compare period
    if 'period' in content['response'] and response[0].get('result')[Period] != content['response']['period']:
        return False, 'Error period'

    if 'isAnomaly' in content['response'] and response[0].get('result')[IsAnomaly] != content['response']['isAnomaly']:
        return False, 'isAnomaly not match'

    if 'isPositiveAnomaly' in content['response'] and response[0].get('result')[IsPositiveAnomaly] != content['response'][
        'isPositiveAnomaly']:
        return False, 'isPositiveAnomaly not match'

    if 'isNegativeAnomaly' in content['response'] and response[0].get('result')[IsNegativeAnomaly] != content['response'][
        'isNegativeAnomaly']:
        return False, 'isNegativeAnomaly not match'

    if 'expectedValue' in content['response']:
        tolerant_ratio = 0.05
        true_exp = content['response']['expectedValue']
        upper = true_exp + max(EPS, tolerant_ratio * abs(true_exp))
        lower = true_exp - max(EPS, tolerant_ratio * abs(true_exp))
        if response[0].get('result')[ExpectedValue] < lower or response[0].get('result')[ExpectedValue] > upper:
            return False, 'expectedValue difference exceed 5 percents'

    return True, "Success"


class TestFunctional(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):

        self.assertEqual([], self.verificationErrors)

    def test_functional(self):
        
        sub_dir = 'cases'
        working_dir = os.path.dirname(os.path.realpath(__file__))
        cases = os.listdir(os.path.join(working_dir, sub_dir))
        for i, case in enumerate(cases):
            if case.startswith("._"):
                continue
            print(f"case {i} {case}")
            file_path = os.path.join(working_dir, sub_dir, case)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
                if not raw.strip():
                    self.fail(f"Empty JSON file: {file_path}")
                try:
                    content = json.loads(raw)   
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {file_path}: {e}")
            
            if content['type'] == 'entire':
                ret = call_entire(content)
                try:
                    self.assertTrue(ret[0])
                except Exception as e:
                    self.verificationErrors.append(f"{case} {ret[1]} {str(e)}")
            
            elif content['type'] == 'last':
                ret = call_last(content)
                try:
                    self.assertTrue(ret[0])
                except Exception as e:
                    self.verificationErrors.append(f"{case} {ret[1]} {str(e)}")


if __name__ == '__main__':
    unittest.main()
