from operator import sub
import os
import json
import unittest
from enum import Enum
import pandas as pd
import sys
sys.path.append(os.path.abspath("anomaly_detector"))
from univariate.util.enum import default_gran_window
from univariate.util.fields import DetectType, IsAnomaly, ExpectedValue, Severity, IsPositiveAnomaly, IsNegativeAnomaly
from univariate.univariate_anomaly_detection import UnivariateAnomalyDetector
from univariate.util.fields import DetectType, IsAnomaly, ExpectedValue, Severity, IsPositiveAnomaly, IsNegativeAnomaly

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
    params_, results_, period_, spectrum_period_, model_id_, do_fill_up_ = detector.predict(data, params)
    
    # compare period
    if 'period' in content['response'] and period_.get('period') != content['response']['period']:
        return False, 'Error period'

    if 'spectrumPeriod' in content['response'] and spectrum_period_.get('spectrum_period') != content['response']['spectrumPeriod']:
        return False, 'Error spectrumPeriod'

    # compare is anomaly
    if 'isAnomaly' in content['response']:
        if len(results_.get('results')[IsAnomaly]) != len(content['response']['isAnomaly']):
            return False, 'isAnomaly length incorrect'
        else:
            for i in range(len(results_.get('results')[IsAnomaly])):
                if results_.get('results')[IsAnomaly].iloc[i] != content['response']['isAnomaly'][i]:
                    return False, 'isAnomaly not match'
    if 'expectedValues' in content['response']:
        tolerant_ratio = 0.05
        for true_exp, new_exp in zip(content['response']['expectedValues'], results_.get('results')[ExpectedValue]):
            upper = true_exp + tolerant_ratio * abs(true_exp)
            lower = true_exp - tolerant_ratio * abs(true_exp)
            if new_exp < lower or new_exp > upper:
                return False, 'expectedValues difference exceed 5 percents'
    
    # if "severity" in content["response"] and "severity" in result:
    if "severity" in content["response"]:
        tolerant_ratio = 0.05
        for true_severity, new_severity in zip(content['response']['severity'], results_.get('results')['Severity']):
            upper = true_severity + tolerant_ratio * abs(true_severity)
            lower = true_severity - tolerant_ratio * abs(true_severity)
            if new_severity < lower or new_severity > upper:
                return False, 'severity difference exceed 5 percents'

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
    params_, results_, period_, spectrum_period_, model_id_, do_fill_up_ = detector.predict(data, params)
    
    # get suggested_window
    if period_.get("period") != 0:
            suggested_window = 4 * period_.get("period") + 1
    elif params_.get('params')['granularity'].name in default_gran_window:
        suggested_window = default_gran_window[params_.get('params')['granularity'].name] + 1
    else:
        suggested_window = 0
    
    # compare period
    if 'period' in content['response'] and period_.get('period') != content['response']['period']:
        return False, 'Error period'

    if 'spectrumPeriod' in content['response'] and spectrum_period_.get('spectrum_period') != content['response']['spectrumPeriod']:
        return False, 'Error spectrumPeriod'

    if 'isAnomaly' in content['response'] and results_.get('results')[IsAnomaly].iloc[-1] != content['response']['isAnomaly']:
        return False, 'isAnomaly not match'

    if 'isPositiveAnomaly' in content['response'] and results_.get('results')[IsPositiveAnomaly].iloc[-1] != content['response'][
        'isPositiveAnomaly']:
        return False, 'isPositiveAnomaly not match'

    if 'isNegativeAnomaly' in content['response'] and results_.get('results')[IsNegativeAnomaly].iloc[-1] != content['response'][
        'isNegativeAnomaly']:
        return False, 'isNegativeAnomaly not match'

    if 'expectedValue' in content['response']:
        tolerant_ratio = 0.05
        true_exp = content['response']['expectedValue']
        upper = true_exp + tolerant_ratio * abs(true_exp)
        lower = true_exp - tolerant_ratio * abs(true_exp)
        if results_.get('results')[ExpectedValue].iloc[-1] < lower or results_.get('results')[ExpectedValue].iloc[-1] > upper:
            return False, 'expectedValue difference exceed 5 percents'

    if 'severity' in content['response']:
        tolerant_ratio = 0.05
        true_severity = content['response']['severity']
        upper = true_severity + tolerant_ratio * abs(true_severity)
        lower = true_severity - tolerant_ratio * abs(true_severity)
        if results_.get('results')[Severity].iloc[-1] < lower or results_.get('results')[Severity].iloc[-1] > upper:
            return False, 'severity difference exceed 5 percents'

    return True, "Success"

class TestFunctional(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_functional(self):
        
        sub_dir = 'tests/cases'
        working_dir = os.getcwd()
        cases = os.listdir(os.path.join(working_dir, sub_dir))
        for i, case in enumerate(cases):
            
            print(f"case {i} {case}")
            with open(os.path.join(working_dir, sub_dir, case), "r") as f:
                content = json.load(f)
            
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