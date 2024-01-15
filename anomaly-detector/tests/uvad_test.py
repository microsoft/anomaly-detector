from operator import sub
import os
import json
import unittest
from enum import Enum
import pandas as pd
import sys
sys.path.append(os.path.abspath("anomaly_detector"))
from univariate.util.enum import default_gran_window
from univariate.util.fields import DetectType
from univariate.univariate_anomaly_detection import UnivariateAnomalyDetector

def call_entire_endpoint(content):
    
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
    
    # refine result
    results_ = results_.get("results").sort_index()
    expected_value, upper_margin, lower_margin, anomaly_neg, anomaly_pos, anomaly, severity, boundary_units, anomaly_scores \
            = detector.get_margins(results_, params_.get('params')['sensitivity'], model_id_.get('model_id'), params_.get('params')['boundaryVersion'], False)

    # compare period
    if 'period' in content['response'] and period_.get('period') != content['response']['period']:
        return False, 'Error period'

    if 'spectrumPeriod' in content['response'] and spectrum_period_.get('spectrum_period') != content['response']['spectrumPeriod']:
        return False, 'Error spectrumPeriod'

    # compare is anomaly
    if 'isAnomaly' in content['response']:
        if len(anomaly) != len(content['response']['isAnomaly']):
            return False, 'isAnomaly length incorrect'
        else:
            for i in range(len(anomaly)):
                if anomaly[i] != content['response']['isAnomaly'][i]:
                    return False, 'isAnomaly not match'
    if 'expectedValues' in content['response']:
        tolerant_ratio = 0.05
        for true_exp, new_exp in zip(content['response']['expectedValues'], expected_value):
            upper = true_exp + tolerant_ratio * abs(true_exp)
            lower = true_exp - tolerant_ratio * abs(true_exp)
            if new_exp < lower or new_exp > upper:
                return False, 'expectedValues difference exceed 5 percents'
    
    # if "severity" in content["response"] and "severity" in result:
    if "severity" in content["response"]:
        tolerant_ratio = 0.05
        for true_severity, new_severity in zip(content['response']['severity'], severity):
            upper = true_severity + tolerant_ratio * abs(true_severity)
            lower = true_severity - tolerant_ratio * abs(true_severity)
            if new_severity < lower or new_severity > upper:
                return False, 'severity difference exceed 5 percents'

    return True, "Success"


def call_last_endpoint(content):
    
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
    
    # refine result
    expected_value, upper_margin, lower_margin, anomaly_neg, anomaly_pos, anomaly, severity, boundary_units, anomaly_scores \
            = detector.get_margins(results_.get("results"), params_.get('params')['sensitivity'], model_id_.get('model_id'), params_.get('params')['boundaryVersion'], last=True)
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

    if 'isAnomaly' in content['response'] and anomaly != content['response']['isAnomaly']:
        return False, 'isAnomaly not match'

    if 'isPositiveAnomaly' in content['response'] and anomaly_pos != content['response'][
        'isPositiveAnomaly']:
        return False, 'isPositiveAnomaly not match'

    if 'isNegativeAnomaly' in content['response'] and anomaly_neg != content['response'][
        'isNegativeAnomaly']:
        return False, 'isNegativeAnomaly not match'

    if 'expectedValue' in content['response']:
        tolerant_ratio = 0.05
        true_exp = content['response']['expectedValue']
        upper = true_exp + tolerant_ratio * abs(true_exp)
        lower = true_exp - tolerant_ratio * abs(true_exp)
        if expected_value < lower or expected_value > upper:
            return False, 'expectedValue difference exceed 5 percents'

    if 'severity' in content['response']:
        tolerant_ratio = 0.05
        true_severity = content['response']['severity']
        upper = true_severity + tolerant_ratio * abs(true_severity)
        lower = true_severity - tolerant_ratio * abs(true_severity)
        if severity < lower or severity > upper:
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
                ret = call_entire_endpoint(content)
                try:
                    self.assertTrue(ret[0])
                except Exception as e:
                    self.verificationErrors.append(f"{case} {ret[1]} {str(e)}")
            
            elif content['type'] == 'last':
                ret = call_last_endpoint(content)
                try:
                    self.assertTrue(ret[0])
                except Exception as e:
                    self.verificationErrors.append(f"{case} {ret[1]} {str(e)}")

if __name__ == '__main__':
    unittest.main()