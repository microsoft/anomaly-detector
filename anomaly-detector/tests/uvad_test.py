from operator import sub
import os
import json
import unittest
from enum import Enum
import sys

sys.path.append(os.path.abspath("anomaly_detector"))
print(sys.path)
# call_local_func=False

call_local_func = True
ad_endpoint = os.environ.get("ad_endpoint", None)
subscription_key = os.environ.get("subscription_key", None)


class APIVersion(Enum):
    V1 = 'v1'
    V1_1_PREVIEW = 'v1.1-preview'

def call_entire_endpoint(content):
    if 'version' in content:
        version = APIVersion(content['version'])
    else:
        version = APIVersion.V1
    print(version)
    if call_local_func:
        from univariate.handlers.service_handler import handle_detect_request_VX, DetectType
        if version == APIVersion.V1_1_PREVIEW:
            result = handle_detect_request_VX(content["request"], DetectType.ENTIRE, output_severity=True,
                                           suppress_score_unit=True).response
        else:
            result = handle_detect_request_VX(content["request"], DetectType.ENTIRE).response
    else:
        pass

    # compare period
    if 'period' in content['response'] and result['period'] != content['response']['period']:
        return False, 'Error period'

    if 'spectrumPeriod' in content['response'] and result['spectrumPeriod'] != content['response']['spectrumPeriod']:
        return False, 'Error spectrumPeriod'

    # compare is anomaly
    if 'isAnomaly' in content['response']:
        if len(result['isAnomaly']) != len(content['response']['isAnomaly']):
            return False, 'isAnomaly length incorrect'
        else:
            for i in range(len(result['isAnomaly'])):
                if result['isAnomaly'][i] != content['response']['isAnomaly'][i]:
                    return False, 'isAnomaly not match'
    if 'expectedValues' in content['response']:
        tolerant_ratio = 0.05
        for true_exp, new_exp in zip(content['response']['expectedValues'], result['expectedValues']):
            upper = true_exp + tolerant_ratio * abs(true_exp)
            lower = true_exp - tolerant_ratio * abs(true_exp)
            if new_exp < lower or new_exp > upper:
                return False, 'expectedValues difference exceed 5 percents'
    
    if "severity" in content["response"] and "severity" in result:
        tolerant_ratio = 0.05
        for true_severity, new_severity in zip(content['response']['severity'], result['severity']):
            upper = true_severity + tolerant_ratio * abs(true_severity)
            lower = true_severity - tolerant_ratio * abs(true_severity)
            if new_severity < lower or new_severity > upper:
                return False, 'severity difference exceed 5 percents'


    return True, "Success"


def call_last_endpoint(content):
    if 'version' in content:
        version = APIVersion(content['version'])
    else:
        version = APIVersion.V1
    print(version)
    if call_local_func:
        from univariate.handlers.service_handler import handle_detect_request_VX, DetectType
        if version == APIVersion.V1_1_PREVIEW:
            result = handle_detect_request_VX(content["request"], DetectType.LATEST, output_severity=True,
                                           suppress_score_unit=True).response
        else:
            result = handle_detect_request_VX(content["request"], DetectType.LATEST).response
    else:
        pass

    # compare period
    if 'period' in content['response'] and result['period'] != content['response']['period']:
        return False, 'Error period'

    if 'spectrumPeriod' in content['response'] and result['spectrumPeriod'] != content['response']['spectrumPeriod']:
        return False, 'Error spectrumPeriod'

    if 'isAnomaly' in content['response'] and result['isAnomaly'] != content['response']['isAnomaly']:
        return False, 'isAnomaly not match'

    if 'isPositiveAnomaly' in content['response'] and result['isPositiveAnomaly'] != content['response'][
        'isPositiveAnomaly']:
        return False, 'isPositiveAnomaly not match'

    if 'isNegativeAnomaly' in content['response'] and result['isNegativeAnomaly'] != content['response'][
        'isNegativeAnomaly']:
        return False, 'isNegativeAnomaly not match'

    if 'expectedValue' in content['response']:
        tolerant_ratio = 0.05
        true_exp = content['response']['expectedValue']
        upper = true_exp + tolerant_ratio * abs(true_exp)
        lower = true_exp - tolerant_ratio * abs(true_exp)
        if result['expectedValue'] < lower or result['expectedValue'] > upper:
            return False, 'expectedValue difference exceed 5 percents'

    if 'severity' in content['response'] and "severity" in result:
        tolerant_ratio = 0.05
        true_severity = content['response']['severity']
        upper = true_severity + tolerant_ratio * abs(true_severity)
        lower = true_severity - tolerant_ratio * abs(true_severity)
        if result['severity'] < lower or result['severity'] > upper:
            return False, 'severity difference exceed 5 percents'

    return True, "Success"

class TestFunctional(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_functional(self):
        sub_dir = 'tests/cases' if call_local_func else 'tests\\cases'
        # sub_dir = 'functional_tests/cases'
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