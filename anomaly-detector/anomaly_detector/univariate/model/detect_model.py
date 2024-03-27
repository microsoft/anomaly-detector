# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from anomaly_detector.univariate.model.dynamic_threshold import dynamic_threshold_detection
from anomaly_detector.univariate.model.seasonal_series import seasonal_series_detection
from anomaly_detector.univariate.resource.error_message import *
from anomaly_detector.univariate.model.series_compete_processor import fill_up_on_demand, period_detection_with_filled_values
from anomaly_detector.univariate.model.spectral_residual_model import spectral_residual_detection
from anomaly_detector.univariate.resource.error_message import NotEnoughPointsForSeasonalData, InvalidDetectorNameWithParameters
from anomaly_detector.univariate.util import Granularity, EPS, FillUpMode
from anomaly_detector.common.exception import AnomalyDetectionRequestError
from anomaly_detector.univariate.util import trend_detection
from anomaly_detector.univariate.util.fields import Value, ExpectedValue, IsAnomaly, IsPositiveAnomaly, IsNegativeAnomaly, Trend, \
    DEFAULT_GRANULARITY_NONE, DEFAULT_MAX_RATIO, DEFAULT_ALPHA, DEFAULT_THRESHOLD, DEFAULT_FILL_UP_MODE
from anomaly_detector.univariate.util.helpers import get_delta, reverse_delta
from anomaly_detector.univariate.filling_up import FillingUpProcess

from statsmodels import robust
from arch.unitroot import ADF, KPSS
import numpy as np
import math
import copy
import traceback


def should_trigger_sr(gran, interval, values):
    # return False
    has_majority = np.abs(robust.mad(values) - 0.0) < EPS
    is_proper_gran = (gran == Granularity.minutely and interval < 60) or \
                     (gran == Granularity.secondly and interval < 60 * 60)
    return is_proper_gran and not has_majority

def should_include_delta(values, max_delta = 2):
    import copy
    new_values = copy.deepcopy(values)
    for delta in range(max_delta + 1):
        # if adfuller(new_values)[1] < 0.05 or kpss(new_values)[1] >= 0.05:
        stationary = False
        try:
            stationary = ADF(new_values).pvalue < 0.05 or KPSS(new_values).pvalue >= 0.05
        except:
            stationary = True
        if stationary:
            ## sequence is stationary
            return delta, new_values
        new_values = get_delta(delta, new_values)
        
    ## cannot get stationary sequence after max_delta difference
    return -1, values


def correct_expectedValue_in_delta(pos, ori, new, delta, values):
    if delta == 2:
        ## get delta1
        new = reverse_delta(values[1]-values[0], delta, new)
    res = copy.deepcopy(ori)
    for p in pos:
        if p == 0: continue
        res[p] = ori[p-1] + new[p]
    return res

def correct_anomaly_direction(results):
    positive_anomaly = np.where((results[IsAnomaly].values == 1) & (results[ExpectedValue].values < results[Value].values))[0]
    results[IsPositiveAnomaly].iloc[positive_anomaly] = True
    results[IsNegativeAnomaly].iloc[positive_anomaly] = False
    negative_anomaly = np.where((results[IsAnomaly].values == 1) & (results[ExpectedValue].values >= results[Value].values))[0]
    results[IsPositiveAnomaly].iloc[negative_anomaly] = False
    results[IsNegativeAnomaly].iloc[negative_anomaly] = True
    return results

def merge_with_delta(results, func, args):
    delta, actual_detection_series_delta = should_include_delta(args["series"])
    if delta <= 0: return results
    args["series"] = actual_detection_series_delta
    num_obs = len(args["series"])
    ad_model = "sr" if func == spectral_residual_detection else "dt"
    max_outliers = max(int(num_obs * args["max_anomaly_ratio"]), 1)
    outlier_ori = np.sum(results[IsAnomaly].values == 1)
    outlier_remainder = max_outliers - outlier_ori
    if outlier_remainder > 0:
        args["max_anomaly_ratio"] = outlier_remainder * 1.0 / num_obs
        differentiate_results, differentiate_model_id = func(**args)
        new_a_p = np.where((results[IsAnomaly].values == 0) & (differentiate_results[IsAnomaly].values == 1))[0]
        for expected_value_column in [Trend, ExpectedValue]:
            if expected_value_column in differentiate_results.columns:
                differentiate_results[expected_value_column] = correct_expectedValue_in_delta(new_a_p, results[expected_value_column].values, differentiate_results[expected_value_column].values, delta, results[Value].values)
                
        differentiate_results[Value] = results[Value].values
        if len(new_a_p): 
            ## correct positive and negative anomaly
            correct_anomaly_direction(differentiate_results)
            
        results.iloc[new_a_p] = differentiate_results.iloc[new_a_p]
    return results

class AnomalyDetectionModel:
    def __init__(self, series=None, max_anomaly_ratio=DEFAULT_MAX_RATIO, alpha=DEFAULT_ALPHA, threshold=DEFAULT_THRESHOLD, granularity=DEFAULT_GRANULARITY_NONE, interval=None, indices=None,
                 fill_up_mode=DEFAULT_FILL_UP_MODE, fixed_value_to_fill=None, need_trend=False, need_spectrum_period=False, detector=dict()):
        self.__values = [float(item["value"]) for item in series]
        self.__max_ratio = max_anomaly_ratio
        self.__alpha = alpha
        self.__threshold = threshold
        self.__gran = granularity
        self.__interval = interval if interval is not None else 1
        self.__majority_ratio = -1
        try:
            histogram_counts = np.histogram(self.__values, bins=20, density=False)[0]
            if np.all(np.isfinite(histogram_counts)):
                self.__majority_ratio = np.max(histogram_counts) / len(self.__values)
        except Exception:
            print("Calculate majority failed with traceback:" + traceback.format_exc())

        self.__has_majority = self.__majority_ratio > 0.6
        self.__indices = indices
        self.__fill_up_mode = fill_up_mode
        self.__fixed_value_to_fill = fixed_value_to_fill
        self.__fill_up = FillingUpProcess(self.__indices, self.__values)
        self.__need_trend = need_trend
        self.__need_spectrum_period = need_spectrum_period
        self.__period_source = None
        self.__detector = detector
        # self.__detect_mode = detect_mode

        # if detect_mode is None:
            # raise ValueError("detect_mode is a required parameter.")

    
    def __refine_max_ratio(self):
        return max((1 - self.__majority_ratio) * self.__max_ratio, min(0.05, self.__max_ratio))

    def __should_trigger_model_selection(self):
        if 'name' not in self.__detector:
            return True
        if 'parameters' not in self.__detector:
            return True

        # indicator that np.histogram may fail
        if self.__detector['name'] == 'hbos' and self.__majority_ratio <= 0:
            return True

    def __detect_without_model_selection(self, period, last_value=None):
        if period is None and 'period' not in self.__detector['parameters']:
            raise AnomalyDetectionRequestError(error_code='InvalidDetector',
                                               error_msg=InvalidDetectorNameWithParameters.format(
                                                   self.__detector['name']
                                               ))

        if 'period' in self.__detector['parameters'] and self.__detector['parameters']['period'] is not None:
            period = self.__detector['parameters']['period']

        # fill up missing values
        do_fill_up = False

        if period > 1:
            if self.__fill_up.missing_ratio > 0.5:
                raise AnomalyDetectionRequestError(error_code='InvalidSeries',
                                                   error_msg=NotEnoughPointsForSeasonalData.format(
                                                       math.ceil(self.__fill_up.all_count * 0.5),
                                                       self.__fill_up.all_count - self.__fill_up.all_missing_count
                                                   ))

        full_values, filled_tags = None, None
        if period > 1 or self.__fill_up_mode == FillUpMode.fixed:
            full_values, filled_tags = fill_up_on_demand(filling_up_process=self.__fill_up, mode=self.__fill_up_mode,
                                                         fixed_value=self.__fixed_value_to_fill, period=period)

        if full_values is not None and filled_tags is not None:
            do_fill_up = True
        print("ADPointCountAfterFillUp", 1, count=self.__fill_up.all_count // 1000)

        actual_detection_series = self.__values if full_values is None else full_values

        if self.__detector['name'] == 'seasonal_series':
            if period <= 0:
                raise AnomalyDetectionRequestError(error_code='InvalidDetector',
                                                   error_msg=InvalidDetectorNameWithParameters.format(
                                                       self.__detector['name']
                                                   ))

            adjust_trend = True if last_value is not None else False
            results, model_id = seasonal_series_detection(
                series=actual_detection_series, period=period,
                alpha=self.__detector['parameters']['alpha'],
                adjust_trend=adjust_trend,
                need_trend=self.__need_trend,
                max_anomaly_ratio=self.__detector['parameters']['maxAnomalyRatio'],
                last_value=last_value)

        elif self.__detector['name'] == 'hbos':
            from anomaly_detector.model.hbos_detection import hbos_detection
            results, model_id = hbos_detection(
                series=actual_detection_series, period=period,
                threshold=self.__detector['parameters']['threshold'],
                outlier_fraction=self.__detector['parameters']['outlierFraction'],
                need_trend=self.__need_trend,
                last_value=last_value
            )
        elif self.__detector['name'] == 'spectral_residual':
            results, model_id = spectral_residual_detection(
                series=actual_detection_series, threshold=self.__detector['parameters']['threshold'],
                max_anomaly_ratio=self.__detector['parameters']['maxAnomalyRatio'],
                need_trend=self.__need_trend,
                last_value=last_value)
        elif self.__detector['name'] == 'dynamic_threshold':
            if period != 0:
                raise AnomalyDetectionRequestError(error_code='InvalidDetector',
                                                   error_msg=InvalidDetectorNameWithParameters.format(
                                                       self.__detector['name']
                                                   ))

            results, model_id = dynamic_threshold_detection(
                series=actual_detection_series, trend_values=trend_detection(actual_detection_series),
                alpha=self.__detector['parameters']['alpha'],
                max_anomaly_ratio=self.__detector['parameters']['maxAnomalyRatio'],
                need_trend=self.__need_trend,
                last_value=last_value)
        else:
            raise AnomalyDetectionRequestError(error_code='InvalidDetector',
                                               error_msg=InvalidDetectorNameWithParameters.format(
                                                   self.__detector['name']
                                               ))

        if full_values is not None and filled_tags is not None:
            results = results.drop(results.index[[i for i in range(len(filled_tags)) if filled_tags[i] is True]])

        return results, period, model_id, do_fill_up

    def __detect_with_model_selection(self, period, last_value=None):
        max_ratio = self.__max_ratio if last_value is None or not self.__has_majority else self.__refine_max_ratio()

        do_fill_up = False

        if period is None:
            if self.__fill_up.missing_ratio > 0.5:
                period = 0
            else:
                period, self.__period_source = period_detection_with_filled_values(filling_up_process=self.__fill_up,
                                                             mode=self.__fill_up_mode,
                                                             fixed_value=self.__fixed_value_to_fill,
                                                             granularity=self.__gran,
                                                             interval=self.__interval, return_period_source = True)

        if period > 1:
            if self.__fill_up.missing_ratio > 0.5:
                raise AnomalyDetectionRequestError(error_code='InvalidSeries',
                                                   error_msg=NotEnoughPointsForSeasonalData.format(
                                                       math.ceil(self.__fill_up.all_count * 0.5),
                                                       self.__fill_up.all_count - self.__fill_up.all_missing_count
                                                   ))

        full_values, filled_tags = None, None
        if period > 1 or self.__fill_up_mode == FillUpMode.fixed or self.__fill_up_mode == FillUpMode.use_last \
                or self.__fill_up_mode == FillUpMode.previous or self.__fill_up_mode == FillUpMode.linear:
            full_values, filled_tags = fill_up_on_demand(filling_up_process=self.__fill_up, mode=self.__fill_up_mode,
                                                         fixed_value=self.__fixed_value_to_fill, period=period)

        if full_values is not None and filled_tags is not None:
            do_fill_up = True

        actual_detection_series = self.__values if full_values is None else full_values

        if period > 1:
            adjust_trend = True if last_value is not None else False
            results, model_id = seasonal_series_detection(
                series=actual_detection_series, period=period, alpha=self.__alpha,
                max_anomaly_ratio=max_ratio, adjust_trend=adjust_trend, need_trend=self.__need_trend, last_value=last_value)

        else:
            if should_trigger_sr(self.__gran, self.__interval, actual_detection_series):
                args = {
                    "series": actual_detection_series,
                    "threshold": self.__threshold,
                    "max_anomaly_ratio": max_ratio,
                    "need_trend": self.__need_trend,
                    "last_value": last_value
                }
                results, model_id = spectral_residual_detection(**args)
                results = merge_with_delta(results, spectral_residual_detection, args) 
            else:
                trend_values = trend_detection(actual_detection_series, period=period)
                args = {
                    "series": actual_detection_series,
                    "trend_values": trend_values,
                    "alpha": self.__alpha,
                    "max_anomaly_ratio": max_ratio,
                    "need_trend": self.__need_trend,
                    "last_value": last_value
                }
                results, model_id = dynamic_threshold_detection(**args)
                results = merge_with_delta(results, dynamic_threshold_detection, args) 
                

        if full_values is not None and filled_tags is not None:
            results = results.drop(results.index[[i for i in range(len(filled_tags)) if filled_tags[i] is True]])

        return results, period, model_id, do_fill_up
    
    def get_spectrum_period(self, period, anomalies, last_value = None):
        if not self.__need_spectrum_period: return None
        if period == 0: return 0
        if self.__period_source == 1: return period

        has_anomaly = False
        if last_value is not None and anomalies[-1]:
            has_anomaly = True
        elif last_value is None:
            for a in anomalies:
                if a:
                    has_anomaly = True
                    break 
        
        if not has_anomaly:
            return None

        return period_detection_with_filled_values(filling_up_process=self.__fill_up, mode=self.__fill_up_mode, fixed_value=self.__fixed_value_to_fill, granularity=self.__gran, interval=self.__interval, skip_simple_detector=True)
            
                        
    def detect(self, period, last_value=None):
        if self.__should_trigger_model_selection():
            results, period, model_id, do_fill_up = self.__detect_with_model_selection(period, last_value)
        else:
            results, period, model_id, do_fill_up = self.__detect_without_model_selection(period, last_value)

        # Adjust the anomaly detection result.
        # If the value and expected value are almost same, then the point is not anomaly.
        anomaly_refine = np.abs(results[Value].values - results[ExpectedValue].values) < EPS
        results.loc[anomaly_refine, IsAnomaly] = False
        results.loc[anomaly_refine, IsNegativeAnomaly] = False
        results.loc[anomaly_refine, IsPositiveAnomaly] = False
        spectrum_period = self.get_spectrum_period(period, results[IsAnomaly].values)

        return results, period, spectrum_period, model_id, do_fill_up
