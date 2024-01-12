from univariate.handlers.anomaly_detect_request import AnomalyDetectRequest
from univariate.util.exceptions import AnomalyDetectionRequestError
from univariate.util import BoundaryVersion
from univariate.util.fields import DetectType
from univariate.handlers.parser.request_parser import RequestParser_VX
from univariate.handlers.anomaly_detect_response import LastDetectResponse, EntireDetectResponse
from enum import Enum
import pandas as pd

IsAnomalyField = "isAnomaly"
IsNegativeAnomalyFiled = "isNegativeAnomaly"
IsPositiveAnomalyField = "isPositiveAnomaly"
PeriodField = "period"
ExpectedValueField = "expectedValues"
LastExpectedValue = "expectedValue"
LastUpperMargin = "upperMargin"
LastLowerMargin = "lowerMargin"
UpperMargin = 'upperMargins'
LowerMargin = 'lowerMargins'
SuggestedWindow = 'suggestedWindow'
DetectorId = 'detectorId'
DoFillUp = 'doFillUp'
BoundaryUnit = 'boundaryUnit'
BoundaryUnits = 'boundaryUnits'
AnomalyScore = 'anomalyScore'
Severity = 'severity'
SpectrumPeriod = "spectrumPeriod"





class AnomalyDetectResponse:
    def __init__(self, response, period, granularity, size, level, model_id, custom_interval, do_fill_up):
        self.response = response
        self.period = period
        self.granularity = granularity
        self.size = size
        self.level = level
        self.model_id = model_id
        self.custom_interval = custom_interval
        self.do_fill_up = do_fill_up


def handle_detect_request_VX(request, method=DetectType.ENTIRE, output_severity=False, suppress_score_unit=False):
    
    # input: DataFrame
    data = pd.DataFrame(request["series"])
    # params: dict()
    params = RequestParser_VX(request)
    params['detect_mode'] = method
    # init model
    from univariate.univariate_anomaly_detection import UnivariateAnomalyDetector
    
    detector = UnivariateAnomalyDetector()
    params_, results_, period_, spectrum_period_, model_id_, do_fill_up_ = detector.predict(data, params)
    # print(params_)

    # handle result
    if method == DetectType.LATEST:
        detect_response = LastDetectResponse(results=results_.get("results"), period=period_.get('period'), spectrum_period = spectrum_period_.get('spectrum_period'), sensitivity=params_.get('params')['sensitivity'], granularity=params_.get('params')['granularity'],
                                size=len(request["series"]), model_id=model_id_.get('model_id'), do_fill_up=do_fill_up_.get('do_fill_up'),
                                boundary_version=params_.get('params')['boundaryVersion'])
        response = {
            LastExpectedValue: detect_response.lastExpectedValue,
            LastUpperMargin: detect_response.lastUpperMargin,
            LastLowerMargin: detect_response.lastLowerMargin,
            IsNegativeAnomalyFiled: bool(detect_response.isNegativeAnomaly),
            IsPositiveAnomalyField: bool(detect_response.isPositiveAnomaly),
            IsAnomalyField: bool(detect_response.isAnomaly),
            PeriodField: detect_response.period,
            SuggestedWindow: detect_response.suggestedWindow,
            # Severity: detect_response.severity    # need to upgrade api version to enable.
        }
        if params['needDetectorId']:
            response[DetectorId] = detect_response.model_id.value
        if params['needFillUpConfirm']:
            response[DoFillUp] = detect_response.do_fill_up
        if params['boundaryVersion'] != BoundaryVersion.V1 and not suppress_score_unit:
            response[BoundaryUnit] = detect_response.boundaryUnit
            response[AnomalyScore] = detect_response.anomalyScore
        if output_severity:
            response[Severity] = detect_response.severity
        if params['needSpectrumPeriod']:
            response[SpectrumPeriod] = detect_response.spectrum_period

        return AnomalyDetectResponse(
            response=response,
            level=detect_response.level,
            period=detect_response.period,
            model_id=detect_response.model_id,
            granularity=params_.get('params')['granularity'],
            size=len(request['series']),
            custom_interval=params_.get('params')['customInterval'],
            do_fill_up=detect_response.do_fill_up
        )
    else:
        detect_response = EntireDetectResponse(results=results_.get("results"), period=period_.get('period'), spectrum_period = spectrum_period_.get('spectrum_period'), sensitivity=params_.get('params')['sensitivity'],
                                size=len(request["series"]), model_id=model_id_.get('model_id'), do_fill_up=do_fill_up_.get('do_fill_up'),
                                boundary_version=params_.get('params')['boundaryVersion'])
        response = {
            ExpectedValueField: detect_response.expectedValue,
            UpperMargin: detect_response.upperMargin,
            LowerMargin: detect_response.lowerMargin,
            IsNegativeAnomalyFiled: detect_response.isNegativeAnomaly,
            IsPositiveAnomalyField: detect_response.isPositiveAnomaly,
            IsAnomalyField: detect_response.isAnomaly,
            PeriodField: detect_response.period,
            # Severity: detect_response.severity    # need to upgrade api version to enable.
        }
        if params['needDetectorId']:
            response[DetectorId] = detect_response.model_id.value
        if params['needFillUpConfirm']:
            response[DoFillUp] = detect_response.do_fill_up
        if params['boundaryVersion'] != BoundaryVersion.V1 and not suppress_score_unit:
            response[BoundaryUnits] = detect_response.boundaryUnits
            response[AnomalyScore] = detect_response.anomalyScore
        if output_severity:
            response[Severity] = detect_response.severity
        if params['needSpectrumPeriod']:
            response[SpectrumPeriod] = detect_response.spectrum_period

        return AnomalyDetectResponse(
            response=response,
            level=detect_response.level,
            period=detect_response.period,
            model_id=detect_response.model_id,
            granularity=params_.get('params')['granularity'],
            size=len(request['series']),
            custom_interval=params_.get('params')['customInterval'],
            do_fill_up=detect_response.do_fill_up
        )
