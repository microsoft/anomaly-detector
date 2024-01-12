from univariate.handlers.anomaly_detect_request import AnomalyDetectRequest
from univariate.handlers.anomaly_detect_response import LastDetectResponse, EntireDetectResponse
from univariate import AnomalyDetectionModel
from univariate.util import BoundaryVersion


def detect(series, period, ratio, alpha, threshold, granularity, interval, indices,
           fill_up_mode, fixed_value_to_fill, need_trend, detector, need_spectrum_period=False, last_value=None):
    model = AnomalyDetectionModel(series=series, max_anomaly_ratio=ratio, alpha=alpha, threshold=threshold,
                                  indices=indices, fill_up_mode=fill_up_mode, fixed_value_to_fill=fixed_value_to_fill,
                                  granularity=granularity, interval=interval, need_trend=need_trend, need_spectrum_period = need_spectrum_period, detector=detector)
    return model.detect(period=period, last_value=last_value)


def detect_latest(request: AnomalyDetectRequest):
    results, period, spectrum_period, model_id, do_fill_up = detect(series=request.series, period=request.period, ratio=request.ratio,
                                                   alpha=request.alpha,
                                                   threshold=request.threshold,
                                                   last_value=request.series[-1],
                                                   granularity=request.granularity, interval=request.custom_interval,
                                                   indices=request.indices,
                                                   fill_up_mode=request.fill_up_mode,
                                                   fixed_value_to_fill=request.fixed_value_to_fill,
                                                   need_trend=request.boundary_version != BoundaryVersion.V1,
                                                   detector=request.detector,
                                                   need_spectrum_period = request.need_spectrum_period
                                                   )
    
    return LastDetectResponse(results=results, period=period, spectrum_period = spectrum_period, sensitivity=request.sensitivity,
                              granularity=request.granularity, size=len(request.series), model_id=model_id,
                              do_fill_up=do_fill_up, boundary_version=request.boundary_version)


def detect_entire(request: AnomalyDetectRequest):
    results, period, spectrum_period, model_id, do_fill_up = detect(series=request.series, period=request.period, ratio=request.ratio,
                                                   alpha=request.alpha,
                                                   threshold=request.threshold,
                                                   granularity=request.granularity, interval=request.custom_interval,
                                                   indices=request.indices,
                                                   fill_up_mode=request.fill_up_mode,
                                                   fixed_value_to_fill=request.fixed_value_to_fill,
                                                   need_trend=request.boundary_version != BoundaryVersion.V1,
                                                   detector=request.detector,
                                                   need_spectrum_period = request.need_spectrum_period
                                                   )
    
    
    return EntireDetectResponse(results=results, period=period, spectrum_period = spectrum_period, sensitivity=request.sensitivity,
                                size=len(request.series), model_id=model_id, do_fill_up=do_fill_up,
                                boundary_version=request.boundary_version)
