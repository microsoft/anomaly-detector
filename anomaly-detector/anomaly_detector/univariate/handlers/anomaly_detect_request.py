class AnomalyDetectRequest(object):
    def __init__(self, series, period, granularity, ratio, alpha, sensitivity, threshold, custom_interval, indices,
                 fill_up_mode, fixed_value_to_fill, boundary_version, detector, need_spectrum_period=False):
        self.series = series
        self.period = period
        self.granularity = granularity
        self.ratio = ratio
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.custom_interval = custom_interval if custom_interval is not None else 1
        self.indices = indices
        self.fill_up_mode = fill_up_mode
        self.fixed_value_to_fill = fixed_value_to_fill
        self.boundary_version = boundary_version
        self.need_spectrum_period = need_spectrum_period
        self.detector = detector
