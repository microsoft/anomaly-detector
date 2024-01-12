from telemetry.telemetry_base import BaseTelemetry
from telemetry.statsd_wrapper import StatsdWrapper
from telemetry.fluentd_wrapper import FluentdWrapper


class Geneva(BaseTelemetry):
    _statsd = None
    _logger = None

    def __init__(self, **kwargs):
        super().__init__()
        self._logger = FluentdWrapper(
            host=kwargs['fluentd_host'], port=kwargs['fluentd_port'])
        self._statsd = StatsdWrapper(logger=self._logger,
                                     host=kwargs['statsd_host'], port=kwargs['statsd_port'], mode=kwargs['statsd_mode'])

    def count(self, name, int_val, **tags):
        self._statsd.gauge(name, int_val, **tags)

    def gauge(self, name, float_val, **tags):
        # TODO currently _statsd only support int type counter, should improve later
        self._statsd.gauge(name, int(float_val), **tags)

    def duration(self, name, time_in_seconds, **tags):
        time_in_ms = int(time_in_seconds * 1000)
        self._statsd.timing(name, time_in_ms, **tags)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def track_exception(self, traceback, properties):
        self._logger.track_exception(traceback=traceback, properties=properties)
