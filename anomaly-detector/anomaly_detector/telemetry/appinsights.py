from applicationinsights import TelemetryClient, channel
from telemetry.telemetry_base import BaseTelemetry


class ApplicationInsights(BaseTelemetry):
    def __init__(self, **kwargs):
        super().__init__()
        self.asyncQueue = channel.AsynchronousQueue(channel.AsynchronousSender())
        self.telemetryChannel = channel.TelemetryChannel(queue=self.asyncQueue)
        self.telemetry = TelemetryClient(kwargs['app_insights_apk'], self.telemetryChannel)
        self.telemetry.channel.sender.send_interval_in_milliseconds = 5 * 1000
        self.telemetry.channel.sender.max_queue_item_count = 10000

    def count(self, name, int_val, **tags):
        self.telemetry.track_metric(name, int_val, properties=tags)
        self.telemetry.flush()

    def gauge(self, name, float_val, **tags):
        self.telemetry.track_metric(name, float_val, properties=tags)
        self.telemetry.flush()

    def duration(self, name, time_in_seconds, **tags):
        self.telemetry.track_request(name=name, duration=time_in_seconds * 1000, url=tags['url'],
                                     success=tags['success'], response_code=tags['response_code'],
                                     properties=tags)
        self.telemetry.flush()

    def info(self, msg, *args, **kwargs):
        self.telemetry.track_trace(msg, properties=kwargs, severity='INFO')
        self.telemetry.flush()

    def warning(self, msg, *args, **kwargs):
        self.telemetry.track_trace(msg, properties=kwargs, severity='WARNING')
        self.telemetry.flush()

    def error(self, msg, *args, **kwargs):
        self.telemetry.track_trace(msg, properties=kwargs, severity='ERROR')
        self.telemetry.flush()

    def track_exception(self, traceback, properties):
        self.telemetry.track_exception(tb=traceback, properties=properties)
        self.telemetry.flush()
