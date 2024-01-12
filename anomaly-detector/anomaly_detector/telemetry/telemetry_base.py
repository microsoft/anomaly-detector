class BaseTelemetry(object):
    def __init__(self):
        pass

    def count(self, name, int_val, **tags):
        return

    def gauge(self, name, float_val, **tags):
        return

    def duration(self, name, time_in_seconds, **tags):
        return

    def info(self, msg, *args, **kwargs):
        return

    def warning(self, msg, *args, **kwargs):
        return

    def error(self, msg, *args, **kwargs):
        return

    def track_exception(self, traceback, properties):
        pass
