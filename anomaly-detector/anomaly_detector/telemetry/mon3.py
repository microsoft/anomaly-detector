from telemetry.telemetry_base import BaseTelemetry
from telemetry.mon3lib.sender import Mon3Sender
from telemetry.mon3lib.logger import Mon3Logger
from telemetry.mon3lib.meter import Mon3Meter


class Mon3(BaseTelemetry):
    def __init__(self, kensho2_profile='INT', mon3_server=None, app=None, service=None):
        super().__init__()
        sender = Mon3Sender(mon3_server, 500, False)
        self.__logger = Mon3Logger(sender, app, service, kensho2_profile)
        self.__meter = Mon3Meter(sender, app, service, kensho2_profile)

    def count(self, name, int_val, **tags):
        self.__meter.count(name, int_val, **tags)

    def gauge(self, name, float_val, **tags):
        self.__meter.gauge(name, float_val, **tags)

    def duration(self, name, time_in_seconds, **tags):
        self.__meter.duration(name, time_in_seconds, **tags)

    def info(self, msg, *args, **kwargs):
        self.__logger.log_info(msg)

    def warning(self, msg, *args, **kwargs):
        self.__logger.log_warn(msg)

    def error(self, msg, *args, **kwargs):
        self.__logger.log_error(msg=msg)

    def track_exception(self, traceback, properties):
        self.__logger.log_error(traceback)
